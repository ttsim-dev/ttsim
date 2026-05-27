from __future__ import annotations

from types import ModuleType
from typing import Any, Literal

import dags.tree as dt
import numpy
import pandas as pd

from ttsim.interface_dag_elements.processed_data import (
    _canonicalize_input_dtype,
)
from ttsim.typing import (
    FlatData,
    NestedData,
    NestedInputsMapper,
    NestedResults,
    NestedStrings,
    QNameData,
)


def _get_p_id_index(data_with_p_id: NestedData | QNameData | FlatData) -> pd.Index:
    if "p_id" in data_with_p_id:
        return pd.Index(data_with_p_id["p_id"], name="p_id")  # ty: ignore[invalid-argument-type]
    if ("p_id",) in data_with_p_id:
        return pd.Index(data_with_p_id[("p_id",)], name="p_id")  # ty: ignore[invalid-argument-type]
    raise ValueError("No p_id found in data_with_p_id")


def nested_data_to_df_with_nested_columns(
    nested_data_to_convert: NestedResults,
    index: pd.Index,
) -> pd.DataFrame:
    """Convert a nested data structure to a DataFrame with a MultiIndex for the columns.

    Args:
        nested_data_to_convert:
            A nested data structure.
        index:
            The index to use for the DataFrame.

    Returns:
        A DataFrame.
    """
    flat_data_to_convert = dt.flatten_to_tree_paths(nested_data_to_convert)

    return pd.DataFrame(
        flat_data_to_convert,
        index=index,
    )


def nested_data_to_df_with_qname_columns(
    nested_data_to_convert: NestedResults,
    index: pd.Index,
) -> pd.DataFrame:
    """Convert a nested data structure to a DataFrame with qname-string columns.

    Each output column is named with the qualified-name string for its tree path
    (parts joined by `__`).

    Args:
        nested_data_to_convert:
            A nested data structure.
        index:
            The index to use for the DataFrame.

    Returns:
        A DataFrame with qname-string columns.
    """
    flat_data_to_convert = dt.flatten_to_qnames(nested_data_to_convert)

    return pd.DataFrame(
        flat_data_to_convert,
        index=index,
    )


def nested_data_to_df_with_mapped_columns(
    nested_data_to_convert: NestedResults,
    nested_outputs_df_column_names: NestedStrings,
    data_with_p_id: NestedData | QNameData | FlatData,
) -> pd.DataFrame:
    """Convert a nested data structure to a DataFrame with columns specified in
    `nested_outputs_df_column_names`.

    Args:
        nested_data_to_convert:
            A nested data structure.
        nested_outputs_df_column_names:
            A tree that maps paths (sequence of keys) to data columns names.
        data_with_p_id:
            Some data structure with a "p_id" column.

    Returns:
        A DataFrame.
    """
    flat_data_to_convert = dt.flatten_to_tree_paths(nested_data_to_convert)
    flat_df_columns = dt.flatten_to_tree_paths(nested_outputs_df_column_names)
    p_id_index = _get_p_id_index(data_with_p_id)

    return pd.DataFrame(
        {flat_df_columns[path]: data for path, data in flat_data_to_convert.items()},
        index=p_id_index,
    )


def df_with_mapped_columns_to_flat_data(
    mapper: NestedInputsMapper,
    df: pd.DataFrame,
    backend: Literal["numpy", "jax"],
    xnp: ModuleType,
) -> FlatData:
    """Transform a pandas DataFrame to a flattened data structure.

    Args:
        mapper: A nested dictionary that defines the structure of the inputs tree. The
            elements of the tree paths are strings. Leaves can be:

            - Strings that reference column names in the DataFrame.
            - Numeric or boolean values (which will be broadcasted to match the
              DataFrame length)
        df: The pandas DataFrame containing the source data.
        backend: Backend to use for array creation.
        xnp: The numpy-compatible module.

    Returns:
        A flattened data structure containing the data organized according to the
        mapping definition.

    Examples:
        >>> df = pd.DataFrame({
        ...     "a": [1, 2, 3],
        ...     "b": [4, 5, 6],
        ...     "c": [7, 8, 9],
        ... })
        >>> inputs_tree_to_df_columns = {
        ...     "n1": {
        ...         "n2": "a",
        ...         "n3": "b",
        ...     },
        ...     "n4": 3,
        ... }
        >>> result = create_data_tree(
        ...     inputs_tree_to_df_columns=inputs_tree_to_df_columns,
        ...     df=df,
        ... )
        >>> result
        {
            ("n1", "n2"): np.array([1, 2, 3]),
            ("n1", "n3"): np.array([4, 5, 6]),
            ("n4",): np.array([3, 3, 3]),
        }

    """
    path_to_array = {}
    for path, mapper_value in dt.flatten_to_tree_paths(mapper).items():
        # Use numpy for array creation regardless of backend for performance reasons,
        # see #34.
        if numpy.isscalar(mapper_value) and not isinstance(mapper_value, str):
            numpy_array = numpy.full(len(df), mapper_value)
        else:
            numpy_array = _canonicalize_input_dtype(
                arr=df[mapper_value],
                xnp=numpy,
                column_label=dt.qname_from_tree_path(path),
            )

        # Keep NA-containing int/bool columns as numpy object arrays so the
        # `input_data_has_int_or_bool_missing_values` fail-if can report them;
        # JAX's `asarray` would crash on `pd.NA` before the fail-if runs.
        if backend == "jax" and numpy_array.dtype != numpy.dtype(object):
            path_to_array[path] = xnp.asarray(numpy_array)
        else:
            path_to_array[path] = numpy_array

    return path_to_array


def df_with_nested_columns_to_flat_data(
    df: pd.DataFrame,
    backend: Literal["numpy", "jax"],
    xnp: ModuleType,
) -> FlatData:
    """Convert a DataFrame with nested columns to a flattened data structure.

    Args:
        df:
            The pandas DataFrame with nested columns.
        xnp:
            The numpy module.

    Returns:
        A flattened data structure.

    Examples:
        >>> df = pd.DataFrame({("a", "b"): [1, 2, 3], ("c",): [4, 5, 6]})
        >>> result = df_with_nested_columns_to_flat_data(df, xnp=np)
        >>> result
        {("a", "b"): np.array([1, 2, 3]), ("c",): np.array([4, 5, 6])}
    """
    result = {}
    # Read each column as a `pd.Series` so the canonicalizer sees the
    # original (possibly nullable / pyarrow) dtype and applies the correct
    # NA-aware conversion.
    for raw_key in df.columns:
        clean_key = _remove_nan_from_keys(
            raw_key if isinstance(raw_key, tuple) else (raw_key,)
        )
        numpy_array = _canonicalize_input_dtype(
            arr=df[raw_key],
            xnp=numpy,
            column_label=dt.qname_from_tree_path(clean_key),
        )
        # See comment in `df_with_mapped_columns_to_flat_data` re: object dtype.
        if backend == "jax" and numpy_array.dtype != numpy.dtype(object):
            result[clean_key] = xnp.asarray(numpy_array)
        else:
            result[clean_key] = numpy_array

    return result


def df_with_qname_columns_to_flat_data(
    df: pd.DataFrame,
    backend: Literal["numpy", "jax"],
    xnp: ModuleType,
) -> FlatData:
    """Convert a DataFrame whose flat column index holds qname strings.

    Each column name is split on ``__`` to recover the tree path; everything
    else mirrors `df_with_nested_columns_to_flat_data`.

    Args:
        df:
            The pandas DataFrame with qname-string columns.
        backend:
            Backend to use for array creation.
        xnp:
            The numpy-compatible module.

    Returns:
        A flattened data structure keyed by tree paths.

    Examples:
        >>> df = pd.DataFrame({"a__b": [1, 2, 3], "c": [4, 5, 6]})
        >>> result = df_with_qname_columns_to_flat_data(df, "numpy", np)
        >>> result
        {("a", "b"): np.array([1, 2, 3]), ("c",): np.array([4, 5, 6])}
    """
    result = {}
    for raw_key in df.columns:
        clean_key = dt.tree_path_from_qname(raw_key)
        numpy_array = _canonicalize_input_dtype(df[raw_key], numpy)
        # See comment in `df_with_mapped_columns_to_flat_data` re: object dtype.
        if backend == "jax" and numpy_array.dtype != numpy.dtype(object):
            result[clean_key] = xnp.asarray(numpy_array)
        else:
            result[clean_key] = numpy_array

    return result


def _remove_nan_from_keys(path: tuple[str | Any, ...]) -> tuple[str, ...]:
    """Remove nan string from string tuples."""
    return tuple(el for el in path if not pd.isna(el))
