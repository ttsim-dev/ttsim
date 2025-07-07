from __future__ import annotations

from typing import TYPE_CHECKING, Any

import dags.tree as dt
import pandas as pd

if TYPE_CHECKING:
    from types import ModuleType

    from ttsim.interface_dag_elements.typing import (
        FlatData,
        NestedData,
        NestedInputsMapper,
        NestedStrings,
        QNameData,
    )


def _get_p_id_index(data_with_p_id: NestedData | QNameData | FlatData) -> pd.Index:
    if "p_id" in data_with_p_id:
        return pd.Index(data_with_p_id["p_id"], name="p_id")
    if ("p_id",) in data_with_p_id:
        return pd.Index(data_with_p_id[("p_id",)], name="p_id")
    raise ValueError("No p_id found in data_with_p_id")


def nested_data_to_df_with_nested_columns(
    nested_data_to_convert: NestedData,
    data_with_p_id: NestedData | QNameData | FlatData,
) -> pd.DataFrame:
    """Convert a nested data structure to a DataFrame.

    Args:
        nested_data_to_convert:
            A nested data structure.
        data_with_p_id:
            Some data structure with a "p_id" column.

    Returns
    -------
        A DataFrame.
    """
    flat_data_to_convert = dt.flatten_to_tree_paths(nested_data_to_convert)

    p_id_index = _get_p_id_index(data_with_p_id)

    return pd.DataFrame(
        flat_data_to_convert,
        index=p_id_index,
    )


def nested_data_to_df_with_mapped_columns(
    nested_data_to_convert: NestedData,
    nested_outputs_df_column_names: NestedStrings,
    data_with_p_id: NestedData | QNameData | FlatData,
) -> pd.DataFrame:
    """Convert a nested data structure to a DataFrame.

    Args:
        nested_data_to_convert:
            A nested data structure.
        nested_outputs_df_column_names:
            A tree that maps paths (sequence of keys) to data columns names.
        data_with_p_id:
            Some data structure with a "p_id" column.

    Returns
    -------
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
    xnp: ModuleType,
) -> FlatData:
    """Transform a pandas DataFrame to a flattened data structure.
    `
        Args
        ----
            inputs_tree_to_df_columns:
                A nested dictionary that defines the structure of the inputs tree. The
                elements of the tree paths are strings. Leaves can be:

                - Strings that reference column names in the DataFrame.
                - Numeric or boolean values (which will be broadcasted to match the
                  DataFrame length)
            df:
                The pandas DataFrame containing the source data.

    Returns
    -------
            A flattened data structure containing the data organized according to the
            mapping definition.

    Examples
    --------
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
        if mapper_value in df.columns:
            path_to_array[path] = xnp.asarray(df[mapper_value])
        elif isinstance(mapper_value, int | float | bool):
            path_to_array[path] = xnp.asarray(
                pd.Series(
                    [mapper_value] * len(df),
                    index=df.index,
                ),
            )
        else:
            msg = (
                f"Value of mapper path {path} is neither a column name in the provided "
                "DataFrame nor a scalar value."
            )
            raise ValueError(msg)

    return path_to_array


def df_with_nested_columns_to_flat_data(
    df: pd.DataFrame,
    xnp: ModuleType,
) -> FlatData:
    """Convert a DataFrame with nested columns to a flattened data structure.

    Args:
        df:
            The pandas DataFrame with nested columns.
        xnp:
            The numpy module.

    Returns
    -------
        A flattened data structure.

    Examples
    --------
        >>> df = pd.DataFrame({("a", "b"): [1, 2, 3], ("c",): [4, 5, 6]})
        >>> result = df_with_nested_columns_to_flat_data(df, xnp=np)
        >>> result
        {("a", "b"): np.array([1, 2, 3]), ("c",): np.array([4, 5, 6])}
    """
    return {
        _remove_nan_from_keys(key): xnp.asarray(value)
        for key, value in df.to_dict(orient="list").items()
    }


def _remove_nan_from_keys(path: tuple[str | Any, ...]) -> tuple[str, ...]:
    """Remove nan string from string tuples."""
    return tuple(el for el in path if not pd.isna(el))
