from __future__ import annotations

from typing import TYPE_CHECKING, Any

import dags.tree as dt
import pandas as pd

if TYPE_CHECKING:
    from types import ModuleType

    from ttsim.interface_dag_elements.typing import (
        NestedData,
        NestedInputsMapper,
        NestedStrings,
        QNameData,
    )


def nested_data_to_df_with_nested_columns(
    nested_data_to_convert: NestedData,
    data_with_p_id: NestedData | QNameData,
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

    return pd.DataFrame(
        flat_data_to_convert,
        index=pd.Index(data_with_p_id["p_id"], name="p_id"),
    )


def nested_data_to_df_with_mapped_columns(
    nested_data_to_convert: NestedData,
    nested_outputs_df_column_names: NestedStrings,
    data_with_p_id: NestedData | QNameData,
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

    return pd.DataFrame(
        {flat_df_columns[path]: data for path, data in flat_data_to_convert.items()},
        index=pd.Index(data_with_p_id["p_id"], name="p_id"),
    )


def df_with_mapped_columns_to_nested_data(
    mapper: NestedInputsMapper,
    df: pd.DataFrame,
    xnp: ModuleType,
) -> NestedData:
    """Transform a pandas DataFrame to a nested dictionary expected by TTSIM.
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
            A nested dictionary structure containing the data organized according to the
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
                "n1": {
                    "n2": np.array([1, 2, 3]),
                    "n3": np.array([4, 5, 6]),
                },
                "n4": np.array([3, 3, 3]),
            }


    """
    qname_to_array = {}
    for qname, df_col_name in dt.flatten_to_qnames(mapper).items():
        if df_col_name in df.columns:
            qname_to_array[qname] = xnp.asarray(df[df_col_name])
        else:
            qname_to_array[qname] = xnp.asarray(
                pd.Series(
                    [df_col_name] * len(df),
                    index=df.index,
                ),
            )

    return dt.unflatten_from_qnames(qname_to_array)


def df_with_nested_columns_to_nested_data(
    df: pd.DataFrame,
    xnp: ModuleType,
) -> NestedData:
    """Convert a DataFrame with nested columns to a nested data structure."""
    flat_dict_with_correct_leaf_types = {
        _remove_nan_from_keys(key): xnp.asarray(value)
        for key, value in df.to_dict(orient="list").items()
    }
    return dt.unflatten_from_tree_paths(flat_dict_with_correct_leaf_types)


def _remove_nan_from_keys(path: tuple[str | Any, ...]) -> tuple[str, ...]:
    """Remove nan string from string tuples."""
    return tuple(el for el in path if not pd.isna(el))
