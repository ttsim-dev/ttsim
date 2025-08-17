from __future__ import annotations

from typing import TYPE_CHECKING

import dags.tree as dt
import numpy
import pandas as pd

from ttsim.interface_dag_elements.data_converters import (
    nested_data_to_df_with_mapped_columns,
    nested_data_to_df_with_nested_columns,
)
from ttsim.interface_dag_elements.interface_node_objects import interface_function

if TYPE_CHECKING:
    from ttsim.typing import (
        FlatData,
        NestedData,
        NestedStrings,
        QNameData,
    )


def _restore_original_row_order(
    df: pd.DataFrame,
    input_data__flat: FlatData,
) -> pd.DataFrame:
    """Restore the original row order of a DataFrame using stored sort indices.

    Args:
        df: The DataFrame with rows in sorted order.
        input_data__flat: The flat data containing the original sort indices.

    Returns:
        DataFrame with rows restored to their original order.
    """
    if ("__original_sort_indices__",) not in input_data__flat:
        return df

    sort_indices = input_data__flat[("__original_sort_indices__",)]

    # Create inverse permutation: restore_order[orig_pos] = sorted_pos
    restore_order = numpy.empty(len(sort_indices), dtype=int)
    restore_order[sort_indices] = numpy.arange(len(sort_indices))

    # Restore both the data rows AND the index values
    index_name = df.index.name
    original_index_values = df.index[restore_order]
    df = df.iloc[restore_order]
    df.index = original_index_values
    df.index.name = index_name

    if "__original_sort_indices__" in df.columns:
        df = df.drop(columns=["__original_sort_indices__"])

    return df


@interface_function()
def tree(raw_results__combined: QNameData, input_data__flat: FlatData) -> NestedData:
    """The combined results as a tree.

    The transformed id's are converted back to their original values.

    """
    out = {}
    for k in raw_results__combined:
        path = dt.tree_path_from_qname(k)
        if path in input_data__flat and (
            path[-1].endswith("_id") or path[-1].startswith("p_id_")
        ):
            out[k] = input_data__flat[path]
        else:
            out[k] = raw_results__combined[k]
    return dt.unflatten_from_qnames(out)


@interface_function()
def df_with_mapper(
    tree: NestedData,
    input_data__flat: FlatData,
    tt_targets__tree: NestedStrings,
) -> pd.DataFrame:
    """The results DataFrame with mapped column names.

    Args:
        tree:
            The results of a TTSIM run.
        input_data__tree:
            The data tree of the TTSIM run.
        nested_outputs_df_column_names:
            A tree that maps paths (sequence of keys) to data columns names.

    Returns
    -------
        A DataFrame.
    """
    df = nested_data_to_df_with_mapped_columns(
        nested_data_to_convert=tree,
        nested_outputs_df_column_names=tt_targets__tree,
        data_with_p_id=input_data__flat,
    )

    # Restore original row order if sort indices are available
    return _restore_original_row_order(df, input_data__flat)


@interface_function()
def df_with_nested_columns(
    tree: NestedData,
    input_data__flat: FlatData,
) -> pd.DataFrame:
    """The results DataFrame with nested column names corresponding to tree paths.

    Args:
        tree:
            The results of a TTSIM run.
        input_data__tree:
            The data tree of the TTSIM run.
        nested_outputs_df_column_names:
            A tree that maps paths (sequence of keys) to data columns names.

    Returns
    -------
    A DataFrame with a hierarchical index in the column dimension.
    """
    df = nested_data_to_df_with_nested_columns(
        nested_data_to_convert=tree,
        index=pd.Index(input_data__flat[("p_id",)], name="p_id"),
    )

    # Restore original row order if sort indices are available
    return _restore_original_row_order(df, input_data__flat)
