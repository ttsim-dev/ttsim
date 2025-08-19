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
        IntColumn,
        NestedData,
        NestedStrings,
        QNameData,
    )


@interface_function()
def tree(
    raw_results__columns: QNameData,
    raw_results__params: QNameData,
    raw_results__from_input_data: QNameData,
    input_data__sort_indices: IntColumn,
) -> NestedData:
    """The combined results as a tree with original row order restored."""
    restore_order = numpy.empty(len(input_data__sort_indices), dtype=int)
    restore_order[input_data__sort_indices] = numpy.arange(
        len(input_data__sort_indices)
    )

    def reorder_arrays(v: object) -> object:
        return v[restore_order] if hasattr(v, "shape") and v.ndim > 0 else v  # type: ignore[index, attr-defined]

    return dt.unflatten_from_qnames(
        {
            **raw_results__params,
            **{k: reorder_arrays(v) for k, v in raw_results__from_input_data.items()},
            **{k: reorder_arrays(v) for k, v in raw_results__columns.items()},
        }
    )


@interface_function()
def df_with_mapper(
    tree: NestedData,
    input_data__flat: FlatData,
    tt_targets__tree: NestedStrings,
) -> pd.DataFrame:
    """The results DataFrame with mapped column names.

    Args:
        tree:
            The results of a TTSIM run with original row order already restored.
        input_data__flat:
            The input data containing original p_ids.
        tt_targets__tree:
            A tree that maps paths (sequence of keys) to data columns names.

    Returns
    -------
        A DataFrame.
    """
    return nested_data_to_df_with_mapped_columns(
        nested_data_to_convert=tree,
        nested_outputs_df_column_names=tt_targets__tree,
        data_with_p_id=input_data__flat,
    )


@interface_function()
def df_with_nested_columns(
    tree: NestedData,
    input_data__flat: FlatData,
) -> pd.DataFrame:
    """The results DataFrame with nested column names corresponding to tree paths.

    Args:
        tree:
            The results of a TTSIM run with original row order already restored.
        input_data__flat:
            The flat input data containing original p_ids.

    Returns
    -------
    A DataFrame with a hierarchical index in the column dimension.
    """
    return nested_data_to_df_with_nested_columns(
        nested_data_to_convert=tree,
        index=pd.Index(input_data__flat[("p_id",)], name="p_id"),
    )
