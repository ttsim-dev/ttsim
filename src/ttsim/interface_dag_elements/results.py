from __future__ import annotations

from typing import TYPE_CHECKING

import dags.tree as dt

from ttsim.interface_dag_elements.data_converters import (
    nested_data_to_df_with_mapped_columns,
    nested_data_to_df_with_nested_columns,
)
from ttsim.interface_dag_elements.interface_node_objects import interface_function

if TYPE_CHECKING:
    import pandas as pd

    from ttsim.interface_dag_elements.typing import (
        FlatData,
        NestedData,
        NestedStrings,
        QNameData,
    )


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
    targets__tree: NestedStrings,
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
    return nested_data_to_df_with_mapped_columns(
        nested_data_to_convert=tree,
        nested_outputs_df_column_names=targets__tree,
        data_with_p_id=input_data__flat,
    )


@interface_function()
def df_with_nested_columns(
    tree: NestedData,
    input_data__flat: FlatData,
) -> pd.DataFrame:
    """The results DataFrame with nested column names corresponding to tree paths..

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
    return nested_data_to_df_with_nested_columns(
        nested_data_to_convert=tree,
        data_with_p_id=input_data__flat,
    )
