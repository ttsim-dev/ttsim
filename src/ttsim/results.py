from __future__ import annotations

from typing import TYPE_CHECKING

import dags.tree as dt

from ttsim.data_converters import nested_data_to_df_with_mapped_columns

if TYPE_CHECKING:
    import pandas as pd

    from ttsim.typing import NestedData, NestedStrings, QNameData


def results__tree(raw_results__combined: QNameData) -> NestedData:
    """The combined results as a tree.

    Note: This is the point where the `p_id`s are converted back to their original
    values.

    """

    return dt.unflatten_from_qual_names(raw_results__combined)


def results__df(
    results__tree: NestedData,
    input_data__tree: NestedData,
    results__df_and_mapper__mapper: NestedStrings,
) -> pd.DataFrame:
    """The results DataFrame with mapped column names.

    Args:
        results__tree:
            The results of a TTSIM run.
        input_data__tree:
            The data tree of the TTSIM run.
        nested_outputs_df_column_names:
            A tree that maps paths (sequence of keys) to data columns names.

    Returns:
        A DataFrame.
    """
    return nested_data_to_df_with_mapped_columns(
        nested_data_to_convert=results__tree,
        nested_outputs_df_column_names=results__df_and_mapper__mapper,
        data_with_p_id=input_data__tree,
    )
