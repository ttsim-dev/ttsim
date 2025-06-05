from __future__ import annotations

from typing import TYPE_CHECKING

from ttsim.data_converters import dataframe_to_nested_data

if TYPE_CHECKING:
    import pandas as pd

    from ttsim.tt_dag_elements.typing import NestedData, NestedStrings


def tree(
    input_data__df_and_mapper__df: pd.DataFrame,
    input_data__df_and_mapper__mapper: NestedStrings,
) -> NestedData:
    """The input DataFrame as a nested data structure.

    Args:
        input_data__df_and_mapper__df:
            The input DataFrame.
        input_data__df_and_mapper__mapper:
            A tree that maps paths (sequence of keys) to data columns names.

    Returns:
        A nested data structure.
    """
    return dataframe_to_nested_data(
        df=input_data__df_and_mapper__df,
        mapper=input_data__df_and_mapper__mapper,
    )
