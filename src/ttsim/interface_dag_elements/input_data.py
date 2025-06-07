from __future__ import annotations

from typing import TYPE_CHECKING

from ttsim.interface_dag_elements.data_converters import dataframe_to_nested_data
from ttsim.interface_dag_elements.interface_node_objects import interface_function

if TYPE_CHECKING:
    import pandas as pd

    from ttsim.interface_dag_elements.typing import NestedData, NestedInputsMapper


@interface_function()
def tree(
    input_data__df_and_mapper__df: pd.DataFrame,
    input_data__df_and_mapper__mapper: NestedInputsMapper,
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
