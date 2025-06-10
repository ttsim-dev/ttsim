from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING

from ttsim.interface_dag_elements.data_converters import dataframe_to_nested_data
from ttsim.interface_dag_elements.interface_node_objects import (
    interface_function,
    interface_input,
)

if TYPE_CHECKING:
    import pandas as pd

    from ttsim.interface_dag_elements.typing import NestedData, NestedInputsMapper


@interface_input()
def df_and_mapper__df() -> pd.DataFrame:
    pass


@interface_input()
def df_and_mapper__mapper() -> NestedInputsMapper:
    pass


@interface_input()
def df_with_nested_columns() -> pd.DataFrame:
    pass


@interface_function()
def tree(
    df_and_mapper__df: pd.DataFrame,
    df_and_mapper__mapper: NestedInputsMapper,
    xnp: ModuleType,
) -> NestedData:
    """The input DataFrame as a nested data structure.

    Args:
        df_and_mapper__df:
            The input DataFrame.
        df_and_mapper__mapper:
            A tree that maps paths (sequence of keys) to data columns names.

    Returns:
        A nested data structure.
    """
    return dataframe_to_nested_data(
        df=df_and_mapper__df,
        mapper=df_and_mapper__mapper,
        xnp=xnp,
    )
