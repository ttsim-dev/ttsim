from __future__ import annotations

from typing import TYPE_CHECKING

import dags.tree as dt

from ttsim.interface_dag_elements.data_converters import dataframe_to_nested_data
from ttsim.interface_dag_elements.interface_node_objects import (
    interface_function,
    interface_input,
)

if TYPE_CHECKING:
    from types import ModuleType

    import pandas as pd

    from ttsim.interface_dag_elements.typing import (
        FlatData,
        NestedData,
        NestedInputsMapper,
    )


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

    Returns
    -------
        A nested data structure.
    """
    return dataframe_to_nested_data(
        df=df_and_mapper__df,
        mapper=df_and_mapper__mapper,
        xnp=xnp,
    )


@interface_function()
def flat(tree: NestedData) -> FlatData:
    """The input DataFrame as a flattened data structure.

    Args:
        tree:
            The input tree.

    Returns
    -------
        Mapping of tree paths to input data.
    """
    return dt.flatten_to_tree_paths(tree)
