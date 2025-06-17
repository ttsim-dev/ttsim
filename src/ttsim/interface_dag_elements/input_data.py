from __future__ import annotations

from typing import TYPE_CHECKING

import dags.tree as dt

from ttsim.interface_dag_elements.data_converters import (
    dataframe_with_nested_columns_to_nested_data,
    mapped_dataframe_to_nested_data,
)
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


@interface_input()
def tree() -> NestedData:
    pass


@interface_function()
def flat(
    xnp: ModuleType,
    tree: NestedData = None,
    df_with_nested_columns: pd.DataFrame = None,
    df_and_mapper__df: pd.DataFrame = None,
    df_and_mapper__mapper: NestedInputsMapper = None,
) -> FlatData:
    """The input DataFrame as a flattened data structure.

    Args:
        xnp:
            The NumPy module to use.
        tree:
            The input tree.
        df_with_nested_columns:
            The input DataFrame with nested columns.
        df_and_mapper__df:
            The input DataFrame.
        df_and_mapper__mapper:
            A tree that maps paths (sequence of keys) to data columns names.

    Returns
    -------
        Mapping of tree paths to input data.
    """
    if tree:
        base = tree
    elif df_with_nested_columns:
        base = dataframe_with_nested_columns_to_nested_data(
            df=df_with_nested_columns,
            xnp=xnp,
        )
    else:
        base = mapped_dataframe_to_nested_data(
            df=df_and_mapper__df,
            mapper=df_and_mapper__mapper,
            xnp=xnp,
        )

    return dt.flatten_to_tree_paths(base)
