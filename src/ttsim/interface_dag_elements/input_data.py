from __future__ import annotations

from typing import TYPE_CHECKING

import dags.tree as dt

from ttsim.interface_dag_elements.data_converters import (
    df_with_mapped_columns_to_nested_data,
    df_with_nested_columns_to_nested_data,
)
from ttsim.interface_dag_elements.interface_node_objects import (
    InterfaceFunctionVariant,
    input_dependent_interface_function,
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


def _mapped_df_to_nested_data(
    input_data__df_and_mapper__df: pd.DataFrame,
    input_data__df_and_mapper__mapper: NestedInputsMapper,
    xnp: ModuleType,
) -> NestedData:
    return df_with_mapped_columns_to_nested_data(
        df=input_data__df_and_mapper__df,
        mapper=input_data__df_and_mapper__mapper,
        xnp=xnp,
    )


def _df_with_nested_columns_to_nested_data(
    input_data__df_with_nested_columns: pd.DataFrame,
    xnp: ModuleType,
) -> NestedData:
    return df_with_nested_columns_to_nested_data(
        df=input_data__df_with_nested_columns,
        xnp=xnp,
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


@input_dependent_interface_function(
    variants=[
        InterfaceFunctionVariant(
            required_input_qnames=["input_data__df_with_nested_columns"],
            function=_df_with_nested_columns_to_nested_data,
        ),
        InterfaceFunctionVariant(
            required_input_qnames=[
                "input_data__df_and_mapper__df",
                "input_data__df_and_mapper__mapper",
            ],
            function=_mapped_df_to_nested_data,
        ),
    ]
)
def tree() -> NestedData:
    pass


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
