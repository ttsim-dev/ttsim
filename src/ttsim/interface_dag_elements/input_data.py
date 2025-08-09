from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import dags.tree as dt

from ttsim.interface_dag_elements.data_converters import (
    df_with_mapped_columns_to_flat_data,
    df_with_nested_columns_to_flat_data,
)
from ttsim.interface_dag_elements.interface_node_objects import (
    input_dependent_interface_function,
    interface_input,
)

if TYPE_CHECKING:
    from types import ModuleType

    import pandas as pd

    from ttsim.typing import (
        FlatData,
        NestedData,
        NestedInputsMapper,
    )


@interface_input(leaf_name="df")
def df_and_mapper__df() -> pd.DataFrame:
    pass


@interface_input(leaf_name="mapper")
def df_and_mapper__mapper() -> NestedInputsMapper:
    pass


@interface_input()
def df_with_nested_columns() -> pd.DataFrame:
    pass


@interface_input()
def tree() -> NestedData:
    pass


@input_dependent_interface_function(
    include_if_all_inputs_present=[
        "input_data__df_and_mapper__df",
        "input_data__df_and_mapper__mapper",
    ],
    leaf_name="flat",
)
def flat_from_df_and_mapper(
    df_and_mapper__df: pd.DataFrame,
    df_and_mapper__mapper: NestedInputsMapper,
    backend: Literal["numpy", "jax"],
    xnp: ModuleType,
) -> FlatData:
    """The input DataFrame as a flattened data structure.

    Args:
        df_and_mapper__df:
            The input DataFrame.
        df_and_mapper__mapper:
            Dictionary mapping tree paths to column names.

    Returns
    -------
        Flattened data structure.
    """
    return df_with_mapped_columns_to_flat_data(
        df=df_and_mapper__df,
        mapper=df_and_mapper__mapper,
        backend=backend,
        xnp=xnp,
    )


@input_dependent_interface_function(
    include_if_all_inputs_present=["input_data__df_with_nested_columns"],
    leaf_name="flat",
)
def flat_from_df_with_nested_columns(
    df_with_nested_columns: pd.DataFrame,
    backend: Literal["numpy", "jax"],
    xnp: ModuleType,
) -> FlatData:
    """The input DataFrame as a flattened data structure.

    Args:
        df_with_nested_columns:
            The input DataFrame with nested column names.

    Returns
    -------
        Flattened data structure.
    """
    return df_with_nested_columns_to_flat_data(
        df=df_with_nested_columns,
        backend=backend,
        xnp=xnp,
    )


@input_dependent_interface_function(
    include_if_all_inputs_present=["input_data__tree"],
    leaf_name="flat",
)
def flat_from_tree(
    tree: NestedData,
    xnp: ModuleType,  # noqa: ARG001
) -> FlatData:
    """The input DataFrame as a flattened data structure.

    Args:
        tree:
            The input tree.
        xnp:
            The backend to use, just put here so that fail_if.input_data_tree_is_invalid
            runs before this.

    Returns
    -------
        Flattened data structure.
    """
    return dt.flatten_to_tree_paths(tree)
