from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import dags.tree as dt

from ttsim.interface_dag_elements.interface_node_objects import (
    interface_function,
    interface_input,
)

if TYPE_CHECKING:
    from ttsim.interface_dag_elements.typing import (
        FlatColumnObjects,
        FlatColumnObjectsParamFunctions,
        NestedTargetDict,
        OrderedQNames,
    )

NamespaceSelector = Literal["all"] | str  # noqa: PYI051


@interface_function()
def tree(
    policy_environment__flat_column_objects_and_param_functions: FlatColumnObjectsParamFunctions,  # noqa: E501
    policy_environment__flat_column_objects: FlatColumnObjects,
    include_param_functions: bool,
    namespace: NamespaceSelector,
) -> NestedTargetDict:
    """All targets as a tree.

    Returns all params and columns functions

    If requesting `df_with_mapper` as the targets, the leaves must be the desired
    column names.
    """
    if include_param_functions:
        base = policy_environment__flat_column_objects_and_param_functions
    else:
        base = policy_environment__flat_column_objects

    if namespace == "all":
        return dt.unflatten_from_tree_paths(dict.fromkeys(base))
    return dt.unflatten_from_tree_paths(
        {k: None for k in base if dt.qual_name_from_tree_path(k).startswith(namespace)}
    )


@interface_function()
def qname(tree: NestedTargetDict) -> OrderedQNames:
    """All targets in their qualified name-representation."""
    return dt.qual_names(tree)


@interface_input()
def namespace() -> NamespaceSelector:
    """Namespace for which all policy and param functions should be targeted."""


@interface_input()
def include_param_functions() -> bool:
    """Whether to include param functions in the targets."""
