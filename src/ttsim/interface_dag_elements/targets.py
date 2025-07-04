from __future__ import annotations

from typing import TYPE_CHECKING

import dags.tree as dt

from ttsim.interface_dag_elements.interface_node_objects import (
    interface_function,
)
from ttsim.tt_dag_elements.column_objects_param_function import ColumnFunction

if TYPE_CHECKING:
    from ttsim.interface_dag_elements.typing import (
        NestedTargetDict,
        OrderedQNames,
        PolicyEnvironment,
    )


@interface_function()
def tree(policy_environment: PolicyEnvironment) -> NestedTargetDict:
    """Targets as a tree. Will typically be provided by the user.

    If requesting `df_with_mapper` as the targets, the leaves must be the desired
    column names.

    If not provided, the targets will be inferred from the policy environment by using
    all ColumnFunctions in the policy environment.
    """
    return dt.unflatten_from_tree_paths(
        {
            k: None
            for k, v in dt.flatten_to_tree_paths(policy_environment).items()
            if isinstance(v, ColumnFunction)
        }
    )


@interface_function()
def qname(tree: NestedTargetDict) -> OrderedQNames:
    """Targets in their qualified name-representation."""
    return dt.flatten_to_qnames(tree)
