from __future__ import annotations

from typing import TYPE_CHECKING

import dags.tree as dt

from ttsim.interface_dag_elements.interface_node_objects import (
    interface_function,
)
from ttsim.tt.column_objects_param_function import ColumnFunction

if TYPE_CHECKING:
    from ttsim.typing import (
        NestedStrings,
        NestedTargetDict,
        OrderedQNames,
        PolicyEnvironment,
    )


@interface_function()
def tree(policy_environment: PolicyEnvironment) -> NestedTargetDict | NestedStrings:
    """Targets as a tree. Will typically be provided by the user.

    If requesting `df_with_mapper` as a main target, the leaves must be the desired
    column names.

    If not provided, the targets will be inferred from the policy environment (i.e.,
    use all ColumnFunctions in the policy environment).
    """
    return dt.unflatten_from_tree_paths(
        {
            k: None
            for k, v in dt.flatten_to_tree_paths(policy_environment).items()
            if isinstance(v, ColumnFunction)
        }
    )


@interface_function()
def qname(tree: NestedTargetDict | NestedStrings) -> OrderedQNames:
    """Targets in their qualified name-representation."""
    return dt.flatten_to_qnames(tree)
