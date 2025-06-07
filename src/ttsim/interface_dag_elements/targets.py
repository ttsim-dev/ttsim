from __future__ import annotations

from typing import TYPE_CHECKING

import dags.tree as dt

from ttsim.interface_dag_elements.interface_node_objects import (
    interface_function,
    interface_input,
)

if TYPE_CHECKING:
    from ttsim.interface_dag_elements.typing import (
        NestedTargetDict,
        OrderedQNames,
    )


@interface_input()
def tree() -> NestedTargetDict:
    """All targets as a tree. If requesting `df_with_mapper` as the targets, the leaves must be the desired column names."""  # noqa: E501


@interface_function()
def qname(tree: NestedTargetDict) -> OrderedQNames:
    """All targets in their qualified name-representation."""
    return dt.qual_names(tree)
