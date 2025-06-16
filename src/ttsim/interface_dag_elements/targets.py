from __future__ import annotations

from typing import TYPE_CHECKING, Literal

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

NamespaceSelector = Literal["all"] | str  # noqa: PYI051


@interface_input()
def tree() -> NestedTargetDict:
    """Targets as a tree.

    If requesting `df_with_mapper` as the targets, the leaves must be the desired
    column names.
    """


@interface_function()
def qname(tree: NestedTargetDict) -> OrderedQNames:
    """Targets in their qualified name-representation."""
    return dt.qnames(tree)
