from __future__ import annotations

from typing import TYPE_CHECKING

import dags.tree as dt

from ttsim.interface_dag_elements.interface_node_objects import interface_function

if TYPE_CHECKING:
    from ttsim.interface_dag_elements.typing import (
        NestedTargetDict,
        OrderedQNames,
    )


@interface_function()
def qname(tree: NestedTargetDict) -> OrderedQNames:
    """All targets in their qualified name-representation."""
    return dt.qual_names(tree)
