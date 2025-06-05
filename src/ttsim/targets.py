from __future__ import annotations

from typing import TYPE_CHECKING

import dags.tree as dt

if TYPE_CHECKING:
    from ttsim.tt_dag_elements.typing import (
        NestedTargetDict,
        QualNameTargetList,
    )


def targets__qname(targets__tree: NestedTargetDict) -> QualNameTargetList:
    """All targets in their qualified name-representation."""
    return dt.qual_names(targets__tree)
