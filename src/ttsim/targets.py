from __future__ import annotations

from typing import TYPE_CHECKING

import dags.tree as dt

if TYPE_CHECKING:
    from ttsim.tt_dag_elements.typing import (
        NestedTargetDict,
        QualNameColumnFunctions,
        QualNamePolicyEnvironment,
        QualNameTargetList,
    )


def targets__qname(targets__tree: NestedTargetDict) -> QualNameTargetList:
    """All targets in their qualified name-representation."""
    return dt.qual_names(targets__tree)


def targets__processed__columns(
    required_column_functions: QualNameColumnFunctions,
    targets__qname: QualNameTargetList,
) -> QualNameTargetList:
    """All targets that are column functions."""
    return [t for t in targets__qname if t in required_column_functions]


def targets__processed__params(
    flat_policy_environment_with_derived_functions_and_without_overridden_functions: QualNamePolicyEnvironment,  # noqa: E501
    targets__qname: QualNameTargetList,
    targets__processed__columns: QualNameTargetList,
) -> QualNameTargetList:
    possible_targets = set(targets__qname) - set(targets__processed__columns)
    return [
        t
        for t in targets__qname
        if t in possible_targets
        and t
        in flat_policy_environment_with_derived_functions_and_without_overridden_functions  # noqa: E501
    ]


def targets__processed__from_input_data(
    targets__qname: QualNameTargetList,
    targets__processed__columns: QualNameTargetList,
    targets__processed__params: QualNameTargetList,
) -> QualNameTargetList:
    possible_targets = (
        set(targets__qname)
        - set(targets__processed__columns)
        - set(targets__processed__params)
    )
    return [t for t in targets__qname if t in possible_targets]
