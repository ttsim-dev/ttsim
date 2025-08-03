"""Builds the specialized environment from policy inputs.

The main difference to `specialized_environment` is that policy inputs are taken as a
basis for derived functions alongside with the user-provided input data. This is useful
for applications where users are interested in the DAG itself (not its execution), e.g.
when creating an input data template or plotting it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal
from dataclasses import dataclass

import dags.tree as dt
from dags import create_dag

from ttsim.interface_dag_elements.interface_node_objects import interface_function, interface_input
from ttsim.interface_dag_elements.shared import dummy_callable
from ttsim.interface_dag_elements.shared import (
    get_re_pattern_for_all_time_units_and_groupings,
)
from ttsim.interface_dag_elements.automatically_added_functions import TIME_UNIT_LABELS
from ttsim.interface_dag_elements.specialized_environment import (
    _remove_tree_logic_from_policy_environment,
    _add_derived_functions,
)


if TYPE_CHECKING:
    import networkx as nx

    from ttsim.typing import (
        OrderedQNames,
        PolicyEnvironment,
        SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
        UnorderedQNames,
    )


@dataclass(frozen=True)
class QNameNodeSelector:
    """Select nodes from the DAG."""

    qnames: set[str]
    type: Literal["neighbors", "descendants", "ancestors", "nodes"]
    order: int | None = None


@interface_input()
def qname_node_selector() -> QNameNodeSelector:
    """The qname node selector."""


@interface_function()
def qnames_to_derive_functions_from(
    labels__input_columns: UnorderedQNames,
    labels__policy_inputs: UnorderedQNames,
    labels__grouping_levels: OrderedQNames,
) -> UnorderedQNames:
    """The qnames to derive functions from.
    
    Derived functions should be created based on the actual input columns, and the
    policy inputs (if their base name is not already in the input columns).
    """
    pattern_all = get_re_pattern_for_all_time_units_and_groupings(
        time_units=list(TIME_UNIT_LABELS),
        grouping_levels=labels__grouping_levels,
    )
    base_names_input_columns = {
        pattern_all.fullmatch(qn).group("base_name")
        for qn in labels__input_columns
    }

    out = set(labels__input_columns)
    for pi in labels__policy_inputs:
        match = pattern_all.fullmatch(pi)
        base_name = match.group("base_name")
        if base_name in base_names_input_columns:
            continue
        out.add(pi)
    return out


@interface_function()
def without_tree_logic_and_with_derived_functions(
    policy_environment: PolicyEnvironment,
    qnames_to_derive_functions_from: UnorderedQNames,
    tt_targets__qname: OrderedQNames,
    labels__top_level_namespace: UnorderedQNames,
    labels__grouping_levels: OrderedQNames,
    qname_node_selector: QNameNodeSelector,
) -> SpecEnvWithoutTreeLogicAndWithDerivedFunctions:
    """Return a flat policy environment with derived functions.

    Two steps:
    1. Remove all tree logic from the policy environment.
    2. Add derived functions to the policy environment.

    """
    qname_env_without_tree_logic = _remove_tree_logic_from_policy_environment(
        qname_env=dt.flatten_to_qnames(policy_environment),
        labels__top_level_namespace=labels__top_level_namespace,
    )
    return _add_derived_functions(
        qname_env_without_tree_logic=qname_env_without_tree_logic,
        tt_targets=tt_targets__qname | qname_node_selector.qnames,
        input_columns=qnames_to_derive_functions_from,
        grouping_levels=labels__grouping_levels,
    )


@interface_function()
def dag_for_plotting(
    without_tree_logic_and_with_derived_functions: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
    labels__column_targets: OrderedQNames,
    qname_node_selector: QNameNodeSelector,
) -> nx.DiGraph:
    """Create a DAG for plotting.
    
    Transforms non-callable nodes into callables to include them as nodes in the DAG.
    """
    base_dag = create_dag(
        functions={
            qn: dummy_callable(obj=n, leaf_name=dt.tree_path_from_qname(qn)[-1])
            if not callable(n)
            else n
            for qn, n in without_tree_logic_and_with_derived_functions.items()
        },
        targets=labels__column_targets,
    )

    if not qname_node_selector.qnames:
        return base_dag
    else:
        selected_dag = ...
        return selected_dag
