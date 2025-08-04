"""Builds the specialized environment from policy inputs.

The main difference to `specialized_environment` is that policy inputs are taken as a
basis for derived functions alongside with the user-provided input data. This is useful
for applications where users are interested in the DAG itself (not its execution), e.g.
when creating an input data template or plotting it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import dags.tree as dt
from dags import create_dag

from ttsim.interface_dag_elements.automatically_added_functions import TIME_UNIT_LABELS
from ttsim.interface_dag_elements.interface_node_objects import (
    InterfaceFunction,
    InterfaceInput,
    interface_function,
)
from ttsim.interface_dag_elements.shared import (
    get_re_pattern_for_all_time_units_and_groupings,
)
from ttsim.interface_dag_elements.specialized_environment import (
    _add_derived_functions,
    _remove_tree_logic_from_policy_environment,
)
from ttsim.tt.column_objects_param_function import (
    ParamFunction,
    PolicyFunction,
    PolicyInput,
    param_function,
    policy_function,
)
from ttsim.tt.param_objects import ParamObject

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType
    from typing import Any

    import networkx as nx

    from ttsim.typing import (
        OrderedQNames,
        PolicyEnvironment,
        SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
        UnorderedQNames,
    )


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
        pattern_all.fullmatch(qn).group("base_name") for qn in labels__input_columns
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
        tt_targets=tt_targets__qname,
        input_columns=qnames_to_derive_functions_from,
        grouping_levels=labels__grouping_levels,
    )


@interface_function()
def complete_dag(
    without_tree_logic_and_with_derived_functions: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
    tt_targets__qname: OrderedQNames,
) -> nx.DiGraph:
    """Create the complete DAG.

    Transforms non-callable nodes into callables to include them as nodes in the DAG.
    """
    functions = {
        qn: dummy_callable(obj=n, leaf_name=dt.tree_path_from_qname(qn)[-1])
        if not callable(n)
        else n
        for qn, n in without_tree_logic_and_with_derived_functions.items()
    }
    return create_dag(
        functions=functions,
        targets=tt_targets__qname,
    )


@overload
def dummy_callable(obj: PolicyInput, leaf_name: str) -> PolicyFunction: ...


@overload
def dummy_callable(obj: ParamObject, leaf_name: str) -> ParamFunction: ...


@overload
def dummy_callable(obj: InterfaceInput, leaf_name: str) -> InterfaceFunction: ...


def dummy_callable(
    obj: ModuleType | str | float | bool, leaf_name: str
) -> Callable[[], Any]:
    """Dummy callable, for plotting and checking DAG completeness."""

    def dummy():  # type: ignore[no-untyped-def]  # noqa: ANN202
        pass

    if isinstance(obj, PolicyInput):
        return policy_function(
            leaf_name=leaf_name,
            start_date=obj.start_date,
            end_date=obj.end_date,
            foreign_key_type=obj.foreign_key_type,
        )(dummy)
    if isinstance(obj, ParamObject):
        return param_function(
            leaf_name=leaf_name,
            start_date=obj.start_date,
            end_date=obj.end_date,
        )(dummy)
    if isinstance(obj, InterfaceInput):
        return interface_function(
            leaf_name=leaf_name,
            in_top_level_namespace=obj.in_top_level_namespace,
        )(dummy)
    return dummy
