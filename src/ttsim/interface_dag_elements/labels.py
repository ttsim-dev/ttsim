from __future__ import annotations

from typing import TYPE_CHECKING

import dags.tree as dt
import networkx as nx

from ttsim.interface_dag_elements.automatically_added_functions import (
    TIME_UNIT_LABELS,
)
from ttsim.interface_dag_elements.interface_node_objects import (
    input_dependent_interface_function,
    interface_function,
)
from ttsim.interface_dag_elements.shared import (
    get_base_name_and_grouping_suffix,
    get_re_pattern_for_all_time_units_and_groupings,
    group_pattern,
)
from ttsim.tt.column_objects_param_function import PolicyInput

if TYPE_CHECKING:
    from types import ModuleType

    from ttsim.typing import (
        OrderedQNames,
        PolicyEnvironment,
        QNameData,
        SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
        SpecEnvWithPartialledParamsAndScalars,
        UnorderedQNames,
    )


@interface_function()
def grouping_levels(
    policy_environment: PolicyEnvironment,
) -> OrderedQNames:
    """The grouping levels of the policy environment."""
    return tuple(
        name.rsplit("_", 1)[0]
        for name in policy_environment
        if name.endswith("_id") and name != "p_id"
    )


@interface_function()
def top_level_namespace(
    policy_environment: PolicyEnvironment,
    grouping_levels: OrderedQNames,
) -> UnorderedQNames:
    """The elements of the top level namespace."""
    time_units = tuple(TIME_UNIT_LABELS)
    direct_top_level_names = set(policy_environment)

    # Do not create variations for lower-level namespaces.
    top_level_objects_for_variations = direct_top_level_names - {
        k for k, v in policy_environment.items() if isinstance(v, dict)
    }

    pattern_all = get_re_pattern_for_all_time_units_and_groupings(
        time_units=time_units,
        grouping_levels=grouping_levels,
    )
    bngs_to_variations = {}
    all_top_level_names = direct_top_level_names.copy()
    for name in top_level_objects_for_variations:
        match = pattern_all.fullmatch(name)
        # We must not find multiple time units for the same base name and group.
        bngs = get_base_name_and_grouping_suffix(match)
        if match.group("time_unit"):
            if bngs not in bngs_to_variations:
                bngs_to_variations[bngs] = [name]
            else:
                bngs_to_variations[bngs].append(name)
            for time_unit in time_units:
                all_top_level_names.add(f"{bngs[0]}_{time_unit}{bngs[1]}")
    fail_if_multiple_time_units_for_same_base_name_and_group(bngs_to_variations)

    gp = group_pattern(grouping_levels)
    potential_base_names = {n for n in all_top_level_names if not gp.match(n)}

    for name in potential_base_names:
        for g in grouping_levels:
            all_top_level_names.add(f"{name}_{g}")

    # Add config variables that are not part of the policy environment.
    return all_top_level_names.union(
        {
            "xnp",
            "dnp",
            "num_segments",
            "backend",
            "evaluation_year",
            "evaluation_month",
            "evaluation_day",
        }
    )


@input_dependent_interface_function(
    include_if_any_input_present=[
        "input_data__flat",
        "input_data__tree",
        "input_data__df_with_nested_columns",
        "input_data__df_and_mapper__df",
        "processed_data",
    ],
    leaf_name="input_columns",
)
def input_columns_from_input_data(processed_data: QNameData) -> UnorderedQNames:
    """The (qualified) column names in the input data."""
    return set(processed_data.keys())


@input_dependent_interface_function(
    include_if_no_input_present=[
        "input_data__flat",
        "input_data__tree",
        "input_data__df_with_nested_columns",
        "input_data__df_and_mapper__df",
        "processed_data",
    ],
    leaf_name="input_columns",
)
def input_columns_is_empty_set(
    xnp: ModuleType,  # fake input # noqa: ARG001
) -> UnorderedQNames:
    """The (qualified) column names in the input data."""
    return set()


@interface_function()
def all_qnames_in_policy_environment(
    policy_environment: PolicyEnvironment,
) -> UnorderedQNames:
    """The (qualified) names of all objects in the policy environment."""
    return set(dt.qnames(policy_environment))


@interface_function()
def policy_inputs(policy_environment: PolicyEnvironment) -> UnorderedQNames:
    """The (qualified) names of the policy inputs in the policy environment."""
    return {
        k
        for k, v in dt.flatten_to_qnames(policy_environment).items()
        if isinstance(v, PolicyInput)
    }


@interface_function()
def root_nodes(
    specialized_environment__tt_dag: nx.DiGraph,
    input_columns: UnorderedQNames,
) -> UnorderedQNames:
    """The (qualified) names of those columns in `processed_data` which are required for
    the tax transfer function."""

    # Obtain root nodes
    root_nodes = nx.subgraph_view(
        specialized_environment__tt_dag,
        filter_node=lambda n: specialized_environment__tt_dag.in_degree(n) == 0,
    ).nodes

    # Restrict the passed data to the subset that is actually used.
    return {k for k in input_columns if k in root_nodes}


def fail_if_multiple_time_units_for_same_base_name_and_group(
    base_names_and_groups_to_variations: dict[tuple[str, str], list[str]],
) -> None:
    invalid = {
        b: q for b, q in base_names_and_groups_to_variations.items() if len(q) > 1
    }
    if invalid:
        raise ValueError(f"Multiple time units for base names: {invalid}")


@interface_function()
def input_data_targets(
    tt_targets__qname: OrderedQNames,
    input_columns: UnorderedQNames,
) -> OrderedQNames:
    """
    The (qualified) names of the targets that are already present in the input data.
    """
    return [t for t in tt_targets__qname if t in input_columns]


@interface_function()
def column_targets(
    specialized_environment__with_partialled_params_and_scalars: SpecEnvWithPartialledParamsAndScalars,  # noqa: E501
    tt_targets__qname: OrderedQNames,
    input_data_targets: OrderedQNames,
) -> OrderedQNames:
    """The (qualified) names of the targets that are column functions."""
    possible_targets = set(tt_targets__qname) - set(input_data_targets)
    return [
        t
        for t in tt_targets__qname
        if t in specialized_environment__with_partialled_params_and_scalars
        and t in possible_targets
    ]


@interface_function()
def param_targets(
    specialized_environment__without_tree_logic_and_with_derived_functions: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,  # noqa: E501
    tt_targets__qname: OrderedQNames,
    column_targets: OrderedQNames,
    input_data_targets: OrderedQNames,
) -> OrderedQNames:
    """The (qualified) names of the targets that are parameters or param functions."""
    possible_targets = (
        set(tt_targets__qname) - set(column_targets) - set(input_data_targets)
    )
    return [
        t
        for t in tt_targets__qname
        if t in specialized_environment__without_tree_logic_and_with_derived_functions
        and t in possible_targets
    ]
