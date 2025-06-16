from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx

from ttsim.interface_dag_elements.automatically_added_functions import (
    TIME_UNIT_LABELS,
)
from ttsim.interface_dag_elements.interface_node_objects import interface_function
from ttsim.interface_dag_elements.shared import (
    get_base_name_and_grouping_suffix,
    get_re_pattern_for_all_time_units_and_groupings,
    group_pattern,
)

if TYPE_CHECKING:
    from ttsim.interface_dag_elements.typing import (
        NestedPolicyEnvironment,
        OrderedQNames,
        QNameData,
        QNamePolicyEnvironment,
        UnorderedQNames,
    )


def fail_if_multiple_time_units_for_same_base_name_and_group(
    base_names_and_groups_to_variations: dict[tuple[str, str], list[str]],
) -> None:
    invalid = {
        b: q for b, q in base_names_and_groups_to_variations.items() if len(q) > 1
    }
    if invalid:
        raise ValueError(f"Multiple time units for base names: {invalid}")


@interface_function()
def column_targets(
    specialized_environment__with_partialled_params_and_scalars: UnorderedQNames,
    targets__qname: OrderedQNames,
) -> OrderedQNames:
    """All targets that are column functions."""
    return [
        t
        for t in targets__qname
        if t in specialized_environment__with_partialled_params_and_scalars
    ]


@interface_function()
def param_targets(
    specialized_environment__without_tree_logic_and_with_derived_functions: QNamePolicyEnvironment,  # noqa: E501
    targets__qname: OrderedQNames,
    column_targets: OrderedQNames,
) -> OrderedQNames:
    possible_targets = set(targets__qname) - set(column_targets)
    return [
        t
        for t in targets__qname
        if t in possible_targets
        and t in specialized_environment__without_tree_logic_and_with_derived_functions
    ]


@interface_function()
def input_data_targets(
    targets__qname: OrderedQNames,
    column_targets: OrderedQNames,
    param_targets: OrderedQNames,
) -> OrderedQNames:
    possible_targets = set(targets__qname) - set(column_targets) - set(param_targets)
    return [t for t in targets__qname if t in possible_targets]


@interface_function()
def grouping_levels(
    policy_environment: QNamePolicyEnvironment,
) -> OrderedQNames:
    """The grouping levels of the policy environment."""
    return tuple(
        name.rsplit("_", 1)[0]
        for name in policy_environment
        if name.endswith("_id") and name != "p_id"
    )


@interface_function()
def top_level_namespace(
    policy_environment: NestedPolicyEnvironment,
    grouping_levels: OrderedQNames,
) -> UnorderedQNames:
    """Get the top level namespace.

    Parameters
    ----------
    policy_environment:
        The policy environment.


    Returns
    -------
    top_level_namespace:
        The top level namespace.
    """
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

    return all_top_level_names


@interface_function()
def processed_data_columns(processed_data: QNameData) -> UnorderedQNames:
    return set(processed_data.keys())


@interface_function()
def root_nodes(
    specialized_environment__tax_transfer_dag: nx.DiGraph,
    processed_data: QNameData,
) -> UnorderedQNames:
    """Names of the columns in `processed_data` required for the tax transfer function.

    Parameters
    ----------
    specialized_environment__tax_transfer_dag:
        The tax transfer DAG.
    processed_data:
        The processed data.

    Returns
    -------
    The names of the columns in `processed_data` required for the tax transfer function.

    """
    # Obtain root nodes
    root_nodes = nx.subgraph_view(
        specialized_environment__tax_transfer_dag,
        filter_node=lambda n: specialized_environment__tax_transfer_dag.in_degree(n)
        == 0,
    ).nodes

    # Restrict the passed data to the subset that is actually used.
    return {k for k in processed_data if k in root_nodes}
