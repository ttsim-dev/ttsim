from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx

from ttsim.automatically_added_functions import (
    TIME_UNIT_LABELS,
)
from ttsim.fail_if import (
    fail_if__multiple_time_units_for_same_base_name_and_group,
)
from ttsim.shared import (
    get_base_name_and_grouping_suffix,
    get_re_pattern_for_all_time_units_and_groupings,
    group_pattern,
)

if TYPE_CHECKING:
    from ttsim.tt_dag_elements.typing import (
        NestedPolicyEnvironment,
        QualNameColumnFunctions,
        QualNameData,
        QualNamePolicyEnvironment,
        QualNameTargetList,
    )


def names__target_columns(
    combined_environment__with_partialled_params_and_scalars: QualNameColumnFunctions,
    targets__qname: QualNameTargetList,
) -> QualNameTargetList:
    """All targets that are column functions."""
    return [
        t
        for t in targets__qname
        if t in combined_environment__with_partialled_params_and_scalars
    ]


def names__target_params(
    combined_environment__with_derived_functions_and_input_nodes: QualNamePolicyEnvironment,  # noqa: E501
    targets__qname: QualNameTargetList,
    names__target_columns: QualNameTargetList,
) -> QualNameTargetList:
    possible_targets = set(targets__qname) - set(names__target_columns)
    return [
        t
        for t in targets__qname
        if t in possible_targets
        and t in combined_environment__with_derived_functions_and_input_nodes
    ]


def names__targets_from_input_data(
    targets__qname: QualNameTargetList,
    names__target_columns: QualNameTargetList,
    names__target_params: QualNameTargetList,
) -> QualNameTargetList:
    possible_targets = (
        set(targets__qname) - set(names__target_columns) - set(names__target_params)
    )
    return [t for t in targets__qname if t in possible_targets]


def names__grouping_levels(
    policy_environment: QualNamePolicyEnvironment,
) -> tuple[str, ...]:
    """The grouping levels of the policy environment."""
    return tuple(
        name.rsplit("_", 1)[0]
        for name in policy_environment
        if name.endswith("_id") and name != "p_id"
    )


def names__top_level_namespace(
    policy_environment: NestedPolicyEnvironment,
    names__grouping_levels: tuple[str, ...],
) -> set[str]:
    """Get the top level namespace.

    Parameters
    ----------
    environment:
        The policy environment.

    Returns
    -------
    names__top_level_namespace:
        The top level namespace.
    """

    time_units = tuple(TIME_UNIT_LABELS.keys())
    direct_top_level_names = set(policy_environment.keys())

    # Do not create variations for lower-level namespaces.
    top_level_objects_for_variations = direct_top_level_names - {
        k for k, v in policy_environment.items() if isinstance(v, dict)
    }

    pattern_all = get_re_pattern_for_all_time_units_and_groupings(
        groupings=names__grouping_levels,
        time_units=time_units,
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
    fail_if__multiple_time_units_for_same_base_name_and_group(bngs_to_variations)

    gp = group_pattern(names__grouping_levels)
    potential_base_names = {n for n in all_top_level_names if not gp.match(n)}

    for name in potential_base_names:
        for g in names__grouping_levels:
            all_top_level_names.add(f"{name}_{g}")

    # Add num_segments to grouping variables
    for g in names__grouping_levels:
        all_top_level_names.add(f"{g}_id_num_segments")
    return all_top_level_names


def names__processed_data_columns(processed_data: QualNameData) -> set[str]:
    return set(processed_data.keys())


def names__root_nodes(
    tax_transfer_dag: nx.DiGraph,
    processed_data: QualNameData,
) -> set[str]:
    """Names of the columns in `processed_data` required for the tax transfer function.

    Parameters
    ----------
    tax_transfer_dag:
        The tax transfer DAG.
    processed_data:
        The processed data.

    Returns
    -------
    The names of the columns in `processed_data` required for the tax transfer function.

    """

    # Obtain root nodes
    root_nodes = nx.subgraph_view(
        tax_transfer_dag, filter_node=lambda n: tax_transfer_dag.in_degree(n) == 0
    ).nodes

    # Restrict the passed data to the subset that is actually used.
    return {k for k in processed_data if k in root_nodes}
