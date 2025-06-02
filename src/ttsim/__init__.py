from __future__ import annotations

from typing import Any

import dags
import networkx as nx

from ttsim.aggregation import AggType
from ttsim.automatically_added_functions import create_time_conversion_functions
from ttsim.column_objects_param_function import (
    AggByGroupFunction,
    AggByPIDFunction,
    FKType,
    GroupCreationFunction,
    ParamFunction,
    PolicyFunction,
    PolicyInput,
    TimeConversionFunction,
    agg_by_group_function,
    agg_by_p_id_function,
    group_creation_function,
    param_function,
    policy_function,
    policy_input,
)
from ttsim.compute_taxes_and_transfers import (
    FunctionsAndDataOverlapWarning,
    _add_derived_functions,
    column_functions_with_processed_params_and_scalars,
    compute_taxes_and_transfers,
    fail_if_any_paths_are_invalid,
    fail_if_data_tree_is_invalid,
    fail_if_foreign_keys_are_invalid_in_data,
    fail_if_group_variables_are_not_constant_within_groups,
    fail_if_root_nodes_are_missing,
    fail_if_targets_are_not_in_policy_environment_or_data,
    fail_if_targets_tree_is_invalid,
    flat_policy_environment_with_derived_functions_and_without_overridden_functions,
    nested_results,
    qual_name_column_targets,
    qual_name_data_columns,
    qual_name_input_data,
    qual_name_own_targets,
    qual_name_param_targets,
    qual_name_results,
    qual_name_targets,
    required_column_functions,
    tax_transfer_dag,
    tax_transfer_function,
    top_level_namespace,
    warn_if_elements_overridden_by_data,
)
from ttsim.convert_nested_data import dataframe_to_nested_data, nested_data_to_dataframe
from ttsim.param_objects import (
    ConsecutiveInt1dLookupTableParam,
    ConsecutiveInt1dLookupTableParamValue,
    ConsecutiveInt2dLookupTableParamValue,
    DictParam,
    ParamObject,
    PiecewisePolynomialParam,
    PiecewisePolynomialParamValue,
    RawParam,
    ScalarParam,
)
from ttsim.piecewise_polynomial import (
    get_piecewise_parameters,
    piecewise_polynomial,
)
from ttsim.plot_dag import plot_dag
from ttsim.policy_environment import (
    OrigTreesWithFileNames,
    active_tree_with_column_objects_and_param_functions,
    active_tree_with_params,
    fail_if_active_periods_overlap,
    fail_if_environment_is_invalid,
    fail_if_group_ids_are_outside_top_level_namespace,
    get_consecutive_int_1d_lookup_table_param_value,
    get_consecutive_int_2d_lookup_table_param_value,
    get_month_based_phase_inout_of_age_thresholds_param_value,
    get_year_based_phase_inout_of_age_thresholds_param_value,
    orig_tree_with_column_objects_and_param_functions,
    orig_tree_with_params,
    policy_environment,
)
from ttsim.rounding import RoundingSpec
from ttsim.shared import (
    insert_path_and_value,
    join,
    merge_trees,
    to_datetime,
    upsert_path_and_value,
    upsert_tree,
)


def possible_targets():
    return {
        "orig_tree_with_column_objects_and_param_functions": orig_tree_with_column_objects_and_param_functions,
        "orig_tree_with_params": orig_tree_with_params,
        "active_tree_with_column_objects_and_param_functions": active_tree_with_column_objects_and_param_functions,
        "active_tree_with_params": active_tree_with_params,
        "fail_if_active_periods_overlap": fail_if_active_periods_overlap,
        "fail_if_environment_is_invalid": fail_if_environment_is_invalid,
        "fail_if_group_ids_are_outside_top_level_namespace": fail_if_group_ids_are_outside_top_level_namespace,
        "policy_environment": policy_environment,
        "compute_taxes_and_transfers": compute_taxes_and_transfers,
        "qual_name_data_columns": qual_name_data_columns,
        "nested_results": nested_results,
        "qual_name_results": qual_name_results,
        "tax_transfer_dag": tax_transfer_dag,
        "tax_transfer_function": tax_transfer_function,
        "qual_name_targets": qual_name_targets,
        "qual_name_column_targets": qual_name_column_targets,
        "qual_name_param_targets": qual_name_param_targets,
        "qual_name_own_targets": qual_name_own_targets,
        "fail_if_any_paths_are_invalid": fail_if_any_paths_are_invalid,
        "flat_policy_environment_with_derived_functions_and_without_overridden_functions": flat_policy_environment_with_derived_functions_and_without_overridden_functions,
        "top_level_namespace": top_level_namespace,
        "column_functions_with_processed_params_and_scalars": column_functions_with_processed_params_and_scalars,
        "required_column_functions": required_column_functions,
        "fail_if_targets_tree_is_invalid": fail_if_targets_tree_is_invalid,
        "fail_if_targets_are_not_in_policy_environment_or_data": fail_if_targets_are_not_in_policy_environment_or_data,
        "fail_if_data_tree_is_invalid": fail_if_data_tree_is_invalid,
        "fail_if_group_variables_are_not_constant_within_groups": fail_if_group_variables_are_not_constant_within_groups,
        "fail_if_foreign_keys_are_invalid_in_data": fail_if_foreign_keys_are_invalid_in_data,
        "warn_if_elements_overridden_by_data": warn_if_elements_overridden_by_data,
        "fail_if_root_nodes_are_missing": fail_if_root_nodes_are_missing,
        "qual_name_input_data": qual_name_input_data,
    }


def main(inputs: dict[str, Any], targets: list[str] | None = None) -> dict[str, Any]:
    """
    Main function that processes the inputs and returns the outputs.
    """
    dag = dags.create_dag(
        functions=possible_targets(),
        targets=targets,
    )
    draw_tax_transfer_dag(dag, "dag.html")

    return inputs


def draw_tax_transfer_dag(
    tax_transfer_dag: nx.DiGraph,
    output_path: str = "tax_transfer_dag.html",
) -> None:
    """Draw the tax transfer DAG and save it as an interactive HTML file.

    Parameters
    ----------
    tax_transfer_dag
        The DAG to draw.
    output_path
        The path where to save the HTML file.
    """
    import plotly.graph_objects as go

    # Get node positions using spring layout
    pos = nx.spring_layout(tax_transfer_dag, k=1, iterations=50)

    # Create edge trace
    edge_x = []
    edge_y = []
    for edge in tax_transfer_dag.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    for node in tax_transfer_dag.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=node_text,
        textposition="top center",
        marker=dict(
            showscale=False,
            colorscale="YlGnBu",
            size=20,
            colorbar=dict(
                thickness=15,
                title="Node Connections",
                xanchor="left",
                titleside="right",
            ),
        ),
    )

    # Create the figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="Tax Transfer DAG",
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    # Save as HTML
    fig.write_html(output_path)


__all__ = [
    "AggByGroupFunction",
    "AggByPIDFunction",
    "AggType",
    "ConsecutiveInt1dLookupTableParam",
    "ConsecutiveInt1dLookupTableParamValue",
    "ConsecutiveInt2dLookupTableParamValue",
    "DictParam",
    "FKType",
    "FunctionsAndDataOverlapWarning",
    "GroupCreationFunction",
    "OrigTreesWithFileNames",
    "ParamFunction",
    "ParamObject",
    "PiecewisePolynomialParam",
    "PiecewisePolynomialParamValue",
    "PolicyFunction",
    "PolicyInput",
    "RawParam",
    "RoundingSpec",
    "ScalarParam",
    "TimeConversionFunction",
    "_add_derived_functions",
    "agg_by_group_function",
    "agg_by_p_id_function",
    "compute_taxes_and_transfers",
    "create_time_conversion_functions",
    "dataframe_to_nested_data",
    "get_consecutive_int_1d_lookup_table_param_value",
    "get_consecutive_int_2d_lookup_table_param_value",
    "get_month_based_phase_inout_of_age_thresholds_param_value",
    "get_piecewise_parameters",
    "get_year_based_phase_inout_of_age_thresholds_param_value",
    "group_creation_function",
    "insert_path_and_value",
    "join",
    "merge_trees",
    "nested_data_to_dataframe",
    "param_function",
    "piecewise_polynomial",
    "plot_dag",
    "policy_environment",
    "policy_function",
    "policy_input",
    "to_datetime",
    "upsert_path_and_value",
    "upsert_tree",
]
