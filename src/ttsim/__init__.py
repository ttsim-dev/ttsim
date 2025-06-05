# ruff: noqa
# type: ignore
from __future__ import annotations

from typing import Any

import dags
import networkx as nx
import numpy as np

from ttsim.tt_dag_elements.aggregation import AggType
from ttsim.automatically_added_functions import create_time_conversion_functions
from ttsim.tt_dag_elements.column_objects_param_function import (
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
from ttsim.combined_environment import (
    combined_environment__with_processed_params_and_scalars,
    combined_environment__with_derived_functions_and_input_nodes,
    combined_environment__with_partialled_params_and_scalars,
    tax_transfer_dag,
    tax_transfer_function,
)
from ttsim.raw_results import (
    raw_results__columns,
    raw_results__params,
    raw_results__from_input_data,
    raw_results__combined,
)
from ttsim.processed_data import processed_data
from ttsim.data_converters import (
    dataframe_to_nested_data,
    nested_data_to_df_with_mapped_columns,
    nested_data_to_df_with_nested_columns,
)
from ttsim.input_data import tree as input_data__tree
from ttsim.results import results__df, results__tree
from ttsim.fail_if import (
    fail_if__active_periods_overlap,
    fail_if__environment_is_invalid,
    fail_if__group_ids_are_outside_top_level_namespace,
    fail_if__any_paths_are_invalid,
    fail_if__input_data_tree_is_invalid,
    fail_if__foreign_keys_are_invalid_in_data,
    fail_if__group_variables_are_not_constant_within_groups,
    fail_if__root_nodes_are_missing,
    fail_if__targets_are_not_in_policy_environment_or_data,
    fail_if__targets_tree_is_invalid,
    format_list_linewise,
)
from ttsim.fail_if import (
    FunctionsAndDataColumnsOverlapWarning,
    warn_if__functions_and_data_columns_overlap,
)
from ttsim.targets import (
    targets__qname,
)
from ttsim.tt_dag_elements.param_objects import (
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
from ttsim.tt_dag_elements.piecewise_polynomial import (
    get_piecewise_parameters,
    piecewise_polynomial,
)
from ttsim.plot_dag import plot_dag
from ttsim.orig_policy_objects import (
    column_objects_and_param_functions,
    param_specs,
)
from ttsim.policy_environment import (
    get_consecutive_int_1d_lookup_table_param_value,
    get_consecutive_int_2d_lookup_table_param_value,
    get_month_based_phase_inout_of_age_thresholds_param_value,
    get_year_based_phase_inout_of_age_thresholds_param_value,
    policy_environment,
)
from ttsim.names import (
    names__grouping_levels,
    names__top_level_namespace,
    names__processed_data_columns,
    names__target_columns,
    names__target_params,
    names__targets_from_input_data,
    names__root_nodes,
)
from ttsim.tt_dag_elements.rounding import RoundingSpec
from ttsim.shared import (
    insert_path_and_value,
    join,
    merge_trees,
    to_datetime,
    upsert_path_and_value,
    upsert_tree,
)


def function_collection():
    return {
        "combined_environment__with_processed_params_and_scalars": combined_environment__with_processed_params_and_scalars,
        "raw_results__columns": raw_results__columns,
        "fail_if__active_periods_overlap": fail_if__active_periods_overlap,
        "fail_if__any_paths_are_invalid": fail_if__any_paths_are_invalid,
        "fail_if__input_data_tree_is_invalid": fail_if__input_data_tree_is_invalid,
        "fail_if__environment_is_invalid": fail_if__environment_is_invalid,
        "fail_if__foreign_keys_are_invalid_in_data": fail_if__foreign_keys_are_invalid_in_data,
        "fail_if__group_ids_are_outside_top_level_namespace": fail_if__group_ids_are_outside_top_level_namespace,
        "fail_if__group_variables_are_not_constant_within_groups": fail_if__group_variables_are_not_constant_within_groups,
        "fail_if__root_nodes_are_missing": fail_if__root_nodes_are_missing,
        "fail_if__targets_are_not_in_policy_environment_or_data": fail_if__targets_are_not_in_policy_environment_or_data,
        "fail_if__targets_tree_is_invalid": fail_if__targets_tree_is_invalid,
        "combined_environment__with_derived_functions_and_input_nodes": combined_environment__with_derived_functions_and_input_nodes,
        "names__grouping_levels": names__grouping_levels,
        "input_data__tree": input_data__tree,
        "results__tree": results__tree,
        "orig_policy_objects__column_objects_and_param_functions": column_objects_and_param_functions,
        "orig_policy_objects__param_specs": param_specs,
        "policy_environment": policy_environment,
        "names__target_columns": names__target_columns,
        "processed_data": processed_data,
        "names__processed_data_columns": names__processed_data_columns,
        "names__root_nodes": names__root_nodes,
        "names__targets_from_input_data": names__targets_from_input_data,
        "names__target_params": names__target_params,
        "raw_results__combined": raw_results__combined,
        "raw_results__params": raw_results__params,
        "raw_results__from_input_data": raw_results__from_input_data,
        "targets__qname": targets__qname,
        "results__df": results__df,
        "combined_environment__with_partialled_params_and_scalars": combined_environment__with_partialled_params_and_scalars,
        "tax_transfer_dag": tax_transfer_dag,
        "tax_transfer_function": tax_transfer_function,
        "names__top_level_namespace": names__top_level_namespace,
        "warn_if__functions_and_data_columns_overlap": warn_if__functions_and_data_columns_overlap,
    }


def main(inputs: dict[str, Any], targets: list[str] | None = None) -> dict[str, Any]:
    """
    Main function that processes the inputs and returns the outputs.
    """
    possible_targets = function_collection()
    for key in inputs:
        if key in function_collection():
            del possible_targets[key]

    # Collect all missing targets first
    missing_targets = []
    for t in targets:
        if t not in possible_targets:
            missing_targets.append(t)

    # Raise error with all missing targets listed nicely
    if missing_targets:
        if len(missing_targets) == 1:
            raise ValueError(f"Target '{missing_targets[0]}' does not exist.")
        else:
            targets_str = format_list_linewise(missing_targets)
            raise ValueError(f"Targets '{targets_str}' do not exist.")

    dag = dags.create_dag(
        functions=possible_targets,
        targets=targets,
    )
    draw_dag(dag)
    f = dags.concatenate_functions(
        dag=dag,
        functions=possible_targets,
        targets=targets,
        return_type="dict",
        enforce_signature=False,
        set_annotations=False,
    )
    return f(**inputs)


def draw_dag(
    dag: nx.DiGraph,
    output_path: str = "tax_transfer_dag.html",
) -> None:
    """Draw the DAG and save it as an interactive HTML file.

    Parameters
    ----------
    dag
        The DAG to draw.
    output_path
        The path where to save the HTML file.
    """
    import plotly.graph_objects as go

    # Use Graphviz's dot layout for proper DAG visualization
    try:
        # Try to use pygraphviz for better DAG layout
        pos = nx.nx_agraph.pygraphviz_layout(dag, prog="dot", args="-Grankdir=LR")
    except (ImportError, FileNotFoundError):
        # Fallback to spring layout if pygraphviz is not available
        print("Warning: pygraphviz not available, using spring layout")
        pos = nx.spring_layout(dag, k=2, iterations=50)
        # Rotate to make it left-to-right
        pos = {node: (y, -x) for node, (x, y) in pos.items()}

    # Create edge traces with arrows
    edge_traces = []
    annotations = []

    for edge in dag.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]

        # Calculate the direction vector
        dx = x1 - x0
        dy = y1 - y0
        length = np.sqrt(dx**2 + dy**2)

        if length > 0:
            # Normalize the direction vector
            dx = dx / length
            dy = dy / length

            # Calculate start and end points with symmetric offsets
            offset = 50  # Offset in pygraphviz coordinate units
            x0 = x0 + dx * offset
            y0 = y0 + dy * offset
            x1 = x1 - dx * offset
            y1 = y1 - dy * offset

            # Create the edge line
            edge_trace = go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                line=dict(width=1.5, color="#888"),
                hoverinfo="none",
                mode="lines",
            )
            edge_traces.append(edge_trace)

            # Add arrow using Plotly annotation
            annotations.append(
                dict(
                    x=x1,
                    y=y1,
                    ax=x0,
                    ay=y0,
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    arrowhead=2,
                    arrowsize=1.25,
                    arrowwidth=2,
                    arrowcolor="#888",
                    showarrow=True,
                    text="",
                )
            )

    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    node_colors = []

    for node in dag.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

        # Color nodes that start with "fail_" in pale red
        if node.startswith("fail_"):
            node_colors.append("#ffb3b3")  # Pale red
        else:
            node_colors.append("#1f77b4")  # Blue

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker=dict(
            showscale=False,
            color=node_colors,
            size=25,
            line=dict(width=2, color="white"),
        ),
    )

    # Create the figure with specified canvas size (600x900)
    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            title="DAG Visualization",
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=40, l=40, r=40, t=60),
            width=1800,
            height=1200,
            annotations=annotations,
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
            ),
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
    "FunctionsAndDataColumnsOverlapWarning",
    "GroupCreationFunction",
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
    "agg_by_group_function",
    "agg_by_p_id_function",
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
    "nested_data_to_df_with_mapped_columns",
    "nested_data_to_df_with_nested_columns",
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
