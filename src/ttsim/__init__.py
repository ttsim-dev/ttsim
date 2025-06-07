# ruff: noqa
# type: ignore
from __future__ import annotations

from typing import Any

import dags
import networkx as nx
import numpy as np

from ttsim.interface_dag_elements.automatically_added_functions import TIME_UNIT_LABELS
from ttsim.interface_dag_elements.specialized_environment import (
    with_processed_params_and_scalars,
    with_derived_functions_and_processed_input_nodes,
    with_partialled_params_and_scalars,
    tax_transfer_dag,
    tax_transfer_function,
)
from ttsim.interface_dag_elements.raw_results import (
    columns,
    params,
    from_input_data,
    combined,
)
from ttsim.interface_dag_elements.processed_data import processed_data
from ttsim.interface_dag_elements.data_converters import (
    dataframe_to_nested_data,
    nested_data_to_df_with_mapped_columns,
    nested_data_to_df_with_nested_columns,
)
from ttsim.interface_dag_elements.input_data import tree as tree
from ttsim.interface_dag_elements.results import df, tree
from ttsim.interface_dag_elements.fail_if import (
    active_periods_overlap,
    environment_is_invalid,
    group_ids_are_outside_top_level_namespace,
    any_paths_are_invalid,
    input_data_tree_is_invalid,
    foreign_keys_are_invalid_in_data,
    group_variables_are_not_constant_within_groups,
    root_nodes_are_missing,
    targets_are_not_in_policy_environment_or_data,
    targets_tree_is_invalid,
    format_list_linewise,
    input_df_with_mapper_has_bool_or_numeric_column_names,
    input_mapper_has_incorrect_format,
    data_paths_are_missing_in_paths_to_column_names,
    non_convertible_objects_in_results_tree,
)
from ttsim.interface_dag_elements.warn_if import (
    FunctionsAndDataColumnsOverlapWarning,
    functions_and_data_columns_overlap,
)
from ttsim.interface_dag_elements.targets import (
    qname,
)
from ttsim.plot_dag import plot_dag
from ttsim.interface_dag_elements.orig_policy_objects import (
    column_objects_and_param_functions,
    param_specs,
)
from ttsim.interface_dag_elements.policy_environment import (
    policy_environment,
)
from ttsim.interface_dag_elements.names import (
    grouping_levels,
    top_level_namespace,
    processed_data_columns,
    target_columns,
    target_params,
    targets_from_input_data,
    root_nodes,
)
from ttsim.interface_dag_elements.shared import (
    insert_path_and_value,
    merge_trees,
    to_datetime,
    upsert_path_and_value,
    upsert_tree,
)
from ttsim.interface_dag import interface_functions_and_inputs
from ttsim.interface_dag_elements.interface_node_objects import InterfaceFunction


def main(inputs: dict[str, Any], targets: list[str] | None = None) -> dict[str, Any]:
    """
    Main function that processes the inputs and returns the outputs.
    """
    nodes = interface_functions_and_inputs()
    for key in inputs:
        if key in nodes:
            del nodes[key]

    functions = {
        name: node
        for name, node in nodes.items()
        if isinstance(node, InterfaceFunction)
    }

    # Collect all missing targets first
    missing_targets = []
    for t in targets:
        if t not in functions:
            missing_targets.append(t)

    # Raise error with all missing targets listed nicely
    if missing_targets:
        if len(missing_targets) == 1:
            raise ValueError(f"Target '{missing_targets[0]}' does not exist.")
        else:
            targets_str = format_list_linewise(missing_targets)
            raise ValueError(f"Targets '{targets_str}' do not exist.")

    dag = dags.create_dag(
        functions=functions,
        targets=targets,
    )
    # draw_dag(dag)
    f = dags.concatenate_functions(
        dag=dag,
        functions=functions,
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
    "TIME_UNIT_LABELS",
    "dataframe_to_nested_data",
    "get_consecutive_int_1d_lookup_table_param_value",
    "get_consecutive_int_2d_lookup_table_param_value",
    "get_month_based_phase_inout_of_age_thresholds_param_value",
    "get_piecewise_parameters",
    "get_year_based_phase_inout_of_age_thresholds_param_value",
    "group_creation_function",
    "insert_path_and_value",
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
    "FunctionsAndDataColumnsOverlapWarning",
]
