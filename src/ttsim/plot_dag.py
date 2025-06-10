from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

import dags
import dags.tree as dt
import networkx as nx
import numpy
import plotly.graph_objects as go

from ttsim.interface_dag import load_interface_functions_and_inputs, main
from ttsim.interface_dag_elements.fail_if import (
    format_list_linewise,
)
from ttsim.interface_dag_elements.interface_node_objects import InterfaceInput
from ttsim.tt_dag_elements import (
    ColumnObject,
    ParamFunction,
    ParamObject,
    PolicyFunction,
    PolicyInput,
)

if TYPE_CHECKING:
    from pathlib import Path


def plot_tt_dag(
    with_params: bool, inputs_for_main: dict[str, Any], title: str, output_path: Path
) -> None:
    """Plot the taxes & transfers DAG, with or without parameters."""

    policy_environment = main(inputs=inputs_for_main, targets=["policy_environment"])[
        "policy_environment"
    ]
    policy_inputs = {
        qn: n
        for qn, n in dt.flatten_to_qual_names(policy_environment).items()
        if isinstance(n, PolicyInput)
    }
    if with_params:
        targets = [
            qn
            for qn, n in dt.flatten_to_qual_names(policy_environment).items()
            if isinstance(n, PolicyFunction | ParamFunction)
        ]
    else:
        targets = [
            qn
            for qn, n in dt.flatten_to_qual_names(policy_environment).items()
            if isinstance(n, PolicyFunction)
        ]
    env = main(
        inputs={
            "policy_environment": policy_environment,
            "input_data__tree": dt.unflatten_from_qual_names(
                dict.fromkeys(policy_inputs, -1),
            ),
            "targets__tree": dt.unflatten_from_qual_names(
                dict.fromkeys(targets),
            ),
        },
        targets=[
            "specialized_environment__with_derived_functions_and_processed_input_nodes"
        ],
    )["specialized_environment__with_derived_functions_and_processed_input_nodes"]
    # Replace input nodes by PolicyInputs again
    env.update(policy_inputs)
    nodes = {
        qn: n.dummy_callable() if isinstance(n, PolicyInput | ParamObject) else n
        for qn, n in env.items()
    }
    dag = dags.create_dag(functions=nodes, targets=targets)
    # Only keep nodes that are column objects
    if not with_params:
        dag.remove_nodes_from(
            [qn for qn, n in env.items() if not isinstance(n, ColumnObject)]
        )
    fig = _plot_dag(dag=dag, title=title)
    if output_path.suffix == ".html":
        fig.write_html(output_path)
    else:
        raise ValueError(f"Unsupported file extension: {output_path.suffix}")

    if with_params:
        f = dags.concatenate_functions(
            dag=dag,
            functions=nodes,
            targets=targets,
            return_type="dict",
            enforce_signature=False,
            set_annotations=False,
        )
        args = inspect.signature(f).parameters
        if args:
            raise ValueError(
                "The policy environment DAG should include all root nodes but requires "
                f"inputs:\n\n{format_list_linewise(args.keys())}"
            )


def plot_full_interface_dag(output_path: Path) -> None:
    """Plot the full interface DAG."""

    nodes = {
        p: n.dummy_callable() if isinstance(n, InterfaceInput) else n
        for p, n in load_interface_functions_and_inputs().items()
    }

    dag = dags.create_dag(functions=nodes, targets=None)
    f = dags.concatenate_functions(
        dag=dag,
        functions=nodes,
        targets=None,
        return_type="dict",
        enforce_signature=False,
        set_annotations=False,
    )
    args = inspect.signature(f).parameters
    if args:
        raise ValueError(
            "The full interface DAG should include all root nodes but requires inputs:"
            f"\n\n{format_list_linewise(args.keys())}"
        )
    fig = _plot_dag(dag=dag, title="Full Interface DAG")
    if output_path.suffix == ".html":
        fig.write_html(output_path)
    else:
        raise ValueError(f"Output path must end with .html: {output_path}")


def _plot_dag(dag: nx.DiGraph, title: str) -> go.Figure:
    """Plot the DAG."""

    nice_dag = nx.relabel_nodes(
        dag, {qn: qn.replace("__", "<br>") for qn in dag.nodes()}
    )
    pos = nx.nx_agraph.pygraphviz_layout(nice_dag, prog="dot", args="-Grankdir=LR")
    # Create edge traces with arrows
    edge_traces = []
    annotations = []

    for edge in nice_dag.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]

        # Calculate the direction vector
        dx = x1 - x0
        dy = y1 - y0
        length = numpy.sqrt(dx**2 + dy**2)

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
                line={"width": 1.5, "color": "#888"},
                hoverinfo="none",
                mode="lines",
            )
            edge_traces.append(edge_trace)

            # Add arrow using Plotly annotation
            annotations.append(
                {
                    "x": x1,
                    "y": y1,
                    "ax": x0,
                    "ay": y0,
                    "xref": "x",
                    "yref": "y",
                    "axref": "x",
                    "ayref": "y",
                    "arrowhead": 2,
                    "arrowsize": 1.25,
                    "arrowwidth": 2,
                    "arrowcolor": "#888",
                    "showarrow": True,
                    "text": "",
                }
            )

    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    node_colors = []

    for node in nice_dag.nodes():
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
        marker={
            "showscale": False,
            "color": node_colors,
            "size": 25,
            "line": {"width": 2, "color": "white"},
        },
    )

    # Create the figure with specified canvas size
    return go.Figure(
        data=[*edge_traces, node_trace],
        layout=go.Layout(
            title=title,
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin={"b": 40, "l": 40, "r": 40, "t": 60},
            width=1800,
            height=1200,
            annotations=annotations,
            xaxis={
                "showgrid": False,
                "zeroline": False,
                "showticklabels": False,
            },
            yaxis={
                "showgrid": False,
                "zeroline": False,
                "showticklabels": False,
            },
        ),
    )
