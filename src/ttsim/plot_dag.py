from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import dags
import dags.tree as dt
import networkx as nx
import numpy
import optree
import plotly.graph_objects as go

from ttsim.interface_dag import load_interface_functions_and_inputs, main
from ttsim.interface_dag_elements.fail_if import (
    format_list_linewise,
)
from ttsim.interface_dag_elements.interface_node_objects import InterfaceInput
from ttsim.tt_dag_elements import (
    ColumnObject,
    ParamObject,
    PolicyInput,
)

if TYPE_CHECKING:
    from pathlib import Path

    from ttsim.interface_dag_elements.typing import (
        QNameCombinedEnvironment0,
    )


@dataclass(frozen=True)
class NodeSelector:
    nodes: list[str]
    type: Literal["neighbors", "descendants", "ancestors", "nodes"]
    order: int | None = None


def plot_tt_dag(
    date_str: str,
    root: Path,
    node_selector: NodeSelector | None = None,
    namespace: str = "all",
    title: str = "",
    include_param_functions: bool = True,
    show_node_metadata: bool = False,  # noqa: ARG001
) -> go.Figure:
    """Plot the TT DAG.

    Parameters
    ----------
    date_str
        The date string.
    root
        The root path.
    node_selector
        The node selector. Default is None, i.e. the entire DAG of the namespace is
        plotted.
    namespace
        The namespace for which the plot should be created. Default is "all", i.e. the
        entire policy environment.
    title
        The title of the plot.
    include_param_functions
        Whether to include param functions.
    show_node_metadata
        Whether to show node metadata.

    Returns
    -------
    The figure.
    """
    inputs_for_main = {
        "date_str": date_str,
        "orig_policy_objects__root": root,
        "targets__include_param_functions": include_param_functions,
        "targets__namespace": namespace,
    }

    all_targets = all_targets_from_namespace(inputs_for_main)
    specialized_environment = specialized_environment_for_targets(inputs_for_main)

    all_nodes = {
        qn: n.dummy_callable() if isinstance(n, PolicyInput | ParamObject) else n
        for qn, n in specialized_environment.items()
    }
    complete_dag = dags.create_dag(functions=all_nodes, targets=all_targets)

    if node_selector is None:
        selected_dag = complete_dag
    else:
        selected_dag = create_dag_with_selected_nodes(
            complete_dag=complete_dag,
            node_selector=node_selector,
        )

    if not include_param_functions:
        selected_dag.remove_nodes_from(
            [
                qn
                for qn, n in specialized_environment.items()
                if not isinstance(n, ColumnObject)
            ]
        )

    return _plot_dag(dag=selected_dag, title=title)


def plot_full_interface_dag() -> go.Figure:
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
    return _plot_dag(dag=dag, title="Full Interface DAG")


def all_targets_from_namespace(
    inputs_for_main: dict[str, Any],
) -> list[str]:
    """Get all targets from the original policy objects / params functions."""
    return main(
        inputs=inputs_for_main,
        targets=["targets__qname"],
    )["targets__qname"]


def specialized_environment_for_targets(
    inputs_for_main: dict[str, Any],
) -> QNameCombinedEnvironment0:
    """Get the specialized environment for the targets."""
    # Replace policy inputs with dummy data
    policy_inputs = main(
        inputs=inputs_for_main,
        targets=["policy_environment__policy_inputs"],
    )["policy_environment__policy_inputs"]

    dummy_inputs = dt.unflatten_from_tree_paths(
        {qn: numpy.array([0]) for qn in dt.flatten_to_tree_paths(policy_inputs)}
    )

    environment_with_overridden_policy_inputs = main(
        inputs={
            **inputs_for_main,
            "input_data__tree": dummy_inputs,
        },
        targets=[
            "specialized_environment__with_derived_functions_and_processed_input_nodes"
        ],
    )["specialized_environment__with_derived_functions_and_processed_input_nodes"]
    return {
        **environment_with_overridden_policy_inputs,
        **dt.flatten_to_qual_names(policy_inputs),
    }


def create_dag_with_selected_nodes(
    complete_dag: nx.DiGraph,
    node_selector: NodeSelector,
) -> nx.DiGraph:
    """Select nodes based on the node selector."""
    if node_selector.type == "nodes":
        selected_nodes = node_selector.nodes
    elif node_selector.type == "ancestors":
        selected_nodes = optree.tree_flatten(
            [
                _kth_order_predecessors(complete_dag, node, order=node_selector.order)  # type: ignore[arg-type]
                if node_selector.order
                else list(nx.ancestors(complete_dag, node))
                for node in node_selector.nodes
            ]
        )[0]
    elif node_selector.type == "descendants":
        selected_nodes = optree.tree_flatten(
            [
                _kth_order_successors(complete_dag, node, order=node_selector.order)  # type: ignore[arg-type]
                if node_selector.order
                else list(nx.descendants(complete_dag, node))
                for node in node_selector.nodes
            ]
        )[0]
    elif node_selector.type == "neighbors":
        order = node_selector.order or 1
        selected_nodes = optree.tree_flatten(
            [
                _kth_order_neighbors(complete_dag, node, order=order)  # type: ignore[arg-type]
                for node in node_selector.nodes
            ]
        )[0]
    else:
        msg = (
            f"Invalid node selector type: {node_selector.type}. "
            "Choose one of 'nodes', 'ancestors', 'descendants', or 'neighbors'."
        )
        raise ValueError(msg)

    return complete_dag.remove_nodes_from(set(complete_dag.nodes) - set(selected_nodes))


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


def _kth_order_neighbors(
    dag: nx.DiGraph, node: str, order: int, base: set[str] | None = None
) -> set[str]:
    base = base or set()
    base.add(node)
    if order >= 1:
        for predecessor in dag.predecessors(node):
            base.update(
                _kth_order_predecessors(dag, predecessor, order=order - 1, base=base)
            )
        for successor in dag.successors(node):
            base.update(
                _kth_order_successors(dag, successor, order=order - 1, base=base)
            )
    return base


def _kth_order_predecessors(
    dag: nx.DiGraph, node: str, order: int, base: set[str] | None = None
) -> set[str]:
    base = base or set()
    base.add(node)
    if order >= 1:
        for predecessor in dag.predecessors(node):
            base.update(
                _kth_order_predecessors(dag, predecessor, order=order - 1, base=base)
            )
    return base


def _kth_order_successors(
    dag: nx.DiGraph, node: str, order: int, base: set[str] | None = None
) -> set[str]:
    base = base or set()
    base.add(node)
    if order >= 1:
        for successor in dag.successors(node):
            base.update(
                _kth_order_successors(dag, successor, order=order - 1, base=base)
            )
    return base
