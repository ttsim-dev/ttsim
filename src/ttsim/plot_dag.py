from __future__ import annotations

import colorsys
import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import dags
import dags.tree as dt
import networkx as nx
import numpy
import plotly.graph_objects as go

from ttsim.interface_dag import load_interface_functions_and_inputs, main
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
        UnorderedQNames,
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
    show_node_metadata: bool = False,
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
    dag = get_tt_dag_to_plot(
        date_str=date_str,
        root=root,
        node_selector=node_selector,
        namespace=namespace,
        include_param_functions=include_param_functions,
    )
    tln = top_level_namespaces(
        date_str=date_str,
        root=root,
        dag=dag,
    )
    return _plot_dag(
        dag=dag,
        title=title,
        top_level_namespaces=tln,
        show_node_metadata=show_node_metadata,
    )


def get_tt_dag_to_plot(
    date_str: str,
    root: Path,
    node_selector: NodeSelector | None = None,
    namespace: str = "all",
    include_param_functions: bool = True,
) -> nx.DiGraph:
    """Get the TT DAG to plot."""

    inputs_for_main = {
        "date_str": date_str,
        "orig_policy_objects__root": root,
        "targets__include_param_functions": include_param_functions,
        "targets__namespace": namespace,
    }

    all_targets = (
        node_selector.nodes + all_targets_from_namespace(inputs_for_main)
        if node_selector
        else all_targets_from_namespace(inputs_for_main)
    )

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

    return selected_dag


def top_level_namespaces(
    date_str: str,
    root: Path,
    dag: nx.DiGraph,
) -> UnorderedQNames:
    """Get the top level namespaces for this DAG.

    Returns the top-level namespaces that
        - actually appear in the DAG
        - collect leaf nodes or sub-namespaces (i.e. does not contain single top-level
          namespace elements)
    """
    top_level_namespace = main(
        inputs={
            "date_str": date_str,
            "orig_policy_objects__root": root,
        },
        targets=["names__top_level_namespace"],
    )["names__top_level_namespace"]
    return {
        n
        for n in top_level_namespace
        if any(node.startswith(f"{n}__") for node in dag.nodes())
    }


def plot_full_interface_dag(show_node_metadata: bool = False) -> go.Figure:
    """Plot the full interface DAG."""
    nodes = {
        p: n.dummy_callable() if isinstance(n, InterfaceInput) else n
        for p, n in load_interface_functions_and_inputs().items()
    }
    dag = dags.create_dag(functions=nodes, targets=None)
    top_level_namespaces = {n.split("__")[0] for n in dag.nodes() if "__" in n}
    return _plot_dag(
        dag=dag,
        title="Full Interface DAG",
        top_level_namespaces=top_level_namespaces,
        show_node_metadata=show_node_metadata,
    )


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
    selected_nodes: set[str] = set()
    if node_selector.type == "nodes":
        selected_nodes.update(node_selector.nodes)
    elif node_selector.type == "ancestors":
        for node in node_selector.nodes:
            selected_nodes.update(
                _kth_order_predecessors(complete_dag, node, order=node_selector.order)
                if node_selector.order
                else list(nx.ancestors(complete_dag, node))
            )
    elif node_selector.type == "descendants":
        for node in node_selector.nodes:
            selected_nodes.update(
                _kth_order_successors(complete_dag, node, order=node_selector.order)
                if node_selector.order
                else list(nx.descendants(complete_dag, node))
            )
    elif node_selector.type == "neighbors":
        order = node_selector.order or 1
        for node in node_selector.nodes:
            selected_nodes.update(_kth_order_neighbors(complete_dag, node, order=order))
    else:
        msg = (
            f"Invalid node selector type: {node_selector.type}. "
            "Choose one of 'nodes', 'ancestors', 'descendants', or 'neighbors'."
        )
        raise ValueError(msg)

    dag_copy = copy.deepcopy(complete_dag)
    dag_copy.remove_nodes_from(set(complete_dag.nodes) - set(selected_nodes))
    return dag_copy


def _plot_dag(
    dag: nx.DiGraph,
    title: str,
    top_level_namespaces: UnorderedQNames,
    show_node_metadata: bool,
) -> go.Figure:
    """Plot the DAG."""

    if show_node_metadata:
        raise NotImplementedError("Showing node metadata is not implemented yet.")

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

    # Create namespace to color mapping with unique colors
    n_namespaces = len(top_level_namespaces)
    namespace_colors = {
        namespace: hsl_to_hex(hue=i / n_namespaces, saturation=0.7, lightness=0.5)
        for i, namespace in enumerate(sorted(top_level_namespaces))
    }

    for node in nice_dag.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

        node_color = "#1f77b4"  # Default blue
        for namespace, color in namespace_colors.items():
            if node.startswith(f"{namespace}<br>"):
                node_color = color
                break
        node_colors.append(node_color)

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


def hsl_to_hex(hue: float, saturation: float, lightness: float) -> str:
    """Convert HSL color values to hexadecimal color code.

    Parameters
    ----------
    hue : float
        Hue value between 0 and 1, representing the position on the color wheel
        (0 = red, 0.33 = green, 0.66 = blue, 1 = red again)
    saturation : float
        Saturation value between 0 and 1, representing color intensity
        (0 = grayscale, 1 = fully saturated)
    lightness : float
        Lightness value between 0 and 1, representing brightness
        (0 = black, 0.5 = normal, 1 = white)

    Returns
    -------
    str
        Hexadecimal color code in the format '#RRGGBB'
    """

    rgb = colorsys.hls_to_rgb(h=hue, l=lightness, s=saturation)
    return f"#{int(rgb[0] * 255):02x}{int(rgb[1] * 255):02x}{int(rgb[2] * 255):02x}"
