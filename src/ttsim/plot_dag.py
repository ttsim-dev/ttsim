from __future__ import annotations

import colorsys
import copy
import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, overload

import dags
import dags.tree as dt
import networkx as nx
import numpy
import plotly.graph_objects as go

from ttsim import main
from ttsim.interface_dag import load_interface_functions_and_inputs
from ttsim.interface_dag_elements.interface_node_objects import (
    InterfaceFunction,
    InterfaceInput,
    interface_function,
)
from ttsim.tt_dag_elements import (
    ColumnObject,
    ParamFunction,
    ParamObject,
    PolicyFunction,
    PolicyInput,
    param_function,
    policy_function,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from types import ModuleType

    from ttsim.interface_dag_elements.typing import QNameSpecializedEnvironment0


@dataclass(frozen=True)
class NodeSelector:
    nodes: list[str]
    type: Literal["neighbors", "descendants", "ancestors", "nodes"]
    order: int | None = None


@dataclass(frozen=True)
class NodeMetaData:
    description: str
    namespace: str


def plot_tt_dag(
    date_str: str,
    root: Path,
    node_selector: NodeSelector | None = None,
    namespace: str = "all",
    title: str = "",
    include_param_functions: bool = True,
    show_node_description: bool = False,
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
    show_node_description
        Whether to show node source code when hovering over a node.

    Returns
    -------
    The figure.
    """
    dag_with_node_metadata = _get_tt_dag_with_node_metadata(
        date_str=date_str,
        root=root,
        node_selector=node_selector,
        namespace=namespace,
        include_param_functions=include_param_functions,
    )
    return _plot_dag(
        dag=dag_with_node_metadata,
        title=title,
        show_node_description=show_node_description,
    )


def plot_interface_dag(show_node_description: bool = False) -> go.Figure:
    """Plot the full interface DAG."""
    nodes = {
        p: dummy_callable(n) if not callable(n) else n
        for p, n in load_interface_functions_and_inputs().items()
    }
    dag = dags.create_dag(functions=nodes, targets=None)

    for name, node_object in nodes.items():
        f = node_object.function if hasattr(node_object, "function") else node_object
        description = inspect.getdoc(f) or "No description available."
        namespace = name.split("__")[0] if "__" in name else "top-level"
        dag.nodes[name]["node_metadata"] = NodeMetaData(
            description=description,
            namespace=namespace,
        )

    return _plot_dag(
        dag=dag,
        title="Full Interface DAG",
        show_node_description=show_node_description,
    )


def _get_tt_dag_with_node_metadata(
    date_str: str,
    root: Path,
    node_selector: NodeSelector | None = None,
    namespace: str = "all",
    # Merge the two arguments above into one argument "target_nodes"
    include_param_functions: bool = True,
) -> nx.DiGraph:
    """Get the TT DAG to plot."""

    inputs_for_main = {
        "date_str": date_str,
        "orig_policy_objects__root": root,
        "targets__include_param_functions": include_param_functions,
        "targets__namespace": namespace,
    }

    all_plottable_nodes = (
        node_selector.nodes + all_targets_from_namespace(inputs_for_main)
        if node_selector
        else all_targets_from_namespace(inputs_for_main)
    )
    # all_plottable_nodes = [
    #     n
    #     for n in targets
    #     if isinstance(n, ColumnObject | ParamFunction | ParamObject)
    # ]

    specialized_environment = specialized_environment_for_plotting(inputs_for_main)

    all_nodes = {
        qn: dummy_callable(n) if not callable(n) else n
        for qn, n in specialized_environment.items()
    }

    complete_dag = dags.create_dag(functions=all_nodes, targets=all_plottable_nodes)

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

    # Add Node Metadata to DAG
    for name, node_object in all_nodes.items():
        if name not in selected_dag.nodes():
            continue

        f = node_object.function if hasattr(node_object, "function") else node_object

        description = inspect.getdoc(f) or "No description available."
        node_namespace = name.split("__")[0] if "__" in name else "top-level"
        selected_dag.nodes[name]["node_metadata"] = NodeMetaData(
            description=description,
            namespace=node_namespace,
        )

    return selected_dag


@overload
def dummy_callable(obj: PolicyInput) -> PolicyFunction: ...


@overload
def dummy_callable(obj: ParamObject) -> ParamFunction: ...  # type: ignore[overload-cannot-match]


@overload
def dummy_callable(obj: InterfaceInput) -> InterfaceFunction: ...  # type: ignore[overload-cannot-match]


def dummy_callable(obj: ModuleType | str | float | bool) -> Callable[[], Any]:
    """Dummy callable, for plotting and checking DAG completeness."""

    def dummy():  # type: ignore[no-untyped-def]  # noqa: ANN202
        pass

    if isinstance(obj, PolicyInput):
        return policy_function(
            leaf_name=obj.leaf_name,
            start_date=obj.start_date,
            end_date=obj.end_date,
            foreign_key_type=obj.foreign_key_type,
        )(dummy)
    if isinstance(obj, ParamObject):
        return param_function(
            leaf_name=obj.leaf_name,
            start_date=obj.start_date,
            end_date=obj.end_date,
        )(dummy)
    if isinstance(obj, InterfaceInput):
        return interface_function(
            leaf_name=obj.leaf_name,
            in_top_level_namespace=obj.in_top_level_namespace,
        )(dummy)
    return dummy


def all_targets_from_namespace(
    inputs_for_main: dict[str, Any],
) -> list[str]:
    """Get all targets from the original policy objects / params functions."""
    return main(
        inputs=inputs_for_main,
        targets=["targets__qname"],
    )["targets__qname"]


def specialized_environment_for_plotting(
    inputs_for_main: dict[str, Any],
) -> QNameSpecializedEnvironment0:
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
            "specialized_environment__without_tree_logic_and_with_derived_functions"
        ],
    )["specialized_environment__without_tree_logic_and_with_derived_functions"]
    return {
        **environment_with_overridden_policy_inputs,
        **dt.flatten_to_qnames(policy_inputs),
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
    show_node_description: bool,
) -> go.Figure:
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

    # Create namespace to color mapping with unique colors
    top_level_namespaces = {
        dag.nodes[node]["node_metadata"].namespace
        for node in dag.nodes()
        if "node_metadata" in dag.nodes[node]
    }
    n_namespaces = len(top_level_namespaces)
    namespace_colors = {
        namespace: hsl_to_hex(hue=i / n_namespaces, saturation=0.7, lightness=0.5)
        for i, namespace in enumerate(sorted(top_level_namespaces))
    }

    for node in nice_dag.nodes():
        metadata: NodeMetaData = nice_dag.nodes[node]["node_metadata"]

        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(
            node + "<br><br>" + metadata.description.replace("\n", "<br>")
            if show_node_description
            else node
        )

        node_color = (
            "#1f77b4"  # blue
            if metadata.namespace == "top-level"
            else namespace_colors[metadata.namespace]
        )
        node_colors.append(node_color)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        hoverlabel={
            "bgcolor": "white",
            "font": {"color": "black"},
            "bordercolor": "lightgray",
        },
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
