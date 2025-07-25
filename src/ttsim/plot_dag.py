from __future__ import annotations

import colorsys
import copy
import inspect
import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, overload

import dags
import dags.tree as dt
import networkx as nx
import numpy
import plotly.graph_objects as go

from ttsim import main
from ttsim.interface_dag_elements.interface_node_objects import (
    FailFunction,
    InputDependentInterfaceFunction,
    InterfaceFunction,
    InterfaceInput,
    WarnFunction,
    interface_function,
)
from ttsim.main import load_flat_interface_functions_and_inputs
from ttsim.tt import (
    ColumnFunction,
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

    from ttsim.typing import (
        PolicyEnvironment,
        SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
    )


@dataclass(frozen=True)
class NodeSelector:
    """Select nodes from the DAG."""

    node_paths: list[tuple[str, ...]]
    type: Literal["neighbors", "descendants", "ancestors", "nodes"]
    order: int | None = None


@dataclass(frozen=True)
class _QNameNodeSelector:
    """Select nodes from the DAG."""

    qnames: list[str]
    type: Literal["neighbors", "descendants", "ancestors", "nodes"]
    order: int | None = None


@dataclass(frozen=True)
class NodeMetaData:
    description: str
    namespace: str


def plot_tt_dag(
    policy_date_str: str,
    root: Path,
    node_selector: NodeSelector | None = None,
    title: str = "",
    include_params: bool = True,
    include_other_objects: bool = False,
    show_node_description: bool = False,
    output_path: Path | None = None,
) -> go.Figure:
    """Plot the TT DAG.

    Parameters
    ----------
    policy_date_str
        The date string.
    root
        The root path.
    node_selector
        The node selector. Default is None, i.e. the entire DAG is plotted.
    title
        The title of the plot.
    include_params
        Include param functions when plotting the DAG.
    include_other_objects
        Include backend policy inputs when plotting the DAG. Most users will not want
        this.
    show_node_description
        Show a description of the node when hovering over it.
    output_path
        If provided, the figure is written to the path.

    Returns
    -------
    The figure.
    """
    environment = main(
        main_target="policy_environment",
        policy_date_str=policy_date_str,
        orig_policy_objects={"root": root},
        backend="numpy",
    )

    if node_selector:
        qname_node_selector = _QNameNodeSelector(
            qnames=[dt.qname_from_tree_path(qn) for qn in node_selector.node_paths],
            type=node_selector.type,
            order=node_selector.order,
        )
    else:
        qname_node_selector = None

    dag_with_node_metadata = _get_tt_dag_with_node_metadata(
        environment=environment,
        node_selector=qname_node_selector,
        include_params=include_params,
        include_other_objects=include_other_objects,
    )
    # Remove backend, xnp, dnp, and num_segments from the TT DAG.
    dag_with_node_metadata.remove_nodes_from(
        [
            "backend",
            "xnp",
            "dnp",
            "num_segments",
        ]
    )
    fig = _plot_dag(
        dag=dag_with_node_metadata,
        title=title,
        show_node_description=show_node_description,
    )
    if output_path:
        fig.write_html(output_path)

    return fig


def plot_interface_dag(
    include_fail_and_warn_nodes: bool = True,
    show_node_description: bool = False,
    output_path: Path | None = None,
    remove_orig_policy_objects__root: bool = True,
) -> go.Figure:
    """Plot the full interface DAG."""
    interface_functions_and_inputs = load_flat_interface_functions_and_inputs()
    nodes_without_idifs = {
        dt.qname_from_tree_path(p): dummy_callable(obj=n, leaf_name=p[-1])
        if not callable(n)
        else n
        for p, n in interface_functions_and_inputs.items()
        if not isinstance(n, InputDependentInterfaceFunction)
    }
    if not include_fail_and_warn_nodes:
        nodes_without_idifs = {
            qn: n
            for qn, n in nodes_without_idifs.items()
            if not isinstance(n, (FailFunction, WarnFunction))
        }

    dag = dags.create_dag(functions=nodes_without_idifs, targets=None)

    # Add edges manually for InputDependentInterfaceFunction
    input_dependent_interface_functions = {
        qn: n
        for qn, n in interface_functions_and_inputs.items()
        if isinstance(n, InputDependentInterfaceFunction)
    }
    qnames_of_idif_to_their_ancestors = _qnames_of_idif_to_their_ancestors(
        input_dependent_interface_functions
    )
    for qn, ancestors in qnames_of_idif_to_their_ancestors.items():
        for ancestor in ancestors:
            dag.add_edge(ancestor, qn)

    for node_name in dag.nodes():
        interface_object = nodes_without_idifs.get(node_name)
        if interface_object:
            f = (
                interface_object.function
                if hasattr(interface_object, "function")
                else interface_object
            )
            description = inspect.getdoc(f)
        namespace = node_name.split("__")[0] if "__" in node_name else "top-level"
        dag.nodes[node_name]["node_metadata"] = NodeMetaData(
            description=description or "No description available.",
            namespace=namespace,
        )
    if remove_orig_policy_objects__root:
        dag.remove_nodes_from(["orig_policy_objects__root"])

    fig = _plot_dag(
        dag=dag,
        title="Full Interface DAG",
        show_node_description=show_node_description,
    )

    if output_path:
        fig.write_html(output_path)

    return fig


def _qnames_of_idif_to_their_ancestors(
    input_dependent_interface_functions: dict[
        tuple[str, ...], InputDependentInterfaceFunction
    ],
) -> dict[str, list[str]]:
    """Get the qnames of the input dependent interface functions to their ancestors."""
    idif_qname_to_idif_inputs: dict[str, list[str]] = {}
    for orig_p, orig_object in input_dependent_interface_functions.items():
        qname = dt.qname_from_tree_path((*orig_p[:-1], orig_object.leaf_name))
        if qname not in idif_qname_to_idif_inputs:
            idif_qname_to_idif_inputs[qname] = []
        ancestors = set(orig_object.include_if_all_inputs_present) | set(
            orig_object.include_if_any_input_present
        )
        idif_qname_to_idif_inputs[qname].extend(list(ancestors))
    return idif_qname_to_idif_inputs


def _get_tt_dag_with_node_metadata(
    environment: PolicyEnvironment,
    node_selector: _QNameNodeSelector | None = None,
    include_params: bool = True,
    include_other_objects: bool = False,
) -> nx.DiGraph:
    """Get the TT DAG to plot."""
    qname_environment = dt.flatten_to_qnames(environment)
    qnames_to_plot = list(qname_environment)
    if node_selector:
        # Node selector might contain derived functions that are not in qnames_to_plot
        qnames_to_plot.extend(node_selector.qnames)

    qnames_policy_inputs = [
        k
        for k, v in qname_environment.items()
        if isinstance(v, PolicyInput) and k in qnames_to_plot
    ]
    env = main(
        main_target="specialized_environment__without_tree_logic_and_with_derived_functions",
        policy_environment=environment,
        labels={"processed_data_columns": qnames_policy_inputs},
        tt_targets={"qname": qnames_to_plot},
        backend="numpy",
    )

    all_nodes = convert_all_nodes_to_callables(env)

    complete_dag = dags.create_dag(functions=all_nodes, targets=qnames_to_plot)

    if node_selector is None:
        selected_dag = complete_dag
    else:
        selected_dag = _create_dag_with_selected_nodes(
            complete_dag=complete_dag,
            node_selector=node_selector,
        )

    if not include_params:
        selected_dag.remove_nodes_from(
            [qn for qn, v in env.items() if isinstance(v, (ParamObject, ParamFunction))]
        )
    if not include_other_objects:
        selected_dag.remove_nodes_from(
            [
                qn
                for qn, n in env.items()
                if not isinstance(n, (ColumnFunction, ParamFunction, ParamObject))
            ]
        )

    node_descriptions = _get_node_descriptions(env)
    # Add Node Metadata to DAG
    for qn in all_nodes:
        if qn not in selected_dag.nodes():
            continue
        description = node_descriptions[qn]
        node_namespace = qn.split("__")[0] if "__" in qn else "top-level"
        selected_dag.nodes[qn]["node_metadata"] = NodeMetaData(
            description=description,
            namespace=node_namespace,
        )

    return selected_dag


def convert_all_nodes_to_callables(
    env: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
) -> SpecEnvWithoutTreeLogicAndWithDerivedFunctions:
    return {
        qn: dummy_callable(obj=n, leaf_name=dt.tree_path_from_qname(qn)[-1])
        if not callable(n)
        else n
        for qn, n in env.items()
    }


def _get_node_descriptions(
    env: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
) -> dict[str, str]:
    """Get the descriptions of the nodes in the environment."""
    out = {}
    for qn, n in env.items():
        descr = None
        if hasattr(n, "description"):
            if isinstance(n.description, str):
                descr = n.description
            elif (
                isinstance(n.description, dict)
                and "en" in n.description
                and n.description["en"] is not None
            ):
                descr = n.description["en"]
        if not descr:
            descr = "No description available."
        # Wrap description at 79 characters
        descr = textwrap.fill(descr, width=79)
        out[qn] = descr
    return out


@overload
def dummy_callable(obj: PolicyInput, leaf_name: str) -> PolicyFunction: ...


@overload
def dummy_callable(obj: ParamObject, leaf_name: str) -> ParamFunction: ...


@overload
def dummy_callable(obj: InterfaceInput, leaf_name: str) -> InterfaceFunction: ...


def dummy_callable(
    obj: ModuleType | str | float | bool, leaf_name: str
) -> Callable[[], Any]:
    """Dummy callable, for plotting and checking DAG completeness."""

    def dummy():  # type: ignore[no-untyped-def]  # noqa: ANN202
        pass

    if isinstance(obj, PolicyInput):
        return policy_function(
            leaf_name=leaf_name,
            start_date=obj.start_date,
            end_date=obj.end_date,
            foreign_key_type=obj.foreign_key_type,
        )(dummy)
    if isinstance(obj, ParamObject):
        return param_function(
            leaf_name=leaf_name,
            start_date=obj.start_date,
            end_date=obj.end_date,
        )(dummy)
    if isinstance(obj, InterfaceInput):
        return interface_function(
            leaf_name=leaf_name,
            in_top_level_namespace=obj.in_top_level_namespace,
        )(dummy)
    return dummy


def _create_dag_with_selected_nodes(
    complete_dag: nx.DiGraph,
    node_selector: _QNameNodeSelector,
) -> nx.DiGraph:
    """Select nodes based on the node selector."""
    selected_nodes: set[str] = set()
    if node_selector.type == "nodes":
        selected_nodes.update(node_selector.qnames)
    elif node_selector.type == "ancestors":
        for node in node_selector.qnames:
            selected_nodes.update(
                _kth_order_predecessors(complete_dag, node, order=node_selector.order)
                if node_selector.order
                else list(nx.ancestors(complete_dag, node))
            )
    elif node_selector.type == "descendants":
        for node in node_selector.qnames:
            selected_nodes.update(
                _kth_order_successors(complete_dag, node, order=node_selector.order)
                if node_selector.order
                else list(nx.descendants(complete_dag, node))
            )
    elif node_selector.type == "neighbors":
        order = node_selector.order or 1
        for node in node_selector.qnames:
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
            title={"text": title, "font": {"size": 16}},
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
