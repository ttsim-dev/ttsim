from __future__ import annotations

import colorsys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, overload

import networkx as nx
import numpy
import plotly.graph_objects as go

from ttsim.interface_dag_elements.interface_node_objects import (
    InterfaceFunction,
    InterfaceInput,
    interface_function,
)
from ttsim.tt import (
    ParamFunction,
    ParamObject,
    PolicyFunction,
    PolicyInput,
    param_function,
    policy_function,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType


@dataclass(frozen=True)
class NodeMetaData:
    description: str
    namespace: str


def get_figure(
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
