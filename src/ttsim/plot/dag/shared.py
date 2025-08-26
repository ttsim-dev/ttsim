from __future__ import annotations

import colorsys
from dataclasses import dataclass
from typing import Any

import dags.tree as dt
import networkx as nx
import numpy
import plotly.graph_objects as go


@dataclass(frozen=True)
class NodeMetaData:
    description: str
    namespace: str


def get_figure(
    dag: nx.DiGraph,
    show_node_description: bool,
    node_colormap: dict[tuple[str, ...], str] | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> go.Figure:
    """Plot the DAG.

    Parameters
    ----------
    dag : nx.DiGraph
        The DAG to plot.
    show_node_description : bool
        Whether to show node descriptions on hover.
    node_colormap : dict[tuple[str, ...], str] | None, optional
        Dictionary mapping namespace tuples to colors. If provided, overrides
        the default automatic color generation. Tuples can represent any level
        of the namespace hierarchy (e.g., ("housing_benefits",) for top-level,
        ("housing_benefits", "eligibility") for second-level).
    **kwargs : Any
        Additional keyword arguments passed to the plotly layout.
    """
    nice_dag = nx.relabel_nodes(dag, {qn: qname_to_label(qn) for qn in dag.nodes()})

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
    individual_node_colormap = {}
    node_colors = []

    # Create namespace to color mapping
    if node_colormap is not None:
        # Use provided colormap
        namespace_colors = {}
        for qname in dag.nodes():
            tp = dt.tree_path_from_qname(qname)

            longest_match = None
            longest_match_length = 0
            for map_tp in node_colormap:
                if tp[: len(map_tp)] == map_tp and len(map_tp) > longest_match_length:
                    longest_match = map_tp
                    longest_match_length = len(map_tp)

            if longest_match is not None:
                individual_node_colormap[qname] = node_colormap[longest_match]
            else:
                if len(tp) == 1:
                    individual_node_colormap[qname] = node_colormap.get(
                        ("top-level",), "dimgray"
                    )
                else:
                    individual_node_colormap[qname] = "black"
    else:
        # Use default automatic color generation
        top_level_namespaces = {
            dag.nodes[node]["node_metadata"].namespace
            for node in dag.nodes()
            if "node_metadata" in dag.nodes[node]
        }
        top_level_namespaces |= {"top-level"}
        n_namespaces = len(top_level_namespaces)
        namespace_colors = {
            namespace: hsl_to_hex(hue=i / n_namespaces, saturation=0.7, lightness=0.5)
            for i, namespace in enumerate(sorted(top_level_namespaces))
        }
        for qname in dag.nodes():
            ns = dt.tree_path_from_qname(dag.nodes[qname]["node_metadata"].namespace)[0]
            individual_node_colormap[qname] = namespace_colors[ns]

    for qname in dag.nodes():
        label = qname_to_label(qname)
        metadata: NodeMetaData = nice_dag.nodes[label]["node_metadata"]

        x, y = pos[label]
        node_x.append(x)
        node_y.append(y)
        node_text.append(
            label + "<br><br>" + metadata.description.replace("\n", "<br>")
            if show_node_description
            else label
        )
        node_colors.append(individual_node_colormap[qname])

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

    layout = go.Layout(
        showlegend=False,
        hovermode="closest",
        margin={"b": 40, "l": 40, "r": 40, "t": 60},
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
    )

    if kwargs:
        layout.update(kwargs)

    return go.Figure(data=[*edge_traces, node_trace], layout=layout)


def qname_to_label(qname: str) -> str:
    return qname.replace("__", "<br>")


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
