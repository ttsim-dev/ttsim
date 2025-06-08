from __future__ import annotations

import networkx as nx
import numpy
import plotly.graph_objects as go


def plot_dag(dag: nx.DiGraph) -> None:
    """Plot the DAG."""
    pos = nx.nx_agraph.pygraphviz_layout(dag, prog="dot", args="-Grankdir=LR")
    # Create edge traces with arrows
    edge_traces = []
    annotations = []

    for edge in dag.edges():
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
        marker={
            "showscale": False,
            "color": node_colors,
            "size": 25,
            "line": {"width": 2, "color": "white"},
        },
    )

    # Create the figure with specified canvas size (600x900)
    return go.Figure(
        data=[*edge_traces, node_trace],
        layout=go.Layout(
            title="DAG Visualization",
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
