from __future__ import annotations

import colorsys
import fnmatch
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
    node_colormap: dict[tuple[str, ...] | str, str] | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> go.Figure:
    """Plot the DAG.

    Parameters
    ----------
    dag : nx.DiGraph
        The DAG to plot.
    show_node_description : bool
        Whether to show node descriptions on hover.
    node_colormap : dict[tuple[str, ...] | str, str] | None, optional
        Dictionary mapping namespace patterns to colors. Patterns can be specified
        as tuples (e.g., ("housing_benefits", "*_m")) or as qualified name strings
        (e.g., "housing_benefits__*_m"). If provided, overrides the default automatic
        color generation.
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
        # Normalize colormap to use tuples as keys
        normalized_colormap = _normalize_colormap(node_colormap)
        # Use provided colormap with glob pattern support
        for qname in dag.nodes():
            color = _find_color_for_qname(qname, normalized_colormap)
            individual_node_colormap[qname] = color
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


def _normalize_colormap(
    node_colormap: dict[tuple[str, ...] | str, str],
) -> dict[tuple[str, ...], str]:
    """Normalize colormap keys to tuples.

    Accepts both tuple patterns and qname strings (with '__' separators).
    Converts qname strings to tuple patterns.

    Parameters
    ----------
    node_colormap
        Dictionary with keys that are either tuples or qname strings.

    Returns
    -------
    dict[tuple[str, ...], str]
        Dictionary with all keys normalized to tuples.
    """
    normalized = {}
    for pattern, color in node_colormap.items():
        if isinstance(pattern, str):
            # Convert qname string to tuple
            normalized[dt.tree_path_from_qname(pattern)] = color
        else:
            normalized[pattern] = color
    return normalized


def _find_color_for_qname(qname: str, node_colormap: dict[tuple[str, ...], str]) -> str:
    """Find the color for a qname using glob pattern matching.

    Matching priority:
    1. Exact matches (no wildcards) take precedence
    2. Longer patterns are more specific than shorter ones
    3. Patterns without wildcards score higher than those with wildcards
    4. Among equal-specificity patterns, first defined wins

    Parameters
    ----------
    qname
        The qualified name to find a color for.
    node_colormap
        Dictionary mapping pattern tuples to colors. Patterns can contain
        glob wildcards (* for any characters, ? for single character).

    Returns
    -------
    str
        The color for the qname, or "black" if no match is found.
    """
    tp = dt.tree_path_from_qname(qname)

    best_match_color = None
    best_match_score = -1

    for pattern_tp, color in node_colormap.items():
        if _matches_glob_pattern(tp, pattern_tp):
            score = _pattern_specificity(tp, pattern_tp)
            if score > best_match_score:
                best_match_color = color
                best_match_score = score

    if best_match_color is not None:
        return best_match_color

    # Fall back to top-level color for single-element tree paths, else black
    if len(tp) == 1:
        return node_colormap.get(("top-level",), "dimgray")
    return "black"


def _matches_glob_pattern(tp: tuple[str, ...], pattern_tp: tuple[str, ...]) -> bool:
    """Check if a tree path matches a glob pattern tuple.

    The pattern tuple is matched against the tree path. Each element in the
    pattern is matched against the corresponding element in the tree path using
    fnmatch (glob-style matching).

    Special patterns:
    - `"**"` matches any number of path segments (including zero)
    - `("top-level",)` matches single-element tree paths only

    Examples:
    - `("housing_benefits",)` matches `housing_benefits__*` (prefix match)
    - `("**", "*_bg")` matches any path ending with `_bg` at any depth
    - `("ns", "**", "*_m")` matches `ns__*__*_m` at any depth

    Parameters
    ----------
    tp
        The tree path to match against.
    pattern_tp
        The pattern tuple, which may contain glob wildcards and `**`.

    Returns
    -------
    bool
        True if the tree path matches the pattern.
    """
    # Special case: ("top-level",) matches single-element tree paths
    if pattern_tp == ("top-level",):
        return len(tp) == 1

    # Handle patterns containing "**"
    if "**" in pattern_tp:
        return _matches_glob_pattern_with_doublestar(tp, pattern_tp)

    # Pattern must not be longer than the tree path for prefix matching
    if len(pattern_tp) > len(tp):
        return False

    return all(fnmatch.fnmatch(t, p) for t, p in zip(tp, pattern_tp, strict=False))


def _matches_glob_pattern_with_doublestar(
    tp: tuple[str, ...], pattern_tp: tuple[str, ...]
) -> bool:
    """Match a tree path against a pattern containing "**".

    The "**" wildcard matches any number of path segments (including zero).
    """
    # Find the position of the first "**"
    doublestar_idx = pattern_tp.index("**")

    # Split pattern into before and after the "**"
    before = pattern_tp[:doublestar_idx]
    after = pattern_tp[doublestar_idx + 1 :]

    # Check prefix (before the **)
    if len(before) > len(tp):
        return False
    if not all(fnmatch.fnmatch(t, p) for t, p in zip(tp, before, strict=False)):
        return False

    # Check if there are more "**" in the remaining pattern
    if "**" in after:
        # Recursively handle multiple "**"
        remaining_tp = tp[len(before) :]
        return any(
            _matches_glob_pattern_with_doublestar(remaining_tp[i:], after)
            for i in range(len(remaining_tp) + 1)
        )

    # Single "**" case: after must match suffix
    if len(after) > len(tp) - len(before):
        return False
    if not after:
        # No suffix pattern, "**" matches everything remaining
        return True

    # Match suffix from the end
    suffix_tp = tp[-len(after) :]
    return all(fnmatch.fnmatch(t, p) for t, p in zip(suffix_tp, after, strict=False))


def _pattern_specificity(tp: tuple[str, ...], pattern_tp: tuple[str, ...]) -> int:
    """Calculate the specificity score of a pattern match.

    Higher scores indicate more specific matches. The scoring prioritizes:
    1. Longer patterns (more levels matched)
    2. Exact matches over wildcard matches at each level
    3. More specific wildcards (fewer * and ? characters)
    4. Patterns with "**" have lower priority than equivalent patterns without
    5. The special ("top-level",) pattern has lowest priority (catch-all)

    Parameters
    ----------
    tp
        The tree path being matched.
    pattern_tp
        The pattern tuple that matched.

    Returns
    -------
    int
        A specificity score (higher = more specific).
    """
    # Special case: ("top-level",) is a catch-all with lowest priority
    if pattern_tp == ("top-level",):
        return -1

    # Count "**" wildcards - each one significantly reduces specificity
    doublestar_count = pattern_tp.count("**")

    # Base score from pattern length (each level worth 1000 points)
    # Subtract "**" from length since they don't represent specific levels
    effective_length = len(pattern_tp) - doublestar_count
    score = effective_length * 1000

    # Heavy penalty for "**" wildcards (they match anything)
    score -= doublestar_count * 500

    # Bonus for exact matches, penalty for wildcards
    for i, pattern_element in enumerate(pattern_tp):
        if pattern_element == "**":
            continue  # Already penalized above
        if i < len(tp):
            if pattern_element == tp[i]:
                # Exact match at this level
                score += 100
            else:
                # Wildcard match - penalize based on wildcard complexity
                wildcard_count = pattern_element.count("*") + pattern_element.count("?")
                score -= wildcard_count * 10

    return score
