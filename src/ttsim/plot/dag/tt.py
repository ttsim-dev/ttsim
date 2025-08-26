from __future__ import annotations

import copy
import itertools
import textwrap
from typing import TYPE_CHECKING, Any, Literal

import dags.tree as dt
import networkx as nx

from ttsim.entry_point import main
from ttsim.interface_dag_elements.fail_if import format_errors_and_warnings
from ttsim.main_args import OrigPolicyObjects
from ttsim.main_target import MainTarget
from ttsim.plot.dag.shared import NodeMetaData, get_figure
from ttsim.tt import (
    ParamFunction,
    ParamObject,
    TimeConversionFunction,
)

if TYPE_CHECKING:
    from pathlib import Path

    import plotly.graph_objects as go

    from ttsim.main_args import InputData, Labels
    from ttsim.typing import (
        DashedISOString,
        PolicyEnvironment,
        QNameData,
        SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
    )


def tt(
    *,
    # Args specific to TTSIM plotting
    root: Path,
    primary_nodes: set[str] | set[tuple[str, str]] | None = None,
    selection_type: Literal["neighbors", "descendants", "ancestors", "all_paths"]
    | None = None,
    selection_depth: int | None = None,
    include_params: bool = True,
    show_node_description: bool = False,
    output_path: Path | None = None,
    node_colormap: dict[tuple[str, ...], str] | None = None,
    # Elements of main
    policy_date_str: DashedISOString | None = None,
    orig_policy_objects: OrigPolicyObjects | None = None,
    input_data: InputData | None = None,
    processed_data: QNameData | None = None,
    labels: Labels | None = None,
    policy_environment: PolicyEnvironment | None = None,
    backend: Literal["numpy", "jax"] = "numpy",
    include_fail_nodes: bool = True,
    include_warn_nodes: bool = True,
    # Args specific to plotly
    **kwargs: Any,  # noqa: ANN401
) -> go.Figure:
    """Plot the TT DAG.

    Parameters
    ----------
    root
        The root path.
    primary_nodes
        The qnames or paths of the primary nodes. Primary nodes are used to determine
        which other nodes to include in the plot based on the selection_type. They may
        be root nodes (for descendants), end nodes (for ancestors), or middle nodes (for
        neighbors). If not provided, the entire DAG is plotted.
    selection_type
        The type of the DAG to plot. Can be one of:
            - "neighbors": Plot the neighbors of the primary nodes.
            - "descendants": Plot the descendants of the primary nodes.
            - "ancestors": Plot the ancestors of the primary nodes.
            - "all_paths": All paths between the primary nodes are displayed (including
              any other nodes lying on these paths). You must pass at least two primary
              nodes.
        If not provided, the entire DAG is plotted.
    selection_depth
        The depth of the selection. Only used if selection_type is "neighbors",
        "descendants", or "ancestors".
    include_params
        Include params and param functions when plotting the DAG. Default is True.
    show_node_description
        Show a description of the node when hovering over it.
    output_path
        If provided, the figure is written to the path.
    node_colormap
        Dictionary mapping namespace tuples to colors.
            - Tuples can represent any level of the namespace hierarchy (e.g.,
              ("payroll_tax",) would be the first level,
              ("payroll_tax", "child_tax_credit") the second level.
            - The tuple ("top-level",) is used to catch all members of the top-level
              namespace.
            - Individual elements or sub-namespaces can be overridden as the longest
              match will be used.
            - Fallback color is black.
            - Use any color from https://plotly.com/python/css-colors/
        If None, cycle through colors at the uppermost level of the namespace hierarchy.
    policy_date_str
        The date for which to plot the DAG.
    orig_policy_objects
        The orig policy objects.
    input_data
        The input data.
    processed_data
        The processed data.
    labels
        The labels.
    policy_environment
        The policy environment.
    backend
        The backend to use when executing main.
    include_fail_nodes
        Whether to include fail nodes when executing main.
    include_warn_nodes
        Whether to include warn nodes when executing main.
    kwargs
        Additional keyword arguments. Will be passed to
        plotly.graph_objects.Figure.layout.

    Returns
    -------
    The figure.
    """
    dag_with_node_metadata = _get_tt_dag_with_node_metadata(
        root=root,
        primary_nodes=primary_nodes,
        selection_type=selection_type,
        selection_depth=selection_depth,
        include_params=include_params,
        input_data=input_data,
        policy_date_str=policy_date_str,
        policy_environment=policy_environment,
        orig_policy_objects=orig_policy_objects,
        processed_data=processed_data,
        labels=labels,
        backend=backend,
        include_fail_nodes=include_fail_nodes,
        include_warn_nodes=include_warn_nodes,
    )

    fig = get_figure(
        dag=dag_with_node_metadata,
        show_node_description=show_node_description,
        node_colormap=node_colormap,
        **kwargs,
    )
    if output_path:
        fig.write_html(output_path)

    return fig


def _get_tt_dag_with_node_metadata(
    root: Path | None = None,
    primary_nodes: set[str] | set[tuple[str, str]] | None = None,
    selection_type: Literal["neighbors", "descendants", "ancestors", "all_paths"]
    | None = None,
    selection_depth: int | None = None,
    include_params: bool = True,
    input_data: InputData | None = None,
    policy_date_str: DashedISOString | None = None,
    policy_environment: PolicyEnvironment | None = None,
    orig_policy_objects: OrigPolicyObjects | None = None,
    processed_data: QNameData | None = None,
    labels: Labels | None = None,
    backend: Literal["numpy", "jax"] = "numpy",
    include_fail_nodes: bool = True,
    include_warn_nodes: bool = True,
) -> nx.DiGraph:
    """Get the TT DAG to plot."""
    qnames_primary_nodes = _get_qnames_primary_nodes(primary_nodes)
    complete_tt_dag_and_specialized_environment = main(
        main_targets=[
            MainTarget.specialized_environment_for_plotting_and_templates.complete_tt_dag,
            MainTarget.specialized_environment_for_plotting_and_templates.without_tree_logic_and_with_derived_functions,
        ],
        policy_date_str=policy_date_str,
        orig_policy_objects=(
            orig_policy_objects if orig_policy_objects else OrigPolicyObjects(root=root)
        ),
        policy_environment=policy_environment,
        tt_targets={
            "qname": dict.fromkeys(qnames_primary_nodes)
            if qnames_primary_nodes
            else None
        },
        input_data=input_data,
        processed_data=processed_data,
        labels=labels,
        backend=backend,
        include_fail_nodes=include_fail_nodes,
        include_warn_nodes=include_warn_nodes,
    )
    complete_tt_dag = complete_tt_dag_and_specialized_environment[
        "specialized_environment_for_plotting_and_templates"
    ]["complete_tt_dag"]
    without_tree_logic_and_with_derived_functions = (
        complete_tt_dag_and_specialized_environment[
            "specialized_environment_for_plotting_and_templates"
        ]["without_tree_logic_and_with_derived_functions"]
    )

    if not selection_type:
        selected_dag = complete_tt_dag
    else:
        _fail_if_primary_nodes_not_specified(qnames_primary_nodes)
        selected_dag = select_nodes_from_dag(
            complete_tt_dag=complete_tt_dag,
            qnames_primary_nodes=qnames_primary_nodes,
            selection_type=selection_type,
            selection_depth=selection_depth,
        )

    if not include_params:
        qnames_params: set[str] = set()
        for qn, v in without_tree_logic_and_with_derived_functions.items():
            if isinstance(v, (ParamObject, ParamFunction)):
                qnames_params.add(qn)
            elif (
                isinstance(v, TimeConversionFunction)
                and hasattr(v, "source")
                and v.source in without_tree_logic_and_with_derived_functions
                and isinstance(
                    without_tree_logic_and_with_derived_functions[v.source],
                    (ParamObject, ParamFunction),
                )
            ):
                # Also schedule time-converted params for removal.
                qnames_params.add(qn)

        selected_dag.remove_nodes_from(qnames_params)

    # Handle 'special' nodes
    ## 1. Remove backend, xnp, dnp, and num_segments
    selected_dag.remove_nodes_from(
        [
            "backend",
            "xnp",
            "dnp",
            "num_segments",
        ]
    )
    ## 2. Remove policy_x, evaluation_x; x \in {year, month, day, date} nodes if they
    # are orphaned
    # This may happen because these are policy inputs that may be inputs to param
    # functions. Because the TT DAG is created **after** param functions are resolved,
    # they show up as orphaned nodes in the DAG if no other column object depends on
    # them. Hence, we remove them here explicitly.
    for x in [
        "policy_year",
        "policy_month",
        "policy_day",
        "policy_date",
        "evaluation_year",
        "evaluation_month",
        "evaluation_day",
        "evaluation_date",
    ]:
        if (
            x in selected_dag.nodes()
            and not set(selected_dag.predecessors(x))
            and not set(selected_dag.successors(x))
        ):
            selected_dag.remove_node(x)

    # Add Node Metadata to DAG
    node_descriptions = _get_node_descriptions(
        without_tree_logic_and_with_derived_functions
    )
    for qn in selected_dag.nodes():
        description = node_descriptions.get(qn, "No description available.")
        node_namespace = qn.split("__")[0] if "__" in qn else "top-level"
        selected_dag.nodes[qn]["node_metadata"] = NodeMetaData(
            description=description,
            namespace=node_namespace,
        )

    return selected_dag


def _get_node_descriptions(
    without_tree_logic_and_with_derived_functions: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,  # noqa: E501
) -> dict[str, str]:
    """Get the descriptions of the nodes in the environment."""
    qn_env = dt.flatten_to_qnames(without_tree_logic_and_with_derived_functions)
    out = {}
    for qn, node in qn_env.items():
        descr = None
        if hasattr(node, "description"):
            if isinstance(node.description, str):
                descr = node.description
            elif (
                isinstance(node.description, dict)
                and "en" in node.description
                and node.description["en"] is not None
            ):
                descr = node.description["en"]
            else:
                continue
            # Wrap description at 79 characters
            descr = textwrap.fill(descr, width=79)
            out[qn] = descr
    return out


def select_nodes_from_dag(
    complete_tt_dag: nx.DiGraph,
    qnames_primary_nodes: set[str],
    selection_type: Literal["neighbors", "descendants", "ancestors", "all_paths"],
    selection_depth: int | None = None,
) -> nx.DiGraph:
    """Select nodes to plot."""
    if selection_type == "neighbors":
        order = selection_depth or 1
        selected_nodes = {
            neighbor
            for node in qnames_primary_nodes
            for neighbor in _kth_order_neighbors(complete_tt_dag, node, order=order)
        }
    elif selection_type == "descendants":
        selected_nodes = {
            descendant
            for node in qnames_primary_nodes
            for descendant in (
                _kth_order_successors(complete_tt_dag, node, order=selection_depth)
                if selection_depth
                else [*list(nx.descendants(complete_tt_dag, node)), node]
            )
        }
    elif selection_type == "ancestors":
        selected_nodes = {
            ancestor
            for node in qnames_primary_nodes
            for ancestor in (
                _kth_order_predecessors(complete_tt_dag, node, order=selection_depth)
                if selection_depth
                else [*list(nx.ancestors(complete_tt_dag, node)), node]
            )
        }
    elif selection_type == "all_paths":
        _fail_if_less_than_two_primary_nodes(qnames_primary_nodes)
        selected_nodes = {
            node
            for start_node, end_node in itertools.permutations(qnames_primary_nodes, 2)
            for path in nx.all_simple_paths(complete_tt_dag, start_node, end_node)
            for node in path
        }
        selected_nodes = selected_nodes.union(qnames_primary_nodes)
    else:
        msg = (
            f"Invalid selection type: {selection_type}. "
            "Choose one of 'nodes', 'ancestors', 'descendants', or 'neighbors'."
        )
        raise ValueError(msg)

    dag_copy = copy.deepcopy(complete_tt_dag)
    dag_copy.remove_nodes_from(set(complete_tt_dag.nodes) - selected_nodes)
    return dag_copy


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


def _get_qnames_primary_nodes(
    primary_nodes: set[str] | set[tuple[str, str]] | None,
) -> set[str]:
    """Get the qnames of the selected nodes."""
    if not primary_nodes:
        return set()
    if all(isinstance(node, str) for node in primary_nodes):
        return primary_nodes  # type: ignore[return-value]
    if all(isinstance(node, tuple) for node in primary_nodes):
        return {dt.qname_from_tree_path(node) for node in primary_nodes}  # type: ignore[arg-type]
    msg = (
        "Primary nodes must be either a set of qnames or a set of tree paths. "
        f"Got {primary_nodes}."
    )
    raise ValueError(msg)


def _fail_if_primary_nodes_not_specified(qnames: set[str] | None) -> None:
    if not qnames:
        msg = format_errors_and_warnings(
            "You must not specify a selection type when no primary nodes are specified."
            " To fix this, either set 'selection_type' to None (this plots the entire "
            "DAG) or provide 'primary_nodes'."
        )
        raise ValueError(msg)


def _fail_if_less_than_two_primary_nodes(qnames_primary_nodes: set[str]) -> None:
    if len(qnames_primary_nodes) < 2:  # noqa: PLR2004
        msg = format_errors_and_warnings(
            "When using the 'all_paths' selection type, you must provide at least two "
            f"primary nodes. Got {len(qnames_primary_nodes)}."
        )
        raise ValueError(msg)
