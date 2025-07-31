from __future__ import annotations

import copy
import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import dags
import dags.tree as dt
import networkx as nx

from ttsim import main
from ttsim.plot.dag.shared import NodeMetaData, dummy_callable, get_figure
from ttsim.tt import (
    ParamFunction,
    ParamObject,
    PolicyInput,
)

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any
    from collections.abc import Callable

    import plotly.graph_objects as go

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
    input_node_paths: list[tuple[str, ...]] | None = None


@dataclass(frozen=True)
class _QNameNodeSelector:
    """Select nodes from the DAG."""

    qnames: list[str]
    type: Literal["neighbors", "descendants", "ancestors", "nodes"]
    order: int | None = None
    input_qnames: list[str] | None = None


def tt(
    policy_date_str: str,
    root: Path,
    node_selector: NodeSelector | None = None,
    title: str = "",
    include_params: bool = True,
    show_node_description: bool = False,
    output_path: Path | None = None,
    input_node_paths: list[tuple[str, ...]] | None = None,
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
    show_node_description
        Show a description of the node when hovering over it.
    output_path
        If provided, the figure is written to the path.
    input_node_paths
        List of node paths to treat as inputs (i.e., exclude from computation and 
        show as leaf nodes). Each path is a tuple of strings representing the path 
        to a node in the policy tree. When provided, these nodes will be excluded 
        from the DAG computation, effectively "pruning" the graph to show what 
        would be computed if these values were provided as inputs.

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

    # Handle input_node_paths parameter - merge it into node_selector
    if input_node_paths is not None:
        if node_selector is None:
            node_selector = NodeSelector(
                node_paths=[],  # Empty list - we only care about the input_node_paths
                type="nodes",   # Default type
                input_node_paths=input_node_paths
            )
        else:
            # Create a new NodeSelector with the input_node_paths added
            node_selector = NodeSelector(
                node_paths=node_selector.node_paths,
                type=node_selector.type,
                order=node_selector.order,
                input_node_paths=input_node_paths,
            )

    if node_selector:
        input_qnames = None
        if node_selector.input_node_paths:
            input_qnames = [dt.qname_from_tree_path(qn) for qn in node_selector.input_node_paths]
        
        qname_node_selector = _QNameNodeSelector(
            qnames=[dt.qname_from_tree_path(qn) for qn in node_selector.node_paths],
            type=node_selector.type,
            order=node_selector.order,
            input_qnames=input_qnames,
        )
    else:
        qname_node_selector = None

    dag_with_node_metadata = _get_tt_dag_with_node_metadata(
        environment=environment,
        node_selector=qname_node_selector,
        include_params=include_params,
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
    fig = get_figure(
        dag=dag_with_node_metadata,
        title=title,
        show_node_description=show_node_description,
    )
    if output_path:
        fig.write_html(output_path)

    return fig


def _get_tt_dag_with_node_metadata(
    environment: PolicyEnvironment,
    node_selector: _QNameNodeSelector | None = None,
    include_params: bool = True,
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

    if node_selector and node_selector.input_qnames:
        # Create pruned environment by excluding input nodes before conversion
        env_without_inputs = {
            qn: n for qn, n in env.items() 
            if qn not in node_selector.input_qnames
        }
        # Also filter targets to exclude input nodes
        targets_without_inputs = [
            qn for qn in qnames_to_plot 
            if qn not in node_selector.input_qnames
        ]
        all_nodes = convert_all_nodes_to_callables(env_without_inputs)
        complete_dag = dags.create_dag(functions=all_nodes, targets=targets_without_inputs)  # type: ignore[arg-type]
        # Use the original env for metadata since complete_dag may still reference pruned nodes
        metadata_env = env
    else:
        all_nodes = convert_all_nodes_to_callables(env)
        complete_dag = dags.create_dag(functions=all_nodes, targets=qnames_to_plot)  # type: ignore[arg-type]
        metadata_env = env

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

    node_descriptions = _get_node_descriptions(metadata_env)
    # Add Node Metadata to DAG
    for qn in selected_dag.nodes():
        if qn in metadata_env:
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
                else [*list(nx.ancestors(complete_dag, node)), node]
            )
    elif node_selector.type == "descendants":
        for node in node_selector.qnames:
            selected_nodes.update(
                _kth_order_successors(complete_dag, node, order=node_selector.order)
                if node_selector.order
                else [*list(nx.descendants(complete_dag, node)), node]
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
