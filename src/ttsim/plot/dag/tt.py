from __future__ import annotations

import copy
import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import dags.tree as dt
import networkx as nx

from ttsim import main
from ttsim.interface_dag_elements.fail_if import format_errors_and_warnings
from ttsim.main_args import OrigPolicyObjects, TTTargets
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

    from ttsim.main_args import InputData
    from ttsim.typing import (
        DashedISOString,
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

    qnames: set[str]
    type: Literal["neighbors", "descendants", "ancestors", "nodes"]
    order: int | None = None


def tt(
    root: Path,
    policy_date_str: DashedISOString | None = None,
    policy_environment: PolicyEnvironment | None = None,
    input_data: InputData | None = None,
    node_selector: NodeSelector | None = None,
    title: str = "",
    include_params: bool = True,
    show_node_description: bool = False,
    output_path: Path | None = None,
) -> go.Figure:
    """Plot the TT DAG.

    Parameters
    ----------
    root
        The root path.
    policy_date_str
        The date string.
    policy_environment
        (Optional) The policy environment.
    input_data
        (Optional) The input data.
    node_selector
        (Optional) The node selector. If not provided, the entire DAG is plotted.
    title
        (Optional) The title of the plot.
    include_params
        (Optional) Include param functions when plotting the DAG. Default is True.
    show_node_description
        (Optional) Show a description of the node when hovering over it.
    output_path
        (Optional) If provided, the figure is written to the path.

    Returns
    -------
    The figure.
    """
    _fail_if_input_data_provided_without_node_selector(
        input_data=input_data, node_selector=node_selector
    )

    if node_selector:
        qname_node_selector = _QNameNodeSelector(
            qnames={dt.qname_from_tree_path(qn) for qn in node_selector.node_paths},
            type=node_selector.type,
            order=node_selector.order,
        )
    else:
        qname_node_selector = None

    dag_with_node_metadata = _get_tt_dag_with_node_metadata(
        root=root,
        policy_date_str=policy_date_str,
        policy_environment=policy_environment,
        input_data=input_data,
        node_selector=qname_node_selector,
        include_params=include_params,
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
    root: Path,
    policy_date_str: DashedISOString | None = None,
    policy_environment: PolicyEnvironment | None = None,
    input_data: InputData | None = None,
    node_selector: _QNameNodeSelector | None = None,
    include_params: bool = True,
) -> nx.DiGraph:
    """Get the TT DAG to plot."""
    if not policy_environment:
        policy_environment = main(
            main_target=MainTarget.policy_environment,
            orig_policy_objects=OrigPolicyObjects(root=root),
            policy_date_str=policy_date_str,
        )
    # It is not sufficient to use node_selector.qnames as tt_targets because of the
    # NodeSelector types 'neighbors' and 'descendants'. We must always create the
    # complete DAG (given input data) before selecting nodes.
    all_tt_targets = (
        [*dt.qnames(policy_environment), *node_selector.qnames]
        if node_selector
        else dt.qnames(policy_environment)
    )
    complete_tt_dag_and_specialized_environment = main(
        main_targets=[
            MainTarget.specialized_environment_from_policy_inputs.complete_tt_dag,
            MainTarget.specialized_environment_from_policy_inputs.without_tree_logic_and_with_derived_functions,
        ],
        orig_policy_objects=OrigPolicyObjects(root=root),
        policy_environment=policy_environment,
        tt_targets=TTTargets(qname=all_tt_targets),
        input_data=input_data,
        policy_date_str=policy_date_str,
    )
    complete_tt_dag = complete_tt_dag_and_specialized_environment[
        "specialized_environment_from_policy_inputs"
    ]["complete_tt_dag"]
    without_tree_logic_and_with_derived_functions = (
        complete_tt_dag_and_specialized_environment[
            "specialized_environment_from_policy_inputs"
        ]["without_tree_logic_and_with_derived_functions"]
    )

    if node_selector is None:
        selected_dag = complete_tt_dag
    else:
        selected_dag = select_nodes_from_dag(
            complete_tt_dag=complete_tt_dag,
            node_selector=node_selector,
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
                # Also add time-converted params
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
    for p, n in qn_env.items():
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
            else:
                continue
            # Wrap description at 79 characters
            descr = textwrap.fill(descr, width=79)
            out[p] = descr
    return out


def select_nodes_from_dag(
    complete_tt_dag: nx.DiGraph,
    node_selector: _QNameNodeSelector,
) -> nx.DiGraph:
    """Select nodes based on the node selector."""
    selected_nodes: set[str] = set()
    if node_selector.type == "nodes":
        selected_nodes.update(node_selector.qnames)
    elif node_selector.type == "ancestors":
        for node in node_selector.qnames:
            selected_nodes.update(
                _kth_order_predecessors(
                    complete_tt_dag, node, order=node_selector.order
                )
                if node_selector.order
                else [*list(nx.ancestors(complete_tt_dag, node)), node]
            )
    elif node_selector.type == "descendants":
        for node in node_selector.qnames:
            selected_nodes.update(
                _kth_order_successors(complete_tt_dag, node, order=node_selector.order)
                if node_selector.order
                else [*list(nx.descendants(complete_tt_dag, node)), node]
            )
    elif node_selector.type == "neighbors":
        order = node_selector.order or 1
        for node in node_selector.qnames:
            selected_nodes.update(
                _kth_order_neighbors(complete_tt_dag, node, order=order)
            )
    else:
        msg = (
            f"Invalid node selector type: {node_selector.type}. "
            "Choose one of 'nodes', 'ancestors', 'descendants', or 'neighbors'."
        )
        raise ValueError(msg)

    dag_copy = copy.deepcopy(complete_tt_dag)
    dag_copy.remove_nodes_from(set(complete_tt_dag.nodes) - set(selected_nodes))
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


def _fail_if_input_data_provided_without_node_selector(
    input_data: InputData | None,
    node_selector: NodeSelector | None,
) -> None:
    if not node_selector and input_data:
        msg = format_errors_and_warnings(
            "When providing input data, you must also provide a node selector."
            "This is because there is no way to tell which nodes should be part of the "
            "DAG when overriding intermediate nodes with input data. "
            "To specify a node selector, use "
            "\n\n >>> from ttsim.plot.dag import NodeSelector"
        )
        raise ValueError(msg)
