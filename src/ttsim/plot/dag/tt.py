from __future__ import annotations

import copy
import textwrap
from typing import TYPE_CHECKING, Any, Literal

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
    selection_type: Literal["neighbors", "descendants", "ancestors", "nodes"]
    | None = None,
    selection_depth: int | None = None,
    include_params: bool = True,
    show_node_description: bool = False,
    output_path: Path | None = None,
    # Elements of main
    policy_date_str: DashedISOString | None = None,
    orig_policy_objects: OrigPolicyObjects | None = None,
    input_data: InputData | None = None,
    tt_targets: TTTargets | None = None,
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
    selection_type
        The type of the DAG to plot. Can be one of:
        - "neighbors": Plot the neighbors of the target nodes.
        - "descendants": Plot the descendants of the target nodes.
        - "ancestors": Plot the ancestors of the target nodes.
        - "nodes": Plot the target nodes.
        If not provided, the entire DAG is plotted.
    selection_depth
        The depth of the selection. Only used if selection_type is "neighbors",
        "descendants", or "ancestors".
    include_params
        Include param functions when plotting the DAG. Default is True.
    show_node_description
        Show a description of the node when hovering over it.
    output_path
        If provided, the figure is written to the path.
    policy_date_str
        The date for which to plot the DAG.
    orig_policy_objects
        The orig policy objects.
    input_data
        The input data.
    tt_targets
        The TT targets.
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
        selection_type=selection_type,
        selection_depth=selection_depth,
        include_params=include_params,
        input_data=input_data,
        tt_targets=tt_targets,
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
        **kwargs,
    )
    if output_path:
        fig.write_html(output_path)

    return fig


def _get_tt_dag_with_node_metadata(
    root: Path | None = None,
    selection_type: Literal["neighbors", "descendants", "ancestors", "nodes"]
    | None = None,
    selection_depth: int | None = None,
    include_params: bool = True,
    input_data: InputData | None = None,
    tt_targets: TTTargets | None = None,
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
        tt_targets=tt_targets,
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
        tt_targets_qnames = _get_tt_targets_qnames(tt_targets)
        _fail_if_tt_targets_not_provided(tt_targets_qnames)
        selected_dag = select_nodes_from_dag(
            complete_tt_dag=complete_tt_dag,
            tt_targets_qnames=tt_targets_qnames,
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
    tt_targets_qnames: set[str],
    selection_type: Literal["neighbors", "descendants", "ancestors", "nodes"],
    selection_depth: int | None = None,
) -> nx.DiGraph:
    """Select nodes based on the node selector."""
    selected_nodes: set[str] = set()
    if selection_type == "nodes":
        selected_nodes.update(tt_targets_qnames)
    elif selection_type == "ancestors":
        for node in tt_targets_qnames:
            selected_nodes.update(
                _kth_order_predecessors(complete_tt_dag, node, order=selection_depth)
                if selection_depth
                else [*list(nx.ancestors(complete_tt_dag, node)), node]
            )
    elif selection_type == "descendants":
        for node in tt_targets_qnames:
            selected_nodes.update(
                _kth_order_successors(complete_tt_dag, node, order=selection_depth)
                if selection_depth
                else [*list(nx.descendants(complete_tt_dag, node)), node]
            )
    elif selection_type == "neighbors":
        order = selection_depth or 1
        for node in tt_targets_qnames:
            selected_nodes.update(
                _kth_order_neighbors(complete_tt_dag, node, order=order)
            )
    else:
        msg = (
            f"Invalid selection type: {selection_type}. "
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


def _get_tt_targets_qnames(tt_targets: TTTargets | dict[str, Any]) -> set[str]:
    """Extract qnames from TTTargets object."""
    if isinstance(tt_targets, TTTargets):
        # It's a TTTargets dataclass
        if tt_targets.qname is not None:
            return set(tt_targets.qname)
        if tt_targets.tree is not None:
            # Convert tree to qnames
            return set(dt.flatten_to_qnames(tt_targets.tree))
        return set()
    # It's a dict-like object with the same structure as TTTargets
    if "qname" in tt_targets and tt_targets["qname"] is not None:
        return set(tt_targets["qname"])
    if "tree" in tt_targets and tt_targets["tree"] is not None:
        # Convert tree to qnames
        return set(dt.flatten_to_qnames(tt_targets["tree"]))
    return set()


def _fail_if_tt_targets_not_provided(qnames: set[str] | None) -> None:
    if not qnames:
        msg = format_errors_and_warnings(
            "TT targets must be provided when using a selection type. "
            "To fix this, either set selection_type to None (this plots the entire DAG)"
            " or provide TT targets."
        )
        raise ValueError(msg)
