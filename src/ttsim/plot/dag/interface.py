from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import dags
import dags.tree as dt

from ttsim.interface_dag_elements.interface_node_objects import (
    FailFunction,
    InputDependentInterfaceFunction,
    WarnFunction,
)
from ttsim.interface_dag_elements.specialized_environment_from_policy_inputs import (
    dummy_callable,
)
from ttsim.main import load_flat_interface_functions_and_inputs
from ttsim.plot.dag.shared import NodeMetaData, get_figure

if TYPE_CHECKING:
    from pathlib import Path

    import plotly.graph_objects as go


def interface(
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

    fig = get_figure(
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
