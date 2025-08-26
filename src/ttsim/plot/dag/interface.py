from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

import dags
import dags.tree as dt

from ttsim.entry_point import load_flat_interface_functions_and_inputs
from ttsim.interface_dag_elements.interface_node_objects import (
    FailFunction,
    InputDependentInterfaceFunction,
    WarnFunction,
)
from ttsim.interface_dag_elements.specialized_environment_for_plotting_and_templates import (  # noqa: E501
    dummy_callable,
)
from ttsim.plot.dag.shared import NodeMetaData, get_figure

if TYPE_CHECKING:
    from pathlib import Path

    import plotly.graph_objects as go

INTERFACE_COLORMAP = {
    ("policy_date",): "gold",
    ("policy_date_str",): "gold",
    ("orig_policy_objects",): "gold",
    ("policy_environment",): "gold",
    ("evaluation_date",): "gold",
    ("evaluation_date_str",): "gold",
    ("rounding",): "teal",
    ("tt_function",): "teal",
    ("tt_function_set_annotations",): "teal",
    ("input_data",): "mediumblue",
    ("input_data", "sort_indices"): "lightblue",
    ("processed_data",): "midnightblue",
    ("labels",): "lemonchiffon",
    ("raw_results",): "lightgreen",
    ("results",): "lime",
    ("tt_targets",): "darkolivegreen",
    ("specialized_environment",): "darkgreen",
    ("specialized_environment_for_plotting_and_templates",): "palegreen",
    ("templates",): "limegreen",
    ("fail_if",): "indianred",
    ("warn_if",): "salmon",
    ("backend",): "lightgray",
    ("dnp",): "lightgray",
    ("num_segments",): "lightgray",
    ("xnp",): "lightgray",
}


def interface(
    include_fail_and_warn_nodes: bool = True,
    include_backend_nodes: bool = True,
    show_node_description: bool = False,
    output_path: Path | None = None,
    remove_orig_policy_objects__root: bool = True,
    node_colormap: dict[tuple[str, ...], str] | None = INTERFACE_COLORMAP,
    **kwargs: Any,  # noqa: ANN401
) -> go.Figure:
    """Plot the full interface DAG.

    Parameters
    ----------
    include_fail_and_warn_nodes
        Whether to include fail and warn nodes in the plot.
    include_backend_nodes
        Whether to include `backend`, `xnp`, and `dnp` in the plot.
    show_node_description
        Whether to show node descriptions on hover.
    output_path
        If provided, the figure is written to the path.
    remove_orig_policy_objects__root
        Whether to remove `orig_policy_objects__root` node from the plot.
    node_colormap
        Dictionary mapping namespace tuples to colors.
            - Tuples can represent any level of the namespace hierarchy (e.g.,
              ("input_data",) would be the first level,
              ("input_data", "df_and_mapper") the second level.
            - The tuple ("top-level",) is used to catch all members of the top-level
              namespace.
            - Individual elements or sub-namespaces can be overridden as the longest
              match will be used.
            - Fallback color is black.
            - Use any color from https://plotly.com/python/css-colors/
        If None, cycle through colors at the uppermost level of the namespace hierarchy.
    **kwargs
        Additional keyword arguments. Will be passed to
        plotly.graph_objects.Figure.layout.
    """
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

    # Will add the leaf names of input-dependent interface functions to the graph.
    dag = dags.create_dag(functions=nodes_without_idifs, targets=None)

    # Add edges manually for InputDependentInterfaceFunction
    input_dependent_interface_functions = {
        tp: n
        for tp, n in interface_functions_and_inputs.items()
        if isinstance(n, InputDependentInterfaceFunction)
    }
    for qn, ancestors in _qnames_of_final_idif_to_their_ancestors(
        input_dependent_interface_functions
    ).items():
        for ancestor in ancestors:
            dag.add_edge(ancestor, qn)

    final_idif_to_one_original = _qnames_of_final_idif_to_one_original(
        input_dependent_interface_functions
    )

    for node_name in dag.nodes():
        interface_object = nodes_without_idifs.get(node_name)
        if not interface_object:
            interface_object = final_idif_to_one_original[node_name]
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
    if not include_backend_nodes:
        dag.remove_nodes_from(["backend", "xnp", "dnp"])
    if remove_orig_policy_objects__root:
        dag.remove_nodes_from(["orig_policy_objects__root"])

    kwargs_with_title = kwargs
    if "title" not in kwargs:
        kwargs_with_title["title"] = "Full Interface DAG"

    fig = get_figure(
        dag=dag,
        show_node_description=show_node_description,
        node_colormap=node_colormap,
        **kwargs_with_title,
    )

    if output_path:
        fig.write_html(output_path)

    return fig


def _qnames_of_final_idif_to_their_ancestors(
    input_dependent_interface_functions: dict[
        tuple[str, ...], InputDependentInterfaceFunction
    ],
) -> dict[str, list[str]]:
    """Return a mapping of the qnames of the input dependent interface functions in the
    final DAG to all possible ancestors.
    """
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


def _qnames_of_final_idif_to_one_original(
    input_dependent_interface_functions: dict[
        tuple[str, ...], InputDependentInterfaceFunction
    ],
) -> dict[str, InputDependentInterfaceFunction]:
    """Return a mapping of the qnames of the input dependent interface functions in the
    final DAG to one of the original input dependent interface functions.

    Will be used for docstrings.
    """
    out: dict[str, InputDependentInterfaceFunction] = {}
    for orig_p, orig_object in input_dependent_interface_functions.items():
        final_qname = dt.qname_from_tree_path((*orig_p[:-1], orig_object.leaf_name))
        out[final_qname] = orig_object
    return out
