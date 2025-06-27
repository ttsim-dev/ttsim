from __future__ import annotations

import inspect
import re
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import dags
import dags.tree as dt
import optree

from ttsim import arg_templates
from ttsim.interface_dag_elements import _InterfaceDAGElements
from ttsim.interface_dag_elements.fail_if import (
    format_errors_and_warnings,
    format_list_linewise,
)
from ttsim.interface_dag_elements.interface_node_objects import (
    FailOrWarnFunction,
    InputDependentInterfaceFunction,
    InterfaceFunction,
    InterfaceInput,
)
from ttsim.interface_dag_elements.orig_policy_objects import load_module

if TYPE_CHECKING:
    import datetime

    from ttsim.interface_dag_elements.typing import (
        FlatInterfaceObjects,
        QNameStrings,
        UnorderedQNames,
    )


def main(
    *,
    date_str: str | None = None,
    output: arg_templates.output.Name | arg_templates.output.Names | None = None,
    input_data: arg_templates.input_data.DfAndMapper
    | arg_templates.input_data.DfWithNestedColumns
    | arg_templates.input_data.Flat
    | arg_templates.input_data.QName
    | arg_templates.input_data.Tree
    | None = None,
    targets: dict[str, Any] | None = None,
    backend: Literal["numpy", "jax"] | None = None,
    rounding: bool = True,
    fail_and_warn: bool = True,
    orig_policy_objects: dict[str, Any] | None = None,
    raw_results: dict[str, Any] | None = None,
    results: dict[str, Any] | None = None,
    specialized_environment: dict[str, Any] | None = None,
    policy_environment: dict[str, Any] | None = None,
    processed_data: dict[str, Any] | None = None,
    dnp: dict[str, Any] | None = None,
    xnp: dict[str, Any] | None = None,
    date: datetime.date | None = None,
    labels: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Main function that processes the inputs and returns the outputs.
    """

    input_qnames = _harmonize_inputs(locals())
    output_qnames = _harmonize_outputs(output)

    if not any(re.match("(input|processed)_data", s) for s in input_qnames):
        input_qnames["processed_data"] = {}
        input_qnames["processed_data_columns"] = None

    nodes = _resolve_dynamic_interface_objects_to_static_nodes(
        flat_interface_objects=load_flat_interface_functions_and_inputs(),
        input_qnames=list(input_qnames),
    )

    _fail_if_requested_nodes_cannot_be_found(
        output_qnames=output_qnames["names"],
        nodes=nodes,
    )

    functions = {
        qn: n
        for qn, n in nodes.items()
        if isinstance(n, InterfaceFunction) and qn not in input_qnames
    }

    # If targets are None, all failures and warnings are included, anyhow.
    if fail_and_warn and output_qnames["names"] is not None:
        output_qnames["names"] = include_fail_and_warn_nodes(
            functions=functions,
            output_qnames=output_qnames["names"],
        )
    if output_qnames["name"] is None:
        f = dags.concatenate_functions(
            functions=functions,
            targets=output_qnames["names"],
            return_type="dict",
            enforce_signature=False,
            set_annotations=False,
        )
    else:
        f = dags.concatenate_functions(
            functions=functions,
            targets=output_qnames["name"],
            enforce_signature=False,
            set_annotations=False,
        )
    return f(**input_qnames)


def _harmonize_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    # Iterate over the skeleton and see whether we need to convert anything to
    # qualified names.
    flat_inputs = {}
    accs, vals = optree.tree_flatten_with_accessor(  # type: ignore[var-annotated]
        asdict(_InterfaceDAGElements()),  # type: ignore[arg-type]
        none_is_leaf=True,
    )[:2]
    if "input_data" in inputs and isinstance(
        inputs["input_data"], arg_templates.input_data.ABC
    ):
        inputs["input_data"] = inputs["input_data"].to_dict()
    for acc, val in zip(accs, vals, strict=False):
        qname = dt.qname_from_tree_path(acc.path)
        if qname in inputs:
            flat_inputs[qname] = inputs[qname]
        else:
            try:
                flat_inputs[qname] = acc(inputs)
            except (KeyError, TypeError):
                flat_inputs[qname] = val
    return {k: v for k, v in flat_inputs.items() if v is not None}


def _harmonize_outputs(
    output: arg_templates.output.Name | arg_templates.output.Names | None,
) -> dict[str, Any]:
    if output is None:
        flat_output = {
            "qname": None,
            "qnames": None,
        }
    elif isinstance(output, arg_templates.output.ABC):
        flat_output = output.to_dict()
        flat_output["name"] = flat_output.get("name")
        if isinstance(flat_output["name"], tuple):
            flat_output["name"] = dt.qname_from_tree_path(flat_output["name"])
        elif isinstance(flat_output["name"], dict):
            if len(flat_output["name"]) > 1:
                raise ValueError(
                    "The output Name must be a single qualified name, a tuple or a "
                    "dict with one element. If you want to output multiple elements, "
                    "use Names."
                )
            flat_output["name"] = dt.qnames(flat_output["name"])[0]
        flat_output["names"] = flat_output.get(
            "names", [flat_output["name"]] if flat_output["name"] is not None else None
        )
        if isinstance(flat_output["names"], dict):
            flat_output["names"] = dt.qnames(flat_output["names"])
        elif isinstance(flat_output["names"][0], tuple):
            flat_output["names"] = [
                dt.qname_from_tree_path(tp) for tp in flat_output["names"]
            ]
        elif isinstance(flat_output["names"][0], dict):
            # Happens if a dict was passed to Name.
            flat_output["names"] = dt.qnames(flat_output["names"][0])

    return flat_output


def _resolve_dynamic_interface_objects_to_static_nodes(
    flat_interface_objects: FlatInterfaceObjects,
    input_qnames: list[str],
) -> dict[str, InterfaceFunction | InterfaceInput]:
    """Resolve dynamic interface objects to static nodes.

    Make InputDependentInterfaceFunctions static by checking the input data and picking
    among the functions with the same leaf name the one that satisfies the include
    condition.

    Fails if multiple functions with the same leaf name satisfy the include condition.

    Parameters
    ----------
    flat_interface_objects
        The interface objects to resolve.
    input_qnames
        The input qnames to check the include conditions against.

    Returns
    -------
    A dictionary of static interface objects.

    """
    static_nodes: dict[str, InterfaceFunction | InterfaceInput] = {}
    path_to_idif: dict[tuple[str, ...], list[InputDependentInterfaceFunction]] = {}
    for orig_p, orig_object in flat_interface_objects.items():
        if isinstance(orig_object, InputDependentInterfaceFunction):
            new_path = orig_p[:-1] + (orig_object.leaf_name,)
            if new_path not in path_to_idif:
                path_to_idif[new_path] = []
            path_to_idif[new_path].append(orig_object)
        else:
            static_nodes[dt.qname_from_tree_path(orig_p)] = orig_object

    for p, funcs in path_to_idif.items():
        funcs_satisfying_include_condition = [
            f for f in funcs if f.include_condition_satisfied(input_qnames)
        ]
        _fail_if_multiple_functions_satisfy_include_condition(
            funcs=funcs_satisfying_include_condition,
            path=p,
        )
        if funcs_satisfying_include_condition:
            static_nodes[dt.qname_from_tree_path(p)] = (
                funcs_satisfying_include_condition[0]
            )
        else:
            # Default to the first function if no function satisfies the include
            # condition.
            static_nodes[dt.qname_from_tree_path(p)] = funcs[0]

    return static_nodes


def _fail_if_multiple_functions_satisfy_include_condition(
    funcs: list[InputDependentInterfaceFunction],
    path: tuple[str, ...],
) -> None:
    """Fail if multiple functions satisfy the include condition."""
    if len(funcs) > 1:
        func_names = "\n".join(f.original_function_name for f in funcs)
        msg = (
            f"Multiple InputDependentInterfaceFunctions with the path {path} "
            "satisfy their include conditions:\n\n"
            f"{func_names}\n\n"
            "Make sure the input data you provide satisfies only one of the include "
            "conditions."
        )
        raise ValueError(msg)


def include_fail_and_warn_nodes(
    functions: dict[str, InterfaceFunction],
    output_qnames: QNameStrings,
) -> list[str]:
    """Extend targets with failures and warnings that can be computed within the graph.

    FailOrWarnFunctions which are included in the targets are treated like regular
    functions.

    """
    fail_or_warn_functions = {
        p: n
        for p, n in functions.items()
        if isinstance(n, FailOrWarnFunction) and p not in output_qnames
    }
    workers_and_their_inputs = dags.create_dag(
        functions={
            p: n
            for p, n in functions.items()
            if not isinstance(n, FailOrWarnFunction) or p in output_qnames
        },
        targets=output_qnames,
    )
    out = output_qnames.copy()
    for p, n in fail_or_warn_functions.items():
        args = inspect.signature(n).parameters
        if all(a in workers_and_their_inputs for a in args) and (
            # all([]) evaluates to True.
            (
                n.include_if_all_elements_present
                and all(
                    a in workers_and_their_inputs
                    for a in n.include_if_all_elements_present
                )
            )
            or any(
                a in workers_and_their_inputs for a in n.include_if_any_element_present
            )
        ):
            out.append(p)
    return out


def load_flat_interface_functions_and_inputs() -> FlatInterfaceObjects:
    """Load the collection of functions and inputs from the current directory."""
    orig_functions = _load_orig_functions()
    return _remove_tree_logic_from_functions_in_collection(
        orig_functions=orig_functions,
        top_level_namespace={path[0] for path in orig_functions},
    )


def _load_orig_functions() -> dict[tuple[str, ...], InterfaceFunction | InterfaceInput]:
    """
    Load the interface functions and inputs from the current directory.

    """
    root = Path(__file__).parent / "interface_dag_elements"
    paths = [
        p for p in root.rglob("*.py") if p.name not in ["__init__.py", "typing.py"]
    ]
    flat_functions: dict[
        tuple[str, ...], InterfaceFunction | InterfaceInput | FailOrWarnFunction
    ] = {}
    for path in paths:
        module = load_module(path=path, root=root)
        for name, obj in inspect.getmembers(module):
            if isinstance(obj, InterfaceFunction | InterfaceInput):
                if obj.in_top_level_namespace:
                    flat_functions[(name,)] = obj
                else:
                    flat_functions[(str(module.__name__), name)] = obj

    return flat_functions


def _remove_tree_logic_from_functions_in_collection(
    orig_functions: dict[tuple[str, ...], InterfaceFunction | InterfaceInput],
    top_level_namespace: UnorderedQNames,
) -> FlatInterfaceObjects:
    """Map paths to column objects / param functions without tree logic."""
    return {
        path: obj.remove_tree_logic(
            tree_path=path,
            top_level_namespace=top_level_namespace,
        )
        for path, obj in orig_functions.items()
    }


def _fail_if_requested_nodes_cannot_be_found(
    output_qnames: list[str] | None,
    nodes: dict[str, InterfaceFunction | InterfaceInput],
) -> None:
    """Fail if some qname is not among nodes."""
    all_qnames = set(nodes.keys())
    interface_function_names = {
        p for p, n in nodes.items() if isinstance(n, InterfaceFunction)
    }
    fail_or_warn_functions = {
        p: n for p, n in nodes.items() if isinstance(n, FailOrWarnFunction)
    }

    # Output qnames not in interface functions
    if output_qnames is not None:
        missing_output_qnames = set(output_qnames) - set(interface_function_names)
    else:
        missing_output_qnames = set()

    # Qnames from include condtions of fail_or_warn functions not in nodes
    for n in fail_or_warn_functions.values():
        qns = {*n.include_if_all_elements_present, *n.include_if_any_element_present}
        missing_qnames_from_include_conditions = qns - all_qnames

    if missing_output_qnames or missing_qnames_from_include_conditions:
        if missing_output_qnames:
            msg = format_errors_and_warnings(
                "The following output names for the interface DAG are not among the "
                "interface functions or inputs:\n"
            ) + format_list_linewise(sorted(missing_output_qnames))
        else:
            msg = ""
        if missing_qnames_from_include_conditions:
            msg += format_errors_and_warnings(
                "\n\nThe following elements specified in some include condition of "
                "`fail_or_warn_function`s are not among the interface functions or "
                "inputs:\n"
            ) + format_list_linewise(sorted(missing_qnames_from_include_conditions))
        raise ValueError(msg)
