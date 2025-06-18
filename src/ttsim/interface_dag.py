from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dags
import dags.tree as dt

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
    from ttsim.interface_dag_elements.typing import (
        NestedTargetDict,
        QNameStrings,
        UnorderedQNames,
    )


def main(
    inputs: dict[str, Any],
    output_names: QNameStrings | NestedTargetDict | None = None,
    fail_and_warn: bool = True,
) -> dict[str, Any]:
    """
    Main function that processes the inputs and returns the outputs.
    """

    output_qnames = _harmonize_output_qnames(output_names)

    if not any(re.match("(input|processed)_data", s) for s in inputs):
        inputs["processed_data"] = {}
        inputs["processed_data_columns"] = None

    nodes = {
        p: n
        for p, n in load_interface_functions_and_inputs().items()
        if p not in inputs
    }

    # Replace InputDependentInterfaceFunction with InterfaceFunction
    for p, n in nodes.items():
        if isinstance(n, InputDependentInterfaceFunction):
            nodes[p] = n.resolve_to_static_interface_function(inputs)

    _fail_if_requested_nodes_cannot_be_found(
        output_qnames=output_qnames,
        nodes=nodes,
    )

    functions = {p: n for p, n in nodes.items() if isinstance(n, InterfaceFunction)}

    # If targets are None, all failures and warnings are included, anyhow.
    if fail_and_warn and output_qnames is not None:
        output_qnames = include_fail_and_warn_nodes(
            functions=functions,
            output_qnames=output_qnames,
        )

    f = dags.concatenate_functions(
        functions=functions,
        targets=output_qnames,
        return_type="dict",
        enforce_signature=False,
        set_annotations=False,
    )
    return f(**inputs)


def _harmonize_output_qnames(
    output_names: QNameStrings | NestedTargetDict | None,
) -> list[str] | None:
    if output_names is None:
        return None
    if isinstance(output_names, dict):
        return dt.qnames(output_names)
    return output_names


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


def load_interface_functions_and_inputs() -> dict[
    str,
    InterfaceFunction | InterfaceInput,
]:
    """Load the collection of functions and inputs from the current directory."""
    orig_functions = _load_orig_functions()
    return _remove_tree_logic_from_function_collection(
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


def _remove_tree_logic_from_function_collection(
    orig_functions: dict[tuple[str, ...], InterfaceFunction | InterfaceInput],
    top_level_namespace: UnorderedQNames,
) -> dict[str, InterfaceFunction | InterfaceInput]:
    """Map qualified names to column objects / param functions without tree logic."""
    return {
        dags.tree.qname_from_tree_path(path): obj.remove_tree_logic(
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
