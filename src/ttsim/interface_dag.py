from __future__ import annotations

import inspect
import re
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import dags
import dags.tree as dt
import optree

from ttsim.interface_dag_elements import AllOutputNames
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
from ttsim.main_args import MainArg

if TYPE_CHECKING:
    import datetime

    from ttsim.interface_dag_elements.typing import (
        DashedISOString,
        FlatInterfaceObjects,
        PolicyEnvironment,
        QNameData,
        QNameStrings,
        UnorderedQNames,
    )
    from ttsim.main_args import (
        InputData,
        Labels,
        OrigPolicyObjects,
        Output,
        RawResults,
        Results,
        SpecializedEnvironment,
        Targets,
    )


def main(
    *,
    output: Output | None = None,
    date_str: DashedISOString | None = None,
    input_data: InputData | None = None,
    targets: Targets | None = None,
    backend: Literal["numpy", "jax"] | None = None,
    rounding: bool = True,
    fail_and_warn: bool = True,
    orig_policy_objects: OrigPolicyObjects | None = None,
    raw_results: RawResults | None = None,
    results: Results | None = None,
    specialized_environment: SpecializedEnvironment | None = None,
    policy_environment: PolicyEnvironment | None = None,
    processed_data: QNameData | None = None,
    date: datetime.date | None = None,
    policy_date_str: DashedISOString | None = None,
    evaluation_date_str: DashedISOString | None = None,
    policy_date: datetime.date | None = None,
    evaluation_date: datetime.date | None = None,
    labels: Labels | None = None,
) -> dict[str, Any]:
    """
    Main function that processes the inputs and returns the outputs.
    """

    input_qnames = _harmonize_inputs(locals())
    output_qnames = _harmonize_output(output)

    # If requesting an input template, we do not require any data.
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

    # Not strictly necessary, but helps with debugging.
    dag = dags.create_dag(
        functions=functions,
        targets=output_qnames["names"],
    )

    def lexsort_key(x: str) -> int:
        return 0 if x.startswith("fail_if") else 1

    if output_qnames["name"]:
        f = dags.concatenate_functions(
            dag=dag,
            functions=functions,
            targets=output_qnames["name"],
            enforce_signature=False,
            set_annotations=False,
            lexsort_key=lexsort_key,
        )
    else:
        f = dags.concatenate_functions(
            dag=dag,
            functions=functions,
            targets=output_qnames["names"],
            return_type="dict",
            enforce_signature=False,
            set_annotations=False,
            lexsort_key=lexsort_key,
        )
    return f(**input_qnames)


def _harmonize_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    dict_inputs = {
        k: v.to_dict() if isinstance(v, MainArg) else v for k, v in inputs.items()
    }
    qname_inputs = {}
    opo = dict_inputs.get("orig_policy_objects")
    if opo and "root" in opo:
        qname_inputs["orig_policy_objects__root"] = opo.pop("root")
    for acc in optree.tree_accessors(AllOutputNames.to_dict(), none_is_leaf=True):
        qname = dt.qname_from_tree_path(acc.path)
        with suppress(KeyError, TypeError):
            qname_inputs[qname] = acc(dict_inputs)
    return {k: v for k, v in qname_inputs.items() if v is not None}


def _harmonize_output(output: Output | None) -> dict[str, Any]:
    if output is None:
        flat_output: dict[str, Any] = {
            "name": None,
            "names": None,
        }
    elif hasattr(output, "to_dict"):  # Check if it's a MainArg-like object
        flat_output = output.to_dict()
    else:
        flat_output = {
            "name": output.get("name"),
            "names": output.get("names"),
        }

    if flat_output["name"] is not None:
        if isinstance(flat_output["name"], tuple):
            flat_output["name"] = dt.qname_from_tree_path(flat_output["name"])
        elif isinstance(flat_output["name"], dict):
            if len(flat_output["name"]) > 1:
                raise ValueError(
                    "The output Name must be a single qualified name, a tuple or a "
                    "dict with one element. If you want to output multiple "
                    "elements, use 'names'."
                )
            flat_output["name"] = dt.qnames(flat_output["name"])[0]
        flat_output["names"] = [flat_output["name"]]
    if isinstance(flat_output["names"], dict):
        flat_output["names"] = dt.qnames(flat_output["names"])
    elif isinstance(flat_output["names"][0], tuple):
        flat_output["names"] = [
            dt.qname_from_tree_path(tp) for tp in flat_output["names"]
        ]

    return flat_output


def _resolve_dynamic_interface_objects_to_static_nodes(
    flat_interface_objects: FlatInterfaceObjects,
    input_qnames: list[str],
) -> dict[str, InterfaceFunction | InterfaceInput]:
    """Resolve dynamic interface objects to static nodes.

    Make InputDependentInterfaceFunctions static by checking the input data and picking
    among the functions with the same leaf name the one that satisfies the include
    condition.

    Fail if multiple functions with the same leaf name satisfy the include condition.

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
            new_path = (*orig_p[:-1], orig_object.leaf_name)
            if new_path not in path_to_idif:
                path_to_idif[new_path] = []
            path_to_idif[new_path].append(orig_object)
        else:
            static_nodes[dt.qname_from_tree_path(orig_p)] = orig_object

    for p, functions in path_to_idif.items():
        functions_satisfying_include_condition = [
            f for f in functions if f.include_condition_satisfied(input_qnames)
        ]
        _fail_if_multiple_functions_satisfy_include_condition(
            funcs=functions_satisfying_include_condition,
            path=p,
        )
        if functions_satisfying_include_condition:
            static_nodes[dt.qname_from_tree_path(p)] = (
                functions_satisfying_include_condition[0]
            )
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
            "Put differently, there are multiple ways to build a specific target. "
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
        if p == "fail_if__root_nodes_are_missing":
            check = all(a in workers_and_their_inputs for a in args)
            if n.include_if_all_elements_present or n.include_if_any_element_present:
                # all(()) evaluates to True, so include first bit
                all_cond = n.include_if_all_elements_present and all(
                    a in workers_and_their_inputs
                    for a in n.include_if_all_elements_present
                )
                any_cond = any(
                    a in workers_and_their_inputs
                    for a in n.include_if_any_element_present
                )
                check = check and (all_cond or any_cond)
            if check:
                out.append(p)
    return out


def load_flat_interface_functions_and_inputs() -> FlatInterfaceObjects:
    """Load the collection of functions and inputs from the current directory."""
    orig_functions = _load_orig_functions()
    return _remove_tree_logic_from_functions_in_collection(
        orig_functions=orig_functions,
        top_level_namespace={
            (*path[:-1], func.leaf_name)[0] for path, func in orig_functions.items()
        },
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
        for qname, obj in inspect.getmembers(module):
            # If nesting happens (e.g., df+mapper), we need to be consistent.
            tree_path = dt.tree_path_from_qname(qname)
            if isinstance(obj, InterfaceFunction | InterfaceInput):
                if obj.in_top_level_namespace:
                    flat_functions[tree_path] = obj
                else:
                    flat_functions[(str(module.__name__), *tree_path)] = obj

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
