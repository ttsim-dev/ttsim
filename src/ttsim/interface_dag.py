from __future__ import annotations

import functools
import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import dags
import dags.tree as dt

from ttsim.interface_dag_elements.fail_if import (
    format_errors_and_warnings,
    format_list_linewise,
)
from ttsim.interface_dag_elements.interface_node_objects import (
    FailOrWarnFunction,
    InterfaceFunction,
    InterfaceInput,
    InterfaceFunctionFromUserInputs,
)
from ttsim.interface_dag_elements.orig_policy_objects import load_module

if TYPE_CHECKING:
    from collections.abc import KeysView

    from ttsim.interface_dag_elements.typing import UnorderedQNames


def main(
    inputs: dict[str, Any],
    targets: list[str] | None = None,
    backend: Literal["numpy", "jax"] = "numpy",
    fail_and_warn: bool = True,
) -> dict[str, Any]:
    """
    Main function that processes the inputs and returns the outputs.
    """
    if "backend" not in inputs:
        inputs["backend"] = backend

    nodes = {
        p: n
        for p, n in load_interface_functions_and_inputs().items()
        if p not in inputs
    }

    explicit_functions = {
        p: n
        for p, n in nodes.items()
        if isinstance(n, InterfaceFunction) and not isinstance(n, InterfaceFunctionFromUserInputs)
    }

    user_input_dependant_functions = generate_user_input_dependant_functions(
        inputs=inputs,
        functions={
            p: n
            for p, n in nodes.items()
            if isinstance(n, InterfaceFunctionFromUserInputs)
        },
    )

    all_functions = {
        **explicit_functions,
        **user_input_dependant_functions,
    }

    _fail_if_targets_are_not_among_interface_functions(
        targets=targets,
        interface_function_names=all_functions.keys(),
    )

    # If targets are None, all failures and warnings are included, anyhow.
    if fail_and_warn and targets is not None:
        targets = include_fail_and_warn_nodes(
            functions=all_functions,
            targets=targets,
        )

    f = dags.concatenate_functions(
        functions=all_functions,
        targets=targets,
        return_type="dict",
        enforce_signature=False,
        set_annotations=False,
    )
    return f(**inputs)


def generate_user_input_dependant_functions(
    inputs: dict[str, Any],
    functions: dict[str, InterfaceFunctionFromUserInputs],
) -> dict[str, InterfaceFunction]:
    flat_inputs = dt.flatten_to_qnames(inputs)
    generated_functions = {}
    for func in functions.values():
        # Pick spec that matches the inputs; fail if no or too many matches
        # Generate function
    return generated_functions


def include_fail_and_warn_nodes(
    functions: dict[str, InterfaceFunction],
    targets: list[str],
) -> list[str]:
    """Extend targets with failures and warnings that can be computed within the graph.

    FailOrWarnFunctions which are included in the targets are treated like regular
    functions.

    """
    fail_or_warn_functions = {
        p: n
        for p, n in functions.items()
        if isinstance(n, FailOrWarnFunction) and p not in targets
    }
    workers_and_their_inputs = dags.create_dag(
        functions={
            p: n
            for p, n in functions.items()
            if not isinstance(n, FailOrWarnFunction) or p in targets
        },
        targets=targets,
    )
    out = targets.copy()
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
    flat_functions: dict[tuple[str, ...], InterfaceFunction | InterfaceInput] = {}
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


def _fail_if_targets_are_not_among_interface_functions(
    targets: list[str] | None,
    interface_function_names: KeysView[str],
) -> None:
    """Fail if some target is not among functions.

    Parameters
    ----------
    targets
        The targets which should be computed.
    interface_function_names
        The names of the interface functions.

    Raises
    ------
    ValueError
        Raised if any member of `targets` is not among functions.

    """
    if targets is not None:
        missing_targets = set(targets) - set(interface_function_names)

        if missing_targets:
            formatted = format_list_linewise(sorted(missing_targets))
            msg = format_errors_and_warnings(
                "The following targets have no corresponding function in the interface "
                f"DAG:\n\n{formatted}",
            )
            raise ValueError(msg)
