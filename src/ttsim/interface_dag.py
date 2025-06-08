from __future__ import annotations

import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dags

from ttsim.interface_dag_elements.fail_if import (
    format_errors_and_warnings,
    format_list_linewise,
)
from ttsim.interface_dag_elements.interface_node_objects import (
    InterfaceFunction,
    InterfaceInput,
)
from ttsim.interface_dag_elements.orig_policy_objects import load_module

if TYPE_CHECKING:
    from collections.abc import KeysView

    from ttsim.interface_dag_elements.typing import UnorderedQNames


def main(inputs: dict[str, Any], targets: list[str] | None = None) -> dict[str, Any]:
    """
    Main function that processes the inputs and returns the outputs.
    """
    nodes = {
        p: n for p, n in _interface_functions_and_inputs().items() if p not in inputs
    }

    functions = {
        name: node
        for name, node in nodes.items()
        if isinstance(node, InterfaceFunction)
    }

    _fail_if_targets_are_not_among_interface_functions(
        targets=targets,
        interface_function_names=functions.keys(),
    )

    dag = dags.create_dag(
        functions=functions,
        targets=targets,
    )
    # draw_dag(dag)
    f = dags.concatenate_functions(
        dag=dag,
        functions=functions,
        targets=targets,
        return_type="dict",
        enforce_signature=False,
        set_annotations=False,
    )
    return f(**inputs)


def _interface_functions_and_inputs() -> dict[str, InterfaceFunction | InterfaceInput]:
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
        dags.tree.qual_name_from_tree_path(path): obj.remove_tree_logic(
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
                f"DAG:\n\n{formatted}"
            )
            raise ValueError(msg)
