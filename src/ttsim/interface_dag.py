from __future__ import annotations

import inspect
from pathlib import Path

from ttsim.interface_dag_elements.interface_node_objects import (
    InterfaceFunction,
    InterfaceInput,
)
from ttsim.interface_dag_elements.orig_policy_objects import load_module


def _load_function_collection() -> dict[str, InterfaceFunction | InterfaceInput]:
    """
    Load the interface functions and inputs from the current directory.

    """
    root = Path(__file__).parent / "interface_dag_elements"
    paths = [
        p for p in root.rglob("*.py") if p.name not in ["__init__.py", "typing.py"]
    ]
    flat_functions = {}
    for path in paths:
        print(path)
        module = load_module(path=path, root=root)
        for name, obj in inspect.getmembers(module):
            if isinstance(obj, InterfaceFunction | InterfaceInput):
                if obj.in_top_level_namespace:
                    flat_functions[(name,)] = obj
                else:
                    flat_functions[(module.__name__, name)] = obj

    breakpoint()
    return flat_functions
