from __future__ import annotations

import inspect
from dataclasses import asdict

import dags
import dags.tree as dt

from ttsim.interface_dag import load_interface_functions_and_inputs
from ttsim.interface_dag_elements import InterfaceDAGElements
from ttsim.interface_dag_elements.fail_if import format_list_linewise
from ttsim.plot_dag import dummy_callable


def test_load_interface_functions_and_inputs() -> None:
    load_interface_functions_and_inputs()


def test_interface_dag_is_complete() -> None:
    nodes = {
        p: dummy_callable(n) if not callable(n) else n
        for p, n in load_interface_functions_and_inputs().items()
    }

    f = dags.concatenate_functions(
        functions=nodes,
        targets=None,
        return_type="dict",
        enforce_signature=False,
        set_annotations=False,
    )
    args = inspect.signature(f).parameters
    if args:
        raise ValueError(
            "The full interface DAG should include all root nodes but requires inputs:"
            f"\n\n{format_list_linewise(args.keys())}"
        )


def test_interface_elements_class_is_complete():
    from_class = set(dt.qnames(asdict(InterfaceDAGElements())))
    loaded = set(load_interface_functions_and_inputs())
    assert from_class.symmetric_difference(loaded) == set()
