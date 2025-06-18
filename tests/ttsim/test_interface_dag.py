from __future__ import annotations

import inspect
from dataclasses import asdict

import dags
import dags.tree as dt
import pandas as pd
import pytest

from ttsim.interface_dag import (
    _fail_if_requested_nodes_cannot_be_found,
    load_interface_functions_and_inputs,
    main,
)
from ttsim.interface_dag_elements import InterfaceDAGElements
from ttsim.interface_dag_elements.fail_if import format_list_linewise
from ttsim.interface_dag_elements.interface_node_objects import fail_or_warn_function
from ttsim.plot_dag import dummy_callable
from ttsim.tt_dag_elements.column_objects_param_function import policy_function


@fail_or_warn_function(
    include_if_all_elements_present=["a"],
    include_if_any_element_present=["b"],
)
def some_fail_or_warn_function() -> None:
    pass


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


@pytest.mark.parametrize(
    ("output_qnames", "nodes", "error_match"),
    [
        (
            ["a"],
            load_interface_functions_and_inputs(),
            r'output\snames[\s\S]+interface\sfunctions\sor\sinputs:[\s\S]+"a"',
        ),
        (
            ["input_data"],
            load_interface_functions_and_inputs(),
            r'output\snames[\s\S]+interface\sfunctions\sor\sinputs:[\s\S]+"input_data"',
        ),
        (
            [],
            {
                **load_interface_functions_and_inputs(),
                "some_fail_or_warn_function": some_fail_or_warn_function,
            },
            r'include\scondition[\s\S]+functions or inputs:[\s\S]+"a",\s+"b"',
        ),
    ],
)
def test_fail_if_requested_nodes_cannot_be_found(
    output_qnames, nodes, error_match
) -> None:
    with pytest.raises(ValueError, match=error_match):
        _fail_if_requested_nodes_cannot_be_found(
            output_qnames=output_qnames,
            nodes=nodes,
        )


def test_interface_elements_class_is_complete():
    from_class = set(dt.qnames(asdict(InterfaceDAGElements())))
    loaded = set(load_interface_functions_and_inputs())
    assert from_class.symmetric_difference(loaded) == set()


@policy_function()
def e(c: int, d: float) -> float:
    return c + d


def test_harmonize_inputs():
    x = InterfaceDAGElements()
    x.input_data.df_and_mapper.df = pd.DataFrame([], columns=["a", "b", "p_id"])
    x.input_data.df_and_mapper.mapper = {"c": "a", "d": "b", "p_id": "p_id"}
    x.targets.tree = {"e": "f"}
    x.date = "2025-01-01"
    x.orig_policy_objects.column_objects_and_param_functions = {("x.py", "e"): e}
    x.orig_policy_objects.param_specs = {}

    main(inputs=x)
