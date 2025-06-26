from __future__ import annotations

import inspect

import dags
import pytest

from ttsim.interface_dag import (
    _fail_if_requested_nodes_cannot_be_found,
    _harmonize_inputs,
    load_interface_functions_and_inputs,
)
from ttsim.interface_dag_elements.fail_if import format_list_linewise
from ttsim.interface_dag_elements.interface_node_objects import (
    InterfaceFunctionVariant,
    fail_or_warn_function,
    input_dependent_interface_function,
)
from ttsim.plot_dag import dummy_callable
from ttsim.tt_dag_elements.column_objects_param_function import policy_function


@fail_or_warn_function(
    include_if_all_elements_present=["a"],
    include_if_any_element_present=["b"],
)
def some_fail_or_warn_function() -> None:
    pass


@input_dependent_interface_function(
    variants=[
        InterfaceFunctionVariant(
            required_input_qnames=["input_1"],
            function=lambda input_1: input_1,
        ),
        InterfaceFunctionVariant(
            required_input_qnames=["input_2", "n1__input_2"],
            function=lambda input_2, n1__input_2: input_2 + n1__input_2,
        ),
    ]
)
def some_input_dependent_interface_function() -> int:
    pass


@input_dependent_interface_function(
    variants=[
        InterfaceFunctionVariant(
            required_input_qnames=["input_1"],
            function=lambda input_1: input_1,
        ),
        InterfaceFunctionVariant(
            required_input_qnames=["input_1", "n1__input_2"],
            function=lambda input_1, n1__input_2: input_1 + n1__input_2,
        ),
    ]
)
def some_input_dependent_interface_function_with_conflicting_variants() -> int:
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


@policy_function()
def e(c: int, d: float) -> float:
    return c + d


def test_harmonize_inputs_qname_input():
    x = {
        "input_data__df_and_mapper__df": {"cannot use df because comparison fails"},
        "input_data__df_and_mapper__mapper": {"c": "a", "d": "b", "p_id": "p_id"},
        "targets__tree": {"e": "f"},
        "date": "2025-01-01",
        "orig_policy_objects__column_objects_and_param_functions": {("x.py", "e"): e},
        "orig_policy_objects__param_specs": {},
    }
    harmonized = _harmonize_inputs(inputs=x)

    assert harmonized == {**x, "backend": "numpy", "rounding": True}


def test_harmonize_inputs_tree_input():
    x = {
        "input_data": {
            "df_and_mapper": {
                "df": {"cannot use df because comparison fails"},
                "mapper": {"c": "a", "d": "b", "p_id": "p_id"},
            }
        },
        "targets": {"tree": {"e": "f"}},
        "date": "2025-01-01",
        "orig_policy_objects": {
            "column_objects_and_param_functions": {("x.py", "e"): e},
            "param_specs": {},
        },
        "backend": "numpy",
        "rounding": True,
    }
    harmonized = _harmonize_inputs(inputs=x)

    assert harmonized == {
        "input_data__df_and_mapper__df": {"cannot use df because comparison fails"},
        "input_data__df_and_mapper__mapper": {"c": "a", "d": "b", "p_id": "p_id"},
        "targets__tree": {"e": "f"},
        "date": "2025-01-01",
        "orig_policy_objects__column_objects_and_param_functions": {("x.py", "e"): e},
        "orig_policy_objects__param_specs": {},
        "backend": "numpy",
        "rounding": True,
    }


@pytest.mark.parametrize(
    (
        "input_qnames",
        "expected_function_args",
    ),
    [
        (
            ["input_2", "n1__input_2"],
            ["input_2", "n1__input_2"],
        ),
        (
            ["input_1"],
            ["input_1"],
        ),
        (
            ["input_1", "n2__input_2"],
            ["input_1"],
        ),
    ],
)
def test_input_dependent_interface_functions_have_correct_args(
    input_qnames, expected_function_args
):
    func = some_input_dependent_interface_function.resolve_to_static_interface_function(
        input_qnames
    )
    assert list(inspect.signature(func).parameters.keys()) == expected_function_args


def test_input_dependent_interface_functions_with_conflicting_variants():
    match = r"Multiple sets of inputs were found that satisfy the requirements:"
    with pytest.raises(ValueError, match=match):
        some_input_dependent_interface_function_with_conflicting_variants.resolve_to_static_interface_function(
            ["input_1", "n1__input_2"]
        )
