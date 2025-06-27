from __future__ import annotations

import inspect

import dags
import dags.tree as dt
import pytest

from ttsim import arg_templates
from ttsim.interface_dag import (
    _fail_if_requested_nodes_cannot_be_found,
    _harmonize_inputs,
    _harmonize_outputs,
    _resolve_dynamic_interface_objects_to_static_nodes,
    load_flat_interface_functions_and_inputs,
)
from ttsim.interface_dag_elements.fail_if import format_list_linewise
from ttsim.interface_dag_elements.interface_node_objects import (
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
    leaf_name="some_idif",
    include_if_all_inputs_present=["input_1"],
)
def some_idif_require_input_1(input_1: int) -> int:
    return input_1


@input_dependent_interface_function(
    leaf_name="some_idif",
    include_if_all_inputs_present=["input_2", "n1__input_2"],
)
def some_idif_require_input_2_and_n1__input_2(input_2: int, n1__input_2: int) -> int:
    return input_2 + n1__input_2


@input_dependent_interface_function(
    leaf_name="some_idif_with_conflicting_conditions",
    include_if_all_inputs_present=["input_1"],
)
def some_idif_with_conflicting_conditions_require_input_1(input_1: int) -> int:
    return input_1


@input_dependent_interface_function(
    leaf_name="some_idif_with_conflicting_conditions",
    include_if_any_input_present=["input_1", "n1__input_2"],
)
def some_idif_with_conflicting_conditions_require_input_1_or_n1__input_2(
    input_1: int, n1__input_2: int
) -> int:
    return input_1 + n1__input_2


def test_load_flat_interface_functions_and_inputs() -> None:
    load_flat_interface_functions_and_inputs()


def test_interface_dag_is_complete() -> None:
    nodes = _resolve_dynamic_interface_objects_to_static_nodes(
        flat_interface_objects=load_flat_interface_functions_and_inputs(),
        input_qnames=[],
    )
    nodes_with_dummy_callables = {
        qn: dummy_callable(n) if not callable(n) else n for qn, n in nodes.items()
    }
    f = dags.concatenate_functions(
        functions=nodes_with_dummy_callables,
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
            {
                dt.qname_from_tree_path(p): n
                for p, n in load_flat_interface_functions_and_inputs().items()
            },
            r'output\snames[\s\S]+interface\sfunctions\sor\sinputs:[\s\S]+"a"',
        ),
        (
            ["input_data"],
            {
                dt.qname_from_tree_path(p): n
                for p, n in load_flat_interface_functions_and_inputs().items()
            },
            r'output\snames[\s\S]+interface\sfunctions\sor\sinputs:[\s\S]+"input_data"',
        ),
        (
            [],
            {
                **{
                    dt.qname_from_tree_path(p): n
                    for p, n in load_flat_interface_functions_and_inputs().items()
                },
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
        "flat_interface_objects",
        "input_qnames",
        "expected_function_name",
    ),
    [
        (
            {
                ("some_idif_require_input_1",): some_idif_require_input_1,
                (
                    "some_idif_require_input_2_and_n1__input_2",
                ): some_idif_require_input_2_and_n1__input_2,
            },
            ["input_2", "n1__input_2"],
            "some_idif_require_input_2_and_n1__input_2",
        ),
        (
            {
                ("some_idif_require_input_1",): some_idif_require_input_1,
                (
                    "some_idif_require_input_2_and_n1__input_2",
                ): some_idif_require_input_2_and_n1__input_2,
            },
            ["input_1"],
            "some_idif_require_input_1",
        ),
        (
            {
                ("some_idif_require_input_1",): some_idif_require_input_1,
                (
                    "some_idif_require_input_2_and_n1__input_2",
                ): some_idif_require_input_2_and_n1__input_2,
            },
            ["input_2", "n1__input_2", "input_3"],
            "some_idif_require_input_2_and_n1__input_2",
        ),
    ],
)
def test_resolve_dynamic_interface_objects_to_static_nodes_returns_correct_function(
    flat_interface_objects, input_qnames, expected_function_name
):
    static_func = next(
        iter(
            _resolve_dynamic_interface_objects_to_static_nodes(
                flat_interface_objects=flat_interface_objects,
                input_qnames=input_qnames,
            ).values()
        )
    )
    assert static_func.original_function_name == expected_function_name


def test_resolve_dynamic_interface_objects_to_static_nodes_with_conflicting_conditions():  # noqa: E501
    match = r"Multiple InputDependentInterfaceFunctions"
    with pytest.raises(ValueError, match=match):
        _resolve_dynamic_interface_objects_to_static_nodes(
            flat_interface_objects={
                (
                    "some_idif_with_conflicting_conditions_require_input_1",
                ): some_idif_with_conflicting_conditions_require_input_1,
                (
                    "some_idif_with_conflicting_conditions_require_input_1_or_n1__input_2",
                ): some_idif_with_conflicting_conditions_require_input_1_or_n1__input_2,
            },
            input_qnames=["input_1", "n1__input_2"],
        )


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        (arg_templates.output.Name("a__b"), {"name": "a__b", "names": ["a__b"]}),
        (arg_templates.output.Name(("a", "b")), {"name": "a__b", "names": ["a__b"]}),
        (
            arg_templates.output.Name({"a": {"b": None}}),
            {"name": "a__b", "names": ["a__b"]},
        ),
        (arg_templates.output.Names(["a__b"]), {"name": None, "names": ["a__b"]}),
        (arg_templates.output.Names([("a", "b")]), {"name": None, "names": ["a__b"]}),
        (
            arg_templates.output.Names({"a": {"b": None}}),
            {"name": None, "names": ["a__b"]},
        ),
    ],
)
def test_harmonize_outputs(output, expected):
    harmonized = _harmonize_outputs(output=output)

    assert harmonized == expected
