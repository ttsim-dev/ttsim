from __future__ import annotations

from pathlib import Path

import numpy

from ttsim import Output, main
from ttsim.testing_utils import (
    load_policy_test_data,
)
from ttsim.tt_dag_elements.column_objects_param_function import (
    policy_function,
    policy_input,
)
from ttsim.tt_dag_elements.param_objects import DictParam, ScalarParam

METTSIM_ROOT = Path(__file__).parent.parent / "mettsim"

TEST_DIR = Path(__file__).parent

POLICY_TEST_IDS_AND_CASES = load_policy_test_data(
    test_dir=TEST_DIR,
    policy_name="",
    xnp=numpy,
)


p1 = ScalarParam(
    value=1,
    leaf_name="some_int_param",
    start_date="2025-01-01",
    end_date="2025-12-31",
    name="some_int_param",
    description="Some int param",
    unit=None,
    reference_period=None,
    note=None,
    reference=None,
)

p2 = DictParam(
    value={"a": 1, "b": 2},
    leaf_name="some_dict_param",
    start_date="2025-01-01",
    end_date="2025-12-31",
    name="some_dict_param",
    description="Some dict param",
    unit=None,
    reference_period=None,
    note=None,
    reference=None,
)


@policy_input()
def inp1() -> int:
    pass


@policy_input()
def inp2() -> float:
    pass


@policy_function()
def x(inp1: int, p1: int, p2: dict[str, int]) -> int:
    return inp1 + p1 + p2["a"] + p2["b"]


@policy_function()
def y(inp2: float, p2: dict[str, int]) -> float:
    return inp2 + p2["b"]


@policy_function()
def z(a__x: int, a__y: float) -> float:
    return a__x + a__y


def test_template_all_outputs_no_inputs(backend):
    actual = main(
        policy_environment={
            "inp1": inp1,
            "p1": p1,
            "a": {"inp2": inp2, "x": x, "y": y, "p2": p2},
            "b": {
                "z": z,
            },
        },
        rounding=True,
        date_str="2025-01-01",
        backend=backend,
        output=Output.name("templates__input_data_dtypes"),
    )
    assert actual == {"a": {"inp2": "FloatColumn"}, "inp1": "IntColumn"}


def test_template_all_outputs_with_inputs(backend):
    actual = main(
        input_data={
            "tree": {
                "p_id": [4, 5, 6],
                "a": {
                    "inp2": [1, 2, 3],
                },
                "inp1": [0, 1, 2],
            }
        },
        policy_environment={
            "inp1": inp1,
            "p1": p1,
            "a": {"inp2": inp2, "x": x, "y": y, "p2": p2},
            "b": {
                "z": z,
            },
        },
        rounding=True,
        date_str="2025-01-01",
        backend=backend,
        output=Output.name("templates__input_data_dtypes"),
    )
    assert actual == {"a": {"inp2": "FloatColumn"}, "inp1": "IntColumn"}


def test_template_output_y_no_inputs(backend):
    actual = main(
        tt_targets={"tree": {"a": {"y": None}}},
        policy_environment={
            "inp1": inp1,
            "p1": p1,
            "a": {"inp2": inp2, "x": x, "y": y, "p2": p2},
            "b": {
                "z": z,
            },
        },
        rounding=True,
        date_str="2025-01-01",
        backend=backend,
        output=Output.name("templates__input_data_dtypes"),
    )
    assert actual == {"a": {"inp2": "FloatColumn"}}


def test_template_output_x_with_inputs(backend):
    actual = main(
        input_data={
            "tree": {
                "p_id": [4, 5, 6],
                "a": {
                    "inp2": [1, 2, 3],
                },
                "inp1": [0, 1, 2],
            }
        },
        tt_targets={"tree": {"a": {"x": None}}},
        policy_environment={
            "inp1": inp1,
            "p1": p1,
            "a": {"inp2": inp2, "x": x, "y": y, "p2": p2},
            "b": {
                "z": z,
            },
        },
        rounding=True,
        date_str="2025-01-01",
        backend=backend,
        output=Output.name("templates__input_data_dtypes"),
    )
    assert actual == {"inp1": "IntColumn"}
