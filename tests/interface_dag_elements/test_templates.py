from __future__ import annotations

from pathlib import Path

import numpy

from ttsim import TTTargets, main
from ttsim.testing_utils import load_policy_test_data
from ttsim.tt.column_objects_param_function import policy_function, policy_input
from ttsim.tt.param_objects import DictParam, ScalarParam

METTSIM_ROOT = Path(__file__).parent.parent / "mettsim"

TEST_DIR = Path(__file__).parent

POLICY_TEST_IDS_AND_CASES = load_policy_test_data(
    test_dir=TEST_DIR,
    policy_name="",
    xnp=numpy,
)


par1 = ScalarParam(
    value=1,
    start_date="2025-01-01",
    end_date="2025-12-31",
    name="Some int param",
    description="Some int param",
    unit=None,
    reference_period=None,
    note=None,
    reference=None,
)

par2 = DictParam(
    value={"a": 1, "b": 2},
    start_date="2025-01-01",
    end_date="2025-12-31",
    name="Some dict param",
    description="Some dict param",
    unit=None,
    reference_period=None,
    note=None,
    reference=None,
)


@policy_input()
def kin_id() -> int:
    pass


@policy_input()
def inp1() -> int:
    pass


@policy_input()
def inp2() -> float:
    pass


@policy_function()
def x(inp1_kin: int, par1: int, par2: dict[str, int]) -> int:
    return inp1_kin + par1 + par2["a"] + par2["b"]


@policy_function()
def y(inp2: float, par2: dict[str, int]) -> float:
    return inp2 + par2["b"]


@policy_function()
def z(a__x: int, a__y: float) -> float:
    return a__x + a__y


def test_template_all_outputs_no_inputs(backend):
    actual = main(
        main_target="templates__input_data_dtypes",
        policy_environment={
            "kin_id": kin_id,
            "inp1": inp1,
            "par1": par1,
            "a": {"inp2": inp2, "x": x, "y": y, "par2": par2},
            "b": {
                "z": z,
            },
        },
        rounding=True,
        evaluation_date_str="2025-01-01",
        backend=backend,
    )
    assert actual == {
        "a": {"inp2": "FloatColumn"},
        "inp1": "IntColumn",
        "kin_id": "IntColumn",
    }


def test_template_all_outputs_with_inputs(backend, xnp):
    actual = main(
        main_target="templates__input_data_dtypes",
        policy_environment={
            "kin_id": kin_id,
            "inp1": inp1,
            "par1": par1,
            "a": {"inp2": inp2, "x": x, "y": y, "par2": par2},
            "b": {
                "z": z,
            },
        },
        evaluation_date_str="2025-01-01",
        input_data={
            "tree": {
                "p_id": xnp.array([4, 5, 6]),
                "a": {
                    "inp2": xnp.array([1, 2, 3]),
                },
                "inp1": xnp.array([0, 1, 2]),
            }
        },
        tt_targets=TTTargets(tree={"a__x": None, "a__y": None, "b__z": None}),
        backend=backend,
    )
    assert actual == {
        "a": {"inp2": "FloatColumn"},
        "inp1": "IntColumn",
        "kin_id": "IntColumn",
    }


def test_template_output_y_no_inputs(backend):
    actual = main(
        main_target="templates__input_data_dtypes",
        tt_targets={"tree": {"a": {"y": None}}},
        policy_environment={
            "kin_id": kin_id,
            "inp1": inp1,
            "par1": par1,
            "a": {"inp2": inp2, "x": x, "y": y, "par2": par2},
            "b": {
                "z": z,
            },
        },
        rounding=True,
        evaluation_date_str="2025-01-01",
        backend=backend,
    )
    assert actual == {"a": {"inp2": "FloatColumn"}}


def test_template_output_x_with_inputs(backend, xnp):
    actual = main(
        main_target="templates__input_data_dtypes",
        input_data={
            "tree": {
                "p_id": xnp.array([4, 5, 6]),
                "a": {
                    "inp2": xnp.array([1, 2, 3]),
                },
                "inp1": xnp.array([0, 1, 2]),
            }
        },
        tt_targets={"tree": {"a": {"x": None}}},
        policy_environment={
            "kin_id": kin_id,
            "inp1": inp1,
            "par1": par1,
            "a": {"inp2": inp2, "x": x, "y": y, "par2": par2},
            "b": {
                "z": z,
            },
        },
        rounding=True,
        evaluation_date_str="2025-01-01",
        backend=backend,
    )
    assert actual == {"inp1": "IntColumn", "kin_id": "IntColumn"}


def test_template_all_outputs_no_input_for_root_of_derived_function(backend, xnp):
    actual = main(
        main_target="templates__input_data_dtypes",
        policy_environment={
            "kin_id": kin_id,
            "inp1": inp1,
            "par1": par1,
            "a": {"inp2": inp2, "x": x, "y": y, "par2": par2},
            "b": {
                "z": z,
            },
        },
        input_data={
            "tree": {
                "p_id": xnp.array([4, 5, 6]),
                "a": {
                    "inp2": xnp.array([1, 2, 3]),
                },
            }
        },
        tt_targets={"tree": {"a": {"x": None, "y": None}, "b": {"z": None}}},
        evaluation_date_str="2025-01-01",
        backend=backend,
    )
    assert actual == {
        "a": {"inp2": "FloatColumn"},
        "inp1": "IntColumn",
    }
