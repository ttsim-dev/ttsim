from __future__ import annotations

from typing import TYPE_CHECKING

from mettsim import middle_earth
from ttsim import InputData, MainTarget, TTTargets, main
from ttsim.tt.column_objects_param_function import policy_function, policy_input
from ttsim.tt.param_objects import DictParam, ScalarParam

if TYPE_CHECKING:
    from types import ModuleType

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


def test_returns_root_nodes_when_injecting_unrelated_input_data(xnp: ModuleType):
    template = main(
        main_target=MainTarget.templates.input_data_dtypes,
        policy_date_str="2000-01-01",
        orig_policy_objects={"root": middle_earth.ROOT_PATH},
        tt_targets={"tree": {"wealth_tax": {"amount_y": None}}},
        input_data=InputData.tree(
            tree={
                "p_id": xnp.array([0, 1, 2]),
                "kin_id": xnp.array([0, 0, 0]),
                "payroll_tax": {
                    "amount_y": xnp.array([1000, 0, 0]),
                },
            }
        ),
    )

    assert "kin_id" in template  # policy input
    assert "fam_id" not in template  # endogenous group creation function
    # Inputs for fam_id
    assert "p_id_spouse" in template
    assert "p_id" in template
    assert "age" in template
    assert "p_id_parent_1" in template
    assert "p_id_parent_2" in template
