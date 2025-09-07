from __future__ import annotations

from typing import TYPE_CHECKING

from ttsim.entry_point import main
from ttsim.main_args import InputData, TTTargets
from ttsim.main_target import MainTarget
from ttsim.tt import group_creation_function, policy_function, policy_input

if TYPE_CHECKING:
    from types import ModuleType
    from typing import Literal

    from ttsim.typing import IntColumn


@policy_input()
def some_input_m() -> int:
    pass


@policy_input()
def another_input_m() -> int:
    pass


@group_creation_function()
def even_id(p_id: IntColumn, xnp: ModuleType) -> IntColumn:  # noqa: ARG001
    return p_id % 2


@policy_function()
def pf_taking_input_and_aggregated_pf_m(
    some_input_m: int, another_input_m_even: int
) -> float:
    return another_input_m_even / 2 + some_input_m


@policy_function()
def pf_taking_input_and_time_converted_pf_y(
    some_input_y: float, another_input_y: float
) -> float:
    return another_input_y / 2 + some_input_y


def test_aggregated_functions_are_overwritten_by_scalar_inputs(
    xnp: ModuleType, backend: Literal["numpy", "jax"]
):
    column_objects_and_param_functions = {
        ("some_input",): some_input_m,
        ("another_input",): another_input_m,
        ("even_id",): even_id,
        ("pf_taking_input_and_aggregated_pf_m",): pf_taking_input_and_aggregated_pf_m,
    }

    tt_function = main(
        main_target=MainTarget.tt_function,
        orig_policy_objects={
            "column_objects_and_param_functions": column_objects_and_param_functions,
            "param_specs": {},
        },
        input_data=InputData.tree(
            {
                "p_id": xnp.array([0]),
                "some_input_m": xnp.array([1]),
                "another_input_m_even": 2,
            }
        ),
        tt_targets=TTTargets.tree({"pf_taking_input_and_aggregated_pf_m": None}),
        policy_date_str="2024-01-01",
        backend=backend,
    )

    # Returns KeyError because 'another_input_m_even' was not partialled in tt_function
    tt_function(
        {
            "some_input_m": xnp.array([1, 2, 3]),
        }
    )


def test_time_converted_functions_are_overwritten_by_scalar_inputs(
    xnp: ModuleType, backend: Literal["numpy", "jax"]
):
    column_objects_and_param_functions = {
        ("some_input_m",): some_input_m,
        ("another_input_m",): another_input_m,
        ("even_id",): even_id,
        (
            "pf_taking_input_and_time_converted_pf_y",
        ): pf_taking_input_and_time_converted_pf_y,
    }

    tt_function = main(
        main_target=MainTarget.tt_function,
        orig_policy_objects={
            "column_objects_and_param_functions": column_objects_and_param_functions,
            "param_specs": {},
        },
        input_data=InputData.tree(
            {
                "p_id": xnp.array([0]),
                "some_input_m": xnp.array([1]),
                "another_input_y": 2,
            }
        ),
        tt_targets=TTTargets.tree({"pf_taking_input_and_time_converted_pf_y": None}),
        policy_date_str="2024-01-01",
        backend=backend,
    )

    # Returns KeyError because 'another_input_y' was not partialled in tt_function
    tt_function(
        {
            "some_input_m": xnp.array([1, 2, 3]),
        }
    )
