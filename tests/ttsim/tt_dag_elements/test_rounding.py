from __future__ import annotations

import datetime

import numpy
import pandas as pd
import pytest
from pandas._testing import assert_series_equal

from ttsim import main
from ttsim.tt_dag_elements import (
    RoundingSpec,
    policy_function,
    policy_input,
)


@policy_input()
def x() -> int:
    pass


@policy_input()
def p_id() -> int:
    pass


rounding_specs_and_exp_results = [
    (
        RoundingSpec(base=1, direction="up"),
        numpy.array([100.24, 100.78]),
        numpy.array([101.0, 101.0]),
    ),
    (
        RoundingSpec(base=1, direction="down"),
        numpy.array([100.24, 100.78]),
        numpy.array([100.0, 100.0]),
    ),
    (
        RoundingSpec(base=1, direction="nearest"),
        numpy.array([100.24, 100.78]),
        numpy.array([100.0, 101.0]),
    ),
    (
        RoundingSpec(base=5, direction="up"),
        numpy.array([100.24, 100.78]),
        numpy.array([105.0, 105.0]),
    ),
    (
        RoundingSpec(base=0.1, direction="down"),
        numpy.array([100.24, 100.78]),
        numpy.array([100.2, 100.7]),
    ),
    (
        RoundingSpec(base=0.001, direction="nearest"),
        numpy.array([100.24, 100.78]),
        numpy.array([100.24, 100.78]),
    ),
    (
        RoundingSpec(base=1, direction="up", to_add_after_rounding=10),
        numpy.array([100.24, 100.78]),
        numpy.array([111.0, 111.0]),
    ),
    (
        RoundingSpec(base=1, direction="down", to_add_after_rounding=10),
        numpy.array([100.24, 100.78]),
        numpy.array([110.0, 110.0]),
    ),
    (
        RoundingSpec(base=1, direction="nearest", to_add_after_rounding=10),
        numpy.array([100.24, 100.78]),
        numpy.array([110.0, 111.0]),
    ),
]


def test_decorator():
    rs = RoundingSpec(base=1, direction="up")

    @policy_function(rounding_spec=rs)
    def test_func():
        return 0

    assert test_func.rounding_spec == rs


def test_malformed_rounding_specs():
    with pytest.raises(AssertionError):

        @policy_function(rounding_spec={"base": 1, "direction": "updsf"})
        def test_func():
            return 0


@pytest.mark.parametrize(
    "rounding_spec, input_values, exp_output",
    rounding_specs_and_exp_results,
)
def test_rounding(rounding_spec, input_values, exp_output, backend):
    """Check if rounding is correct."""

    # Define function that should be rounded
    @policy_function(rounding_spec=rounding_spec)
    def test_func(x):
        return x

    input_data__tree = {
        "p_id": numpy.array([1, 2]),
        "namespace": {"x": numpy.array(input_values)},
    }
    policy_environment = {"namespace": {"test_func": test_func, "x": x}, "p_id": p_id}

    results__tree = main(
        input_data={"tree": input_data__tree},
        policy_environment=policy_environment,
        date=datetime.date(2024, 1, 1),
        tt_targets={"tree": {"namespace": {"test_func": None}}},
        rounding=True,
        backend=backend,
        main_target=("results__tree"),
    )
    assert_series_equal(
        pd.Series(results__tree["namespace"]["test_func"]),
        pd.Series(exp_output),
        check_names=False,
        check_dtype=False,
    )


def test_rounding_with_time_conversion(backend, xnp):
    """Check if rounding is correct for time-converted functions."""

    # Define function that should be rounded
    @policy_function(rounding_spec=RoundingSpec(base=1, direction="down"))
    def test_func_m(x: float) -> float:
        return x

    data = {
        "p_id": xnp.array([1, 2]),
        "x": xnp.array([1.2, 1.5]),
    }

    policy_environment = {
        "test_func_m": test_func_m,
        "x": x,
        "p_id": p_id,
    }

    results__tree = main(
        input_data={"tree": data},
        policy_environment=policy_environment,
        date=datetime.date(2024, 1, 1),
        tt_targets={"tree": {"test_func_y": None}},
        rounding=True,
        backend=backend,
        main_target=("results__tree"),
    )
    assert_series_equal(
        pd.Series(results__tree["test_func_y"]),
        pd.Series([12.0, 12.0]),
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "rounding_spec, input_values_exp_output, ignore_since_no_rounding",
    rounding_specs_and_exp_results,
)
def test_no_rounding(
    rounding_spec,
    input_values_exp_output,
    ignore_since_no_rounding,  # noqa: ARG001
    backend,
):
    # Define function that should be rounded
    @policy_function(rounding_spec=rounding_spec)
    def test_func(x):
        return x

    data = {"p_id": numpy.array([1, 2])}
    data["x"] = numpy.array(input_values_exp_output)
    policy_environment = {
        "test_func": test_func,
        "x": x,
        "p_id": p_id,
    }

    results__tree = main(
        input_data={"tree": data},
        policy_environment=policy_environment,
        date=datetime.date(2024, 1, 1),
        tt_targets={"tree": {"test_func": None}},
        rounding=False,
        backend=backend,
        main_target=("results__tree"),
    )
    assert_series_equal(
        pd.Series(results__tree["test_func"]),
        pd.Series(input_values_exp_output),
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "rounding_spec, input_values, exp_output",
    rounding_specs_and_exp_results,
)
def test_rounding_callable(rounding_spec, input_values, exp_output, xnp):
    """Check if callable is rounded correctly."""

    def test_func(income):
        return income

    func_with_rounding = rounding_spec.apply_rounding(test_func, xnp=xnp)

    assert_series_equal(
        pd.Series(func_with_rounding(input_values)),
        pd.Series(exp_output),
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "rounding_spec, input_values, exp_output",
    rounding_specs_and_exp_results,
)
def test_rounding_spec(rounding_spec, input_values, exp_output, xnp):
    """Test RoundingSpec directly."""

    def test_func(income):
        return income

    rounded_func = rounding_spec.apply_rounding(test_func, xnp=xnp)
    result = rounded_func(input_values)

    assert_series_equal(
        pd.Series(result),
        pd.Series(exp_output),
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "base, direction, to_add_after_rounding, match",
    [
        (1, "upper", 0, "`direction` must be one of"),
        (5, "closest", 0, "`direction` must be one of"),
        ("0.1", "down", 0, "base needs to be a number"),
        (5, "up", "0", "Additive part must be a number"),
    ],
)
def test_rounding_spec_validation(base, direction, to_add_after_rounding, match):
    """Test validation of RoundingSpec parameters."""
    with pytest.raises(ValueError, match=match):
        RoundingSpec(
            base=base,
            direction=direction,
            to_add_after_rounding=to_add_after_rounding,
        )
