from __future__ import annotations

import datetime

import numpy
import pandas as pd
import pytest
from pandas._testing import assert_series_equal

from ttsim import InputData, TTTargets, main
from ttsim.tt import (
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
    with pytest.raises(TypeError):

        @policy_function(rounding_spec={"base": 1, "direction": "updsf"})  # ty: ignore[invalid-argument-type]
        def test_func():
            return 0


@pytest.mark.parametrize(
    ("rounding_spec", "input_values", "exp_output"),
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
        main_target="results__tree",
        input_data=InputData.tree(input_data__tree),
        policy_environment=policy_environment,
        evaluation_date=datetime.date(2024, 1, 1),
        tt_targets=TTTargets.tree({"namespace": {"test_func": None}}),
        rounding=True,
        include_fail_nodes=False,
        include_warn_nodes=False,
        backend=backend,
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
        main_target="results__tree",
        input_data=InputData.tree(data),
        policy_environment=policy_environment,
        evaluation_date=datetime.date(2024, 1, 1),
        tt_targets=TTTargets.tree({"test_func_y": None}),
        rounding=True,
        include_fail_nodes=False,
        include_warn_nodes=False,
        backend=backend,
    )
    assert_series_equal(
        pd.Series(results__tree["test_func_y"]),
        pd.Series([12.0, 12.0]),
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.parametrize(
    ("rounding_spec", "input_values_exp_output", "ignore_since_no_rounding"),
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
        main_target="results__tree",
        input_data=InputData.tree(data),
        policy_environment=policy_environment,
        evaluation_date=datetime.date(2024, 1, 1),
        tt_targets=TTTargets.tree({"test_func": None}),
        include_fail_nodes=False,
        include_warn_nodes=False,
        rounding=False,
        backend=backend,
    )
    assert_series_equal(
        pd.Series(results__tree["test_func"]),
        pd.Series(input_values_exp_output),
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.parametrize(
    ("rounding_spec", "input_values", "exp_output"),
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
    ("rounding_spec", "input_values", "exp_output"),
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
    ("base", "direction", "to_add_after_rounding", "match"),
    [
        (1, "upper", 0, "`direction` must be one of"),
        (5, "closest", 0, "`direction` must be one of"),
        ("0.1", "down", 0, "base needs to be a number"),
        (5, "up", "0", "Additive part must be a number"),
    ],
)
def test_rounding_spec_validation(base, direction, to_add_after_rounding, match):
    """Test validation of RoundingSpec parameters."""
    expected_exception = TypeError if "be a number" in match else ValueError
    with pytest.raises(expected_exception, match=match):
        RoundingSpec(
            base=base,
            direction=direction,
            to_add_after_rounding=to_add_after_rounding,
        )


def test_rounding_spec_base_zero_behavior(xnp):
    """Test RoundingSpec with base=0 (should cause division by zero or special handling)."""
    # Note: base=0 is technically allowed by type system but will cause issues at runtime
    # when rounding is actually applied (division by zero)
    rs = RoundingSpec(base=0, direction="up")
    assert rs.base == 0

    def test_func(x):
        return x

    # Applying rounding with base=0 will cause issues
    rounded_func = rs.apply_rounding(test_func, xnp=xnp)
    # This should produce inf or nan due to division by zero
    result = rounded_func(numpy.array([1.0, 2.0]))
    assert numpy.all(numpy.isinf(result) | numpy.isnan(result))


def test_rounding_spec_very_small_base(xnp):
    """Test RoundingSpec with very small base value."""
    rs = RoundingSpec(base=0.0001, direction="nearest")

    def test_func(x):
        return x

    rounded_func = rs.apply_rounding(test_func, xnp=xnp)
    result = rounded_func(numpy.array([1.23456789]))

    # Should round to nearest 0.0001
    expected = numpy.array([1.2346])
    numpy.testing.assert_allclose(result, expected, atol=0.00005)


def test_rounding_spec_very_large_base(xnp):
    """Test RoundingSpec with very large base value."""
    rs = RoundingSpec(base=1000, direction="down")

    def test_func(x):
        return x

    rounded_func = rs.apply_rounding(test_func, xnp=xnp)
    result = rounded_func(numpy.array([1234.0, 5678.0, 9999.0]))

    # Should round down to nearest 1000
    expected = numpy.array([1000.0, 5000.0, 9000.0])
    numpy.testing.assert_array_equal(result, expected)


def test_rounding_negative_values_up(xnp):
    """Test rounding negative values up (toward zero or away from zero)."""
    rs = RoundingSpec(base=1, direction="up")

    def test_func(x):
        return x

    rounded_func = rs.apply_rounding(test_func, xnp=xnp)
    result = rounded_func(numpy.array([-1.5, -1.1, -0.9, -0.1]))

    # ceil(-1.5) = -1, ceil(-1.1) = -1, ceil(-0.9) = 0, ceil(-0.1) = 0
    expected = numpy.array([-1.0, -1.0, 0.0, 0.0])
    numpy.testing.assert_array_equal(result, expected)


def test_rounding_negative_values_down(xnp):
    """Test rounding negative values down (away from zero)."""
    rs = RoundingSpec(base=1, direction="down")

    def test_func(x):
        return x

    rounded_func = rs.apply_rounding(test_func, xnp=xnp)
    result = rounded_func(numpy.array([-1.5, -1.1, -0.9, -0.1]))

    # floor(-1.5) = -2, floor(-1.1) = -2, floor(-0.9) = -1, floor(-0.1) = -1
    expected = numpy.array([-2.0, -2.0, -1.0, -1.0])
    numpy.testing.assert_array_equal(result, expected)


def test_rounding_negative_values_nearest(xnp):
    """Test rounding negative values to nearest."""
    rs = RoundingSpec(base=1, direction="nearest")

    def test_func(x):
        return x

    rounded_func = rs.apply_rounding(test_func, xnp=xnp)
    result = rounded_func(numpy.array([-1.6, -1.4, -0.6, -0.4]))

    # round(-1.6) = -2, round(-1.4) = -1, round(-0.6) = -1, round(-0.4) = 0
    expected = numpy.array([-2.0, -1.0, -1.0, 0.0])
    numpy.testing.assert_array_equal(result, expected)


def test_rounding_mixed_positive_negative(xnp):
    """Test rounding with mixed positive and negative values."""
    rs = RoundingSpec(base=5, direction="nearest")

    def test_func(x):
        return x

    rounded_func = rs.apply_rounding(test_func, xnp=xnp)
    result = rounded_func(numpy.array([-12.0, -8.0, -2.0, 2.0, 8.0, 12.0]))

    # round(-12/5)*5 = -2*5 = -10, round(-8/5)*5 = -2*5 = -10
    # round(-2/5)*5 = 0*5 = 0, round(2/5)*5 = 0*5 = 0
    # round(8/5)*5 = 2*5 = 10, round(12/5)*5 = 2*5 = 10
    expected = numpy.array([-10.0, -10.0, 0.0, 0.0, 10.0, 10.0])
    numpy.testing.assert_array_equal(result, expected)


def test_rounding_value_exactly_on_boundary(xnp):
    """Test rounding when value is exactly on a boundary."""
    rs = RoundingSpec(base=10, direction="nearest")

    def test_func(x):
        return x

    rounded_func = rs.apply_rounding(test_func, xnp=xnp)
    result = rounded_func(numpy.array([10.0, 20.0, 30.0, 0.0, -10.0]))

    # Values already on boundary should stay the same
    expected = numpy.array([10.0, 20.0, 30.0, 0.0, -10.0])
    numpy.testing.assert_array_equal(result, expected)


def test_rounding_to_add_after_negative(xnp):
    """Test rounding with negative to_add_after_rounding."""
    rs = RoundingSpec(base=10, direction="up", to_add_after_rounding=-5)

    def test_func(x):
        return x

    rounded_func = rs.apply_rounding(test_func, xnp=xnp)
    result = rounded_func(numpy.array([12.0, 25.0]))

    # ceil(12/10)*10 = 20, ceil(25/10)*10 = 30
    # Then subtract 5: 20-5 = 15, 30-5 = 25
    expected = numpy.array([15.0, 25.0])
    numpy.testing.assert_array_equal(result, expected)


def test_rounding_spec_float_base(xnp):
    """Test RoundingSpec with float base."""
    rs = RoundingSpec(base=2.5, direction="down")

    def test_func(x):
        return x

    rounded_func = rs.apply_rounding(test_func, xnp=xnp)
    result = rounded_func(numpy.array([3.0, 5.0, 7.5, 10.0]))

    # floor(3/2.5)*2.5 = 2.5, floor(5/2.5)*2.5 = 5.0
    # floor(7.5/2.5)*2.5 = 7.5, floor(10/2.5)*2.5 = 10.0
    expected = numpy.array([2.5, 5.0, 7.5, 10.0])
    numpy.testing.assert_array_equal(result, expected)


def test_rounding_preserves_function_name(xnp):
    """Test that apply_rounding preserves the wrapped function's name."""
    rs = RoundingSpec(base=1, direction="up")

    def my_custom_function(x):
        return x

    rounded_func = rs.apply_rounding(my_custom_function, xnp=xnp)

    assert rounded_func.__name__ == "my_custom_function"
