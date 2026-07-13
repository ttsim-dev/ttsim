from __future__ import annotations

import datetime
import inspect
from typing import Any, cast

# Importing the mettsim package registers the castar (the base currency), the
# silver penny, and the statutory-currency mapping, so concrete currency tokens
# exist for the statutory-guard tests regardless of test-collection order
# (GEP 10).
import mettsim.middle_earth  # noqa: F401
import numpy
import pandas as pd
import pytest
from beartype.roar import BeartypeCallHintViolation
from pandas._testing import assert_series_equal

from ttsim import InputData, TTTargets, main
from ttsim.exceptions import (
    PolicyFunctionDefinitionError,
    RoundingSpecError,
    UnitDefinitionError,
)
from ttsim.interface_dag_elements.policy_environment import (
    _active_column_objects_and_param_functions,
)
from ttsim.tt import (
    RoundingSpec,
    Unit,
    policy_function,
    policy_input,
)
from ttsim.typing import FloatColumn, IntColumn


@policy_input(unit=Unit.DIMENSIONLESS)
def x() -> int:
    pass


@policy_input(unit=Unit.DIMENSIONLESS)
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

    @policy_function(rounding_spec=rs, unit=Unit.DIMENSIONLESS)
    def test_func() -> int:
        return 0

    assert test_func.rounding_spec == rs


def test_malformed_rounding_specs():
    with pytest.raises(PolicyFunctionDefinitionError):

        @policy_function(
            rounding_spec=cast("RoundingSpec", {"base": 1, "direction": "updsf"}),
            unit=Unit.DIMENSIONLESS,
        )
        def test_func() -> int:
            return 0


@pytest.mark.parametrize(
    ("rounding_spec", "input_values", "exp_output"),
    rounding_specs_and_exp_results,
)
def test_rounding(rounding_spec, input_values, exp_output, backend):
    """Check if rounding is correct."""

    # Define function that should be rounded
    @policy_function(rounding_spec=rounding_spec, unit=Unit.DIMENSIONLESS)
    def test_func(x: float) -> float:
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
        policy_date=datetime.date(2024, 1, 1),
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
    @policy_function(
        rounding_spec=RoundingSpec(base=1, direction="down"), unit=Unit.DIMENSIONLESS
    )
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
        policy_date=datetime.date(2024, 1, 1),
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
    @policy_function(rounding_spec=rounding_spec, unit=Unit.DIMENSIONLESS)
    def test_func(x: float) -> float:
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
        policy_date=datetime.date(2024, 1, 1),
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
        (1, "upper", 0, "parameter direction='upper' violates type hint"),
        (5, "closest", 0, "parameter direction='closest' violates type hint"),
        ("0.1", "down", 0, "parameter base='0.1' violates type hint"),
        (5, "up", "0", "parameter to_add_after_rounding='0' violates type hint"),
    ],
)
def test_rounding_spec_validation(base, direction, to_add_after_rounding, match):
    """Reject `RoundingSpec` arguments whose type violates the field annotations.

    The `@beartype`-checked constructor raises `RoundingSpecError` for a bad
    `direction` literal or non-numeric `base` / `to_add_after_rounding`.
    """
    with pytest.raises(RoundingSpecError, match=match):
        RoundingSpec(
            base=base,
            direction=direction,
            to_add_after_rounding=to_add_after_rounding,
        )


def test_rounding_spec_base_zero_raises():
    """Test that RoundingSpec raises ValueError when base=0."""
    with pytest.raises(ValueError, match="base must be positive, got 0"):
        RoundingSpec(base=0, direction="up")


def test_rounding_spec_base_negative_raises():
    """Test that RoundingSpec raises ValueError when base is negative."""
    with pytest.raises(ValueError, match="base must be positive, got -1"):
        RoundingSpec(base=-1, direction="up")


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

    assert rounded_func.__name__ == "my_custom_function"  # ty: ignore[unresolved-attribute]


def test_rounded_wrapper_signature_preserves_param_annotations(xnp) -> None:
    """The rounding wrapper exposes the wrapped function's parameter annotations
    on its own `__signature__`.
    """

    def underlying(a: IntColumn, b: FloatColumn) -> IntColumn:  # noqa: ARG001
        return b

    rs = RoundingSpec(base=1, direction="up")
    rounded = rs.apply_rounding(underlying, xnp=xnp)

    sig = inspect.signature(rounded)
    param_annotations = {
        name: param.annotation for name, param in sig.parameters.items()
    }
    assert param_annotations == {"a": "IntColumn", "b": "FloatColumn"}


def test_rounded_wrapper_signature_forces_return_to_float_column(xnp) -> None:
    """The rounding wrapper forces its `__signature__` return annotation to
    `FloatColumn` because rounding always produces a float column, regardless
    of the wrapped function's declared return type.
    """

    def underlying(a: IntColumn, b: FloatColumn) -> IntColumn:  # noqa: ARG001
        return b

    rs = RoundingSpec(base=1, direction="up")
    rounded = rs.apply_rounding(underlying, xnp=xnp)

    assert inspect.signature(rounded).return_annotation == "FloatColumn"


def test_beartype_catches_structural_misuse_at_rounded_boundary(xnp) -> None:
    """Beartype rejects a structurally wrong argument (a string here) at the
    outer rounded-wrapper boundary, not just at the inner wrapped function.
    """

    def underlying(x: FloatColumn) -> FloatColumn:
        return x

    rs = RoundingSpec(base=1, direction="up")
    rounded = rs.apply_rounding(underlying, xnp=xnp)

    # Route the bogus value through `typing.cast` so ty's literal-narrowing
    # does not surface it as `Literal["not a column"]` (which ty-jax would
    # otherwise flag against the tighter JAX `Array` parameter type).
    # beartype rejects the structural mismatch at runtime regardless.
    bogus = cast("Any", "not a column")
    with pytest.raises(BeartypeCallHintViolation):
        rounded(bogus)


# ----------------------------------------------------------------------------
# Currency-denominated rounding specs (GEP 10): the magnitudes are statutory
# numbers written in a concrete currency and never converted — the declared
# currency must be the statutory one at the policy date.
# ----------------------------------------------------------------------------


def test_policy_environment_rejects_non_statutory_rounding_spec_currency():
    @policy_function(
        rounding_spec=RoundingSpec(
            base=4, direction="down", unit=Unit.SILVER_PENNY.PER_MONTH
        ),
        unit=Unit.CURRENCY.PER_MONTH,
    )
    def amount_m(x: float) -> float:
        return x

    # mettsim's statutory currency at 2024 is the castar; a silver-penny spec
    # must be restated by splitting the function at the changeover.
    with pytest.raises(UnitDefinitionError, match="never converted"):
        _active_column_objects_and_param_functions(
            orig={("income.py", "amount_m"): amount_m},
            policy_date=datetime.date(2024, 1, 1),
            computation_currency="CASTAR",
        )


def test_policy_environment_accepts_statutory_rounding_spec_currency():
    spec = RoundingSpec(base=4, direction="down", unit=Unit.SILVER_PENNY.PER_MONTH)

    @policy_function(
        rounding_spec=spec,
        unit=Unit.CURRENCY.PER_MONTH,
    )
    def amount_m(x: float) -> float:
        return x

    active = _active_column_objects_and_param_functions(
        orig={("income.py", "amount_m"): amount_m},
        policy_date=datetime.date(1950, 1, 1),
        computation_currency="SILVER_PENNY",
    )
    # The spec passes through untouched: rounding happens in the statutory
    # currency natively.
    assert active["amount_m"].rounding_spec is spec
