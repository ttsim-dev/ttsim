"""
Tests for `piecewise_polynomial`
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy
import pytest

if TYPE_CHECKING:
    from types import ModuleType


from ttsim.tt.param_objects import PiecewisePolynomialParamValue
from ttsim.tt.piecewise_polynomial import (
    _create_intercepts,
    get_piecewise_parameters,
    get_piecewise_thresholds,
    piecewise_polynomial,
)


@pytest.fixture
def parameters(xnp: ModuleType):
    return PiecewisePolynomialParamValue(
        thresholds=xnp.array([-xnp.inf, 9168.0, 14254.0, 55960.0, 265326.0, xnp.inf]),
        rates=xnp.array(
            [
                [
                    0.00000000e00,
                    1.40000000e-01,
                    2.39700000e-01,
                    4.20000000e-01,
                    4.50000000e-01,
                ],
                [
                    0.00000000e00,
                    9.80141565e-06,
                    2.16155949e-06,
                    0.00000000e00,
                    0.00000000e00,
                ],
            ],
        ),
        intercepts=xnp.array([0.0, 0.0, 965.5771, 14722.3012, 102656.0212]),
    )


def test_get_piecewise_parameters_all_intercepts_supplied(xnp: ModuleType):
    parameter_dict = {
        0: {
            "lower_threshold": "-inf",
            "upper_threshold": 2005,
            "rate_linear": 0,
            "intercept_at_lower_threshold": 0.27,
        },
        1: {
            "lower_threshold": 2005,
            "upper_threshold": 2021,
            "rate_linear": 0.02,
            "intercept_at_lower_threshold": 0.5,
        },
        2: {
            "lower_threshold": 2021,
            "upper_threshold": 2041,
            "rate_linear": 0.01,
            "intercept_at_lower_threshold": 0.8,
        },
        3: {
            "lower_threshold": 2041,
            "upper_threshold": "inf",
            "rate_linear": 0,
            "intercept_at_lower_threshold": 1,
        },
    }

    actual = get_piecewise_parameters(
        leaf_name="test",
        func_type="piecewise_linear",
        parameter_dict=parameter_dict,
        xnp=xnp,
    )
    expected = xnp.array([0.27, 0.5, 0.8, 1])

    numpy.testing.assert_allclose(actual.intercepts, expected, atol=1e-7)


def test_piecewise_polynomial(
    parameters: PiecewisePolynomialParamValue,
    xnp: ModuleType,
):
    x = xnp.array([-1_000, 1_000, 10_000, 30_000, 100_000, 1_000_000])
    expected = xnp.array([0.0, 0.0, 246.53, 10551.65, 66438.2, 866518.64])

    actual = piecewise_polynomial(
        x=x,
        parameters=parameters,
        rates_multiplier=2,
        xnp=xnp,
    )
    numpy.testing.assert_allclose(xnp.array(actual), expected, atol=0.01)


def test_piecewise_constant(xnp: ModuleType):
    """Test piecewise constant function (no rates)."""
    parameter_dict = {
        0: {
            "lower_threshold": "-inf",
            "upper_threshold": 0,
            "intercept_at_lower_threshold": 100.0,
        },
        1: {
            "lower_threshold": 0,
            "upper_threshold": "inf",
            "intercept_at_lower_threshold": 200.0,
        },
    }

    params = get_piecewise_parameters(
        leaf_name="test_const",
        func_type="piecewise_constant",
        parameter_dict=parameter_dict,
        xnp=xnp,
    )

    x = xnp.array([-100.0, -1.0, 0.0, 1.0, 100.0])
    result = piecewise_polynomial(x=x, parameters=params, xnp=xnp)

    expected = xnp.array([100.0, 100.0, 200.0, 200.0, 200.0])
    numpy.testing.assert_allclose(result, expected)


def test_piecewise_quadratic(xnp: ModuleType):
    """Test piecewise quadratic function."""
    parameter_dict = {
        0: {
            "lower_threshold": "-inf",
            "upper_threshold": 0,
            "rate_linear": 0.0,
            "rate_quadratic": 0.0,
            "intercept_at_lower_threshold": 0.0,
        },
        1: {
            "lower_threshold": 0,
            "upper_threshold": "inf",
            "rate_linear": 0.0,
            "rate_quadratic": 1.0,
            "intercept_at_lower_threshold": 0.0,
        },
    }

    params = get_piecewise_parameters(
        leaf_name="test_quad",
        func_type="piecewise_quadratic",
        parameter_dict=parameter_dict,
        xnp=xnp,
    )

    x = xnp.array([0.0, 1.0, 2.0, 3.0])
    result = piecewise_polynomial(x=x, parameters=params, xnp=xnp)

    # Quadratic: intercept + rate_linear * x + rate_quadratic * x^2
    # = 0 + 0*x + 1*x^2 = x^2
    expected = xnp.array([0.0, 1.0, 4.0, 9.0])
    numpy.testing.assert_allclose(result, expected)


def test_piecewise_cubic(xnp: ModuleType):
    """Test piecewise cubic function."""
    parameter_dict = {
        0: {
            "lower_threshold": "-inf",
            "upper_threshold": 0,
            "rate_linear": 0.0,
            "rate_quadratic": 0.0,
            "rate_cubic": 0.0,
            "intercept_at_lower_threshold": 0.0,
        },
        1: {
            "lower_threshold": 0,
            "upper_threshold": "inf",
            "rate_linear": 0.0,
            "rate_quadratic": 0.0,
            "rate_cubic": 1.0,
            "intercept_at_lower_threshold": 0.0,
        },
    }

    params = get_piecewise_parameters(
        leaf_name="test_cubic",
        func_type="piecewise_cubic",
        parameter_dict=parameter_dict,
        xnp=xnp,
    )

    x = xnp.array([0.0, 1.0, 2.0, 3.0])
    result = piecewise_polynomial(x=x, parameters=params, xnp=xnp)

    # Cubic: intercept + rate_linear*x + rate_quadratic*x^2 + rate_cubic*x^3
    # = 0 + 0*x + 0*x^2 + 1*x^3 = x^3
    expected = xnp.array([0.0, 1.0, 8.0, 27.0])
    numpy.testing.assert_allclose(result, expected)


def test_piecewise_value_exactly_on_threshold(xnp: ModuleType):
    """Test that values exactly on thresholds are handled correctly."""
    parameter_dict = {
        0: {
            "lower_threshold": "-inf",
            "upper_threshold": 100.0,
            "rate_linear": 0.1,
            "intercept_at_lower_threshold": 0.0,
        },
        1: {
            "lower_threshold": 100.0,
            "upper_threshold": "inf",
            "rate_linear": 0.2,
            "intercept_at_lower_threshold": 10.0,
        },
    }

    params = get_piecewise_parameters(
        leaf_name="test_threshold",
        func_type="piecewise_linear",
        parameter_dict=parameter_dict,
        xnp=xnp,
    )

    x = xnp.array([99.99, 100.0, 100.01])
    result = piecewise_polynomial(x=x, parameters=params, xnp=xnp)

    # x=99.99 -> first segment: 0.1 * (99.99 - (-inf)) but since -inf, increment=0
    # x=100 -> second segment: 10 + 0.2 * 0 = 10
    # x=100.01 -> second segment: 10 + 0.2 * 0.01 = 10.002
    numpy.testing.assert_allclose(result[1], 10.0)
    numpy.testing.assert_allclose(result[2], 10.002, atol=0.001)


def test_piecewise_single_segment(xnp: ModuleType):
    """Test piecewise with single segment (entire real line)."""
    parameter_dict = {
        0: {
            "lower_threshold": "-inf",
            "upper_threshold": "inf",
            "rate_linear": 0.5,
            "intercept_at_lower_threshold": 10.0,
        },
    }

    params = get_piecewise_parameters(
        leaf_name="test_single",
        func_type="piecewise_linear",
        parameter_dict=parameter_dict,
        xnp=xnp,
    )

    x = xnp.array([-100.0, 0.0, 100.0])
    result = piecewise_polynomial(x=x, parameters=params, xnp=xnp)

    # Since lower threshold is -inf, increment is 0, so result is just intercept
    expected = xnp.array([10.0, 10.0, 10.0])
    numpy.testing.assert_allclose(result, expected)


def test_piecewise_rates_multiplier_zero(xnp: ModuleType):
    """Test piecewise with rates_multiplier=0."""
    parameter_dict = {
        0: {
            "lower_threshold": "-inf",
            "upper_threshold": 0,
            "rate_linear": 1.0,
            "intercept_at_lower_threshold": 5.0,
        },
        1: {
            "lower_threshold": 0,
            "upper_threshold": "inf",
            "rate_linear": 2.0,
            "intercept_at_lower_threshold": 10.0,
        },
    }

    params = get_piecewise_parameters(
        leaf_name="test_mult_zero",
        func_type="piecewise_linear",
        parameter_dict=parameter_dict,
        xnp=xnp,
    )

    x = xnp.array([-10.0, 0.0, 10.0])
    result = piecewise_polynomial(x=x, parameters=params, xnp=xnp, rates_multiplier=0.0)

    # With rates_multiplier=0, only intercepts remain (but entire expression is multiplied)
    # Actually looking at code: return rates_multiplier * (intercepts + polynomial)
    # So with 0, result is 0
    expected = xnp.array([0.0, 0.0, 0.0])
    numpy.testing.assert_allclose(result, expected)


# =============================================================================
# get_piecewise_parameters validation tests
# =============================================================================


def test_get_piecewise_params_non_consecutive_keys_raises(xnp: ModuleType):
    """Test that non-consecutive keys raise ValueError."""
    parameter_dict = {
        0: {
            "lower_threshold": "-inf",
            "upper_threshold": 0,
            "rate_linear": 0.0,
            "intercept_at_lower_threshold": 0.0,
        },
        2: {  # Skip 1
            "lower_threshold": 0,
            "upper_threshold": "inf",
            "rate_linear": 0.0,
            "intercept_at_lower_threshold": 0.0,
        },
    }

    with pytest.raises(ValueError, match="not start with 0 or are not consecutive"):
        get_piecewise_parameters(
            leaf_name="test_non_consec",
            func_type="piecewise_linear",
            parameter_dict=parameter_dict,
            xnp=xnp,
        )


def test_get_piecewise_params_keys_not_starting_at_zero_raises(xnp: ModuleType):
    """Test that keys not starting at 0 raise ValueError."""
    parameter_dict = {
        1: {
            "lower_threshold": "-inf",
            "upper_threshold": 0,
            "rate_linear": 0.0,
            "intercept_at_lower_threshold": 0.0,
        },
        2: {
            "lower_threshold": 0,
            "upper_threshold": "inf",
            "rate_linear": 0.0,
            "intercept_at_lower_threshold": 0.0,
        },
    }

    with pytest.raises(ValueError, match="not start with 0"):
        get_piecewise_parameters(
            leaf_name="test_not_zero",
            func_type="piecewise_linear",
            parameter_dict=parameter_dict,
            xnp=xnp,
        )


def test_get_piecewise_params_missing_first_lower_threshold_raises(xnp: ModuleType):
    """Test that missing lower_threshold in first piece raises ValueError."""
    parameter_dict = {
        0: {
            # Missing lower_threshold
            "upper_threshold": 0,
            "rate_linear": 0.0,
            "intercept_at_lower_threshold": 0.0,
        },
        1: {
            "lower_threshold": 0,
            "upper_threshold": "inf",
            "rate_linear": 0.0,
            "intercept_at_lower_threshold": 0.0,
        },
    }

    with pytest.raises(ValueError, match="first piece.*lower_threshold"):
        get_piecewise_parameters(
            leaf_name="test_missing_first_lower",
            func_type="piecewise_linear",
            parameter_dict=parameter_dict,
            xnp=xnp,
        )


def test_get_piecewise_params_missing_last_upper_threshold_raises(xnp: ModuleType):
    """Test that missing upper_threshold in last piece raises ValueError."""
    parameter_dict = {
        0: {
            "lower_threshold": "-inf",
            "upper_threshold": 0,
            "rate_linear": 0.0,
            "intercept_at_lower_threshold": 0.0,
        },
        1: {
            "lower_threshold": 0,
            # Missing upper_threshold
            "rate_linear": 0.0,
            "intercept_at_lower_threshold": 0.0,
        },
    }

    with pytest.raises(ValueError, match="last piece.*upper_threshold"):
        get_piecewise_parameters(
            leaf_name="test_missing_last_upper",
            func_type="piecewise_linear",
            parameter_dict=parameter_dict,
            xnp=xnp,
        )


def test_get_piecewise_params_not_defined_on_real_line_raises(xnp: ModuleType):
    """Test that function not defined on entire real line raises ValueError."""
    parameter_dict = {
        0: {
            "lower_threshold": 0,  # Not -inf
            "upper_threshold": 100,
            "rate_linear": 0.0,
            "intercept_at_lower_threshold": 0.0,
        },
        1: {
            "lower_threshold": 100,
            "upper_threshold": "inf",
            "rate_linear": 0.0,
            "intercept_at_lower_threshold": 0.0,
        },
    }

    with pytest.raises(ValueError, match="defined on the entire real line"):
        get_piecewise_parameters(
            leaf_name="test_not_real_line",
            func_type="piecewise_linear",
            parameter_dict=parameter_dict,
            xnp=xnp,
        )


def test_get_piecewise_params_missing_rate_raises(xnp: ModuleType):
    """Test that missing rate in a piece raises ValueError."""
    parameter_dict = {
        0: {
            "lower_threshold": "-inf",
            "upper_threshold": 0,
            "rate_linear": 0.0,
            "intercept_at_lower_threshold": 0.0,
        },
        1: {
            "lower_threshold": 0,
            "upper_threshold": "inf",
            # Missing rate_linear
            "intercept_at_lower_threshold": 0.0,
        },
    }

    with pytest.raises(ValueError, match="rate_linear is missing"):
        get_piecewise_parameters(
            leaf_name="test_missing_rate",
            func_type="piecewise_linear",
            parameter_dict=parameter_dict,
            xnp=xnp,
        )


def test_get_piecewise_params_thresholds_dont_coincide_raises(xnp: ModuleType):
    """Test that non-coinciding thresholds raise ValueError."""
    parameter_dict = {
        0: {
            "lower_threshold": "-inf",
            "upper_threshold": 100,
            "rate_linear": 0.0,
            "intercept_at_lower_threshold": 0.0,
        },
        1: {
            "lower_threshold": 200,  # Doesn't match upper of piece 0
            "upper_threshold": "inf",
            "rate_linear": 0.0,
            "intercept_at_lower_threshold": 0.0,
        },
    }

    with pytest.raises(ValueError, match="have to coincide"):
        get_piecewise_parameters(
            leaf_name="test_non_coincide",
            func_type="piecewise_linear",
            parameter_dict=parameter_dict,
            xnp=xnp,
        )


def test_get_piecewise_params_partial_intercepts_raises(xnp: ModuleType):
    """Test that providing some but not all intercepts raises ValueError."""
    parameter_dict = {
        0: {
            "lower_threshold": "-inf",
            "upper_threshold": 0,
            "rate_linear": 0.1,
            "intercept_at_lower_threshold": 0.0,
        },
        1: {
            "lower_threshold": 0,
            "upper_threshold": 100,
            "rate_linear": 0.2,
            "intercept_at_lower_threshold": 10.0,
        },
        2: {
            "lower_threshold": 100,
            "upper_threshold": "inf",
            "rate_linear": 0.3,
            # Missing intercept_at_lower_threshold
        },
    }

    with pytest.raises(ValueError, match="not all intercepts are supplied"):
        get_piecewise_parameters(
            leaf_name="test_partial_intercepts",
            func_type="piecewise_linear",
            parameter_dict=parameter_dict,
            xnp=xnp,
        )


# =============================================================================
# _create_intercepts tests
# =============================================================================


def test_create_intercepts_linear_continuity(xnp: ModuleType):
    """Test that _create_intercepts ensures continuity for linear functions."""
    lower_thresholds = xnp.array([-numpy.inf, 0.0, 100.0])
    upper_thresholds = xnp.array([0.0, 100.0, numpy.inf])
    rates = xnp.array([[0.0, 0.1, 0.2]])  # Linear rates only
    intercept_at_lowest = 5.0  # Starting value

    intercepts = _create_intercepts(
        lower_thresholds=lower_thresholds,
        upper_thresholds=upper_thresholds,
        rates=rates,
        intercept_at_lowest_threshold=intercept_at_lowest,
        xnp=xnp,
    )

    # First intercept is given
    numpy.testing.assert_allclose(float(intercepts[0]), 5.0)
    # At threshold 0: first segment has rate 0, so value stays 5.0
    # Second segment starts at 5.0
    numpy.testing.assert_allclose(float(intercepts[1]), 5.0)
    # At threshold 100: second segment value = 5 + 0.1 * 100 = 15
    numpy.testing.assert_allclose(float(intercepts[2]), 15.0)


def test_create_intercepts_quadratic_continuity(xnp: ModuleType):
    """Test that _create_intercepts ensures continuity for quadratic functions."""
    lower_thresholds = xnp.array([-numpy.inf, 0.0, 10.0])
    upper_thresholds = xnp.array([0.0, 10.0, numpy.inf])
    # [rate_linear, rate_quadratic] per segment
    rates = xnp.array([
        [0.0, 1.0, 0.0],  # Linear rates
        [0.0, 0.1, 0.0],  # Quadratic rates
    ])
    intercept_at_lowest = 5.0

    intercepts = _create_intercepts(
        lower_thresholds=lower_thresholds,
        upper_thresholds=upper_thresholds,
        rates=rates,
        intercept_at_lowest_threshold=intercept_at_lowest,
        xnp=xnp,
    )

    # First intercept is given
    numpy.testing.assert_allclose(float(intercepts[0]), 5.0)

    # At x=0: first segment value = 5 + 0*(0-(-inf)) = 5 (since -inf, increment=0)
    # Second segment should start at this value
    numpy.testing.assert_allclose(float(intercepts[1]), 5.0)

    # At x=10: second segment value = 5 + 1.0*10 + 0.1*100 = 5 + 10 + 10 = 25
    numpy.testing.assert_allclose(float(intercepts[2]), 25.0)


# =============================================================================
# get_piecewise_thresholds tests
# =============================================================================


def test_get_piecewise_thresholds_inferred_from_adjacent(xnp: ModuleType):
    """Test that thresholds are correctly inferred from adjacent pieces."""
    parameter_dict = {
        0: {
            "lower_threshold": "-inf",
            "upper_threshold": 100,
        },
        1: {
            # lower_threshold inferred from piece 0's upper_threshold
            "upper_threshold": 200,
        },
        2: {
            "lower_threshold": 200,
            "upper_threshold": "inf",
        },
    }

    lower_thr, upper_thr, thresholds = get_piecewise_thresholds(
        leaf_name="test_infer",
        parameter_dict=parameter_dict,
        xnp=xnp,
    )

    numpy.testing.assert_array_equal(lower_thr, xnp.array([-numpy.inf, 100.0, 200.0]))
    numpy.testing.assert_array_equal(upper_thr, xnp.array([100.0, 200.0, numpy.inf]))


def test_get_piecewise_thresholds_missing_both_raises(xnp: ModuleType):
    """Test that missing both thresholds in middle piece raises ValueError."""
    parameter_dict = {
        0: {
            "lower_threshold": "-inf",
            # No upper_threshold
        },
        1: {
            # No lower_threshold, no upper_threshold in piece 0
            "upper_threshold": "inf",
        },
    }

    with pytest.raises(ValueError, match="no lower.*upper"):
        get_piecewise_thresholds(
            leaf_name="test_missing_both",
            parameter_dict=parameter_dict,
            xnp=xnp,
        )
