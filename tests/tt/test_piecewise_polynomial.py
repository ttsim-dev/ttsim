"""
Tests for `piecewise_polynomial`
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy
import pytest

if TYPE_CHECKING:
    from types import ModuleType


from ttsim.tt.interval_utils import merge_piecewise_intervals
from ttsim.tt.param_objects import PiecewisePolynomialInterval
from ttsim.tt.piecewise_polynomial import (
    PiecewisePolynomialParamValue,
    get_piecewise_parameters,
    piecewise_polynomial,
)


@pytest.fixture
def parameters(xnp: ModuleType):
    return PiecewisePolynomialParamValue(
        thresholds=xnp.array([-xnp.inf, 9168.0, 14254.0, 55960.0, 265326.0, xnp.inf]),
        coefficients=xnp.array(
            [
                [0.00000000e00, 0.00000000e00],
                [1.40000000e-01, 9.80141565e-06],
                [2.39700000e-01, 2.16155949e-06],
                [4.20000000e-01, 0.00000000e00],
                [4.50000000e-01, 0.00000000e00],
            ],
        ),
        intercepts=xnp.array([0.0, 0.0, 965.5771, 14722.3012, 102656.0212]),
    )


def test_get_piecewise_parameters_all_intercepts_supplied(xnp: ModuleType):
    parameter_list: list[dict[str, int | float | str]] = [
        {
            "interval": "(-inf, 2005)",
            "slope": 0,
            "intercept": 0.27,
        },
        {
            "interval": "[2005, 2021)",
            "slope": 0.02,
            "intercept": 0.5,
        },
        {
            "interval": "[2021, 2041)",
            "slope": 0.01,
            "intercept": 0.8,
        },
        {
            "interval": "[2041, inf)",
            "slope": 0,
            "intercept": 1,
        },
    ]

    actual = get_piecewise_parameters(
        leaf_name="test",
        func_type="piecewise_linear",
        parameter_list=parameter_list,
        xnp=xnp,
    )
    expected = xnp.array([0.27, 0.5, 0.8, 1])

    numpy.testing.assert_allclose(actual.intercepts, expected, atol=1e-7)


def test_piecewise_polynomial(
    parameters: PiecewisePolynomialParamValue,
    xnp: ModuleType,
):
    x = xnp.array([-1_000, 1_000, 10_000, 30_000, 100_000, 1_000_000])
    expected = xnp.array([0.0, 0.0, 123.265, 5275.825, 33219.1, 433259.32])

    actual = piecewise_polynomial(
        x=x,
        parameters=parameters,
        xnp=xnp,
    )
    numpy.testing.assert_allclose(xnp.array(actual), expected, atol=0.01)


def test_piecewise_polynomial_scalar_input(
    parameters: PiecewisePolynomialParamValue,
    xnp: ModuleType,
):
    """piecewise_polynomial accepts a scalar float and returns a 1-element array."""
    result = piecewise_polynomial(x=30_000.0, parameters=parameters, xnp=xnp)
    assert result.shape == (1,)
    numpy.testing.assert_allclose(result[0], 5275.825, atol=0.01)


def test_partial_domain_returns_nan(xnp: ModuleType):
    """Values outside partial domain should return NaN."""
    parameter_list = [
        {
            "interval": "[0, 100)",
            "slope": 0.1,
            "intercept": 0,
        },
        {
            "interval": "[100, 200)",
            "slope": 0.2,
            "intercept": 10,
        },
    ]

    params = get_piecewise_parameters(
        leaf_name="test_partial",
        func_type="piecewise_linear",
        parameter_list=parameter_list,
        xnp=xnp,
    )

    x = xnp.array([-10.0, 50.0, 150.0, 300.0])
    result = piecewise_polynomial(x=x, parameters=params, xnp=xnp)

    assert numpy.isnan(result[0]), "Value below domain should be NaN"
    numpy.testing.assert_allclose(result[1], 5.0, atol=1e-7)
    numpy.testing.assert_allclose(result[2], 20.0, atol=1e-7)
    assert numpy.isnan(result[3]), "Value above domain should be NaN"


def test_interval_parsing_edge_cases(xnp: ModuleType):
    """Test various interval bracket combinations."""
    parameter_list: list[dict[str, int | float | str]] = [
        {
            "interval": "(-inf, 0)",
            "slope": 0,
            "intercept": 0,
        },
        {
            "interval": "[0, inf)",
            "slope": 0.5,
            "intercept": 0,
        },
    ]

    params = get_piecewise_parameters(
        leaf_name="test_brackets",
        func_type="piecewise_linear",
        parameter_list=parameter_list,
        xnp=xnp,
    )

    x = xnp.array([-10.0, 0.0, 10.0])
    result = piecewise_polynomial(x=x, parameters=params, xnp=xnp)

    numpy.testing.assert_allclose(result[0], 0.0, atol=1e-7)
    numpy.testing.assert_allclose(result[1], 0.0, atol=1e-7)
    numpy.testing.assert_allclose(result[2], 5.0, atol=1e-7)


def test_validation_error_gaps(xnp: ModuleType):
    """Gaps between intervals should raise ValueError."""
    parameter_list: list[dict[str, int | float | str]] = [
        {
            "interval": "(-inf, 0)",
            "slope": 0,
            "intercept": 0,
        },
        {
            "interval": "[5, inf)",
            "slope": 0.5,
            "intercept": 0,
        },
    ]
    with pytest.raises(ValueError, match="Gap between intervals"):
        get_piecewise_parameters(
            leaf_name="test_gaps",
            func_type="piecewise_linear",
            parameter_list=parameter_list,
            xnp=xnp,
        )


def test_validation_error_wrong_order(xnp: ModuleType):
    """Intervals not in ascending order should raise ValueError."""
    parameter_list: list[dict[str, int | float | str]] = [
        {
            "interval": "[5, inf)",
            "slope": 0.5,
            "intercept": 0,
        },
        {
            "interval": "(-inf, 5)",
            "slope": 0,
            "intercept": 0,
        },
    ]
    with pytest.raises(ValueError, match="not in ascending order"):
        get_piecewise_parameters(
            leaf_name="test_order",
            func_type="piecewise_linear",
            parameter_list=parameter_list,
            xnp=xnp,
        )


# --- Tests for merge_piecewise_intervals ---


def _base_intervals():
    return [
        {"interval": "(-inf, 0)", "slope": 0, "intercept": 0},
        {"interval": "[0, 100)", "slope": 0.1, "intercept": 0},
        {"interval": "[100, inf)", "slope": 0.2, "intercept": 10},
    ]


def test_merge_single_interval_update():
    """Replace one of three intervals."""
    base = _base_intervals()
    update = [{"interval": "[0, 100)", "slope": 0.5, "intercept": 0}]
    result = merge_piecewise_intervals(base, update)
    assert result[0] == base[0]
    assert result[1] == update[0]
    assert result[2] == base[2]


def test_merge_multiple_intervals_update():
    """Replace two of three intervals."""
    base = _base_intervals()
    update = [
        {"interval": "(-inf, 0)", "slope": 0.3, "intercept": 5},
        {"interval": "[100, inf)", "slope": 0.9, "intercept": 20},
    ]
    result = merge_piecewise_intervals(base, update)
    assert result[0] == update[0]
    assert result[1] == base[1]
    assert result[2] == update[1]


def test_merge_no_match_raises():
    """Update with bounds not in base should raise ValueError."""
    base = _base_intervals()
    update = [{"interval": "[50, 150)", "slope": 0.5, "intercept": 0}]
    with pytest.raises(ValueError, match="does not match any base interval"):
        merge_piecewise_intervals(base, update)


def test_merge_preserves_order():
    """Output order matches base order regardless of update order."""
    base = _base_intervals()
    update = [
        {"interval": "[100, inf)", "slope": 0.9, "intercept": 20},
        {"interval": "(-inf, 0)", "slope": 0.3, "intercept": 5},
    ]
    result = merge_piecewise_intervals(base, update)
    assert result[0] == update[1]  # (-inf, 0) at position 0
    assert result[1] == base[1]  # [0, 100) unchanged
    assert result[2] == update[0]  # [100, inf) at position 2


def test_merge_full_replacement():
    """All intervals updated."""
    base = _base_intervals()
    update = [
        {"interval": "(-inf, 0)", "slope": 1, "intercept": 1},
        {"interval": "[0, 100)", "slope": 2, "intercept": 2},
        {"interval": "[100, inf)", "slope": 3, "intercept": 3},
    ]
    result = merge_piecewise_intervals(base, update)
    assert result == update


# --- Tests for overlapping intervals ---


def test_validation_error_double_open_gap(xnp: ModuleType):
    """Two intervals both open at a shared boundary should raise ValueError."""
    parameter_list: list[dict[str, int | float | str]] = [
        {
            "interval": "(-inf, 0)",
            "slope": 0,
            "intercept": 0,
        },
        {
            "interval": "(0, inf)",
            "slope": 0.5,
            "intercept": 0,
        },
    ]
    with pytest.raises(ValueError, match="Gap at boundary"):
        get_piecewise_parameters(
            leaf_name="test_double_open",
            func_type="piecewise_linear",
            parameter_list=parameter_list,
            xnp=xnp,
        )


def test_nonzero_coefficient_on_neg_inf_interval_raises(xnp: ModuleType):
    """Non-zero slope on (-inf, b) interval should raise ValueError."""
    parameter_list: list[dict[str, int | float | str]] = [
        {
            "interval": "(-inf, 0)",
            "slope": 0.5,
            "intercept": 0,
        },
        {
            "interval": "[0, inf)",
            "slope": 0.3,
            "intercept": 0,
        },
    ]
    with pytest.raises(ValueError, match="has no effect"):
        get_piecewise_parameters(
            leaf_name="test_neg_inf_coeff",
            func_type="piecewise_linear",
            parameter_list=parameter_list,
            xnp=xnp,
        )


def test_validation_error_overlapping(xnp: ModuleType):
    """Overlapping intervals should raise ValueError."""
    parameter_list = [
        {"interval": "[0, 100)", "slope": 0.1, "intercept": 0},
        {"interval": "[50, 200)", "slope": 0.2, "intercept": 10},
    ]
    with pytest.raises(ValueError, match="Overlapping intervals"):
        get_piecewise_parameters(
            leaf_name="test_overlap",
            func_type="piecewise_linear",
            parameter_list=parameter_list,
            xnp=xnp,
        )


# --- Tests for PiecewisePolynomialParamValue.__getitem__ ---


def test_param_value_getitem(xnp: ModuleType):
    """__getitem__ returns correct PiecewisePolynomialInterval."""
    params = get_piecewise_parameters(
        leaf_name="test_getitem",
        func_type="piecewise_linear",
        parameter_list=[
            {"interval": "[0, 100)", "slope": 0.5, "intercept": 0},
            {"interval": "[100, 200)", "slope": 0.4, "intercept": 5},
        ],
        xnp=xnp,
    )
    interval_0 = params[0]
    assert isinstance(interval_0, PiecewisePolynomialInterval)
    assert float(interval_0.intercept) == pytest.approx(0.0)
    assert float(interval_0.slope) == pytest.approx(0.5)

    interval_1 = params[1]
    assert float(interval_1.intercept) == pytest.approx(5.0)
    assert float(interval_1.slope) == pytest.approx(0.4)


# --- Tests for PiecewisePolynomialInterval property guards ---


def test_interval_slope_guard(xnp: ModuleType):
    """Accessing slope on a piecewise_constant interval raises AttributeError."""
    interval = PiecewisePolynomialInterval(
        intercept=1.0,
        coefficients=xnp.array([]),
    )
    with pytest.raises(AttributeError, match="piecewise_constant"):
        _ = interval.slope


def test_interval_quadratic_guard(xnp: ModuleType):
    """Accessing quadratic on piecewise_linear raises AttributeError."""
    interval = PiecewisePolynomialInterval(
        intercept=1.0,
        coefficients=xnp.array([0.5]),
    )
    with pytest.raises(AttributeError, match="piecewise_quadratic"):
        _ = interval.quadratic


def test_interval_cubic_guard(xnp: ModuleType):
    """Accessing cubic on piecewise_quadratic raises AttributeError."""
    interval = PiecewisePolynomialInterval(
        intercept=1.0,
        coefficients=xnp.array([0.5, 0.1]),
    )
    with pytest.raises(AttributeError, match="piecewise_cubic"):
        _ = interval.cubic


# --- Tests for piecewise_constant with n_coefficients=0 ---


def test_piecewise_constant(xnp: ModuleType):
    """piecewise_constant uses only intercepts, no coefficients."""
    parameter_list = [
        {"interval": "(-inf, 100)", "intercept": 5.0},
        {"interval": "[100, 200)", "intercept": 10.0},
        {"interval": "[200, inf)", "intercept": 15.0},
    ]
    params = get_piecewise_parameters(
        leaf_name="test_const",
        func_type="piecewise_constant",
        parameter_list=parameter_list,
        xnp=xnp,
    )
    assert params.coefficients.shape == (3, 0)

    x = xnp.array([50.0, 150.0, 250.0])
    result = piecewise_polynomial(x=x, parameters=params, xnp=xnp)
    numpy.testing.assert_allclose(result, xnp.array([5.0, 10.0, 15.0]), atol=1e-7)
