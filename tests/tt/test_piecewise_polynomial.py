"""
Tests for `piecewise_polynomial`
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy
import pytest

if TYPE_CHECKING:
    from types import ModuleType


from ttsim.tt.piecewise_polynomial import (
    PiecewisePolynomialParamValue,
    get_piecewise_parameters,
    piecewise_polynomial,
)


@pytest.fixture
def parameters(xnp: ModuleType):
    return PiecewisePolynomialParamValue(
        thresholds=xnp.array([-xnp.inf, 9168.0, 14254.0, 55960.0, 265326.0, xnp.inf]),
        rates=xnp.array(
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
    parameter_list = [
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
    expected = xnp.array([0.0, 0.0, 246.53, 10551.65, 66438.2, 866518.64])

    actual = piecewise_polynomial(
        x=x,
        parameters=parameters,
        rates_multiplier=2,
        xnp=xnp,
    )
    numpy.testing.assert_allclose(xnp.array(actual), expected, atol=0.01)


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
    parameter_list = [
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
    parameter_list = [
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
    parameter_list = [
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
