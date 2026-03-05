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
    expected = xnp.array([0.0, 0.0, 123.265, 5275.825, 33219.1, 433259.32])

    actual = piecewise_polynomial(
        x=x,
        parameters=parameters,
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


def test_merge_piecewise_intervals_full_replacement():
    base = [
        {"interval": "[0, 100)", "slope": 0.1, "intercept": 0},
        {"interval": "[100, inf)", "slope": 0.2, "intercept": 10},
    ]
    update = [
        {"interval": "[0, 200)", "slope": 0.15, "intercept": 0},
        {"interval": "[200, inf)", "slope": 0.25, "intercept": 30},
    ]
    result = merge_piecewise_intervals(base=base, update=update)
    assert len(result) == 2
    assert result[0]["slope"] == 0.15
    assert result[1]["slope"] == 0.25


def test_merge_piecewise_intervals_partial_update_trims_carried_over():
    base = [
        {"interval": "[0, 16956)", "intercept": 0, "slope": 0},
        {"interval": "[16956, 31528)", "slope": 0.119},
        {"interval": "[31528, inf)", "slope": 0.055},
    ]
    update = [
        {"interval": "[0, 17543)"},
        {"interval": "[17543, 32619)", "slope": 0.119},
    ]
    result = merge_piecewise_intervals(base=base, update=update)
    assert len(result) == 3
    # Carried-over third interval should be trimmed to start at 32619
    assert "32619" in result[2]["interval"]
    assert result[2]["slope"] == 0.055


def test_merge_piecewise_intervals_unspecified_coefficients_not_inherited():
    base = [
        {"interval": "[0, 100)", "intercept": 0, "slope": 0},
        {"interval": "[100, inf)", "slope": 0.5, "intercept": 50},
    ]
    update = [
        {"interval": "[0, 150)"},  # No coefficients specified
    ]
    result = merge_piecewise_intervals(base=base, update=update)
    # Update interval should only have "interval" key, no inherited coefficients
    assert "intercept" not in result[0]
    assert "slope" not in result[0]
    # Carried-over interval trimmed
    assert result[1]["slope"] == 0.5


def test_merge_piecewise_intervals_empty_update_returns_base():
    base = [
        {"interval": "[0, inf)", "slope": 0.1, "intercept": 0},
    ]
    result = merge_piecewise_intervals(base=base, update=[])
    assert result == base


def test_merge_piecewise_intervals_chained_updates():
    """Multiple sequential updates should compose correctly."""
    base = [
        {"interval": "[0, 100)", "intercept": 0, "slope": 0.1},
        {"interval": "[100, 200)", "slope": 0.2},
        {"interval": "[200, inf)", "slope": 0.3},
    ]
    update1 = [
        {"interval": "[0, 120)", "slope": 0.15},
    ]
    after_first = merge_piecewise_intervals(base=base, update=update1)
    update2 = [
        {"interval": "[0, 130)"},
    ]
    result = merge_piecewise_intervals(base=after_first, update=update2)
    # Update interval has no coefficients, so none should be present
    assert "slope" not in result[0]
    assert len(result) >= 2
