"""
Tests for `piecewise_polynomial`
"""

import numpy
import pytest

from ttsim.piecewise_polynomial import (
    get_piecewise_parameters,
    piecewise_polynomial_jax,
    piecewise_polynomial_numpy,
)


@pytest.fixture
def eink_st_params():
    params = {
        "thresholds": numpy.array(
            [-numpy.inf, 9168.0, 14254.0, 55960.0, 265326.0, numpy.inf]
        ),
        "rates": numpy.array(
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
            ]
        ),
        "intercepts_at_lower_thresholds": numpy.array(
            [0.0, 0.0, 965.5771, 14722.3012, 102656.0212]
        ),
    }
    return params


def test_get_piecewise_parameters_all_intercepts_supplied():
    params_dict = {
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
        parameter_dict=params_dict,
        parameter="test",
        func_type="linear",
    )["intercepts_at_lower_thresholds"]
    expected = numpy.array([0.27, 0.5, 0.8, 1])

    numpy.testing.assert_almost_equal(actual, expected, decimal=10)


def test_piecewise_polynomial(eink_st_params):
    x_linspace = numpy.linspace(100, 100_000, 100)

    y_numpy = piecewise_polynomial_numpy(
        x=x_linspace,
        thresholds=eink_st_params["thresholds"],
        rates=eink_st_params["rates"],
        intercepts_at_lower_thresholds=eink_st_params["intercepts_at_lower_thresholds"],
        rates_multiplier=2,
    )
    y_legacy = numpy.array(
        [
            piecewise_polynomial_legacy(
                x=xi,
                thresholds=eink_st_params["thresholds"],
                rates=eink_st_params["rates"],
                intercepts_at_lower_thresholds=eink_st_params[
                    "intercepts_at_lower_thresholds"
                ],
                rates_multiplier=2,
            )
            for xi in x_linspace
        ]
    )
    numpy.testing.assert_allclose(y_numpy, y_legacy,rtol=1e-6)


def test_piecewise_polynomial_jax(eink_st_params):
    x_linspace = numpy.linspace(100, 100_000, 100)
    y_jax = piecewise_polynomial_jax(
        x=x_linspace,
        thresholds=eink_st_params["thresholds"],
        rates=eink_st_params["rates"],
        intercepts_at_lower_thresholds=eink_st_params["intercepts_at_lower_thresholds"],
        rates_multiplier=2,
    )
    y_legacy = numpy.array(
        [
            piecewise_polynomial_legacy(
                x=xi,
                thresholds=eink_st_params["thresholds"],
                rates=eink_st_params["rates"],
                intercepts_at_lower_thresholds=eink_st_params[
                    "intercepts_at_lower_thresholds"
                ],
                rates_multiplier=2,
            )
            for xi in x_linspace
        ]
    )
    numpy.testing.assert_allclose(y_jax, y_legacy, rtol=1e-6)


def piecewise_polynomial_legacy(
    x, thresholds, rates, intercepts_at_lower_thresholds, rates_multiplier=None
):
    """Calculate value of the piecewise function at `x`.

    Parameters
    ----------
    x : pd.Series
        Series with values which piecewise polynomial is applied to.
    thresholds : numpy.array
                A one-dimensional array containing the thresholds for all intervals.
    rates : numpy.ndarray
            A two-dimensional array where columns are interval sections and rows
            correspond to the nth polynomial.
    intercepts_at_lower_thresholds : numpy.ndarray
        The intercepts at the lower threshold of each interval.
    rates_multiplier : pd.Series, float
                       Multiplier to create individual or scaled rates. If given and
                       not equal to 1, the function also calculates new intercepts.

    Returns
    -------
    out : float
        The value of `x` under the piecewise function.

    """
    num_intervals = len(thresholds) - 1
    degree_polynomial = rates.shape[0]

    # Check in which interval each individual is. The thresholds are not exclusive on
    # the right side.
    selected_bin = numpy.searchsorted(thresholds, x, side="right") - 1

    # Calc last threshold for each individual
    threshold = thresholds[selected_bin]

    # Increment for each individual in the corresponding interval.
    increment_to_calc = x - threshold

    # If each individual has its own rates or the rates are scaled, we can't use the
    # intercept, which was generated in the parameter loading.
    if rates_multiplier is not None:
        # Initialize Series containing 0 for all individuals.
        out = intercepts_at_lower_thresholds[0]

        # Go through all intervals except the first and last.
        for i in range(2, num_intervals):
            threshold_incr = thresholds[i] - thresholds[i - 1]
            for pol in range(1, degree_polynomial + 1):
                # We only calculate the intercepts for individuals who are in this or
                # higher interval. Hence we have to use the individual rates.
                if selected_bin >= i:
                    out += (
                        rates_multiplier * rates[pol - 1, i - 1] * threshold_incr**pol
                    )
    # If rates remain the same, everything is a lot easier.
    else:
        # We assign each individual the pre-calculated intercept.
        out = intercepts_at_lower_thresholds[selected_bin]
    # Intialize a multiplyer for 1 if it is not given.
    rates_multiplier = 1 if rates_multiplier is None else rates_multiplier

    if selected_bin > 0:
        # Now add the evaluation of the increment
        for pol in range(1, degree_polynomial + 1):
            out += (
                rates[pol - 1][selected_bin]
                * rates_multiplier
                * (increment_to_calc**pol)
            )

    return out
