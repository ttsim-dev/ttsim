from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, get_args

import numpy
import portion

from ttsim.tt.interval_utils import (
    intervals_to_thresholds,
    validate_intervals,
)
from ttsim.tt.param_objects import PiecewisePolynomialParamValue

if TYPE_CHECKING:
    from types import ModuleType

    from jaxtyping import Array, Float

FUNC_TYPES = Literal[
    "piecewise_constant",
    "piecewise_linear",
    "piecewise_quadratic",
    "piecewise_cubic",
]


@dataclass(frozen=True)
class RatesOptions:
    required_keys: tuple[Literal["slope", "quadratic", "cubic"], ...]
    rates_size: int


OPTIONS_REGISTRY = {
    "piecewise_constant": RatesOptions(
        required_keys=(),
        rates_size=1,
    ),
    "piecewise_linear": RatesOptions(
        required_keys=("slope",),
        rates_size=1,
    ),
    "piecewise_quadratic": RatesOptions(
        required_keys=("slope", "quadratic"),
        rates_size=2,
    ),
    "piecewise_cubic": RatesOptions(
        required_keys=("slope", "quadratic", "cubic"),
        rates_size=3,
    ),
}

if set(OPTIONS_REGISTRY.keys()) != set(get_args(FUNC_TYPES)):
    raise ValueError("Keys in OPTIONS_REGISTRY must match FUNC_TYPES")


def piecewise_polynomial(
    x: Float[Array, " n_pp_values"],
    parameters: PiecewisePolynomialParamValue,
    xnp: ModuleType,
    rates_multiplier: Float[Array, " n_segments"] | float = 1.0,
) -> Float[Array, " n_pp_values"]:
    """Calculate value of the piecewise function at `x`.

    Values outside the defined domain return NaN.

    Parameters
    ----------
    x:
        Array with values at which the piecewise polynomial is to be calculated.
    parameters:
        Thresholds defining the pieces and coefficients on each piece.
    xnp:
        The backend module to use for calculations.
    rates_multiplier:
        Multiplier to create individual or scaled rates.

    Returns
    -------
    out:
        The value of `x` under the piecewise function.

    """
    n_intervals = parameters.rates.shape[0]
    order = parameters.rates.shape[1]
    # Get interval of requested value
    selected_bin = xnp.searchsorted(parameters.thresholds, x, side="right") - 1

    # Clamp to valid range for indexing (we'll mask out-of-domain later)
    clamped_bin = xnp.clip(selected_bin, 0, n_intervals - 1)

    coefficients = parameters.rates[clamped_bin]
    # Calculate distance from x to lower threshold
    increment_to_calc = xnp.where(
        parameters.thresholds[clamped_bin] == -xnp.inf,
        0,
        x - parameters.thresholds[clamped_bin],
    )
    # Evaluate polynomial at x
    result = rates_multiplier * (
        parameters.intercepts[clamped_bin]
        + (
            increment_to_calc.reshape(-1, 1) ** xnp.arange(1, order + 1, 1)
            * coefficients
        ).sum(axis=1)
    )

    # NaN for out-of-domain values
    out_of_domain = (selected_bin < 0) | (selected_bin >= n_intervals)
    return xnp.where(out_of_domain, xnp.array(float("nan")), result)


def get_piecewise_parameters(
    leaf_name: str,
    func_type: FUNC_TYPES,
    parameter_list: list[dict[str, float | str]],
    xnp: ModuleType,
) -> PiecewisePolynomialParamValue:
    """Create the objects for piecewise polynomial from a list of interval specs.

    Parameters
    ----------
    leaf_name:
        Name of the parameter (for error messages).
    func_type:
        The type of piecewise function.
    parameter_list:
        List of dicts, each with an 'interval' string and coefficient keys.
    xnp:
        The backend module to use for calculations.

    Returns
    -------
    PiecewisePolynomialParamValue

    """
    # Parse intervals
    intervals = [
        portion.from_string(item["interval"], conv=float) for item in parameter_list
    ]
    validate_intervals(intervals, leaf_name)

    # Extract thresholds
    lower_thresholds, upper_thresholds, thresholds = intervals_to_thresholds(
        intervals=intervals,
        xnp=xnp,
    )

    # Create and fill rates-array
    rates = _check_and_get_rates(
        parameter_list=parameter_list,
        leaf_name=leaf_name,
        func_type=func_type,
        xnp=xnp,
    )
    # Create and fill intercept-array
    intercepts = _check_and_get_intercepts(
        parameter_list=parameter_list,
        leaf_name=leaf_name,
        lower_thresholds=lower_thresholds,
        upper_thresholds=upper_thresholds,
        rates=rates,
        xnp=xnp,
    )
    return PiecewisePolynomialParamValue(
        thresholds=thresholds,
        rates=rates,
        intercepts=intercepts,
    )


def get_piecewise_thresholds(
    leaf_name: str,
    parameter_list: list[dict[str, float | str]],
    xnp: ModuleType,
) -> tuple[
    Float[Array, " n_segments"],
    Float[Array, " n_segments"],
    Float[Array, " n_segments"],
]:
    """Check and extract threshold data from list-of-dicts format.

    Parameters
    ----------
    leaf_name:
        Name of the parameter (for error messages).
    parameter_list:
        List of dicts, each with an 'interval' string.
    xnp:
        The numpy module to use for calculations.

    Returns
    -------
    (lower_thresholds, upper_thresholds, thresholds)

    """
    intervals = [
        portion.from_string(item["interval"], conv=float) for item in parameter_list
    ]
    validate_intervals(intervals, leaf_name)
    return intervals_to_thresholds(intervals=intervals, xnp=xnp)


def _check_and_get_rates(
    leaf_name: str,
    func_type: FUNC_TYPES,
    parameter_list: list[dict[str, float | str]],
    xnp: ModuleType,
) -> Float[Array, "n_intervals n_coefficients"]:
    """Check and extract rates data from the list-of-dicts format.

    Returns rates with shape (n_intervals, n_coefficients).
    """
    n_intervals = len(parameter_list)
    options = OPTIONS_REGISTRY[func_type]
    rates = numpy.zeros((n_intervals, options.rates_size))
    for i, item in enumerate(parameter_list):
        for j, rate_type in enumerate(options.required_keys):
            if rate_type not in item:
                raise ValueError(
                    f"In interval {i} of {leaf_name}, {rate_type} is missing.",
                )
            rates[i, j] = item[rate_type]
    return xnp.array(rates)


def _check_and_get_intercepts(
    leaf_name: str,
    parameter_list: list[dict[str, float | str]],
    lower_thresholds: Float[Array, " n_segments"],
    upper_thresholds: Float[Array, " n_segments"],
    rates: Float[Array, "n_intervals n_coefficients"],
    xnp: ModuleType,
) -> Float[Array, " n_segments"]:
    """Check and extract intercept data. If necessary create intercepts."""
    n_intervals = len(parameter_list)
    intercepts = numpy.zeros(n_intervals)
    count_intercepts_supplied = 1

    if "intercept" not in parameter_list[0]:
        raise ValueError(f"The first piece of {leaf_name} needs an intercept.")
    intercepts[0] = parameter_list[0]["intercept"]
    # Check if all intercepts are supplied.
    for i in range(1, n_intervals):
        if "intercept" in parameter_list[i]:
            count_intercepts_supplied += 1
            intercepts[i] = parameter_list[i]["intercept"]
    if 1 < count_intercepts_supplied < n_intervals:
        raise ValueError(
            "More than one, but not all intercepts are supplied. "
            "The dictionaries should contain either only the lowest intercept "
            "or all intercepts.",
        )
    if count_intercepts_supplied < n_intervals:
        intercepts = _create_intercepts(
            lower_thresholds,
            upper_thresholds,
            rates,
            intercepts[0],
            xnp=xnp,
        )
    return xnp.array(intercepts)


def _create_intercepts(
    lower_thresholds: Float[Array, " n_segments"],
    upper_thresholds: Float[Array, " n_segments"],
    rates: Float[Array, "n_intervals n_coefficients"],
    intercept_at_lowest_threshold: float,
    xnp: ModuleType,
) -> Float[Array, " n_segments"]:
    """Create intercepts from raw data."""
    intercepts = numpy.full_like(upper_thresholds, numpy.nan)
    intercepts[0] = intercept_at_lowest_threshold
    for i, up_thr in enumerate(upper_thresholds[:-1]):
        intercepts[i + 1] = _calculate_one_intercept(
            x=up_thr,
            lower_thresholds=lower_thresholds,
            upper_thresholds=upper_thresholds,
            rates=rates,
            intercepts=intercepts,
        )
    return xnp.array(intercepts)


def _calculate_one_intercept(
    x: float,
    lower_thresholds: Float[Array, " n_segments"],
    upper_thresholds: Float[Array, " n_segments"],
    rates: Float[Array, "n_intervals n_coefficients"],
    intercepts: Float[Array, " n_segments"],
) -> float:
    """Calculate the intercept for the segment `x` lies in."""
    # Check if value lies within the defined range.
    if (x < lower_thresholds[0]) or (x > upper_thresholds[-1]) or numpy.isnan(x):
        return numpy.nan
    index_interval = numpy.searchsorted(upper_thresholds, x, side="left")
    intercept_interval = intercepts[index_interval]

    # Select threshold and calculate corresponding increment into interval
    lower_threshold_interval = lower_thresholds[index_interval]

    if lower_threshold_interval == -numpy.inf:
        return intercept_interval

    increment_to_calc = x - lower_threshold_interval

    out = intercept_interval
    for j in range(rates.shape[1]):
        out += rates[index_interval, j] * increment_to_calc ** (j + 1)
    return out
