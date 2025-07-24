from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, get_args

import numpy

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
    required_keys: tuple[Literal["rate_linear", "rate_quadratic", "rate_cubic"], ...]
    rates_size: int


OPTIONS_REGISTRY = {
    "piecewise_constant": RatesOptions(
        required_keys=(),
        rates_size=1,
    ),
    "piecewise_linear": RatesOptions(
        required_keys=("rate_linear",),
        rates_size=1,
    ),
    "piecewise_quadratic": RatesOptions(
        required_keys=("rate_linear", "rate_quadratic"),
        rates_size=2,
    ),
    "piecewise_cubic": RatesOptions(
        required_keys=("rate_linear", "rate_quadratic", "rate_cubic"),
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
    """Calculate value of the piecewise function at `x`. If the first interval begins
    at -inf the polynomial of that interval can only have slope of 0. Requesting a
    value outside of the provided thresholds will lead to undefined behaviour.

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
    order = parameters.rates.shape[0]
    # Get interval of requested value
    selected_bin = xnp.searchsorted(parameters.thresholds, x, side="right") - 1
    coefficients = parameters.rates[:, selected_bin].T
    # Calculate distance from x to lower threshold
    increment_to_calc = xnp.where(
        parameters.thresholds[selected_bin] == -xnp.inf,
        0,
        x - parameters.thresholds[selected_bin],
    )
    # Evaluate polynomial at x
    return rates_multiplier * (
        parameters.intercepts[selected_bin]
        + (
            ((increment_to_calc.reshape(-1, 1)) ** xnp.arange(1, order + 1, 1))
            * (coefficients)
        ).sum(axis=1)
    )


def get_piecewise_parameters(
    leaf_name: str,
    func_type: FUNC_TYPES,
    parameter_dict: dict[int, dict[str, float]],
    xnp: ModuleType,
) -> PiecewisePolynomialParamValue:
    """Create the objects for piecewise polynomial.

    Parameters
    ----------
    parameter_dict
    leaf_name
    func_type

    Returns
    -------

    """
    # Check if keys are consecutive numbers and starting at 0.
    if sorted(parameter_dict) != list(range(len(parameter_dict))):
        raise ValueError(
            f"The keys of {leaf_name} do not start with 0 or are not consecutive"
            f" numbers.",
        )

    # Extract lower thresholds.
    lower_thresholds, upper_thresholds, thresholds = get_piecewise_thresholds(
        leaf_name=leaf_name,
        parameter_dict=parameter_dict,
        xnp=xnp,
    )

    # Create and fill rates-array
    rates = _check_and_get_rates(
        parameter_dict=parameter_dict,
        leaf_name=leaf_name,
        func_type=func_type,
        xnp=xnp,
    )
    # Create and fill intercept-array
    intercepts = _check_and_get_intercepts(
        parameter_dict=parameter_dict,
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


def get_piecewise_thresholds(  # noqa: C901
    leaf_name: str,
    parameter_dict: dict[int, dict[str, float]],
    xnp: ModuleType,
) -> tuple[
    Float[Array, " n_segments"],
    Float[Array, " n_segments"],
    Float[Array, " n_segments"],
]:
    """Check and transfer raw threshold data.

    Transfer and check raw threshold data, which needs to be specified in a
    piecewise_polynomial layout in the yaml file.

    Parameters
    ----------
    parameter_dict
    leaf_name
    keys
    xnp : ModuleType
        The numpy module to use for calculations.

    Returns
    -------

    """
    keys = sorted(parameter_dict.keys())
    lower_thresholds = numpy.zeros(len(parameter_dict))
    upper_thresholds = numpy.zeros(len(parameter_dict))

    # Check if lowest threshold exists.
    if "lower_threshold" not in parameter_dict[0]:
        raise ValueError(
            f"The first piece of {leaf_name} needs to contain a lower_threshold value.",
        )
    lower_thresholds[0] = parameter_dict[0]["lower_threshold"]

    # Check if highest upper_threshold exists.
    if "upper_threshold" not in parameter_dict[keys[-1]]:
        raise ValueError(
            f"The last piece of {leaf_name} needs to contain an upper_threshold value.",
        )
    upper_thresholds[keys[-1]] = parameter_dict[keys[-1]]["upper_threshold"]

    # Check if the function is defined on the complete real line
    if (upper_thresholds[keys[-1]] != numpy.inf) | (lower_thresholds[0] != -numpy.inf):
        raise ValueError(f"{leaf_name} needs to be defined on the entire real line.")

    for interval in keys[1:]:
        if "lower_threshold" in parameter_dict[interval]:
            lower_thresholds[interval] = parameter_dict[interval]["lower_threshold"]
        elif "upper_threshold" in parameter_dict[interval - 1]:
            lower_thresholds[interval] = parameter_dict[interval - 1]["upper_threshold"]
        else:
            raise ValueError(
                f"In {interval} of {leaf_name} is no lower upper threshold or an upper"
                f" in the piece before.",
            )

    for interval in keys[:-1]:
        if "upper_threshold" in parameter_dict[interval]:
            upper_thresholds[interval] = parameter_dict[interval]["upper_threshold"]
        elif "lower_threshold" in parameter_dict[interval + 1]:
            upper_thresholds[interval] = parameter_dict[interval + 1]["lower_threshold"]
        else:
            raise ValueError(
                f"In {interval} of {leaf_name} is no upper threshold or a lower"
                f" threshold in the piece after.",
            )

    if not numpy.allclose(lower_thresholds[1:], upper_thresholds[:-1]):
        raise ValueError(
            f"The lower and upper thresholds of {leaf_name} have to coincide",
        )
    thresholds = sorted([lower_thresholds[0], *upper_thresholds])
    return (
        xnp.array(lower_thresholds),
        xnp.array(upper_thresholds),
        xnp.array(thresholds),
    )


def _check_and_get_rates(
    leaf_name: str,
    func_type: FUNC_TYPES,
    parameter_dict: dict[int, dict[str, float]],
    xnp: ModuleType,
) -> Float[Array, " n_segments"]:
    """Check and transfer raw rates data.

    Transfer and check raw rates data, which needs to be specified in a
    piecewise_polynomial layout in the yaml file.

    Parameters
    ----------
    parameter_dict
    leaf_name
    keys
    func_type
    xnp : ModuleType
        The numpy module to use for calculations.

    Returns
    -------

    """
    keys = sorted(parameter_dict.keys())
    rates = numpy.zeros((OPTIONS_REGISTRY[func_type].rates_size, len(keys)))
    for i, rate_type in enumerate(OPTIONS_REGISTRY[func_type].required_keys):
        for interval in keys:
            if rate_type in parameter_dict[interval]:
                rates[i, interval] = parameter_dict[interval][rate_type]
            else:
                raise ValueError(
                    f"In interval {interval} of {leaf_name}, {rate_type} is missing.",
                )
    return xnp.array(rates)


def _check_and_get_intercepts(
    leaf_name: str,
    parameter_dict: dict[int, dict[str, float]],
    lower_thresholds: Float[Array, " n_segments"],
    upper_thresholds: Float[Array, " n_segments"],
    rates: Float[Array, " n_segments"],
    xnp: ModuleType,
) -> Float[Array, " n_segments"]:
    """Check and transfer raw intercept data. If necessary create intercepts.

    Transfer and check raw rates data, which needs to be specified in a
    piecewise_polynomial layout in the yaml file.
    """
    keys = sorted(parameter_dict.keys())
    intercepts = numpy.zeros(len(keys))
    count_intercepts_supplied = 1

    if "intercept_at_lower_threshold" not in parameter_dict[0]:
        raise ValueError(f"The first piece of {leaf_name} needs an intercept.")
    intercepts[0] = parameter_dict[0]["intercept_at_lower_threshold"]
    # Check if all intercepts are supplied.
    for interval in keys[1:]:
        if "intercept_at_lower_threshold" in parameter_dict[interval]:
            count_intercepts_supplied += 1
            intercepts[interval] = parameter_dict[interval][
                "intercept_at_lower_threshold"
            ]
    if (count_intercepts_supplied > 1) & (count_intercepts_supplied != len(keys)):
        raise ValueError(
            "More than one, but not all intercepts are supplied. "
            "The dictionaries should contain either only the lowest intercept "
            "or all intercepts.",
        )
    if count_intercepts_supplied == len(keys):
        pass

    else:
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
    rates: Float[Array, " n_segments"],
    intercept_at_lowest_threshold: float,
    xnp: ModuleType,
) -> Float[Array, " n_segments"]:
    """Create intercepts from raw data.

    Parameters
    ----------
    lower_thresholds:
        The lower thresholds defining the intervals

    upper_thresholds:
        The upper thresholds defining the intervals

    rates:
        The slope in the interval below the corresponding element of *upper_thresholds*.

    intercept_at_lowest_threshold:
        Intercept at the lowest threshold

    xnp: ModuleType
        The module to use for calculations.

    Returns
    -------

    """
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
    rates: Float[Array, " n_segments"],
    intercepts: Float[Array, " n_segments"],
) -> float:
    """Calculate the intercept for the segment `x` lies in.

    Parameters
    ----------
    x
        The value that the function is applied to.
    lower_thresholds
        A one-dimensional array containing lower thresholds of each interval.
    upper_thresholds
        A one-dimensional array containing upper thresholds each interval.
    rates
        A two-dimensional array where columns are interval sections and rows correspond
        to the nth polynomial.
    intercepts
        The intercepts at the lower threshold of each interval.

    Returns
    -------
    out
        The value of `x` under the piecewise function.

    """
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
    for pol in range(1, rates.shape[0] + 1):
        out += rates[pol - 1, index_interval] * (increment_to_calc**pol)

    return out
