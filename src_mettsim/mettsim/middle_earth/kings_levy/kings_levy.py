from __future__ import annotations

from typing import TYPE_CHECKING

from ttsim.tt import (
    UNSET_UNIT,
    PiecewisePolynomialParamValue,
    TTSIMUnit,
    get_piecewise_parameters,
    param_function,
    piecewise_polynomial,
    policy_function,
)

if TYPE_CHECKING:
    from types import ModuleType

    from ttsim.typing import RawParamValue


@param_function(unit=UNSET_UNIT)
def kings_levy_schedule(
    raw_kings_levy_schedule: RawParamValue,
    xnp: ModuleType,
) -> PiecewisePolynomialParamValue:
    """Build the king's levy schedule from raw rates.

    The marginal rate rises linearly from ``entry_rate`` at zero wealth to
    ``top_rate`` at ``bracket_ceiling``; this converts to a quadratic
    coefficient (a Progressionsfaktor) of units 1/currency. ttsim cannot read
    that convention out of the raw blob, so the parameter declares per-axis
    units and the conversion happens on this typed output — the quadratic term
    scaling by ``1 / f_in``, not by a single uniform factor.
    """
    ceiling = raw_kings_levy_schedule["bracket_ceiling"]
    entry_rate = raw_kings_levy_schedule["entry_rate"]
    top_rate = raw_kings_levy_schedule["top_rate"]
    progressionsfaktor = (top_rate - entry_rate) / (2 * ceiling)
    return get_piecewise_parameters(
        leaf_name="kings_levy_schedule",
        func_type="piecewise_quadratic",
        parameter_list=[
            {"interval": "(-inf, 0)", "slope": 0.0, "quadratic": 0.0, "intercept": 0.0},
            {
                "interval": f"[0, {ceiling})",
                "slope": entry_rate,
                "quadratic": progressionsfaktor,
            },
            {"interval": f"[{ceiling}, inf)", "slope": top_rate, "quadratic": 0.0},
        ],
        xnp=xnp,
    )


@policy_function(unit=TTSIMUnit.CURRENCY.PER_YEAR)
def amount_y(
    wealth: float,
    kings_levy_schedule: PiecewisePolynomialParamValue,
    xnp: ModuleType,
) -> float:
    """The king's levy, a progressive function of a person's wealth."""
    return piecewise_polynomial(x=wealth, parameters=kings_levy_schedule, xnp=xnp)
