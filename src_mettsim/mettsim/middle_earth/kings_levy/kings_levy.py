from __future__ import annotations

from typing import TYPE_CHECKING

from ttsim.tt import (
    InputOutputUnit,
    PiecewisePolynomialParamValue,
    TTSIMUnit,
    get_consecutive_int_lookup_table_param_value,
    get_piecewise_parameters,
    param_function,
    piecewise_polynomial,
    policy_function,
)

if TYPE_CHECKING:
    from types import ModuleType

    from ttsim.tt import ConsecutiveIntLookupTableParamValue
    from ttsim.typing import RawParamValue


@param_function(
    unit=InputOutputUnit(
        input_unit=TTSIMUnit.CURRENCY,
        output_unit=TTSIMUnit.CURRENCY.PER_YEAR,
    ),
    verify_units=False,
)
def kings_levy_schedule(
    raw_kings_levy_schedule: RawParamValue,
    xnp: ModuleType,
) -> PiecewisePolynomialParamValue:
    """Build the king's levy schedule from raw rates.

    The marginal rate rises linearly from ``entry_rate`` at zero wealth to
    ``top_rate`` at ``bracket_ceiling``; this converts to a quadratic
    coefficient (a Progressionsfaktor) of units 1/currency. ttsim cannot read
    that convention out of the raw blob, so the builder declares its two axes
    with ``InputOutputUnit`` — a currency wealth in, a yearly currency flow out —
    and opts out of body verification, because the quadratic term scales by
    ``1 / f_in`` rather than by a single uniform factor.
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


@param_function(
    unit=InputOutputUnit(
        input_unit=TTSIMUnit.DIMENSIONLESS.PER_KIN,
        output_unit=TTSIMUnit.CURRENCY.PER_YEAR.PER_KIN,
    ),
    verify_units=False,
)
def child_rebate_schedule(
    raw_kings_levy_child_rebate: RawParamValue,
    xnp: ModuleType,
) -> ConsecutiveIntLookupTableParamValue:
    """Build the kinstead's levy rebate keyed by its number of dependants.

    A lookup table is keyed by consecutive integers — here the kinstead's count
    of dependent children — so its input axis is a (kin-level) dimensionless
    count, never a currency; the output is a yearly currency rebate for the
    kinstead. The builder declares both axes with ``InputOutputUnit`` and opts
    out of body verification, since ttsim cannot read the table's units off the
    raw integer-keyed blob.
    """
    # The raw blob is keyed by the (integer) number of dependants; RawParamValue
    # types its keys as `str | int`, so narrow to the int-keyed lookup dict.
    return get_consecutive_int_lookup_table_param_value(
        raw=raw_kings_levy_child_rebate,  # ty: ignore[invalid-argument-type]
        xnp=xnp,
    )


@policy_function(unit=TTSIMUnit.CURRENCY.PER_YEAR.PER_KIN)
def child_rebate_y_kin(
    number_of_dependants_kin: int,
    child_rebate_schedule: ConsecutiveIntLookupTableParamValue,
) -> float:
    """The kinstead's king's-levy rebate for its dependent children."""
    return child_rebate_schedule.look_up(number_of_dependants_kin)
