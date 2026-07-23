from __future__ import annotations

from ttsim.tt import AggType, TTSIMUnit, agg_by_group_function, policy_function


@policy_function(unit=TTSIMUnit.CURRENCY.PER_YEAR)
def amount_y(
    wealth: float,
    tax_rate_y: float,
    exempt_from_wealth_tax: bool,
) -> float:
    return 0.0 if exempt_from_wealth_tax else wealth * tax_rate_y


@agg_by_group_function(
    agg_type=AggType.MEAN,
    unit=TTSIMUnit.CURRENCY.PER_KIN,
    # A MEAN derives as the person's, but the average kin wealth reads as a
    # property of the kinstead; the declaration states that and skips the
    # declared-vs-derived check (GEP 10).
    verify_units=False,
)
def average_wealth_kin(kin_id: int, wealth: float) -> float:
    """The average wealth of the kinstead."""


@policy_function(unit=TTSIMUnit.DIMENSIONLESS)
def exempt_from_wealth_tax(
    wealth_kin: float,
    wealth_fam: float,
    wealth: float,
    wealth_above_which_kin_is_exempt: float,
    wealth_above_which_family_is_exempt: float,
    wealth_above_which_individual_is_exempt: float,
) -> bool:
    return (
        wealth_kin >= wealth_above_which_kin_is_exempt
        or wealth_fam >= wealth_above_which_family_is_exempt
        or wealth >= wealth_above_which_individual_is_exempt
    )
