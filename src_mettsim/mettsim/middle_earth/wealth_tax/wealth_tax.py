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
    agg_type=AggType.MEAN, unit=TTSIMUnit.CURRENCY.PER_KIN, verify_units=False
)
def average_wealth_kin(kin_id: int, wealth: float) -> float:
    """The average wealth of the kinstead — a property of the kin group.

    The unit algebra derives a ``MEAN`` as the person's (``MEAN = SUM / COUNT``
    cancels the group), but the average kin wealth reads more naturally as a kin
    property, so it is declared ``PER_KIN`` and opts out of the declared-vs-derived
    aggregation check (GEP 10).
    """


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
