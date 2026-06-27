from __future__ import annotations

from ttsim.tt import Unit, policy_function


@policy_function(unit=Unit.CURRENCY.PER_YEAR)
def amount_y(
    wealth: float,
    tax_rate_y: float,
    exempt_from_wealth_tax: bool,
) -> float:
    return 0.0 if exempt_from_wealth_tax else wealth * tax_rate_y


@policy_function(unit=Unit.DIMENSIONLESS)
def exempt_from_wealth_tax(
    wealth_kin: float,
    wealth_fam: float,
    wealth: float,
    wealth_above_which_kin_is_exempt: float,
    wealth_above_which_family_is_exempt: float,
    wealth_above_which_individual_is_exempt: float,
) -> bool:
    # Each comparison is a *leveled* boolean — kin, fam, and person — because its
    # operands carry those levels (the thresholds spell them, e.g. `..._PER_FAM`,
    # GEP 10).
    # Combining them with bitwise `|` applies the boolean combine rule: mismatched
    # levels downcast to the per-person level, matching this function's unsuffixed
    # (person-level) name. Python `or` would instead return a single, mis-levelled
    # comparison and be rejected by the result check.
    return (
        (wealth_kin >= wealth_above_which_kin_is_exempt)
        | (wealth_fam >= wealth_above_which_family_is_exempt)
        | (wealth >= wealth_above_which_individual_is_exempt)
    )
