from __future__ import annotations

from ttsim.tt import policy_function


@policy_function()
def amount_y(
    wealth: float,
    tax_rate: float,
    exempt_from_wealth_tax: bool,
) -> float:
    return 0.0 if exempt_from_wealth_tax else wealth * tax_rate


@policy_function()
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
