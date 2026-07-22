from __future__ import annotations

from typing import TYPE_CHECKING

from ttsim.tt import (
    TTSIMUnit,
    cast_ttsim_unit,
    param_function,
    policy_function,
)

if TYPE_CHECKING:
    from types import ModuleType

    from ttsim.tt import ConsecutiveIntLookupTableParamValue


@policy_function(
    vectorization_strategy="vectorize",
    unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_FAM,
)
def amount_m_fam(
    eligibility__requirement_fulfilled_fam: bool,
    income__amount_m_fam: float,
    assistance_rate: float,
    max_amount_m_fam: float,
    xnp: ModuleType,
) -> float:
    if eligibility__requirement_fulfilled_fam:
        return xnp.minimum(income__amount_m_fam * assistance_rate, max_amount_m_fam)
    else:
        return 0


@policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH)
def benefit_per_member_m(
    amount_m_fam: float,
    eligibility__number_of_individuals_fam: int,
) -> float:
    """The housing benefit each family member receives."""
    return amount_m_fam / eligibility__number_of_individuals_fam


@param_function(unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_FAM)
def max_amount_m_fam(
    policy_year: int,
    max_amount_m_fam_by_policy_year: ConsecutiveIntLookupTableParamValue,
) -> float:
    return max_amount_m_fam_by_policy_year.look_up(policy_year)


@policy_function(
    vectorization_strategy="vectorize",
    unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_FAM,
)
def income_after_taper_m_fam(
    income__amount_m_fam: float,
    taper_coefficients: dict[str, float],
    xnp: ModuleType,
) -> float:
    """The family's income after the fitted taper is applied.

    The taper rate is a fitted expression: `b` multiplies a monthly currency
    amount, and the product is a pure number because that is what the fit says,
    not because the units cancel. Tagging just that product keeps the rest of the
    expression — including the currency the result is denominated in — checked.
    """
    rate = taper_coefficients["a"] - cast_ttsim_unit(
        taper_coefficients["b"] * income__amount_m_fam, TTSIMUnit.DIMENSIONLESS
    )
    return income__amount_m_fam * xnp.maximum(rate, 0.0)


@policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_FAM)
def benefit_share_m_fam(
    amount_m_fam: float,
    eligibility__number_of_individuals_fam: int,
) -> float:
    """Each member's share of the family's housing benefit, held at family level.

    Dividing a family total by the family's head count cancels `[fam]` in the
    algebra, but the share is reported for the family rather than for whichever
    member is looked at, so it is tagged back to the level it is consumed at.
    """
    return cast_ttsim_unit(
        amount_m_fam / eligibility__number_of_individuals_fam,
        TTSIMUnit.CURRENCY.PER_MONTH.PER_FAM,
    )
