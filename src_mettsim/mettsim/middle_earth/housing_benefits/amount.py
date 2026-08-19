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
def income_after_deduction_m_fam(
    income__amount_m_fam: float,
    deduction_coefficients: dict[str, float],
    xnp: ModuleType,
) -> float:
    """The family's income after the statutory deduction.

    The share of income that remains, `a - b * income`, shrinks as income rises
    and never falls below zero.
    """
    # `b` is an empirically fitted coefficient: its product with a monthly
    # currency amount is a pure number by construction of the fit, not by unit
    # algebra, so it is the product alone that states its unit.
    remaining_share = deduction_coefficients["a"] - cast_ttsim_unit(
        deduction_coefficients["b"] * income__amount_m_fam,
        unit=TTSIMUnit.DIMENSIONLESS,
    )
    return income__amount_m_fam * xnp.maximum(remaining_share, 0.0)


@policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_FAM)
def benefit_share_m_fam(
    amount_m_fam: float,
    eligibility__number_of_individuals_fam: int,
) -> float:
    """Each member's share of the family's housing benefit, held at family level."""
    # The share is reported for the family, not for whichever member is looked
    # at, so it is tagged back to the level it is consumed at.
    return cast_ttsim_unit(
        amount_m_fam / eligibility__number_of_individuals_fam,
        unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_FAM,
    )
