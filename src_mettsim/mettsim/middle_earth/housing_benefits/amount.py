from __future__ import annotations

from typing import TYPE_CHECKING

from ttsim.tt import TTSIMUnit, param_function, policy_function

if TYPE_CHECKING:
    from types import ModuleType

    from ttsim.tt import ConsecutiveIntLookupTableParamValue


@policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_FAM)
def income_after_taper_m_fam(
    income__amount_m_fam: float,
    assistance_rate: float,
) -> float:
    """The family's income once the assistance rate has tapered it."""
    return income__amount_m_fam * assistance_rate


@policy_function(
    vectorization_strategy="vectorize",
    unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_FAM,
)
def benefit_share_m_fam(
    income_after_taper_m_fam: float,
    max_amount_m_fam: float,
    xnp: ModuleType,
) -> float:
    """The tapered income, capped at the maximum the statute grants a family."""
    return xnp.minimum(income_after_taper_m_fam, max_amount_m_fam)


@policy_function(
    vectorization_strategy="vectorize",
    unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_FAM,
)
def amount_m_fam(
    eligibility__requirement_fulfilled_fam: bool,
    benefit_share_m_fam: float,
) -> float:
    """The housing benefit a family receives, zero unless it is eligible."""
    if eligibility__requirement_fulfilled_fam:
        return benefit_share_m_fam
    else:
        return 0


@policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH)
def benefit_per_member_m(
    amount_m_fam: float,
    eligibility__number_of_individuals_fam: int,
) -> float:
    """The family's benefit split evenly across its members."""
    return amount_m_fam / eligibility__number_of_individuals_fam


@param_function(unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_FAM)
def max_amount_m_fam(
    policy_year: int,
    max_amount_m_fam_by_policy_year: ConsecutiveIntLookupTableParamValue,
) -> float:
    return max_amount_m_fam_by_policy_year.look_up(policy_year)
