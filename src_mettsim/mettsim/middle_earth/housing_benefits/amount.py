from __future__ import annotations

from typing import TYPE_CHECKING

from ttsim.tt import TTSIMUnit, param_function, policy_function

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
    """The housing benefit each family member receives.

    A genuine per-capita split, not an average: the family's benefit is a fam-level
    amount computed from the family's situation, so dividing it by the head count
    cancels the level — ``(CURRENCY/month/[fam]) / (1/[fam]) = CURRENCY/month`` —
    and the per-person result is bare (GEP 10).
    """
    return amount_m_fam / eligibility__number_of_individuals_fam


@param_function(unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_FAM)
def max_amount_m_fam(
    policy_year: int,
    max_amount_m_fam_by_policy_year: ConsecutiveIntLookupTableParamValue,
) -> float:
    return max_amount_m_fam_by_policy_year.look_up(policy_year)
