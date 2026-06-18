from __future__ import annotations

from typing import TYPE_CHECKING

from ttsim.tt import Unit, param_function, policy_function

if TYPE_CHECKING:
    from types import ModuleType

    from ttsim.tt import ConsecutiveIntLookupTableParamValue


@policy_function(
    vectorization_strategy="vectorize", unit=Unit.CURRENCY_FLOW, verify_units=False
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


@param_function(unit=Unit.CURRENCY_FLOW, verify_units=False)
def max_amount_m_fam(
    policy_year: int,
    max_amount_m_fam_by_policy_year: ConsecutiveIntLookupTableParamValue,
) -> float:
    return max_amount_m_fam_by_policy_year.look_up(policy_year)
