from __future__ import annotations

from ttsim.tt_dag_elements import policy_function


@policy_function(vectorization_strategy="vectorize")
def amount_m_fam(
    eligibility__requirement_fulfilled_fam: bool,
    income__amount_m_fam: float,
    assistance_rate: float,
) -> float:
    if eligibility__requirement_fulfilled_fam:
        return income__amount_m_fam * assistance_rate
    else:
        return 0
