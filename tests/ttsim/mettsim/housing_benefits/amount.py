from ttsim import policy_function


@policy_function(vectorization_strategy="vectorize")
def amount_m_fam(
    eligibility__requirement_fulfilled_fam: bool,
    income__amount_m_fam: float,
    housing_benefits_params: dict,
) -> float:
    if eligibility__requirement_fulfilled_fam:
        return income__amount_m_fam * housing_benefits_params["assistance_rate"]
    else:
        return 0
