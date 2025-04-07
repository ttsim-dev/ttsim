from ttsim import policy_function


@policy_function(
    params_key_for_rounding="housing_benefits__eligibility__income__amount_m"
)
def amount_m(
    gross_wage_m: float,
    payroll_tax__amount_m: float,
) -> float:
    return gross_wage_m - payroll_tax__amount_m
