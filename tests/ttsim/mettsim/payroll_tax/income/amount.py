from ttsim import policy_function


@policy_function()
def amount_y(
    gross_wage_y: float,
    deductions_y: float,
) -> float:
    return max(gross_wage_y - deductions_y, 0.0)
