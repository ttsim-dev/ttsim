from ttsim import policy_function


@policy_function()
def deductions_y(
    lump_sum_deduction_y: float,
    payroll_tax__child_tax_credit__amount_y: float,
) -> float:
    return lump_sum_deduction_y + payroll_tax__child_tax_credit__amount_y
