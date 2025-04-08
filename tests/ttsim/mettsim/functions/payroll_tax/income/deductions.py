from ttsim import policy_function


@policy_function()
def deductions_y(
    payroll_tax__child_tax_credit__amount_y: float,
    payroll_tax_params: dict,
) -> float:
    return (
        payroll_tax_params["lump_sum_deduction_y"]
        + payroll_tax__child_tax_credit__amount_y
    )
