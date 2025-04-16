from ttsim import policy_function


@policy_function()
def amount_y(
    income__amount_y: float,
    payroll_tax_params: dict,
) -> float:
    return income__amount_y * payroll_tax_params["income"]["rate"]
