from ttsim import policy_function


@policy_function()
def amount_y(
    income__amount_y: float,
    rate: float,
) -> float:
    return income__amount_y * rate
