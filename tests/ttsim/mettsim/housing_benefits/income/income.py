from ttsim import RoundingSpec, policy_function


@policy_function(
    rounding_spec=RoundingSpec(
        base=1,
        direction="down",
        reference="§ 4 Gondorian Housing Benefit Law",
    )
)
def amount_m(
    payroll_tax__income__gross_wage_m: float,
    payroll_tax__amount_m: float,
) -> float:
    return payroll_tax__income__gross_wage_m - payroll_tax__amount_m
