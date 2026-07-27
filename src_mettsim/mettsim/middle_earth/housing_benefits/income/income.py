from __future__ import annotations

from ttsim.tt import RoundingSpec, policy_function


@policy_function(
    rounding_spec=RoundingSpec(
        base=1,
        direction="down",
        reference="§ 4 Gondorian Housing Benefit Law",
    ),
)
def amount_m(
    payroll_tax__income__gross_wage_m: float,
    payroll_tax__amount_m: float,
    housing_benefits__eligibility__child: bool,
) -> float:
    if housing_benefits__eligibility__child:
        return 0.0
    else:
        return payroll_tax__income__gross_wage_m - payroll_tax__amount_m
