from __future__ import annotations

from ttsim.tt import RoundingSpec, TTSIMUnit, policy_function


@policy_function(
    leaf_name="amount_m",
    end_date="2019-12-31",
    rounding_spec=RoundingSpec(
        base=1,
        direction="down",
        reference="§ 4 Gondorian Housing Benefit Law",
        unit=TTSIMUnit.SILVER_PENNY.PER_MONTH,
    ),
    unit=TTSIMUnit.CURRENCY.PER_MONTH,
)
def amount_m_before_currency_reform(
    payroll_tax__income__gross_wage_m: float,
    payroll_tax__amount_m: float,
    housing_benefits__eligibility__child: bool,
) -> float:
    if housing_benefits__eligibility__child:
        return 0.0
    else:
        return payroll_tax__income__gross_wage_m - payroll_tax__amount_m


@policy_function(
    leaf_name="amount_m",
    start_date="2020-01-01",
    rounding_spec=RoundingSpec(
        base=0.25,
        direction="down",
        reference="§ 4 Gondorian Housing Benefit Law",
        unit=TTSIMUnit.CASTAR.PER_MONTH,
    ),
    unit=TTSIMUnit.CURRENCY.PER_MONTH,
)
def amount_m_after_currency_reform(
    payroll_tax__income__gross_wage_m: float,
    payroll_tax__amount_m: float,
    housing_benefits__eligibility__child: bool,
) -> float:
    if housing_benefits__eligibility__child:
        return 0.0
    else:
        return payroll_tax__income__gross_wage_m - payroll_tax__amount_m
