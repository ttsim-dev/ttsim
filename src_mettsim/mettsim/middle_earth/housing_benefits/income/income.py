"""Income relevant for housing benefits.

The rounding step is statutory: § 4 rounds down to whole silver pennies until
the currency reform of 2020 and to quarter-castars — the same magnitude,
restated — from 2020 on, so the function is split at the changeover.
"""

from __future__ import annotations

from ttsim.tt import RoundingSpec, TTSIMUnit, policy_function


@policy_function(
    end_date="2019-12-31",
    leaf_name="amount_m",
    rounding_spec=RoundingSpec(
        base=1,
        direction="down",
        reference="§ 4 Gondorian Housing Benefit Law",
        unit=TTSIMUnit.SILVER_PENNY.PER_MONTH,
    ),
    unit=TTSIMUnit.CURRENCY.PER_MONTH,
)
def amount_m_rounded_to_silver_pennies(
    payroll_tax__income__gross_wage_m: float,
    payroll_tax__amount_m: float,
    housing_benefits__eligibility__child: bool,
) -> float:
    if housing_benefits__eligibility__child:
        return 0.0
    else:
        return payroll_tax__income__gross_wage_m - payroll_tax__amount_m


@policy_function(
    start_date="2020-01-01",
    leaf_name="amount_m",
    rounding_spec=RoundingSpec(
        base=0.25,
        direction="down",
        reference="§ 4 Gondorian Housing Benefit Law, castar restatement (2020)",
        unit=TTSIMUnit.CASTAR.PER_MONTH,
    ),
    unit=TTSIMUnit.CURRENCY.PER_MONTH,
)
def amount_m_rounded_to_quarter_castars(
    payroll_tax__income__gross_wage_m: float,
    payroll_tax__amount_m: float,
    housing_benefits__eligibility__child: bool,
) -> float:
    if housing_benefits__eligibility__child:
        return 0.0
    else:
        return payroll_tax__income__gross_wage_m - payroll_tax__amount_m
