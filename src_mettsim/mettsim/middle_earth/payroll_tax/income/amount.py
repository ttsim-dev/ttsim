from __future__ import annotations

from ttsim.tt import Unit, policy_function


@policy_function(vectorization_strategy="vectorize", unit=Unit.CURRENCY.PER_YEAR)
def amount_y(
    gross_wage_y: float,
    deductions_y: float,
) -> float:
    return max(gross_wage_y - deductions_y, 0.0)
