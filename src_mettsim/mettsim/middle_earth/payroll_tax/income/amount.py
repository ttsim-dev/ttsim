from __future__ import annotations

from ttsim.tt import policy_function


@policy_function(vectorization_strategy="vectorize")
def amount_y(
    gross_wage_y: float,
    deductions_y: float,
) -> float:
    return max(gross_wage_y - deductions_y, 0.0)
