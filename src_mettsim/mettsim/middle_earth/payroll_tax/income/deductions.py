from __future__ import annotations

from ttsim.tt import TTSIMUnit, policy_function


@policy_function(vectorization_strategy="vectorize", unit=TTSIMUnit.CURRENCY.PER_YEAR)
def deductions_y(
    payroll_tax__child_tax_credit__amount_y: float,
    lump_sum_deduction_y: float,
) -> float:
    return lump_sum_deduction_y + payroll_tax__child_tax_credit__amount_y
