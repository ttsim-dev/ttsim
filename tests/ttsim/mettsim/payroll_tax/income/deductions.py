from ttsim import DictTTSIMParam, policy_function


@policy_function(vectorization_strategy="vectorize")
def deductions_y(
    payroll_tax__child_tax_credit__amount_y: float,
    schedule: DictTTSIMParam,
) -> float:
    return (
        schedule.value["lump_sum_deduction_y"] + payroll_tax__child_tax_credit__amount_y
    )
