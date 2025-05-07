from ttsim import (
    AggType,
    PiecewisePolynomialParameters,
    agg_by_group_function,
    piecewise_polynomial,
    policy_function,
)


@policy_function(vectorization_strategy="vectorize")
def amount_y(
    amount_standard_y: float,
    amount_reduced_y: float,
    parent_is_noble_fam: bool,
    wealth_fam: float,
    wealth_threshold_for_reduced_tax_rate: float,
) -> float:
    if parent_is_noble_fam:
        return 0.0
    elif wealth_fam >= wealth_threshold_for_reduced_tax_rate:
        return amount_standard_y
    else:
        return amount_reduced_y


@policy_function(vectorization_strategy="vectorize")
def amount_standard_y(
    income__amount_y: float,
    tax_schedule_standard: PiecewisePolynomialParameters,
) -> float:
    """Payroll tax amount for the standard tax schedule."""
    return piecewise_polynomial(
        x=income__amount_y,
        parameters=tax_schedule_standard,
    )


@policy_function(vectorization_strategy="vectorize")
def amount_reduced_y(
    income__amount_y: float,
    tax_schedule_reduced: PiecewisePolynomialParameters,
) -> float:
    """Payroll tax amount for the reduced tax schedule."""
    return piecewise_polynomial(
        x=income__amount_y,
        parameters=tax_schedule_reduced,
    )


@agg_by_group_function(agg_type=AggType.ANY)
def parent_is_noble_fam(
    parent_is_noble: bool,
    fam_id: int,
) -> bool:
    pass
