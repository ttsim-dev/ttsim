from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

from ttsim.tt import (
    PiecewisePolynomialParamValue,
    piecewise_polynomial,
    policy_function,
)


@policy_function()
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
        return amount_reduced_y
    else:
        return amount_standard_y


@policy_function()
def amount_standard_y(
    income__amount_y: float,
    tax_schedule_standard: PiecewisePolynomialParamValue,
    xnp: ModuleType,
) -> float:
    """Payroll tax amount for the standard tax schedule."""
    return piecewise_polynomial(
        x=income__amount_y,
        parameters=tax_schedule_standard,
        xnp=xnp,
    )


@policy_function()
def amount_reduced_y(
    income__amount_y: float,
    tax_schedule_reduced: PiecewisePolynomialParamValue,
    xnp: ModuleType,
) -> float:
    """Payroll tax amount for the reduced tax schedule."""
    return piecewise_polynomial(
        x=income__amount_y,
        parameters=tax_schedule_reduced,
        xnp=xnp,
    )
