"""Property tax.

Three tax brackets:
    - No acre
    - small acre
    - large acre
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

from ttsim.tt import (
    PiecewisePolynomialParamValue,
    Unit,
    piecewise_polynomial,
    policy_function,
    policy_input,
)


@policy_input(unit=Unit.HECTARES)
def acre_size() -> float:
    """The size of the acre in hectares."""


@policy_function(
    vectorization_strategy="vectorize", unit=Unit.CURRENCY_FLOW, verify_units=False
)
def amount_y(
    acre_size_after_cap: float,
    tax_schedule: PiecewisePolynomialParamValue,
    xnp: ModuleType,
) -> float:
    """Property tax amount for the standard tax schedule."""
    return piecewise_polynomial(
        x=acre_size_after_cap,
        parameters=tax_schedule,
        xnp=xnp,
    )


@policy_function(unit=Unit.HECTARES)
def acre_size_after_cap(
    acre_size: float,
    cap: float,
    year_from_which_cap_is_applied: int,
    evaluation_year: int,
) -> float:
    """The size of the acre in hectares after the cap is applied."""
    if evaluation_year < year_from_which_cap_is_applied:
        return acre_size
    else:
        return min(acre_size, cap)
