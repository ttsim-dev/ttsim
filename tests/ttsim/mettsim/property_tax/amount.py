"""Property tax.

Three tax brackets:
    - No acre
    - small acre
    - large acre
"""

from __future__ import annotations

from ttsim.tt_dag_elements import (
    PiecewisePolynomialParamValue,
    piecewise_polynomial,
    policy_function,
    policy_input,
)


@policy_input()
def acre_size_in_hectares() -> float:
    """The size of the acre in hectares."""


@policy_function(vectorization_strategy="vectorize")
def amount_y(
    acre_size_in_hectares: float,
    tax_schedule: PiecewisePolynomialParamValue,
) -> float:
    """Property tax amount for the standard tax schedule."""
    return piecewise_polynomial(
        x=acre_size_in_hectares,
        parameters=tax_schedule,
    )
