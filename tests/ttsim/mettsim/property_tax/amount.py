"""Property tax.

Three tax brackets:
    - No acre
    - small acre
    - large acre
"""

from ttsim import policy_function, piecewise_polynomial, PiecewisePolynomialParameters


@policy_function(vectorization_strategy="vectorize")
def amount_y(
    acre_size_in_hectares: float,
    tax_schedule: PiecewisePolynomialParameters,
) -> float:
    """Property tax amount for the standard tax schedule."""
    return piecewise_polynomial(
        x=acre_size_in_hectares,
        parameters=tax_schedule,
    )