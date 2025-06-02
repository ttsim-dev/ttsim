"""Tax allowances for the disabled."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ttsim import piecewise_polynomial, policy_function

if TYPE_CHECKING:
    from ttsim import PiecewisePolynomialParam


@policy_function()
def pauschbetrag_behinderung_y(
    behinderungsgrad: int, parameter_behindertenpauschbetrag: PiecewisePolynomialParam
) -> float:
    """Assign tax deduction allowance for handicaped to different handicap degrees."""
    return piecewise_polynomial(
        x=behinderungsgrad,
        parameters=parameter_behindertenpauschbetrag,
    )
