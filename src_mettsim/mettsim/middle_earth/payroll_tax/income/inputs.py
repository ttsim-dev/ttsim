"""Input columns."""

from __future__ import annotations

from ttsim.tt import Unit, policy_input


@policy_input(unit=Unit.CURRENCY_FLOW)
def gross_wage_y() -> float:
    """Annual gross wage."""
