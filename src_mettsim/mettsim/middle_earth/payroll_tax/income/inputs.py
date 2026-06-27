"""Input columns."""

from __future__ import annotations

from ttsim.tt import Unit, policy_input


@policy_input(unit=Unit.CURRENCY.PER_YEAR)
def gross_wage_y() -> float:
    """Annual gross wage."""
