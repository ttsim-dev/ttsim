"""Input columns."""

from __future__ import annotations

from ttsim.tt import TTSIMUnit, policy_input


@policy_input(unit=TTSIMUnit.CURRENCY.PER_YEAR)
def gross_wage_y() -> float:
    """Annual gross wage."""
