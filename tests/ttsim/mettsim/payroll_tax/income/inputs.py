"""Input columns."""

from __future__ import annotations

from ttsim import policy_input


@policy_input()
def gross_wage_y() -> float:
    """Annual gross wage."""
