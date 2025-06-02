"""Input columns."""

from __future__ import annotations

from ttsim import policy_input


@policy_input()
def kapitalerträge_m() -> float:
    """Monthly capital income."""
