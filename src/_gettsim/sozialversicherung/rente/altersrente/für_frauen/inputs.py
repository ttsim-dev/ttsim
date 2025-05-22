"""Input columns."""

from __future__ import annotations

from ttsim import policy_input


@policy_input()
def pflichtsbeitragsjahre_ab_alter_40() -> float:
    """Total years of mandatory contributions after age 40."""
