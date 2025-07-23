"""Input columns."""

from __future__ import annotations

from ttsim.tt_dag_elements import policy_input


@policy_input()
def betrag_m() -> float:
    """Monthly business income."""
