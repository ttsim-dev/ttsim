"""Input columns."""

from __future__ import annotations

from ttsim.tt_dag_elements import policy_input


@policy_input()
def gross_wage_y() -> float:
    """Annual gross wage."""
