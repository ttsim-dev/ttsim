"""Input columns."""

from __future__ import annotations

from ttsim.tt_dag_elements import policy_input


@policy_input()
def mietstufe_hh() -> int:
    """Municipality's rent classification."""
