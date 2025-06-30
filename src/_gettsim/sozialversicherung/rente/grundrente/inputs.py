"""Input columns."""

from __future__ import annotations

from ttsim.tt_dag_elements import policy_input


@policy_input(start_date="2021-01-01")
def bewertungszeiten_monate() -> int:
    """Number of months determining amount of Grundrente."""


@policy_input(start_date="2021-01-01")
def grundrentenzeiten_monate() -> int:
    """Number of months determining eligibility for Grundrente."""


@policy_input(start_date="2021-01-01")
def mean_entgeltpunkte() -> float:
    """Mean Entgeltpunkte during Bewertungszeiten."""
