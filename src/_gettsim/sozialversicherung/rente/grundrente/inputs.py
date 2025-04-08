"""Input columns."""

from ttsim import policy_input


@policy_input()
def bewertungszeiten_monate() -> int:
    """Number of months determining amount of Grundrente."""
    return 0


@policy_input()
def grundrentenzeiten_monate() -> int:
    """Number of months determining eligibility for Grundrente."""
    return 0


@policy_input()
def mean_entgeltpunkte() -> float:
    """Mean Entgeltpunkte during Bewertungszeiten."""
    return 0
