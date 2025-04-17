"""Input columns."""

from ttsim import policy_input


@policy_input()
def betrag_m() -> float:
    """Monthly business income."""
