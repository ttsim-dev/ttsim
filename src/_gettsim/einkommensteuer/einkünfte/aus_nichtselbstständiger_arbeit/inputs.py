"""Input columns."""

from ttsim import policy_input


@policy_input()
def bruttolohn_m() -> float:
    """Monthly wage."""


@policy_input()
def bruttolohn_vorjahr_m() -> float:
    """Monthly wage of previous year."""
