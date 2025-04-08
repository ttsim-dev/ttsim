"""Input columns."""

from ttsim import policy_input


@policy_input()
def ist_selbstständig() -> bool:
    """Self-employed (main profession)."""
    return False
