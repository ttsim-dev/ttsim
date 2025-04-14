"""Input columns."""

from ttsim import policy_input


@policy_input()
def privat_versichert() -> bool:
    """Has (only) a private health insurance contract."""
