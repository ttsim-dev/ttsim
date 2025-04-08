"""Input columns."""

from ttsim import policy_input


@policy_input()
def hat_kinder() -> bool:
    """Has kids (incl. not in hh)."""
    return False
