"""Input columns."""

from ttsim import policy_input


@policy_input()
def mietstufe() -> int:
    """Municipality's rent classification."""
    return 0
