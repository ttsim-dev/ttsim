"""Input columns."""

from ttsim import policy_input


@policy_input()
def betrag_m() -> float:
    """Additional income: includes private and public transfers that are not yet implemented in GETTSIM (e.g., BAföG, Kriegsopferfürsorge)"""
    return 0
