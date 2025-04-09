"""Input columns."""

from ttsim import policy_input


@policy_input()
def ohne_renten_m() -> float:
    """Additional income: includes private and public transfers that are not yet
    implemented in GETTSIM (e.g., BAföG, Kriegsopferfürsorge).

    Excludes income from public and private pensions.
    """
