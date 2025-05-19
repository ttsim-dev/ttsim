"""Input columns."""

from ttsim import policy_input


@policy_input()
def ohne_renten_m() -> float:
    """Additional income: includes private and public transfers that are not yet
    implemented in GETTSIM (e.g., BAföG, Kriegsopferfürsorge).

    Excludes income from public and private pensions.
    """


@policy_input()
def renteneinkünfte_vorjahr_m() -> float:
    """Income from private and public pensions in the previous year.

    GETTSIM can calculate this input based on the data of the previous year using the
    target `("einkommensteuer", "einkünfte", "sonstige", "renteneinkünfte_m")`.
    """
