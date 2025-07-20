"""Input columns."""

from __future__ import annotations

from ttsim.tt_dag_elements import policy_input


@policy_input()
def alle_weiteren_m() -> float:
    """Additional income: includes private and public transfers that are not yet
    implemented in GETTSIM (e.g., BAföG, Kriegsopferfürsorge).

    Excludes income from public and private pensions.
    """


@policy_input(start_date="2021-01-01")
def gesamtbetrag_vorjahr_m() -> float:
    """Income from private and public pensions in the previous year.

    GETTSIM can calculate this input based on the data of the previous year using the
    target `("einkommensteuer", "einkünfte", "sonstige", "betrag_m")`.
    """


@policy_input()
def sonstige_private_vorsorge_m() -> float:
    """Monthly payout from private pensions without tax-favored contributions.

    This refers to pension payments from plans where the original
    contributions were not tax-deductible (or tax-exempt).
    """


@policy_input()
def geförderte_private_vorsorge_m() -> float:
    """Monthly payout from private pensions with tax-favored contributions.

    This refers to pension payments from plans where the original
    contributions were tax-deductible (or tax-exempt). Primarily Riesterrente.
    """


@policy_input()
def betriebliche_altersvorsorge_m() -> float:
    """Amount of monthly occupational pension."""
