"""Unterhalt (child support)."""

from __future__ import annotations

from ttsim import policy_function


@policy_function()
def kind_festgelegter_zahlbetrag_m(
    anspruch_m: float,
    kindergeld__betrag_m: float,
    familie__kind: bool,
    abzugsrate_kindergeld: dict[str, float],
) -> float:
    """Monthly actual child alimony payments to be received by the child after
    deductions."""
    if familie__kind:
        abzugsrate = abzugsrate_kindergeld["minderjährig"]
    else:
        abzugsrate = abzugsrate_kindergeld["volljährig"]

    return anspruch_m - abzugsrate * kindergeld__betrag_m
