"""Einkünfte aus nichtselbstständiger Arbeit."""

from __future__ import annotations

from ttsim import policy_function


@policy_function()
def betrag_y(
    betrag_ohne_minijob_y: float,
    sozialversicherung__geringfügig_beschäftigt: bool,
) -> float:
    """Taxable income from dependent employment. In particular, taxable income is set to
    0 for marginally employed persons.

    Parameters
    ----------
    betrag_ohne_minijob_y
        See :func:`betrag_ohne_minijob_y`.
    sozialversicherung__geringfügig_beschäftigt
        See :func:`sozialversicherung__geringfügig_beschäftigt`.

    Returns
    -------

    """
    if sozialversicherung__geringfügig_beschäftigt:
        out = 0.0
    else:
        out = betrag_ohne_minijob_y

    return out


@policy_function()
def betrag_ohne_minijob_y(bruttolohn_y: float, werbungskostenpauschale: float) -> float:
    """Take gross wage and deduct Werbungskostenpauschale."""

    return max(bruttolohn_y - werbungskostenpauschale, 0.0)
