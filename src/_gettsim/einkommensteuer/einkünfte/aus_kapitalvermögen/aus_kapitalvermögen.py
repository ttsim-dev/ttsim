"""Einkünfte aus Kapitalvermögen."""

from __future__ import annotations

from ttsim import policy_function


@policy_function(end_date="2008-12-31", leaf_name="betrag_y")
def betrag_y_mit_sparerfreibetrag_und_werbungskostenpauschbetrag(
    kapitalerträge_y: float,
    eink_st_abzuege_params: dict,
) -> float:
    """Calculate taxable capital income on Steuernummer level.

    Parameters
    ----------
    kapitalerträge_y
        See :func:`kapitalerträge_y`.
    eink_st_abzuege_params
        See params documentation :ref:`eink_st_abzuege_params <eink_st_abzuege_params>`.

    Returns
    -------

    """
    out = kapitalerträge_y - (
        eink_st_abzuege_params["sparerfreibetrag"]
        + eink_st_abzuege_params["sparer_werbungskostenpauschbetrag"]
    )

    return max(out, 0.0)


@policy_function(start_date="2009-01-01", leaf_name="betrag_y")
def betrag_y_mit_sparerpauschbetrag(
    kapitalerträge_y: float,
    eink_st_abzuege_params: dict,
) -> float:
    """Calculate taxable capital income on Steuernummer level.

    Parameters
    ----------
    kapitalerträge_y
        See :func:`kapitalerträge_y`.
    eink_st_abzuege_params
        See params documentation :ref:`eink_st_abzuege_params <eink_st_abzuege_params>`.

    Returns
    -------

    """
    out = kapitalerträge_y - eink_st_abzuege_params["sparerpauschbetrag"]

    return max(out, 0.0)
