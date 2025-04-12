"""Tax allowances for the elderly."""

from ttsim import policy_function
from ttsim.config import numpy_or_jax as np


@policy_function(end_date="2004-12-31", leaf_name="altersfreibetrag_y")
def altersfreibetrag_y_bis_2004(
    alter: int,
    einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_y: float,
    einkommensteuer__einkünfte__aus_kapitalvermögen__kapitalerträge_y: float,
    einkommensteuer__einkünfte__aus_selbstständiger_arbeit__betrag_y: float,
    einkommensteuer__einkünfte__aus_vermietung_und_verpachtung__betrag_y: float,
    eink_st_abzuege_params: dict,
) -> float:
    """Calculate tax deduction allowance for elderly until 2004.

    Parameters
    ----------
    einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_y
        See basic input variable :ref:`einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_y <einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_y>`.
    alter
        See basic input variable :ref:`alter <alter>`.
    einkommensteuer__einkünfte__aus_kapitalvermögen__kapitalerträge_y
        See basic input variable :ref:`einkommensteuer__einkünfte__aus_kapitalvermögen__kapitalerträge_y <einkommensteuer__einkünfte__aus_kapitalvermögen__kapitalerträge_y>`.
    einkommensteuer__einkünfte__aus_selbstständiger_arbeit__betrag_y
        See :func:`einkommensteuer__einkünfte__aus_selbstständiger_arbeit__betrag_y`.
    einkommensteuer__einkünfte__aus_vermietung_und_verpachtung__betrag_y
        See basic input variable :ref:`einkommensteuer__einkünfte__aus_vermietung_und_verpachtung__betrag_y <einkommensteuer__einkünfte__aus_vermietung_und_verpachtung__betrag_y>`.
    eink_st_abzuege_params
        See params documentation :ref:`eink_st_abzuege_params <eink_st_abzuege_params>`.

    Returns
    -------

    """
    altersgrenze = eink_st_abzuege_params["altersentlastungsbetrag_altersgrenze"]
    weiteres_einkommen = max(
        einkommensteuer__einkünfte__aus_kapitalvermögen__kapitalerträge_y
        + einkommensteuer__einkünfte__aus_selbstständiger_arbeit__betrag_y
        + einkommensteuer__einkünfte__aus_vermietung_und_verpachtung__betrag_y,
        0.0,
    )
    if alter > altersgrenze:
        out = min(
            eink_st_abzuege_params["altersentlastung_quote"]
            * (
                einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_y
                + weiteres_einkommen
            ),
            eink_st_abzuege_params["altersentlastungsbetrag_max"],
        )
    else:
        out = 0.0

    return out


@policy_function(start_date="2005-01-01", leaf_name="altersfreibetrag_y")
def altersfreibetrag_y_ab_2005(
    alter: int,
    geburtsjahr: int,
    sozialversicherung__geringfügig_beschäftigt: bool,
    einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_y: float,
    einkommensteuer__einkünfte__aus_kapitalvermögen__kapitalerträge_y: float,
    einkommensteuer__einkünfte__aus_selbstständiger_arbeit__betrag_y: float,
    einkommensteuer__einkünfte__aus_vermietung_und_verpachtung__betrag_y: float,
    eink_st_abzuege_params: dict,
) -> float:
    """Calculate tax deduction allowance for elderly since 2005.

    Parameters
    ----------
    einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_y
        See basic input variable :ref:`einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_y <einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_y>`.
    alter
        See basic input variable :ref:`alter <alter>`.
    geburtsjahr
        See basic input variable :ref:`geburtsjahr <geburtsjahr>`.
    einkommensteuer__einkünfte__aus_kapitalvermögen__kapitalerträge_y
        See basic input variable :ref:`einkommensteuer__einkünfte__aus_kapitalvermögen__kapitalerträge_y <einkommensteuer__einkünfte__aus_kapitalvermögen__kapitalerträge_y>`.
    einkommensteuer__einkünfte__aus_selbstständiger_arbeit__betrag_y
        See :func:`einkommensteuer__einkünfte__aus_selbstständiger_arbeit__betrag_y`.
    einkommensteuer__einkünfte__aus_vermietung_und_verpachtung__betrag_y
        See basic input variable :ref:`einkommensteuer__einkünfte__aus_vermietung_und_verpachtung__betrag_y <einkommensteuer__einkünfte__aus_vermietung_und_verpachtung__betrag_y>`.
    eink_st_abzuege_params
        See params documentation :ref:`eink_st_abzuege_params <eink_st_abzuege_params>`.
    sozialversicherung__geringfügig_beschäftigt
        See :func:`sozialversicherung__geringfügig_beschäftigt`.

    Returns
    -------

    """
    # Maximum tax credit by birth year.
    bins = sorted(eink_st_abzuege_params["altersentlastungsbetrag_max"])
    if geburtsjahr <= 1939:
        selected_bin = 1940
    else:
        # Select corresponding bin.
        selected_bin = bins[
            np.searchsorted(np.asarray([*bins, np.inf]), geburtsjahr, side="right") - 1
        ]

    # Select appropriate tax credit threshold and quota.
    out_max = eink_st_abzuege_params["altersentlastungsbetrag_max"][selected_bin]

    einkommen_lohn = (
        0
        if sozialversicherung__geringfügig_beschäftigt
        else einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_y
    )
    weiteres_einkommen = max(
        einkommensteuer__einkünfte__aus_kapitalvermögen__kapitalerträge_y
        + einkommensteuer__einkünfte__aus_selbstständiger_arbeit__betrag_y
        + einkommensteuer__einkünfte__aus_vermietung_und_verpachtung__betrag_y,
        0.0,
    )
    out_quote = eink_st_abzuege_params["altersentlastung_quote"][selected_bin] * (
        einkommen_lohn + weiteres_einkommen
    )

    if alter > eink_st_abzuege_params["altersentlastungsbetrag_altersgrenze"]:
        out = min(out_quote, out_max)
    else:
        out = 0.0

    return out
