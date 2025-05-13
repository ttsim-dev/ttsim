"""Taxes on capital income (Abgeltungssteuer)."""

from ttsim import policy_function


@policy_function(start_date="2009-01-01")
def betrag_y_sn(zu_versteuerndes_kapitaleinkommen_y_sn: float, satz: float) -> float:
    """Abgeltungssteuer on Steuernummer level."""
    return satz * zu_versteuerndes_kapitaleinkommen_y_sn


@policy_function(start_date="2009-01-01")
def zu_versteuerndes_kapitaleinkommen_y_sn(
    einkommensteuer__einkünfte__aus_kapitalvermögen__kapitalerträge_y_sn: float,
    einkommensteuer__anzahl_personen_sn: float,
    eink_st_abzuege_params: dict,
) -> float:
    """Taxable capital income for Abgeltungssteuer.

    TODO(@MImmesberger): Find out whether Sparerpauschbetrag is
    transferable to partner with same sn_id.
    https://github.com/iza-institute-of-labor-economics/gettsim/issues/843

    Parameters
    ----------
    einkommensteuer__einkünfte__aus_kapitalvermögen__kapitalerträge_y_sn
        See :func:`einkommensteuer__einkünfte__aus_kapitalvermögen__kapitalerträge_y_sn`.
    einkommensteuer__anzahl_personen_sn
        See :func:`einkommensteuer__anzahl_personen_sn`.
    eink_st_abzuege_params
        See params documentation :ref:`eink_st_abzuege_params <eink_st_abzuege_params>`.

    Returns
    -------

    """
    out = (
        einkommensteuer__einkünfte__aus_kapitalvermögen__kapitalerträge_y_sn
        - einkommensteuer__anzahl_personen_sn
        * eink_st_abzuege_params["sparerpauschbetrag"]
    )
    return max(out, 0.0)
