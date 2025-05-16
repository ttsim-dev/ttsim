"""Taxable income."""

from __future__ import annotations

from ttsim import RoundingSpec, policy_function


@policy_function(
    rounding_spec=RoundingSpec(
        base=1, direction="down", reference="§ 32a Abs. 1 S.1 EStG"
    ),
    start_date="2004-01-01",
    leaf_name="zu_versteuerndes_einkommen_y_sn",
)
def zu_versteuerndes_einkommen_y_sn_mit_abrundungsregel(
    zu_versteuerndes_einkommen_mit_kinderfreibetrag_y_sn: float,
    einkommensteuer__gesamteinkommen_y: float,
    kinderfreibetrag_günstiger_sn: bool,
) -> float:
    """Calculate taxable income on Steuernummer level.

    Parameters
    ----------
    zu_versteuerndes_einkommen_mit_kinderfreibetrag_y_sn
        See :func:`zu_versteuerndes_einkommen_mit_kinderfreibetrag_y_sn`.
    einkommensteuer__gesamteinkommen_y
        See :func:`einkommensteuer__gesamteinkommen_y`.
    kinderfreibetrag_günstiger_sn
        See :func:`kinderfreibetrag_günstiger_sn`.

    Returns
    -------

    """
    if kinderfreibetrag_günstiger_sn:
        out = zu_versteuerndes_einkommen_mit_kinderfreibetrag_y_sn
    else:
        out = einkommensteuer__gesamteinkommen_y

    return out


@policy_function(
    rounding_spec=RoundingSpec(
        base=36,
        direction="down",
        to_add_after_rounding=18,
        reference="§ 32a Abs. 2 EStG",
    ),
    start_date="2002-01-01",
    end_date="2003-12-31",
    leaf_name="zu_versteuerndes_einkommen_y_sn",
)
def zu_versteuerndes_einkommen_y_sn_mit_grober_54er_rundungsregel(
    zu_versteuerndes_einkommen_mit_kinderfreibetrag_y_sn: float,
    einkommensteuer__gesamteinkommen_y: float,
    kinderfreibetrag_günstiger_sn: bool,
) -> float:
    """Calculate taxable income on Steuernummer level.

    Parameters
    ----------
    zu_versteuerndes_einkommen_mit_kinderfreibetrag_y_sn
        See :func:`zu_versteuerndes_einkommen_mit_kinderfreibetrag_y_sn`.
    einkommensteuer__gesamteinkommen_y
        See :func:`einkommensteuer__gesamteinkommen_y`.
    kinderfreibetrag_günstiger_sn
        See :func:`kinderfreibetrag_günstiger_sn`.

    Returns
    -------

    """
    if kinderfreibetrag_günstiger_sn:
        out = zu_versteuerndes_einkommen_mit_kinderfreibetrag_y_sn
    else:
        out = einkommensteuer__gesamteinkommen_y

    return out


@policy_function(
    rounding_spec=RoundingSpec(
        base=27.609762,
        direction="down",
        to_add_after_rounding=13.804881,
        reference="§ 32a Abs. 2 EStG",
    ),
    end_date="2001-12-31",
    leaf_name="zu_versteuerndes_einkommen_y_sn",
)
def zu_versteuerndes_einkommen_y_sn_mit_dmark_rundungsregel(
    zu_versteuerndes_einkommen_mit_kinderfreibetrag_y_sn: float,
    einkommensteuer__gesamteinkommen_y: float,
    kinderfreibetrag_günstiger_sn: bool,
) -> float:
    """Calculate taxable income on Steuernummer level.

    Parameters
    ----------
    zu_versteuerndes_einkommen_mit_kinderfreibetrag_y_sn
        See :func:`zu_versteuerndes_einkommen_mit_kinderfreibetrag_y_sn`.
    einkommensteuer__gesamteinkommen_y
        See :func:`einkommensteuer__gesamteinkommen_y`.
    kinderfreibetrag_günstiger_sn
        See :func:`kinderfreibetrag_günstiger_sn`.

    Returns
    -------

    """
    if kinderfreibetrag_günstiger_sn:
        out = zu_versteuerndes_einkommen_mit_kinderfreibetrag_y_sn
    else:
        out = einkommensteuer__gesamteinkommen_y

    return out


@policy_function()
def zu_versteuerndes_einkommen_mit_kinderfreibetrag_y_sn(
    einkommensteuer__gesamteinkommen_y: float,
    kinderfreibetrag_y_sn: float,
) -> float:
    """Calculate taxable income with child allowance on Steuernummer level.

    Parameters
    ----------
    einkommensteuer__gesamteinkommen_y
        See :func:`einkommensteuer__gesamteinkommen_y`.
    kinderfreibetrag_y_sn
        See :func:`kinderfreibetrag_y_sn`.

    Returns
    -------

    """

    out = einkommensteuer__gesamteinkommen_y - kinderfreibetrag_y_sn
    return max(out, 0.0)
