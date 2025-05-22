"""Contributions to the unemployment insurance."""

from __future__ import annotations

from ttsim import policy_function


@policy_function(end_date="2003-03-31", leaf_name="betrag_versicherter_m")
def betrag_versicherter_m_ohne_midijob(
    sozialversicherung__geringfügig_beschäftigt: bool,
    sozialversicherung__rente__beitrag__einkommen_m: float,
    beitragssatz: float,
) -> float:
    """Unemployment insurance contributions paid by the insured person."""
    betrag_arbeitgeber_regulär_beschäftigt_m = (
        sozialversicherung__rente__beitrag__einkommen_m * beitragssatz
    )

    # Set to 0 for minijobs
    if sozialversicherung__geringfügig_beschäftigt:
        out = 0.0
    else:
        out = betrag_arbeitgeber_regulär_beschäftigt_m

    return out


@policy_function(start_date="2003-04-01", leaf_name="betrag_versicherter_m")
def betrag_versicherter_m_mit_midijob(
    sozialversicherung__geringfügig_beschäftigt: bool,
    sozialversicherung__in_gleitzone: bool,
    betrag_versicherter_midijob_m: float,
    sozialversicherung__rente__beitrag__einkommen_m: float,
    beitragssatz: float,
) -> float:
    """Unemployment insurance contributions paid by the insured person."""
    betrag_arbeitgeber_regulär_beschäftigt_m = (
        sozialversicherung__rente__beitrag__einkommen_m * beitragssatz
    )

    # Set to 0 for minijobs
    if sozialversicherung__geringfügig_beschäftigt:
        out = 0.0
    elif sozialversicherung__in_gleitzone:
        out = betrag_versicherter_midijob_m
    else:
        out = betrag_arbeitgeber_regulär_beschäftigt_m

    return out


@policy_function(end_date="2003-03-31", leaf_name="betrag_arbeitgeber_m")
def betrag_arbeitgeber_m_ohne_midijob(
    sozialversicherung__geringfügig_beschäftigt: bool,
    sozialversicherung__rente__beitrag__einkommen_m: float,
    beitragssatz: float,
) -> float:
    """Employer's unemployment insurance contribution until March 2003."""
    betrag_arbeitgeber_regulär_beschäftigt_m = (
        sozialversicherung__rente__beitrag__einkommen_m * beitragssatz
    )

    # Set to 0 for minijobs
    if sozialversicherung__geringfügig_beschäftigt:
        out = 0.0
    else:
        out = betrag_arbeitgeber_regulär_beschäftigt_m

    return out


@policy_function(start_date="2003-04-01", leaf_name="betrag_arbeitgeber_m")
def betrag_arbeitgeber_m_mit_midijob(
    sozialversicherung__geringfügig_beschäftigt: bool,
    sozialversicherung__in_gleitzone: bool,
    betrag_arbeitgeber_midijob_m: float,
    sozialversicherung__rente__beitrag__einkommen_m: float,
    beitragssatz: float,
) -> float:
    """Employer's unemployment insurance contribution since April 2003."""
    betrag_arbeitgeber_regulär_beschäftigt_m = (
        sozialversicherung__rente__beitrag__einkommen_m * beitragssatz
    )

    # Set to 0 for minijobs
    if sozialversicherung__geringfügig_beschäftigt:
        out = 0.0
    elif sozialversicherung__in_gleitzone:
        out = betrag_arbeitgeber_midijob_m
    else:
        out = betrag_arbeitgeber_regulär_beschäftigt_m

    return out


@policy_function(start_date="2003-04-01")
def betrag_gesamt_midijob_m(
    sozialversicherung__midijob_bemessungsentgelt_m: float,
    beitragssatz: float,
) -> float:
    """Sum of employee's and employer's unemployment insurance contribution
    for midijobs."""
    return sozialversicherung__midijob_bemessungsentgelt_m * 2 * beitragssatz


@policy_function(
    start_date="2003-04-01",
    end_date="2022-09-30",
    leaf_name="betrag_arbeitgeber_midijob_m",
)
def betrag_arbeitgeber_midijob_m_anteil_bruttolohn(
    einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m: float,
    beitragssatz: float,
) -> float:
    """Employers' unemployment insurance contribution for Midijobs until September
    2022."""
    return (
        einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m
        * beitragssatz
    )


@policy_function(start_date="2022-10-01", leaf_name="betrag_arbeitgeber_midijob_m")
def betrag_arbeitgeber_midijob_m_als_differenz_von_gesamt_und_versichertenbeitrag(
    betrag_gesamt_midijob_m: float,
    betrag_versicherter_midijob_m: float,
) -> float:
    """Employer's unemployment insurance contribution since October 2022.

    Parameters
    ----------
    betrag_gesamt_midijob_m
        See :func:`betrag_gesamt_midijob_m`.
    betrag_versicherter_midijob_m
        See :func:`betrag_versicherter_midijob_m`.

    Returns
    -------

    """
    return betrag_gesamt_midijob_m - betrag_versicherter_midijob_m


@policy_function(
    start_date="2003-04-01",
    end_date="2022-09-30",
    leaf_name="betrag_versicherter_midijob_m",
)
def betrag_versicherter_midijob_m_als_differenz_von_gesamt_und_arbeitgeberbeitrag(
    betrag_gesamt_midijob_m: float,
    betrag_arbeitgeber_midijob_m: float,
) -> float:
    """Employee's unemployment insurance contribution for Midijobs until September
    2022.

    Parameters
    ----------
    betrag_gesamt_midijob_m
        See :func:`betrag_gesamt_midijob_m`.
    betrag_arbeitgeber_midijob_m
        See :func:`betrag_arbeitgeber_midijob_m`.

    Returns
    -------

    """
    return betrag_gesamt_midijob_m - betrag_arbeitgeber_midijob_m


@policy_function(start_date="2022-10-01", leaf_name="betrag_versicherter_midijob_m")
def betrag_versicherter_midijob_m_mit_festem_beitragssatz(
    sozialversicherung__beitragspflichtige_einnahmen_aus_midijob_arbeitnehmer_m: float,
    beitragssatz: float,
) -> float:
    """Employee's unemployment insurance contribution since October 2022."""
    return (
        sozialversicherung__beitragspflichtige_einnahmen_aus_midijob_arbeitnehmer_m
        * beitragssatz
    )
