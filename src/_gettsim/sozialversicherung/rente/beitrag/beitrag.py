"""Public pension insurance contributions."""

from ttsim import policy_function


@policy_function(end_date="2003-03-31", leaf_name="betrag_versicherter_m")
def betrag_versicherter_m_ohne_midijob(
    sozialversicherung__geringfügig_beschäftigt: bool,
    einkommen_m: float,
    parameter_beitragssatz: float,
) -> float:
    """Public pension insurance contributions paid by the insured person.

    Before Midijob introduction in April 2003.
    """
    ges_rentenv_beitr_regular_job_m = einkommen_m * parameter_beitragssatz

    if sozialversicherung__geringfügig_beschäftigt:
        out = 0.0
    else:
        out = ges_rentenv_beitr_regular_job_m

    return out


@policy_function(start_date="2003-04-01", leaf_name="betrag_versicherter_m")
def betrag_versicherter_m_mit_midijob(
    sozialversicherung__geringfügig_beschäftigt: bool,
    betrag_midijob_arbeitnehmer_m: float,
    einkommen_m: float,
    parameter_beitragssatz: float,
    sozialversicherung__in_gleitzone: bool,
) -> float:
    """Public pension insurance contributions paid by the insured person.

    After Midijob introduction in April 2003.
    """
    ges_rentenv_beitr_regular_job_m = einkommen_m * parameter_beitragssatz

    if sozialversicherung__geringfügig_beschäftigt:
        out = 0.0
    elif sozialversicherung__in_gleitzone:
        out = betrag_midijob_arbeitnehmer_m
    else:
        out = ges_rentenv_beitr_regular_job_m

    return out


@policy_function(end_date="2003-03-31", leaf_name="betrag_arbeitgeber_m")
def betrag_arbeitgeber_m_ohne_midijob(
    sozialversicherung__geringfügig_beschäftigt: bool,
    einkommen_m: float,
    parameter_beitragssatz: float,
    sozialversicherung__rente__beitrag__arbeitgeberpauschale_bei_geringfügiger_beschäftigung: float,
    einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m: float,
) -> float:
    """Employer's public pension insurance contribution.

    Before Midijob introduction in April 2003.
    """
    ges_rentenv_beitr_regular_job_m = einkommen_m * parameter_beitragssatz

    if sozialversicherung__geringfügig_beschäftigt:
        out = (
            einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m
            * sozialversicherung__rente__beitrag__arbeitgeberpauschale_bei_geringfügiger_beschäftigung
        )
    else:
        out = ges_rentenv_beitr_regular_job_m

    return out


@policy_function(start_date="2003-04-01", leaf_name="betrag_arbeitgeber_m")
def betrag_arbeitgeber_m_mit_midijob(
    sozialversicherung__geringfügig_beschäftigt: bool,
    betrag_midijob_arbeitgeber_m: float,
    einkommen_m: float,
    parameter_beitragssatz: float,
    sozialversicherung__in_gleitzone: bool,
    einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m: float,
    sozialversicherung__rente__beitrag__arbeitgeberpauschale_bei_geringfügiger_beschäftigung: float,
) -> float:
    """Employer's public pension insurance contribution.

    After Midijob introduction in April 2003.
    """
    ges_rentenv_beitr_regular_job_m = einkommen_m * parameter_beitragssatz

    if sozialversicherung__geringfügig_beschäftigt:
        out = (
            einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m
            * sozialversicherung__rente__beitrag__arbeitgeberpauschale_bei_geringfügiger_beschäftigung
        )
    elif sozialversicherung__in_gleitzone:
        out = betrag_midijob_arbeitgeber_m
    else:
        out = ges_rentenv_beitr_regular_job_m

    return out


@policy_function()
def einkommen_m(
    einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m: float,
    beitragsbemessungsgrenze_m: float,
) -> float:
    """Wage subject to pension and unemployment insurance contributions."""
    return min(
        einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m,
        beitragsbemessungsgrenze_m,
    )


# TODO: Differentiate by regime, i.e., just a parameter in years where we do not have
# Ost/West distinction. Same for all Sozialversicherunbsbeiträge!
@policy_function()
def beitragsbemessungsgrenze_m(
    wohnort_ost: bool, parameter_beitragsbemessungsgrenze: dict
) -> float:
    """Income threshold up to which pension insurance payments apply."""
    return (
        parameter_beitragsbemessungsgrenze["ost"]
        if wohnort_ost
        else parameter_beitragsbemessungsgrenze["west"]
    )


@policy_function(start_date="2003-04-01")
def betrag_midijob_gesamt_m(
    sozialversicherung__midijob_bemessungsentgelt_m: float,
    parameter_beitragssatz: float,
) -> float:
    """Sum of employer and employee pension insurance contribution for midijobs.
    Midijobs were introduced in April 2003.
    """
    return sozialversicherung__midijob_bemessungsentgelt_m * 2 * parameter_beitragssatz


@policy_function(
    end_date="2022-09-30",
    leaf_name="betrag_midijob_arbeitgeber_m",
)
def betrag_midijob_arbeitgeber_m_mit_festem_beitragssatz(
    einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m: float,
    parameter_beitragssatz: float,
) -> float:
    """Employer's unemployment insurance contribution until September 2022."""
    return (
        einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m
        * parameter_beitragssatz
    )


@policy_function(start_date="2022-10-01", leaf_name="betrag_midijob_arbeitgeber_m")
def betrag_midijob_arbeitgeber_m_als_differenz_von_gesamt_und_arbeitnehmerbeitrag(
    betrag_midijob_gesamt_m: float,
    betrag_midijob_arbeitnehmer_m: float,
) -> float:
    """Employer's unemployment insurance contribution since October 2022."""
    return betrag_midijob_gesamt_m - betrag_midijob_arbeitnehmer_m


@policy_function(
    end_date="2022-09-30",
    leaf_name="betrag_midijob_arbeitnehmer_m",
)
def betrag_midijob_arbeitnehmer_m_als_differenz_von_gesamt_und_arbeitgeberbeitrag(
    betrag_midijob_arbeitgeber_m: float,
    betrag_midijob_gesamt_m: float,
) -> float:
    """Employee's unemployment insurance contribution for midijobs until September 2022."""
    return betrag_midijob_gesamt_m - betrag_midijob_arbeitgeber_m


@policy_function(start_date="2022-10-01", leaf_name="betrag_midijob_arbeitnehmer_m")
def betrag_midijob_arbeitnehmer_m_mit_festem_beitragssatz(
    sozialversicherung__beitragspflichtige_einnahmen_aus_midijob_arbeitnehmer_m: float,
    parameter_beitragssatz: float,
) -> float:
    """Employee's unemployment insurance contribution for midijobs since October 2022."""
    return (
        sozialversicherung__beitragspflichtige_einnahmen_aus_midijob_arbeitnehmer_m
        * parameter_beitragssatz
    )
