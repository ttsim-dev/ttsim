"""Public pension insurance contributions."""

from ttsim import policy_function


@policy_function(end_date="2003-03-31", leaf_name="betrag_versicherter_m")
def betrag_versicherter_m_ohne_midijob(
    sozialversicherung__geringfügig_beschäftigt: bool,
    einkommen_m: float,
    ges_rentenv_params: dict,
) -> float:
    """Public pension insurance contributions paid by the insured person.

    Before Midijob introduction in April 2003.

    Parameters
    ----------
    sozialversicherung__geringfügig_beschäftigt
        See :func:`sozialversicherung__geringfügig_beschäftigt`.
    einkommen_m
        See :func:`einkommen_m`.
    sozialv_beitr_params
        See params documentation :ref:`sozialv_beitr_params <sozialv_beitr_params>`.

    Returns
    -------

    """
    ges_rentenv_beitr_regular_job_m = einkommen_m * ges_rentenv_params["beitragssatz"]

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
    ges_rentenv_params: dict,
    sozialversicherung__in_gleitzone: bool,
) -> float:
    """Public pension insurance contributions paid by the insured person.

    After Midijob introduction in April 2003.

    Parameters
    ----------
    sozialversicherung__geringfügig_beschäftigt
        See :func:`sozialversicherung__geringfügig_beschäftigt`.
    betrag_midijob_arbeitnehmer_m
        See :func:`betrag_midijob_arbeitnehmer_m`.
    einkommen_m
        See :func:`einkommen_m`.
    sozialv_beitr_params
        See params documentation :ref:`sozialv_beitr_params <sozialv_beitr_params>`.
    sozialversicherung__in_gleitzone
        See :func:`sozialversicherung__in_gleitzone`.

    Returns
    -------

    """
    ges_rentenv_beitr_regular_job_m = einkommen_m * ges_rentenv_params["beitragssatz"]

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
    ges_rentenv_params: dict,
    einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m: float,
) -> float:
    """Employer's public pension insurance contribution.

    Before Midijob introduction in April 2003.

    Parameters
    ----------
    sozialversicherung__geringfügig_beschäftigt
        See :func:`sozialversicherung__geringfügig_beschäftigt`.
    einkommen_m
        See :func:`einkommen_m`.
    sozialv_beitr_params
        See params documentation :ref:`sozialv_beitr_params <sozialv_beitr_params>`.
    einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m
        See basic input variable :ref:`einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m <einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m>`.

    Returns
    -------

    """
    ges_rentenv_beitr_regular_job_m = einkommen_m * ges_rentenv_params["beitragssatz"]

    if sozialversicherung__geringfügig_beschäftigt:
        out = (
            einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m
            * ges_rentenv_params["arbeitgeberpauschale_bei_geringfügiger_beschäftigung"]
        )
    else:
        out = ges_rentenv_beitr_regular_job_m

    return out


@policy_function(start_date="2003-04-01", leaf_name="betrag_arbeitgeber_m")
def betrag_arbeitgeber_m_mit_midijob(
    sozialversicherung__geringfügig_beschäftigt: bool,
    betrag_midijob_arbeitgeber_m: float,
    einkommen_m: float,
    ges_rentenv_params: dict,
    sozialversicherung__in_gleitzone: bool,
    einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m: float,
) -> float:
    """Employer's public pension insurance contribution.

    After Midijob introduction in April 2003.

    Parameters
    ----------
    sozialversicherung__geringfügig_beschäftigt
        See :func:`sozialversicherung__geringfügig_beschäftigt`.
    betrag_midijob_arbeitgeber_m
        See :func:`betrag_midijob_arbeitgeber_m`.
    einkommen_m
        See :func:`einkommen_m`.
    sozialv_beitr_params
        See params documentation :ref:`sozialv_beitr_params <sozialv_beitr_params>`.
    sozialversicherung__in_gleitzone
        See :func:`sozialversicherung__in_gleitzone`.
    einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m
        See basic input variable :ref:`einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m <einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m>`.

    Returns
    -------

    """
    ges_rentenv_beitr_regular_job_m = einkommen_m * ges_rentenv_params["beitragssatz"]

    if sozialversicherung__geringfügig_beschäftigt:
        out = (
            einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m
            * ges_rentenv_params["arbeitgeberpauschale_bei_geringfügiger_beschäftigung"]
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
    """Wage subject to pension and unemployment insurance contributions.

    Parameters
    ----------
    einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m
        See params documentation :ref:`sozialv_beitr_params <sozialv_beitr_params>`.
    beitragsbemessungsgrenze_m
        See :func:`beitragsbemessungsgrenze_m`.


    Returns
    -------

    """
    return min(
        einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m,
        beitragsbemessungsgrenze_m,
    )


# TODO: Differentiate by regime, i.e., just a parameter in years where we do not have
# Ost/West distinction. Same for all Sozialversicherunbsbeiträge!
@policy_function()
def beitragsbemessungsgrenze_m(wohnort_ost: bool, ges_rentenv_params: dict) -> float:
    """Income threshold up to which pension insurance payments apply.

    Parameters
    ----------
    wohnort_ost
        See :func:`wohnort_ost`.
    sozialv_beitr_params
        See params documentation :ref:`sozialv_beitr_params <sozialv_beitr_params>`.

    Returns
    -------

    """
    params = ges_rentenv_params["beitragsbemessungsgrenze_m"]
    out = params["ost"] if wohnort_ost else params["west"]

    return out


@policy_function(start_date="2003-04-01")
def betrag_midijob_gesamt_m(
    sozialversicherung__midijob_bemessungsentgelt_m: float,
    ges_rentenv_params: dict,
) -> float:
    """Sum of employer and employee pension insurance contribution for midijobs.
    Midijobs were introduced in April 2003.

    Parameters
    ----------
    sozialversicherung__midijob_bemessungsentgelt_m
        See :func:`sozialversicherung__midijob_bemessungsentgelt_m`.
    sozialv_beitr_params
        See params documentation :ref:`sozialv_beitr_params <sozialv_beitr_params>`.

    Returns
    -------

    """
    return (
        sozialversicherung__midijob_bemessungsentgelt_m
        * 2
        * ges_rentenv_params["beitragssatz"]
    )


@policy_function(
    end_date="2022-09-30",
    leaf_name="betrag_midijob_arbeitgeber_m",
)
def betrag_midijob_arbeitgeber_m_mit_festem_beitragssatz(
    einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m: float,
    ges_rentenv_params: dict,
) -> float:
    """Employer's unemployment insurance contribution until September 2022.

    Parameters
    ----------
    einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m
        See basic input variable :ref:`einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m <einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m>`.
    sozialv_beitr_params
        See params documentation :ref:`sozialv_beitr_params <sozialv_beitr_params>`.

    Returns
    -------

    """
    return (
        einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m
        * ges_rentenv_params["beitragssatz"]
    )


@policy_function(start_date="2022-10-01", leaf_name="betrag_midijob_arbeitgeber_m")
def betrag_midijob_arbeitgeber_m_als_differenz_von_gesamt_und_arbeitnehmerbeitrag(
    betrag_midijob_gesamt_m: float,
    betrag_midijob_arbeitnehmer_m: float,
) -> float:
    """Employer's unemployment insurance contribution since October 2022.

    Parameters
    ----------
    betrag_midijob_gesamt_m
        See :func:`betrag_midijob_gesamt_m`.
    betrag_midijob_arbeitnehmer_m
        See :func:`betrag_midijob_arbeitnehmer_m`.

    Returns
    -------

    """
    return betrag_midijob_gesamt_m - betrag_midijob_arbeitnehmer_m


@policy_function(
    end_date="2022-09-30",
    leaf_name="betrag_midijob_arbeitnehmer_m",
)
def betrag_midijob_arbeitnehmer_m_als_differenz_von_gesamt_und_arbeitgeberbeitrag(
    betrag_midijob_arbeitgeber_m: float,
    betrag_midijob_gesamt_m: float,
) -> float:
    """Employee's unemployment insurance contribution for midijobs until September 2022.

    Parameters
    ----------
    betrag_midijob_arbeitgeber_m
        See :func:`betrag_midijob_arbeitgeber_m`.
    betrag_midijob_gesamt_m
        See :func:`betrag_midijob_gesamt_m`.

    Returns
    -------

    """
    return betrag_midijob_gesamt_m - betrag_midijob_arbeitgeber_m


@policy_function(start_date="2022-10-01", leaf_name="betrag_midijob_arbeitnehmer_m")
def betrag_midijob_arbeitnehmer_m_mit_festem_beitragssatz(
    sozialversicherung__beitragspflichtige_einnahmen_aus_midijob_arbeitnehmer_m: float,
    ges_rentenv_params: dict,
) -> float:
    """Employee's unemployment insurance contribution for midijobs since October 2022.

    Parameters
    ----------
    sozialversicherung__beitragspflichtige_einnahmen_aus_midijob_arbeitnehmer_m
        See :func:`sozialversicherung__beitragspflichtige_einnahmen_aus_midijob_arbeitnehmer_m`.
    sozialv_beitr_params
        See params documentation :ref:`sozialv_beitr_params <sozialv_beitr_params>`.

    Returns
    -------

    """
    return (
        sozialversicherung__beitragspflichtige_einnahmen_aus_midijob_arbeitnehmer_m
        * ges_rentenv_params["beitragssatz"]
    )
