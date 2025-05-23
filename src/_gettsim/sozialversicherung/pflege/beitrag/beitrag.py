"""Contributions to public long-term care insurance."""

from __future__ import annotations

from ttsim import policy_function


@policy_function(
    start_date="1995-01-01",
    end_date="2003-03-31",
    leaf_name="betrag_versicherter_m",
)
def betrag_versicherter_m_ohne_midijob(
    einkommensteuer__einkünfte__ist_selbstständig: bool,
    betrag_selbstständig_m: float,
    sozialversicherung__geringfügig_beschäftigt: bool,
    betrag_versicherter_regulär_beschäftigt_m: float,
    betrag_rentner_m: float,
) -> float:
    """Long-term care insurance contributions paid by the insured person."""

    if einkommensteuer__einkünfte__ist_selbstständig:
        out = betrag_selbstständig_m
    elif sozialversicherung__geringfügig_beschäftigt:
        out = 0.0
    else:
        out = betrag_versicherter_regulär_beschäftigt_m

    # Add the care insurance contribution for pensions
    return out + betrag_rentner_m


@policy_function(
    start_date="2003-04-01",
    leaf_name="betrag_versicherter_m",
)
def betrag_versicherter_m_mit_midijob(
    einkommensteuer__einkünfte__ist_selbstständig: bool,
    betrag_selbstständig_m: float,
    sozialversicherung__geringfügig_beschäftigt: bool,
    sozialversicherung__in_gleitzone: bool,
    betrag_versicherter_in_gleitzone_m: float,
    betrag_versicherter_regulär_beschäftigt_m: float,
    betrag_rentner_m: float,
) -> float:
    """Long-term care insurance contributions paid by the insured person."""

    if einkommensteuer__einkünfte__ist_selbstständig:
        out = betrag_selbstständig_m
    elif sozialversicherung__geringfügig_beschäftigt:
        out = 0.0
    elif sozialversicherung__in_gleitzone:
        out = betrag_versicherter_in_gleitzone_m
    else:
        out = betrag_versicherter_regulär_beschäftigt_m

    # Add the care insurance contribution for pensions
    return out + betrag_rentner_m


@policy_function(
    start_date="1995-01-01",
    end_date="2003-03-31",
    leaf_name="betrag_arbeitgeber_m",
)
def betrag_arbeitgeber_m_ohne_midijob(
    einkommensteuer__einkünfte__ist_selbstständig: bool,
    sozialversicherung__geringfügig_beschäftigt: bool,
    betrag_arbeitgeber_regulär_beschäftigt_m: float,
) -> float:
    """Employer's long-term care insurance contribution.

    Before Midijob introduction in April 2003.
    """

    if (
        einkommensteuer__einkünfte__ist_selbstständig
        or sozialversicherung__geringfügig_beschäftigt
    ):
        out = 0.0
    else:
        out = betrag_arbeitgeber_regulär_beschäftigt_m

    return out


@policy_function(
    start_date="2003-04-01",
    leaf_name="betrag_arbeitgeber_m",
)
def betrag_arbeitgeber_m_mit_midijob(
    einkommensteuer__einkünfte__ist_selbstständig: bool,
    sozialversicherung__geringfügig_beschäftigt: bool,
    sozialversicherung__in_gleitzone: bool,
    betrag_arbeitgeber_in_gleitzone_m: float,
    betrag_arbeitgeber_regulär_beschäftigt_m: float,
) -> float:
    """Employer's long-term care insurance contribution.

    After Midijob introduction in April 2003.
    """

    if (
        einkommensteuer__einkünfte__ist_selbstständig
        or sozialversicherung__geringfügig_beschäftigt
    ):
        out = 0.0
    elif sozialversicherung__in_gleitzone:
        out = betrag_arbeitgeber_in_gleitzone_m
    else:
        out = betrag_arbeitgeber_regulär_beschäftigt_m

    return out


@policy_function(start_date="1995-01-01")
def betrag_selbstständig_m(
    sozialversicherung__kranken__beitrag__bemessungsgrundlage_selbstständig_m: float,
    beitragssatz_arbeitnehmer: float,
    beitragssatz_arbeitgeber: float,
) -> float:
    """Self-employed individuals' long-term care insurance contribution until 2004.

    Self-employed pay the full contribution (employer + employee), which is either
    assessed on their self-employement income or 3/4 of the 'Bezugsgröße'
    """
    return sozialversicherung__kranken__beitrag__bemessungsgrundlage_selbstständig_m * (
        beitragssatz_arbeitnehmer + beitragssatz_arbeitgeber
    )


@policy_function(start_date="1995-01-01")
def betrag_versicherter_regulär_beschäftigt_m(
    sozialversicherung__kranken__beitrag__einkommen_m: float,
    beitragssatz_arbeitnehmer: float,
) -> float:
    """Long-term care insurance contributions paid by the insured person if regularly
    employed.
    """

    return sozialversicherung__kranken__beitrag__einkommen_m * beitragssatz_arbeitnehmer


@policy_function(start_date="1995-01-01")
def betrag_arbeitgeber_regulär_beschäftigt_m(
    sozialversicherung__kranken__beitrag__einkommen_m: float,
    beitragssatz_arbeitgeber: float,
) -> float:
    """Long-term care insurance contributions paid by the employer under regular
    employment.
    """

    return sozialversicherung__kranken__beitrag__einkommen_m * beitragssatz_arbeitgeber


@policy_function(
    start_date="2003-04-01",
)
def betrag_gesamt_in_gleitzone_m(
    sozialversicherung__midijob_bemessungsentgelt_m: float,
    beitragssatz_arbeitnehmer: float,
    beitragssatz_arbeitgeber: float,
) -> float:
    """Sum of employee and employer long-term care insurance contributions until 2004."""

    return sozialversicherung__midijob_bemessungsentgelt_m * (
        beitragssatz_arbeitnehmer + beitragssatz_arbeitgeber
    )


@policy_function(
    start_date="2003-04-01",
    end_date="2022-09-30",
    leaf_name="betrag_versicherter_in_gleitzone_m",
)
def betrag_versicherter_in_gleitzone_m_als_differenz_von_gesamt_und_arbeitgeberbeitrag(
    betrag_arbeitgeber_in_gleitzone_m: float,
    betrag_gesamt_in_gleitzone_m: float,
) -> float:
    """Employee's long-term care insurance contribution for Midijobs
    until September 2022.
    """
    return betrag_gesamt_in_gleitzone_m - betrag_arbeitgeber_in_gleitzone_m


@policy_function(
    start_date="2022-10-01",
    end_date="2023-06-30",
    leaf_name="betrag_versicherter_in_gleitzone_m",
)
def betrag_versicherter_in_gleitzone_m_direkt(
    sozialversicherung__beitragspflichtige_einnahmen_aus_midijob_arbeitnehmer_m: float,
    beitragssatz_arbeitnehmer: float,
) -> float:
    """Employee's long-term care insurance contribution between October 2022 and
    June 2023.
    """
    return (
        sozialversicherung__beitragspflichtige_einnahmen_aus_midijob_arbeitnehmer_m
        * beitragssatz_arbeitnehmer
    )


@policy_function(
    start_date="2023-07-01",
    leaf_name="betrag_versicherter_in_gleitzone_m",
)
def betrag_versicherter_midijob_m_mit_verringertem_beitrag_für_eltern_mit_mehreren_kindern(
    anzahl_kinder_bis_24: int,
    zahlt_zusatzbetrag_kinderlos: bool,
    sozialversicherung__beitragspflichtige_einnahmen_aus_midijob_arbeitnehmer_m: float,
    sozialversicherung__midijob_bemessungsentgelt_m: float,
    beitragssatz_nach_kinderzahl: dict[str, float],
) -> float:
    """Employee's long-term care insurance contribution since July 2023."""

    base = (
        sozialversicherung__beitragspflichtige_einnahmen_aus_midijob_arbeitnehmer_m
        * beitragssatz_nach_kinderzahl["standard"]
        / 2
    )

    add = 0.0
    if zahlt_zusatzbetrag_kinderlos:
        add = (
            add
            + sozialversicherung__midijob_bemessungsentgelt_m
            * beitragssatz_nach_kinderzahl["zusatz_kinderlos"]
        )
    if anzahl_kinder_bis_24 >= 2:
        add = add + (
            sozialversicherung__beitragspflichtige_einnahmen_aus_midijob_arbeitnehmer_m
            * beitragssatz_nach_kinderzahl["abschlag_für_kinder_bis_24"]
            * min(anzahl_kinder_bis_24 - 1, 4)
        )

    return base + add


@policy_function(
    start_date="2003-04-01",
    end_date="2022-09-30",
    leaf_name="betrag_arbeitgeber_in_gleitzone_m",
)
def betrag_arbeitgeber_in_gleitzone_m_mit_festem_beitragssatz(
    einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m: float,
    beitragssatz_arbeitgeber: float,
) -> float:
    """Employer's long-term care insurance contribution in Midijob range."""

    return (
        einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m
        * beitragssatz_arbeitgeber
    )


@policy_function(start_date="2022-10-01", leaf_name="betrag_arbeitgeber_in_gleitzone_m")
def betrag_arbeitgeber_in_gleitzone_m_als_differenz_von_gesamt_und_versichertenbeitrag(
    betrag_gesamt_in_gleitzone_m: float,
    betrag_versicherter_m: float,
) -> float:
    """Employer's long-term care insurance contribution since October 2022."""
    return betrag_gesamt_in_gleitzone_m - betrag_versicherter_m


@policy_function(
    start_date="1995-01-01",
    end_date="2004-03-31",
    leaf_name="betrag_rentner_m",
)
def betrag_rentner_m_reduzierter_beitrag(
    sozialversicherung__kranken__beitrag__bemessungsgrundlage_rente_m: float,
    beitragssatz_arbeitnehmer: float,
) -> float:
    """Long-term care insurance contribution from pension income from 1995 until March
    2004.

    Pensioners pay the same contribution as employees.
    """
    return (
        sozialversicherung__kranken__beitrag__bemessungsgrundlage_rente_m
        * beitragssatz_arbeitnehmer
    )


@policy_function(
    start_date="2004-04-01",
    end_date="2004-12-31",
    leaf_name="betrag_rentner_m",
)
def betrag_rentner_m_ohne_zusatz_für_kinderlose(
    sozialversicherung__kranken__beitrag__bemessungsgrundlage_rente_m: float,
    beitragssatz: float,
) -> float:
    """Health insurance contribution from pension income from April until December 2004.

    Pensioners pay twice the contribution of employees.
    """
    return sozialversicherung__kranken__beitrag__bemessungsgrundlage_rente_m * (
        beitragssatz
    )


@policy_function(
    start_date="2005-01-01", leaf_name="betrag_rentner_m", vectorization_strategy="loop"
)
def betrag_rentner_m_mit_zusatz_für_kinderlose(
    sozialversicherung__kranken__beitrag__bemessungsgrundlage_rente_m: float,
    beitragssatz_arbeitnehmer: float,
    beitragssatz_nach_kinderzahl: dict[str, float],
) -> float:
    """Health insurance contribution from pension income since 2005.

    Pensioners pay twice the contribution of employees, but only once the additional
    charge for childless individuals.
    """
    return sozialversicherung__kranken__beitrag__bemessungsgrundlage_rente_m * (
        beitragssatz_arbeitnehmer + beitragssatz_nach_kinderzahl["standard"] / 2
    )
