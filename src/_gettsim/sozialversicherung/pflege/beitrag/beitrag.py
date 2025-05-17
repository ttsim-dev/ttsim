"""Contributions to public long-term care insurance."""

from ttsim import policy_function


@policy_function(
    end_date="2003-03-31",
    leaf_name="betrag_versicherter_m",
    vectorization_strategy="loop",
)
def betrag_versicherter_m_ohne_midijob(
    betrag_versicherter_regulär_beschäftigt_m: float,
    sozialversicherung__geringfügig_beschäftigt: bool,
    betrag_rentner_m: float,
    betrag_selbstständig_m: float,
    einkommensteuer__einkünfte__ist_selbstständig: bool,
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
    vectorization_strategy="loop",
)
def betrag_versicherter_m_mit_midijob(
    betrag_versicherter_regulär_beschäftigt_m: float,
    sozialversicherung__geringfügig_beschäftigt: bool,
    betrag_rentner_m: float,
    betrag_selbstständig_m: float,
    betrag_versicherter_midijob_m: float,
    sozialversicherung__in_gleitzone: bool,
    einkommensteuer__einkünfte__ist_selbstständig: bool,
) -> float:
    """Long-term care insurance contributions paid by the insured person."""

    if einkommensteuer__einkünfte__ist_selbstständig:
        out = betrag_selbstständig_m
    elif sozialversicherung__geringfügig_beschäftigt:
        out = 0.0
    elif sozialversicherung__in_gleitzone:
        out = betrag_versicherter_midijob_m
    else:
        out = betrag_versicherter_regulär_beschäftigt_m

    # Add the care insurance contribution for pensions
    return out + betrag_rentner_m


@policy_function(vectorization_strategy="loop")
def betrag_versicherter_regulär_beschäftigt_m(
    sozialversicherung__kranken__beitrag__einkommen_m: float,
    beitragssatz: float,
) -> float:
    """Long-term care insurance contributions paid by the insured person if regularly
    employed.
    """

    return sozialversicherung__kranken__beitrag__einkommen_m * beitragssatz


@policy_function(
    end_date="2003-03-31",
    leaf_name="betrag_arbeitgeber_m",
    vectorization_strategy="loop",
)
def betrag_arbeitgeber_m_ohne_midijob(
    sozialversicherung__geringfügig_beschäftigt: bool,
    sozialversicherung__kranken__beitrag__einkommen_m: float,
    beitragssatz_einheitlich: float,
    einkommensteuer__einkünfte__ist_selbstständig: bool,
) -> float:
    """Employer's long-term care insurance contribution.

    Before Midijob introduction in April 2003.
    """
    # Calculate care insurance contributions for regular jobs.
    beitr_regulär_beschäftigt_m = (
        sozialversicherung__kranken__beitrag__einkommen_m * beitragssatz_einheitlich
    )

    if (
        einkommensteuer__einkünfte__ist_selbstständig
        or sozialversicherung__geringfügig_beschäftigt
    ):
        out = 0.0
    else:
        out = beitr_regulär_beschäftigt_m

    return out


@policy_function(
    start_date="2003-04-01",
    end_date="2004-12-31",
    leaf_name="betrag_arbeitgeber_m",
    vectorization_strategy="loop",
)
def betrag_arbeitgeber_m_mit_midijob_einheitlicher_beitragssatz(
    sozialversicherung__geringfügig_beschäftigt: bool,
    betrag_arbeitgeber_midijob_m: float,
    sozialversicherung__kranken__beitrag__einkommen_m: float,
    beitragssatz_einheitlich: float,
    sozialversicherung__in_gleitzone: bool,
    einkommensteuer__einkünfte__ist_selbstständig: bool,
) -> float:
    """Employer's long-term care insurance contribution.

    After Midijob introduction in April 2003.
    """
    # Calculate care insurance contributions for regular jobs.
    beitr_regulär_beschäftigt_m = (
        sozialversicherung__kranken__beitrag__einkommen_m * beitragssatz_einheitlich
    )

    if (
        einkommensteuer__einkünfte__ist_selbstständig
        or sozialversicherung__geringfügig_beschäftigt
    ):
        out = 0.0
    elif sozialversicherung__in_gleitzone:
        out = betrag_arbeitgeber_midijob_m
    else:
        out = beitr_regulär_beschäftigt_m

    return out


@policy_function(
    start_date="2005-01-01",
    leaf_name="betrag_arbeitgeber_m",
    vectorization_strategy="loop",
)
def betrag_arbeitgeber_m_mit_midijob_beitragssatz_abhängig_von_anzahl_kinder(
    sozialversicherung__geringfügig_beschäftigt: bool,
    betrag_arbeitgeber_midijob_m: float,
    sozialversicherung__kranken__beitrag__einkommen_m: float,
    beitragssatz_abhängig_von_anzahl_kinder: dict[str, float],
    sozialversicherung__in_gleitzone: bool,
    einkommensteuer__einkünfte__ist_selbstständig: bool,
) -> float:
    """Employer's long-term care insurance contribution.

    After Midijob introduction in April 2003.
    """
    # Calculate care insurance contributions for regular jobs.
    beitr_regulär_beschäftigt_m = (
        sozialversicherung__kranken__beitrag__einkommen_m
        * beitragssatz_abhängig_von_anzahl_kinder["standard"]
    )

    if (
        einkommensteuer__einkünfte__ist_selbstständig
        or sozialversicherung__geringfügig_beschäftigt
    ):
        out = 0.0
    elif sozialversicherung__in_gleitzone:
        out = betrag_arbeitgeber_midijob_m
    else:
        out = beitr_regulär_beschäftigt_m

    return out


@policy_function(
    start_date="1995-01-01",
    end_date="2004-12-31",
    leaf_name="betrag_selbstständig_m",
    vectorization_strategy="loop",
)
def betrag_selbstständig_m_ohne_zusatz_für_kinderlose(
    sozialversicherung__kranken__beitrag__bemessungsgrundlage_selbstständig_m: float,
    beitragssatz: float,
) -> float:
    """Self-employed individuals' long-term care insurance contribution until 2004.

    Self-employed pay the full contribution (employer + employee), which is either
    assessed on their self-employement income or 3/4 of the 'Bezugsgröße'
    """
    return sozialversicherung__kranken__beitrag__bemessungsgrundlage_selbstständig_m * (
        beitragssatz * 2
    )


@policy_function(
    start_date="2005-01-01",
    leaf_name="betrag_selbstständig_m",
    vectorization_strategy="loop",
)
def betrag_selbstständig_m_mit_zusatz_für_kinderlose(
    sozialversicherung__kranken__beitrag__bemessungsgrundlage_selbstständig_m: float,
    beitragssatz: float,
    beitragssatz_abhängig_von_anzahl_kinder: dict[str, float],
) -> float:
    """Self-employed individuals' long-term care insurance contribution since 2005.

    Self-employed pay the full contribution (employer + employee), which is either
    assessed on their self-employement income or 3/4 of the 'Bezugsgröße'
    """
    return sozialversicherung__kranken__beitrag__bemessungsgrundlage_selbstständig_m * (
        beitragssatz + beitragssatz_abhängig_von_anzahl_kinder["standard"]
    )


@policy_function(
    start_date="1995-01-01",
    end_date="2004-03-31",
    leaf_name="betrag_rentner_m",
    vectorization_strategy="loop",
)
def betrag_rentner_m_reduzierter_beitrag(
    sozialversicherung__kranken__beitrag__bemessungsgrundlage_rente_m: float,
    beitragssatz: float,
) -> float:
    """Long-term care insurance contribution from pension income from 1995 until March
    2004.

    Pensioners pay the same contribution as employees.
    """
    return (
        sozialversicherung__kranken__beitrag__bemessungsgrundlage_rente_m * beitragssatz
    )


@policy_function(
    start_date="2004-04-01",
    end_date="2004-12-31",
    leaf_name="betrag_rentner_m",
    vectorization_strategy="loop",
)
def betrag_rentner_m_ohne_zusatz_für_kinderlose(
    sozialversicherung__kranken__beitrag__bemessungsgrundlage_rente_m: float,
    beitragssatz: float,
) -> float:
    """Health insurance contribution from pension income from April until December 2004.

    Pensioners pay twice the contribution of employees.
    """
    return sozialversicherung__kranken__beitrag__bemessungsgrundlage_rente_m * (
        beitragssatz * 2
    )


@policy_function(
    start_date="2005-01-01", leaf_name="betrag_rentner_m", vectorization_strategy="loop"
)
def betrag_rentner_m_mit_zusatz_für_kinderlose(
    sozialversicherung__kranken__beitrag__bemessungsgrundlage_rente_m: float,
    beitragssatz: float,
    beitragssatz_abhängig_von_anzahl_kinder: dict[str, float],
) -> float:
    """Health insurance contribution from pension income since 2005.

    Pensioners pay twice the contribution of employees, but only once the additional
    charge for childless individuals.
    """
    return sozialversicherung__kranken__beitrag__bemessungsgrundlage_rente_m * (
        beitragssatz + beitragssatz_abhängig_von_anzahl_kinder["standard"]
    )


@policy_function(
    start_date="2003-04-01",
    end_date="2004-12-31",
    leaf_name="betrag_gesamt_m",
    vectorization_strategy="loop",
)
def betrag_gesamt_m_bis_2004(
    sozialversicherung__midijob_bemessungsentgelt_m: float,
    beitragssatz: float,
    beitragssatz_einheitlich: float,
) -> float:
    """Sum of employee and employer long-term care insurance contributions until 2004."""

    return sozialversicherung__midijob_bemessungsentgelt_m * (
        beitragssatz + beitragssatz_einheitlich
    )


@policy_function(
    start_date="2005-01-01",
    leaf_name="betrag_gesamt_m",
    vectorization_strategy="loop",
)
def betrag_gesamt_m_ab_2005(
    sozialversicherung__midijob_bemessungsentgelt_m: float,
    beitragssatz: float,
    beitragssatz_abhängig_von_anzahl_kinder: dict[str, float],
) -> float:
    """Sum of employee and employer long-term care insurance contributions since 2005."""

    return sozialversicherung__midijob_bemessungsentgelt_m * (
        beitragssatz + beitragssatz_abhängig_von_anzahl_kinder["standard"]
    )


@policy_function(
    end_date="2004-12-31",
    leaf_name="betrag_arbeitgeber_midijob_m",
    vectorization_strategy="loop",
)
def betrag_arbeitgeber_midijob_m_mit_festem_beitragssatz_bis_2004(
    einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m: float,
    beitragssatz_einheitlich: float,
) -> float:
    """Employer's long-term care insurance contribution until December 2004."""

    return (
        einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m
        * beitragssatz_einheitlich
    )


@policy_function(
    start_date="2005-01-01",
    end_date="2022-09-30",
    leaf_name="betrag_arbeitgeber_midijob_m",
    vectorization_strategy="loop",
)
def betrag_arbeitgeber_midijob_m_mit_festem_beitragssatz_ab_2005(
    einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m: float,
    beitragssatz_abhängig_von_anzahl_kinder: dict[str, float],
) -> float:
    """Employers' contribution to long-term care insurance between 2005 and September
    2022.
    """
    return (
        einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m
        * beitragssatz_abhängig_von_anzahl_kinder["standard"]
    )


@policy_function(start_date="2022-10-01", leaf_name="betrag_arbeitgeber_midijob_m")
def betrag_arbeitgeber_midijob_m_als_differenz_von_gesamt_und_versichertenbeitrag(
    betrag_gesamt_m: float,
    betrag_versicherter_m: float,
) -> float:
    """Employer's long-term care insurance contribution since October 2022."""
    return betrag_gesamt_m - betrag_versicherter_m


@policy_function(
    end_date="2022-09-30",
    leaf_name="betrag_versicherter_midijob_m",
    vectorization_strategy="loop",
)
def betrag_versicherter_midijob_m_als_differenz_von_gesamt_und_arbeitgeberbeitrag(
    betrag_arbeitgeber_midijob_m: float,
    betrag_gesamt_m: float,
) -> float:
    """Employee's long-term care insurance contribution for Midijobs
    until September 2022.
    """
    return betrag_gesamt_m - betrag_arbeitgeber_midijob_m


@policy_function(
    start_date="2022-10-01",
    end_date="2023-06-30",
    leaf_name="betrag_versicherter_midijob_m",
    vectorization_strategy="loop",
)
def betrag_versicherter_midijob_m_mit_zusatzbeitrag_für_kinderlos(
    zusatzbetrag_kinderlos: bool,
    sozialversicherung__beitragspflichtige_einnahmen_aus_midijob_arbeitnehmer_m: float,
    sozialversicherung__midijob_bemessungsentgelt_m: float,
    beitragssatz_abhängig_von_anzahl_kinder: dict[str, float],
) -> float:
    """Employee's long-term care insurance contribution between October 2022 and
    June 2023.
    """
    # Calculate the employee care insurance contribution
    an_beitr_midijob_m = (
        sozialversicherung__beitragspflichtige_einnahmen_aus_midijob_arbeitnehmer_m
        * beitragssatz_abhängig_von_anzahl_kinder["standard"]
    )

    # Add additional contribution for childless individuals
    if zusatzbetrag_kinderlos:
        an_beitr_midijob_m += (
            sozialversicherung__midijob_bemessungsentgelt_m
            * beitragssatz_abhängig_von_anzahl_kinder["zusatz_kinderlos"]
        )

    return an_beitr_midijob_m


@policy_function(
    start_date="2023-07-01",
    leaf_name="betrag_versicherter_midijob_m",
    vectorization_strategy="loop",
)
def betrag_versicherter_midijob_m_mit_verringertem_beitrag_für_eltern_mit_mehreren_kindern(
    anzahl_kinder_bis_24: int,
    zusatzbetrag_kinderlos: bool,
    sozialversicherung__beitragspflichtige_einnahmen_aus_midijob_arbeitnehmer_m: float,
    sozialversicherung__midijob_bemessungsentgelt_m: float,
    beitragssatz_abhängig_von_anzahl_kinder: dict[str, float],
) -> float:
    """Employee's long-term care insurance contribution since July 2023."""
    # Calculate the employee care insurance rate
    ges_pflegev_rate = beitragssatz_abhängig_von_anzahl_kinder["standard"]

    # Reduced contribution for individuals with two or more children under 25
    if anzahl_kinder_bis_24 >= 2:
        ges_pflegev_rate -= beitragssatz_abhängig_von_anzahl_kinder[
            "abschlag_kinder"
        ] * min(anzahl_kinder_bis_24 - 1, 4)

    # Calculate the employee care insurance contribution
    an_beitr_midijob_m = (
        sozialversicherung__beitragspflichtige_einnahmen_aus_midijob_arbeitnehmer_m
        * ges_pflegev_rate
    )

    # Add additional contribution for childless individuals
    if zusatzbetrag_kinderlos:
        an_beitr_midijob_m += (
            sozialversicherung__midijob_bemessungsentgelt_m
            * beitragssatz_abhängig_von_anzahl_kinder["zusatz_kinderlos"]
        )

    return an_beitr_midijob_m
