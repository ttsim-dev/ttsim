"""Midijob."""

from ttsim import RoundingSpec, policy_function


@policy_function()
def beitragspflichtige_einnahmen_aus_midijob_arbeitnehmer_m(
    einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m: float,
    minijob_grenze: float,
    geringfügige_einkommen_params: dict,
) -> float:
    """Income subject to employee social insurance contributions for midijob since
    October 2022.

    Gesonderte Beitragspflichtige Einnahme is the reference income for midijobs subject
    to employee social insurance contribution.

    Legal reference: Changes in § 20 SGB IV from 01.10.2022
    """
    midijob_grenze = geringfügige_einkommen_params["grenzen_m"]["midijob"]

    quotient = midijob_grenze / (midijob_grenze - minijob_grenze)
    einkommen_diff = (
        einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m
        - minijob_grenze
    )

    return quotient * einkommen_diff


@policy_function(
    start_date="2003-04-01",
    end_date="2004-12-31",
    leaf_name="midijob_faktor_f",
    rounding_spec=RoundingSpec(base=0.0001, direction="nearest"),
)
def midijob_faktor_f_mit_minijob_steuerpauschale_bis_2004(
    sozialversicherung__kranken__beitrag__beitragssatz_arbeitnehmer_jahresanfang: float,
    sozialversicherung__kranken__beitrag__beitragssatz_arbeitgeber_jahresanfang: float,
    sozialversicherung__rente__beitrag__parameter_beitragssatz_jahresanfang: float,
    sozialversicherung__arbeitslosen__beitrag__parameter_beitragssatz_jahresanfang: float,
    sozialversicherung__pflege__beitrag__beitragssatz_einheitlich: float,
    geringfügige_einkommen_params: dict,
    sozialversicherung__kranken__beitrag__arbeitgeberpauschale_bei_geringfügiger_beschäftigung: float,
    sozialversicherung__rente__beitrag__arbeitgeberpauschale_bei_geringfügiger_beschäftigung: float,
) -> float:
    """Midijob Faktor F until December 2004.

    Legal reference: § 163 Abs. 10 SGB VI
    """
    # First calculate the factor F from the formula in § 163 (10) SGB VI
    # Therefore sum the contributions which are the same for employee and employer
    allg_sozialv_beitr = (
        sozialversicherung__rente__beitrag__parameter_beitragssatz_jahresanfang
        + sozialversicherung__arbeitslosen__beitrag__parameter_beitragssatz_jahresanfang
        + sozialversicherung__pflege__beitrag__beitragssatz_einheitlich
    )

    # Then calculate specific shares
    an_anteil = (
        allg_sozialv_beitr
        + sozialversicherung__kranken__beitrag__beitragssatz_arbeitnehmer_jahresanfang
    )
    ag_anteil = (
        allg_sozialv_beitr
        + sozialversicherung__kranken__beitrag__beitragssatz_arbeitgeber_jahresanfang
    )

    # Sum over the shares which are specific for midijobs.
    pausch_mini = (
        sozialversicherung__kranken__beitrag__arbeitgeberpauschale_bei_geringfügiger_beschäftigung
        + sozialversicherung__rente__beitrag__arbeitgeberpauschale_bei_geringfügiger_beschäftigung
        + geringfügige_einkommen_params["arbeitgeberpauschale_lohnsteuer"]
    )

    # Now calculate final factor
    return pausch_mini / (an_anteil + ag_anteil)


@policy_function(
    start_date="2005-01-01",
    end_date="2022-09-30",
    leaf_name="midijob_faktor_f",
    rounding_spec=RoundingSpec(base=0.0001, direction="nearest"),
)
def midijob_faktor_f_mit_minijob_steuerpauschale_ab_2005_bis_2022_09(
    sozialversicherung__kranken__beitrag__beitragssatz_arbeitnehmer_jahresanfang: float,
    sozialversicherung__kranken__beitrag__beitragssatz_arbeitgeber_jahresanfang: float,
    sozialversicherung__rente__beitrag__parameter_beitragssatz_jahresanfang: float,
    sozialversicherung__arbeitslosen__beitrag__parameter_beitragssatz_jahresanfang: float,
    sozialversicherung__pflege__beitrag__beitragssatz_abhängig_von_anzahl_kinder_jahresanfang: dict[
        str, float
    ],
    geringfügige_einkommen_params: dict,
    sozialversicherung__kranken__beitrag__arbeitgeberpauschale_bei_geringfügiger_beschäftigung: float,
    sozialversicherung__rente__beitrag__arbeitgeberpauschale_bei_geringfügiger_beschäftigung: float,
) -> float:
    """Midijob Faktor F between 2005 and September 2025.

    Legal reference: § 163 Abs. 10 SGB VI

    """
    # First calculate the factor F from the formula in § 163 (10) SGB VI
    # Therefore sum the contributions which are the same for employee and employer
    allg_sozialv_beitr = (
        sozialversicherung__rente__beitrag__parameter_beitragssatz_jahresanfang
        + sozialversicherung__arbeitslosen__beitrag__parameter_beitragssatz_jahresanfang
        + sozialversicherung__pflege__beitrag__beitragssatz_abhängig_von_anzahl_kinder_jahresanfang[
            "standard"
        ]
    )

    an_anteil = (
        allg_sozialv_beitr
        + sozialversicherung__kranken__beitrag__beitragssatz_arbeitnehmer_jahresanfang
    )
    ag_anteil = (
        allg_sozialv_beitr
        + sozialversicherung__kranken__beitrag__beitragssatz_arbeitgeber_jahresanfang
    )

    # Sum over the shares which are specific for midijobs.
    pausch_mini = (
        sozialversicherung__kranken__beitrag__arbeitgeberpauschale_bei_geringfügiger_beschäftigung
        + sozialversicherung__rente__beitrag__arbeitgeberpauschale_bei_geringfügiger_beschäftigung
        + geringfügige_einkommen_params["arbeitgeberpauschale_lohnsteuer"]
    )

    # Now calculate final factor
    return pausch_mini / (an_anteil + ag_anteil)


@policy_function(
    start_date="2022-10-01",
    leaf_name="midijob_faktor_f",
    rounding_spec=RoundingSpec(base=0.0001, direction="nearest"),
    vectorization_strategy="loop",
)
def midijob_faktor_f_ohne_minijob_steuerpauschale(
    sozialversicherung__kranken__beitrag__beitragssatz_arbeitnehmer_jahresanfang: float,
    sozialversicherung__kranken__beitrag__beitragssatz_arbeitgeber_jahresanfang: float,
    sozialversicherung__rente__beitrag__parameter_beitragssatz_jahresanfang: float,
    sozialversicherung__pflege__beitrag__beitragssatz_abhängig_von_anzahl_kinder_jahresanfang: dict[
        str, float
    ],
    sozialversicherung__arbeitslosen__beitrag__parameter_beitragssatz_jahresanfang: float,
    sozialversicherung__kranken__beitrag__arbeitgeberpauschale_bei_geringfügiger_beschäftigung: float,
    sozialversicherung__rente__beitrag__arbeitgeberpauschale_bei_geringfügiger_beschäftigung: float,
) -> float:
    """Midijob Faktor F since October 2022.

    Legal reference: § 163 Abs. 10 SGB VI
    """
    # Calculate the Gesamtsozialversicherungsbeitragssatz by summing social
    # insurance contributions for employer and employee and
    # adding the mean Zusatzbeitrag
    # First calculate the factor F from the formula in § 163 (10) SGB VI
    # Therefore sum the contributions which are the same for employee and employer
    allg_sozialv_beitr = (
        sozialversicherung__rente__beitrag__parameter_beitragssatz_jahresanfang
        + sozialversicherung__pflege__beitrag__beitragssatz_abhängig_von_anzahl_kinder_jahresanfang[
            "standard"
        ]
        + sozialversicherung__arbeitslosen__beitrag__parameter_beitragssatz_jahresanfang
    )

    # Then calculate specific shares
    an_anteil = (
        allg_sozialv_beitr
        + sozialversicherung__kranken__beitrag__beitragssatz_arbeitnehmer_jahresanfang
    )
    ag_anteil = (
        allg_sozialv_beitr
        + sozialversicherung__kranken__beitrag__beitragssatz_arbeitgeber_jahresanfang
    )

    # Sum over the shares which are specific for midijobs.
    # New formula only inludes the lump-sum contributions to health care
    # and pension insurance
    pausch_mini = (
        sozialversicherung__kranken__beitrag__arbeitgeberpauschale_bei_geringfügiger_beschäftigung
        + sozialversicherung__rente__beitrag__arbeitgeberpauschale_bei_geringfügiger_beschäftigung
    )

    # Now calculate final factor f
    return pausch_mini / (an_anteil + ag_anteil)


@policy_function(
    start_date="2003-04-01",
    end_date="2022-09-30",
    leaf_name="midijob_bemessungsentgelt_m",
)
def midijob_bemessungsentgelt_m_bis_09_2022(
    einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m: float,
    midijob_faktor_f: float,
    minijob_grenze: float,
    geringfügige_einkommen_params: dict,
) -> float:
    """Income subject to social insurance contributions for midijob until September
    2022.

    Bemessungsgeld (Gleitzonenentgelt) is the reference income for midijobs subject to
    social insurance contribution.

    Legal reference: § 163 Abs. 10 SGB VI

    """
    # Now use the factor to calculate the overall bemessungsentgelt
    minijob_anteil = midijob_faktor_f * minijob_grenze
    lohn_über_mini = (
        einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m
        - minijob_grenze
    )
    gewichtete_midijob_rate = (
        geringfügige_einkommen_params["grenzen_m"]["midijob"]
        / (geringfügige_einkommen_params["grenzen_m"]["midijob"] - minijob_grenze)
    ) - (
        minijob_grenze
        / (geringfügige_einkommen_params["grenzen_m"]["midijob"] - minijob_grenze)
        * midijob_faktor_f
    )

    return minijob_anteil + lohn_über_mini * gewichtete_midijob_rate


@policy_function(start_date="2022-10-01", leaf_name="midijob_bemessungsentgelt_m")
def midijob_bemessungsentgelt_m_ab_10_2022(
    einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m: float,
    midijob_faktor_f: float,
    minijob_grenze: float,
    geringfügige_einkommen_params: dict,
) -> float:
    """Total income subject to social insurance contributions for midijobs since October
    2022.

    In the law, the considered income is referred to as "beitragspflichtige Einnahme".

    Beitragspflichtige Einnahme is the reference income for midijobs subject to employer
    and employee social insurance contribution.

    Legal reference: Changes in § 20 SGB IV from 01.10.2022

    """
    midijob_grenze = geringfügige_einkommen_params["grenzen_m"]["midijob"]

    quotient1 = (midijob_grenze) / (midijob_grenze - minijob_grenze)
    quotient2 = (minijob_grenze) / (midijob_grenze - minijob_grenze)
    einkommen_diff = (
        einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m
        - minijob_grenze
    )

    faktor1 = midijob_faktor_f * minijob_grenze
    faktor2 = (quotient1 - quotient2 * midijob_faktor_f) * einkommen_diff

    return faktor1 + faktor2


@policy_function(start_date="2003-04-01")
def in_gleitzone(
    einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m: float,
    geringfügig_beschäftigt: bool,
    geringfügige_einkommen_params: dict,
) -> bool:
    """Individual's income is in midi-job range.

    Employed people with their wage in the range of gleitzone pay reduced social
    insurance contributions.

    Legal reference: § 20 Abs. 2 SGB IV

    """
    return (
        einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m
        <= geringfügige_einkommen_params["grenzen_m"]["midijob"]
    ) and (not geringfügig_beschäftigt)
