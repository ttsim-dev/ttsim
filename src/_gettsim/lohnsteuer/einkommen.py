"""Income relevant for withholding tax on earnings (Lohnsteuer)."""

from typing import Any

from ttsim import RoundingSpec, params_function, piecewise_polynomial, policy_function
from ttsim.piecewise_polynomial import PiecewisePolynomialParameters


@policy_function(rounding_spec=RoundingSpec(base=1, direction="down"))
def einkommen_y(
    einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_y: float,
    steuerklasse: int,
    vorsorgepauschale_y: float,
    einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__werbungskostenpauschale: float,
    einkommensteuer__abzüge__alleinerziehendenfreibetrag_basis: float,
    einkommensteuer__abzüge__sonderausgabenpauschbetrag: float,
) -> float:
    """Calculate tax base for Lohnsteuer (withholding tax on earnings)."""
    if steuerklasse == 6:
        werbungskosten = 0.0
    else:
        werbungskosten = einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__werbungskostenpauschale

    if steuerklasse == 6:
        sonderausgaben = 0.0
    else:
        sonderausgaben = einkommensteuer__abzüge__sonderausgabenpauschbetrag

    if steuerklasse == 2:
        alleinerziehendenfreibetrag = (
            einkommensteuer__abzüge__alleinerziehendenfreibetrag_basis
        )
    else:
        alleinerziehendenfreibetrag = 0.0

    out = max(
        einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_y
        - werbungskosten
        - sonderausgaben
        - alleinerziehendenfreibetrag
        - vorsorgepauschale_y,
        0.0,
    )

    return out


@policy_function(start_date="2010-01-01")
def vorsorge_krankenversicherungsbeiträge_option_a(
    sozialversicherung__kranken__beitrag__einkommen_regulär_beschäftigt_y: float,
    steuerklasse: int,
    vorsorgepauschale_mindestanteil: float,
    maximal_absetzbare_krankenversicherungskosten: dict[str, float],
) -> float:
    """Option a for calculating deductible health insurance contributions.

    This function calculates option a where at least 12% of earnings can be deducted,
    but only up to a certain threshold.

    """

    vorsorge_krankenversicherungsbeiträge_option_a_basis = (
        vorsorgepauschale_mindestanteil
        * sozialversicherung__kranken__beitrag__einkommen_regulär_beschäftigt_y
    )

    if steuerklasse == 3:
        vorsorge_krankenversicherungsbeiträge_option_a_max = (
            maximal_absetzbare_krankenversicherungskosten["steuerklasse_3"]
        )
    else:
        vorsorge_krankenversicherungsbeiträge_option_a_max = (
            maximal_absetzbare_krankenversicherungskosten["steuerklasse_nicht_3"]
        )

    out = min(
        vorsorge_krankenversicherungsbeiträge_option_a_max,
        vorsorge_krankenversicherungsbeiträge_option_a_basis,
    )

    return out


@policy_function(
    start_date="2015-01-01",
    end_date="2018-12-31",
    leaf_name="vorsorge_krankenversicherungsbeiträge_option_b",
)
def vorsorge_krankenversicherungsbeiträge_option_b_ab_2015_bis_2018(
    sozialversicherung__kranken__beitrag__einkommen_regulär_beschäftigt_y: float,
    sozialversicherung__kranken__beitrag__zusatzbeitragssatz: float,
    sozialversicherung__pflege__beitrag__beitragssatz: float,
    sozialversicherung__kranken__beitrag__parameter_beitragssatz: dict,
) -> float:
    """Option b for calculating deductible health insurance cont.

    For health care deductions, there are two ways to calculate the deductions: "Option
    a" and "Option b". This function calculates option b where the actual contributions
    are used.
    """
    out = sozialversicherung__kranken__beitrag__einkommen_regulär_beschäftigt_y * (
        sozialversicherung__kranken__beitrag__parameter_beitragssatz["ermäßigt"] / 2
        + sozialversicherung__kranken__beitrag__zusatzbeitragssatz
        + sozialversicherung__pflege__beitrag__beitragssatz
    )

    return out


@policy_function(
    start_date="2019-01-01",
    leaf_name="vorsorge_krankenversicherungsbeiträge_option_b",
)
def vorsorge_krankenversicherungsbeiträge_option_b_ab_2019(
    sozialversicherung__kranken__beitrag__einkommen_regulär_beschäftigt_y: float,
    sozialversicherung__kranken__beitrag__zusatzbeitragssatz: float,
    sozialversicherung__pflege__beitrag__beitragssatz: float,
    sozialversicherung__kranken__beitrag__parameter_beitragssatz: dict,
) -> float:
    """Option b for calculating deductible health insurance cont.

    For health care deductions, there are two ways to calculate the deductions: "Option
    a" and "Option b". This function calculates option b where the actual contributions
    are used.
    """

    out = sozialversicherung__kranken__beitrag__einkommen_regulär_beschäftigt_y * (
        sozialversicherung__kranken__beitrag__parameter_beitragssatz["ermäßigt"] / 2
        + sozialversicherung__kranken__beitrag__zusatzbeitragssatz / 2
        + sozialversicherung__pflege__beitrag__beitragssatz
    )

    return out


@params_function(start_date="2005-01-01", end_date="2022-12-31")
def einführungsfaktor_rentenversicherungsaufwendungen(
    evaluationsjahr: int,
    parameter_einführungsfaktor_rentenversicherungsaufwendungen: PiecewisePolynomialParameters,
) -> dict[str, Any]:
    """Calculate introductory factor for pension expense deductions which depends on the
    current year as follows:

    In the years 2005-2025 the share of deductible contributions increases by
    2 percentage points each year from 60% in 2005 to 100% in 2025.

    Reference: § 10 Abs. 1 Nr. 2 Buchst. a und b EStG


    """
    return piecewise_polynomial(
        x=evaluationsjahr,
        parameters=parameter_einführungsfaktor_rentenversicherungsaufwendungen,
    )


@policy_function(
    start_date="2010-01-01",
    end_date="2022-12-31",
    leaf_name="vorsorgepauschale_y",
    rounding_spec=RoundingSpec(base=1, direction="up"),
)
def vorsorgepauschale_y_ab_2010_bis_2022(
    sozialversicherung__rente__beitrag__einkommen_y: float,
    ges_rentenv_params: dict,
    vorsorge_krankenversicherungsbeiträge_option_a: float,
    vorsorge_krankenversicherungsbeiträge_option_b: float,
    einführungsfaktor_rentenversicherungsaufwendungen: float,
) -> float:
    """Calculate Vorsorgepauschale for Lohnsteuer valid since 2010. Those are deducted
    from gross earnings. Idea is similar, but not identical, to Vorsorgeaufwendungen
    used when calculating Einkommensteuer.

    """

    rente = (
        sozialversicherung__rente__beitrag__einkommen_y
        * ges_rentenv_params["parameter_beitragssatz"]
        * einführungsfaktor_rentenversicherungsaufwendungen
    )
    kranken = max(
        vorsorge_krankenversicherungsbeiträge_option_a,
        vorsorge_krankenversicherungsbeiträge_option_b,
    )

    return rente + kranken


@policy_function(
    start_date="2023-01-01",
    leaf_name="vorsorgepauschale_y",
    rounding_spec=RoundingSpec(base=1, direction="up"),
)
def vorsorgepauschale_y_ab_2023(
    sozialversicherung__rente__beitrag__einkommen_y: float,
    ges_rentenv_params: dict,
    vorsorge_krankenversicherungsbeiträge_option_a: float,
    vorsorge_krankenversicherungsbeiträge_option_b: float,
) -> float:
    """Calculate Vorsorgepauschale for Lohnsteuer valid since 2010. Those are deducted
    from gross earnings. Idea is similar, but not identical, to Vorsorgeaufwendungen
    used when calculating Einkommensteuer.

    """

    rente = (
        sozialversicherung__rente__beitrag__einkommen_y
        * ges_rentenv_params["parameter_beitragssatz"]
    )
    kranken = max(
        vorsorge_krankenversicherungsbeiträge_option_a,
        vorsorge_krankenversicherungsbeiträge_option_b,
    )

    return rente + kranken


@policy_function(
    start_date="2005-01-01",
    end_date="2009-12-31",
    leaf_name="vorsorgepauschale_y",
)
def vorsorgepauschale_y_ab_2005_bis_2009() -> float:
    return 0.0
