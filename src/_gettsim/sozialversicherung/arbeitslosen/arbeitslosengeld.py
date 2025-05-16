"""Unemployment benefits (Arbeitslosengeld)."""

from __future__ import annotations

from _gettsim.einkommensteuer.einkommensteuer import einkommensteuertarif
from ttsim import PiecewisePolynomialParameters, piecewise_polynomial, policy_function


@policy_function(vectorization_strategy="loop")
def betrag_m(
    einkommensteuer__anzahl_kinderfreibeträge: int,
    grundsätzlich_anspruchsberechtigt: bool,
    einkommen_vorjahr_proxy_m: float,
    satz: dict[str, float],
) -> float:
    """Calculate individual unemployment benefit."""

    if einkommensteuer__anzahl_kinderfreibeträge == 0:
        arbeitsl_geld_satz = satz["allgemein"]
    elif einkommensteuer__anzahl_kinderfreibeträge > 0:
        arbeitsl_geld_satz = satz["erhöht"]

    if grundsätzlich_anspruchsberechtigt:
        out = einkommen_vorjahr_proxy_m * arbeitsl_geld_satz
    else:
        out = 0.0

    return out


@policy_function(vectorization_strategy="loop")
def monate_verbleibender_anspruchsdauer(
    alter: int,
    monate_sozialversicherungspflichtiger_beschäftigung_in_letzten_5_jahren: float,
    anwartschaftszeit: bool,
    monate_durchgängigen_bezugs_von_arbeitslosengeld: float,
    anspruchsdauer_nach_alter: PiecewisePolynomialParameters,
    anspruchsdauer_nach_versicherungspflichtigen_monaten: PiecewisePolynomialParameters,
) -> int:
    """Calculate the remaining amount of months a person can receive unemployment
    benefits.

    """
    nach_alter = piecewise_polynomial(
        alter,
        parameters=anspruchsdauer_nach_alter,
    )
    nach_versich_pfl = piecewise_polynomial(
        monate_sozialversicherungspflichtiger_beschäftigung_in_letzten_5_jahren,
        parameters=anspruchsdauer_nach_versicherungspflichtigen_monaten,
    )
    if anwartschaftszeit:
        anspruchsdauer_gesamt = min(nach_alter, nach_versich_pfl)

    if anwartschaftszeit:
        out = max(
            anspruchsdauer_gesamt - monate_durchgängigen_bezugs_von_arbeitslosengeld,
            0,
        )
    else:
        out = 0

    return out


@policy_function()
def grundsätzlich_anspruchsberechtigt(
    alter: int,
    arbeitssuchend: bool,
    monate_verbleibender_anspruchsdauer: int,
    arbeitsstunden_w: float,
    sozialversicherung__rente__altersrente__regelaltersrente__altersgrenze: float,
    stundengrenze: float,
) -> bool:
    """Check eligibility for unemployment benefit."""
    regelaltersgrenze = (
        sozialversicherung__rente__altersrente__regelaltersrente__altersgrenze
    )

    out = (
        arbeitssuchend
        and (monate_verbleibender_anspruchsdauer > 0)
        and (alter < regelaltersgrenze)
        and (arbeitsstunden_w < stundengrenze)
    )

    return out


@policy_function(vectorization_strategy="loop")
def einkommen_vorjahr_proxy_m(
    sozialversicherung__rente__beitrag__beitragsbemessungsgrenze_m: float,
    einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_vorjahr_m: float,
    sozialversicherungspauschale: float,
    einkommensteuer__parameter_einkommensteuertarif: PiecewisePolynomialParameters,
    einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__werbungskostenpauschale: float,
    solidaritätszuschlag__parameter_solidaritätszuschlag: PiecewisePolynomialParameters,
) -> float:
    """Approximate last years income for unemployment benefit."""
    # Relevant wage is capped at the contribution thresholds
    max_wage = min(
        einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_vorjahr_m,
        sozialversicherung__rente__beitrag__beitragsbemessungsgrenze_m,
    )

    # We need to deduct lump-sum amounts for contributions, taxes and soli
    prox_ssc = sozialversicherungspauschale * max_wage

    # Fictive taxes (Lohnsteuer) are approximated by applying the wage to the tax tariff
    # Caution: currently wrong calculation due to
    # 12 * max_wage - einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__werbungskostenpauschale not being
    # the same as zu versteuerndes einkommen
    # waiting for PR Lohnsteuer #150 to be merged to correct this problem
    prox_tax = einkommensteuertarif(
        12 * max_wage
        - einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__werbungskostenpauschale,
        einkommensteuer__parameter_einkommensteuertarif,
    )
    prox_soli = piecewise_polynomial(
        x=prox_tax,
        parameters=solidaritätszuschlag__parameter_solidaritätszuschlag,
    )
    out = max_wage - prox_ssc - prox_tax / 12 - prox_soli / 12
    out = max(out, 0.0)
    return out
