"""Unemployment benefits (Arbeitslosengeld)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ttsim import (
    get_consecutive_int_1d_lookup_table_param_value,
    param_function,
    piecewise_polynomial,
    policy_function,
)

if TYPE_CHECKING:
    from ttsim import (
        ConsecutiveInt1dLookupTableParamValue,
        PiecewisePolynomialParamValue,
    )


@policy_function(end_date="1998-07-31", leaf_name="betrag_m")
def betrag_m_not_implemented() -> float:
    """Calculate individual unemployment benefit."""
    raise NotImplementedError("Not implemented yet.")


@policy_function(start_date="1998-08-01")
def betrag_m(
    einkommensteuer__anzahl_kinderfreibeträge: int,
    grundsätzlich_anspruchsberechtigt: bool,
    einkommen_vorjahr_proxy_m: float,
    satz: dict[str, float],
) -> float:
    """Calculate individual unemployment benefit."""

    if einkommensteuer__anzahl_kinderfreibeträge == 0:
        arbeitsl_geld_satz = satz["allgemein"]
    else:
        arbeitsl_geld_satz = satz["erhöht"]

    if grundsätzlich_anspruchsberechtigt:
        out = einkommen_vorjahr_proxy_m * arbeitsl_geld_satz
    else:
        out = 0.0

    return out


@policy_function()
def monate_verbleibender_anspruchsdauer(
    alter: int,
    monate_sozialversicherungspflichtiger_beschäftigung_in_letzten_5_jahren: float,
    min_12_monate_beitragspflichtig_versichert_in_letzten_30_monaten: bool,
    monate_durchgängigen_bezugs_von_arbeitslosengeld: float,
    anspruchsdauer_nach_alter: ConsecutiveInt1dLookupTableParamValue,
    anspruchsdauer_nach_versicherungspflichtigen_monaten: ConsecutiveInt1dLookupTableParamValue,
) -> float:
    """Remaining amount of months of potential unemployment benefit claims."""
    auf_altersbasis = anspruchsdauer_nach_alter.values_to_look_up[
        alter - anspruchsdauer_nach_alter.base_to_subtract
    ]
    auf_basis_versicherungspflichtiger_monate = (
        anspruchsdauer_nach_versicherungspflichtigen_monaten.values_to_look_up[
            monate_sozialversicherungspflichtiger_beschäftigung_in_letzten_5_jahren
            - anspruchsdauer_nach_versicherungspflichtigen_monaten.base_to_subtract
        ]
    )

    if min_12_monate_beitragspflichtig_versichert_in_letzten_30_monaten:
        out = max(
            min(auf_altersbasis, auf_basis_versicherungspflichtiger_monate)
            - monate_durchgängigen_bezugs_von_arbeitslosengeld,
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

    return (
        arbeitssuchend
        and (monate_verbleibender_anspruchsdauer > 0)
        and (alter < regelaltersgrenze)
        and (arbeitsstunden_w < stundengrenze)
    )


@policy_function()
def einkommen_vorjahr_proxy_m(
    sozialversicherung__rente__beitrag__beitragsbemessungsgrenze_m: float,
    einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_vorjahr_m: float,
    sozialversicherungspauschale: float,
    einkommensteuer__parameter_einkommensteuertarif: PiecewisePolynomialParamValue,
    einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__werbungskostenpauschale: float,
    solidaritätszuschlag__parameter_solidaritätszuschlag: PiecewisePolynomialParamValue,
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
    prox_tax = piecewise_polynomial(
        x=12 * max_wage
        - einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__werbungskostenpauschale,
        parameters=einkommensteuer__parameter_einkommensteuertarif,
    )
    prox_soli = piecewise_polynomial(
        x=prox_tax,
        parameters=solidaritätszuschlag__parameter_solidaritätszuschlag,
    )
    out = max_wage - prox_ssc - prox_tax / 12 - prox_soli / 12
    return max(out, 0.0)


@param_function(start_date="1997-03-24")
def anspruchsdauer_nach_alter(
    raw_anspruchsdauer_nach_alter: dict[int, int],
) -> ConsecutiveInt1dLookupTableParamValue:
    """Amount of potential months of unemployment benefit claims by age."""
    max_age = 120
    ages_in_spec = sorted(raw_anspruchsdauer_nach_alter.keys())

    full_spec: dict[int, int] = {}
    for a in range(max_age):
        if a not in ages_in_spec:
            # Find the highest age in raw_anspruchsdauer_nach_alter that is less than current age
            threshold_age_for_this_age = max(age for age in ages_in_spec if age < a)
            full_spec[a] = full_spec[threshold_age_for_this_age]
        else:
            full_spec[a] = raw_anspruchsdauer_nach_alter[a]

    return get_consecutive_int_1d_lookup_table_param_value(full_spec)


@param_function(start_date="1997-03-24")
def anspruchsdauer_nach_versicherungspflichtigen_monaten(
    raw_anspruchsdauer_nach_versicherungspflichtigen_monaten: dict[int, int],
) -> ConsecutiveInt1dLookupTableParamValue:
    """Amount of potential months of unemployment benefit claims by age."""
    max_age = 120
    ages_in_spec = sorted(
        raw_anspruchsdauer_nach_versicherungspflichtigen_monaten.keys()
    )

    full_spec: dict[int, int] = {}
    for a in range(max_age):
        if a not in ages_in_spec:
            # Find the highest age in raw_anspruchsdauer_nach_alter that is less than current age
            threshold_age_for_this_age = max(age for age in ages_in_spec if age < a)
            full_spec[a] = full_spec[threshold_age_for_this_age]
        else:
            full_spec[a] = raw_anspruchsdauer_nach_versicherungspflichtigen_monaten[a]

    return get_consecutive_int_1d_lookup_table_param_value(full_spec)
