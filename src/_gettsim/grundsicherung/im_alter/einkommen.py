"""Income considered in the calculation of Grundsicherung im Alter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ttsim import piecewise_polynomial, policy_function

if TYPE_CHECKING:
    from _gettsim.arbeitslosengeld_2.regelbedarf import RegelsatzNachRegelbedarfsstufen


@policy_function()
def einkommen_m(
    erwerbseinkommen_m: float,
    private_rente_betrag_m: float,
    gesetzliche_rente_m: float,
    einkommensteuer__einkünfte__sonstige__ohne_renten_m: float,
    einkommensteuer__einkünfte__aus_vermietung_und_verpachtung__betrag_m: float,
    kapitaleinkommen_brutto_m: float,
    einkommensteuer__betrag_m_sn: float,
    solidaritätszuschlag__betrag_m_sn: float,
    einkommensteuer__anzahl_personen_sn: int,
    sozialversicherung__beiträge_versicherter_m: float,
    elterngeld__anrechenbarer_betrag_m: float,
) -> float:
    """Calculate individual income considered in the calculation of Grundsicherung im
    Alter.
    """

    # Income
    total_income = (
        erwerbseinkommen_m
        + gesetzliche_rente_m
        + private_rente_betrag_m
        + einkommensteuer__einkünfte__sonstige__ohne_renten_m
        + einkommensteuer__einkünfte__aus_vermietung_und_verpachtung__betrag_m
        + kapitaleinkommen_brutto_m
        + elterngeld__anrechenbarer_betrag_m
    )

    out = (
        total_income
        - (einkommensteuer__betrag_m_sn / einkommensteuer__anzahl_personen_sn)
        - (solidaritätszuschlag__betrag_m_sn / einkommensteuer__anzahl_personen_sn)
        - sozialversicherung__beiträge_versicherter_m
    )

    return max(out, 0.0)


@policy_function(start_date="2011-01-01")
def erwerbseinkommen_m(
    einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m: float,
    einkommensteuer__einkünfte__aus_selbstständiger_arbeit__betrag_m: float,
    grunds_im_alter_params: dict,
    arbeitslosengeld_2__regelsatz_nach_regelbedarfsstufen: RegelsatzNachRegelbedarfsstufen,
) -> float:
    """Calculate individual earnings considered in the calculation of Grundsicherung im
    Alter.

    Legal reference: § 82 SGB XII Abs. 3

    Notes:

    - Freibeträge for income are currently not considered
    - Start date is 2011 because of the reference to regelsatz_nach_regelbedarfsstufen,
      which was introduced in 2011.
    - The cap at 1/2 of Regelbedarf was only introduced in 2006 (which is currently
      not implemented): https://www.buzer.de/gesetz/3415/al3764-0.htm
    """
    earnings = (
        einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m
        + einkommensteuer__einkünfte__aus_selbstständiger_arbeit__betrag_m
    )

    # Can deduct 30% of earnings (but no more than 1/2 of regelbedarf)
    earnings_after_max_deduction = (
        earnings
        - arbeitslosengeld_2__regelsatz_nach_regelbedarfsstufen.rbs_1.regelsatz / 2
    )
    earnings = (
        1 - grunds_im_alter_params["anrechnungsfreier_anteil_erwerbseinkünfte"]
    ) * earnings

    return max(earnings, earnings_after_max_deduction)


@policy_function()
def kapitaleinkommen_brutto_m(
    einkommensteuer__einkünfte__aus_kapitalvermögen__kapitalerträge_y: float,
    grunds_im_alter_params: dict,
) -> float:
    """Calculate individual capital income considered in the calculation of
    Grundsicherung im Alter.

    Legal reference: § 82 SGB XII Abs. 2
    """
    # Can deduct allowance from yearly capital income
    capital_income_y = (
        einkommensteuer__einkünfte__aus_kapitalvermögen__kapitalerträge_y
        - grunds_im_alter_params["anrechnungsfreie_kapitaleinkünfte"]
    )

    # Calculate and return monthly capital income (after deduction)
    return max(0.0, capital_income_y / 12)


@policy_function(start_date="2011-01-01")
def private_rente_betrag_m(
    sozialversicherung__rente__private_rente_betrag_m: float,
    grunds_im_alter_params: dict,
    arbeitslosengeld_2__regelsatz_nach_regelbedarfsstufen: RegelsatzNachRegelbedarfsstufen,
) -> float:
    """Calculate individual private pension benefits considered in the calculation of
    Grundsicherung im Alter.

    Legal reference: § 82 SGB XII Abs. 4
    """
    sozialversicherung__rente__private_rente_betrag_m_amount_exempt = (
        piecewise_polynomial(
            x=sozialversicherung__rente__private_rente_betrag_m,
            parameters=grunds_im_alter_params[
                "anrechnungsfreier_anteil_private_renteneinkünfte"
            ],
        )
    )
    upper = arbeitslosengeld_2__regelsatz_nach_regelbedarfsstufen.rbs_1.regelsatz / 2

    return sozialversicherung__rente__private_rente_betrag_m - min(
        sozialversicherung__rente__private_rente_betrag_m_amount_exempt, upper
    )


@policy_function(end_date="2020-12-31", leaf_name="gesetzliche_rente_m")
def gesetzliche_rente_m_bis_2020(
    sozialversicherung__rente__altersrente__betrag_m: float,
) -> float:
    """Calculate individual public pension benefits which are considered in the
    calculation of Grundsicherung im Alter until 2020.

    Until 2020: No deduction is possible.
    """
    return sozialversicherung__rente__altersrente__betrag_m


@policy_function(start_date="2021-01-01", leaf_name="gesetzliche_rente_m")
def gesetzliche_rente_m_ab_2021(
    sozialversicherung__rente__altersrente__betrag_m: float,
    sozialversicherung__rente__grundrente__grundsätzlich_anspruchsberechtigt: bool,
    arbeitslosengeld_2__regelsatz_nach_regelbedarfsstufen: RegelsatzNachRegelbedarfsstufen,
    grunds_im_alter_params: dict,
) -> float:
    """Calculate individual public pension benefits which are considered in the
    calculation of Grundsicherung im Alter since 2021.

    Starting from 2021: If eligible for Grundrente, can deduct 100€ completely and 30%
    of private pension above 100 (but no more than 1/2 of regelbedarf)
    """

    angerechnete_rente = piecewise_polynomial(
        x=sozialversicherung__rente__altersrente__betrag_m,
        parameters=grunds_im_alter_params["anrechnungsfreier_anteil_gesetzliche_rente"],
    )

    upper = arbeitslosengeld_2__regelsatz_nach_regelbedarfsstufen.rbs_1.regelsatz / 2
    if sozialversicherung__rente__grundrente__grundsätzlich_anspruchsberechtigt:
        angerechnete_rente = min(angerechnete_rente, upper)
    else:
        angerechnete_rente = 0.0

    return sozialversicherung__rente__altersrente__betrag_m - angerechnete_rente
