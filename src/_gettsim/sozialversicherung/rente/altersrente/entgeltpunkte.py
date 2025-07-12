from __future__ import annotations

from ttsim.tt_dag_elements import policy_function


@policy_function(end_date="2023-06-30")
def entgeltpunkte_west_updated(
    wohnort_ost_hh: bool,
    sozialversicherung__rente__entgeltpunkte_west: float,
    neue_entgeltpunkte: float,
) -> float:
    """Updated Entgeltpunkte from West Germany based on current income.

    Given earnings, social insurance rules, average earnings in a particular year and
    potentially other variables (e.g., benefits for raising children, informal care),
    return the new earnings points.
    """
    if wohnort_ost_hh:
        out = sozialversicherung__rente__entgeltpunkte_west
    else:
        out = sozialversicherung__rente__entgeltpunkte_west + neue_entgeltpunkte
    return out


@policy_function(end_date="2023-06-30")
def entgeltpunkte_ost_updated(
    wohnort_ost_hh: bool,
    sozialversicherung__rente__entgeltpunkte_ost: float,
    neue_entgeltpunkte: float,
) -> float:
    """Updated Entgeltpunkte from East Germany based on current income.

    Given earnings, social insurance rules, average earnings in a particular year and
    potentially other variables (e.g., benefits for raising children, informal care),
    return the new earnings points.
    """
    if wohnort_ost_hh:
        out = sozialversicherung__rente__entgeltpunkte_ost + neue_entgeltpunkte
    else:
        out = sozialversicherung__rente__entgeltpunkte_ost
    return out


@policy_function(start_date="2023-07-01")
def entgeltpunkte_updated(
    sozialversicherung__rente__entgeltpunkte: float,
    neue_entgeltpunkte: float,
) -> float:
    """Updated Entgeltpunkte based on current income."""
    return sozialversicherung__rente__entgeltpunkte + neue_entgeltpunkte


@policy_function(end_date="2024-12-31", leaf_name="neue_entgeltpunkte")
def neue_entgeltpunkte_nach_wohnort(
    einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m: float,
    wohnort_ost_hh: bool,
    sozialversicherung__rente__beitrag__beitragsbemessungsgrenze_m: float,
    beitragspflichtiges_durchschnittsentgelt_y: float,
    umrechnung_entgeltpunkte_beitrittsgebiet: float,
) -> float:
    """Earning points for the wages earned in this year."""
    # Scale bruttolohn up if earned in eastern Germany
    if wohnort_ost_hh:
        umgerechneter_bruttolohn = (
            einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m
            * umrechnung_entgeltpunkte_beitrittsgebiet
        )
    else:
        umgerechneter_bruttolohn = (
            einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m
        )

    # Calculate the (scaled) wage, which is subject to pension contributions.
    if (
        umgerechneter_bruttolohn
        > sozialversicherung__rente__beitrag__beitragsbemessungsgrenze_m
    ):
        versicherungspflichtiger_bruttolohn = (
            sozialversicherung__rente__beitrag__beitragsbemessungsgrenze_m
        )
    else:
        versicherungspflichtiger_bruttolohn = umgerechneter_bruttolohn

    return versicherungspflichtiger_bruttolohn / (
        beitragspflichtiges_durchschnittsentgelt_y / 12
    )


@policy_function(start_date="2025-01-01", leaf_name="neue_entgeltpunkte")
def neue_entgeltpunkte_einheitlich(
    einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m: float,
    sozialversicherung__rente__beitrag__beitragsbemessungsgrenze_m: float,
    beitragspflichtiges_durchschnittsentgelt_y: float,
) -> float:
    """Earning points for the wages earned in this year."""
    if (
        einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m
        > sozialversicherung__rente__beitrag__beitragsbemessungsgrenze_m
    ):
        versicherungspflichtiger_bruttolohn = (
            sozialversicherung__rente__beitrag__beitragsbemessungsgrenze_m
        )
    else:
        versicherungspflichtiger_bruttolohn = (
            einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m
        )

    return versicherungspflichtiger_bruttolohn / (
        beitragspflichtiges_durchschnittsentgelt_y / 12
    )
