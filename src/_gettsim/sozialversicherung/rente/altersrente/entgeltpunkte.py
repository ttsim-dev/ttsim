from ttsim.tt_dag_elements import policy_function

# TODO(@MImmesberger): Do not distinguish between Entgeltpunkte from West and East
# Germany starting in July 2023.
# https://github.com/iza-institute-of-labor-economics/gettsim/issues/925


@policy_function()
def entgeltpunkte_west_updated(
    wohnort_ost: bool,
    sozialversicherung__rente__entgeltpunkte_west: float,
    neue_entgeltpunkte: float,
) -> float:
    """Updated Entgeltpunkte from West Germany based on current income.

    Given earnings, social insurance rules, average earnings in a particular year and
    potentially other variables (e.g., benefits for raising children, informal care),
    return the new earnings points.
    """
    if wohnort_ost:
        out = sozialversicherung__rente__entgeltpunkte_west
    else:
        out = sozialversicherung__rente__entgeltpunkte_west + neue_entgeltpunkte
    return out


@policy_function()
def entgeltpunkte_ost_updated(
    wohnort_ost: bool,
    sozialversicherung__rente__entgeltpunkte_ost: float,
    neue_entgeltpunkte: float,
) -> float:
    """Updated Entgeltpunkte from East Germany based on current income.

    Given earnings, social insurance rules, average earnings in a particular year and
    potentially other variables (e.g., benefits for raising children, informal care),
    return the new earnings points.
    """
    if wohnort_ost:
        out = sozialversicherung__rente__entgeltpunkte_ost + neue_entgeltpunkte
    else:
        out = sozialversicherung__rente__entgeltpunkte_ost
    return out


@policy_function()
def neue_entgeltpunkte(
    einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m: float,
    wohnort_ost: bool,
    sozialversicherung__rente__beitrag__beitragsbemessungsgrenze_m: float,
    beitragspflichtiges_durchschnittsentgelt_y: float,
    umrechnung_entgeltpunkte_beitrittsgebiet: float,
) -> float:
    """Return earning points for the wages earned in the last year."""

    # Scale bruttolohn up if earned in eastern Germany
    if wohnort_ost:
        bruttolohn_scaled_east = (
            einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m
            * umrechnung_entgeltpunkte_beitrittsgebiet
        )
    else:
        bruttolohn_scaled_east = (
            einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_m
        )

    # Calculate the (scaled) wage, which is subject to pension contributions.
    if (
        bruttolohn_scaled_east
        > sozialversicherung__rente__beitrag__beitragsbemessungsgrenze_m
    ):
        bruttolohn_scaled_rentenv = (
            sozialversicherung__rente__beitrag__beitragsbemessungsgrenze_m
        )
    else:
        bruttolohn_scaled_rentenv = bruttolohn_scaled_east

    return bruttolohn_scaled_rentenv / (beitragspflichtiges_durchschnittsentgelt_y / 12)
