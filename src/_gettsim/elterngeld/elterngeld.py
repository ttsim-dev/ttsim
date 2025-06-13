"""Elterngeld (parental leave benefit)."""

from __future__ import annotations

from ttsim.tt_dag_elements import (
    AggType,
    RoundingSpec,
    agg_by_group_function,
    agg_by_p_id_function,
    policy_function,
)


@agg_by_group_function(agg_type=AggType.ANY)
def kind_grundsätzlich_anspruchsberechtigt_fg(
    kind_grundsätzlich_anspruchsberechtigt: bool,
    fg_id: int,
) -> bool:
    pass


@agg_by_group_function(agg_type=AggType.SUM)
def anzahl_anträge_fg(claimed: bool, fg_id: int) -> int:
    pass


@agg_by_p_id_function(agg_type=AggType.SUM)
def bezugsmonate_partner(
    bisherige_bezugsmonate: int,
    arbeitslosengeld_2__p_id_einstandspartner: int,
    p_id: int,
) -> int:
    pass


@agg_by_group_function(agg_type=AggType.MIN)
def alter_monate_jüngstes_mitglied_fg(alter_monate: int, fg_id: int) -> float:
    pass


@agg_by_group_function(agg_type=AggType.SUM)
def anzahl_kinder_bis_2_fg(familie__kind_bis_2: bool, fg_id: int) -> int:
    pass


@agg_by_group_function(agg_type=AggType.SUM)
def anzahl_kinder_bis_5_fg(familie__kind_bis_5: bool, fg_id: int) -> int:
    pass


@agg_by_group_function(agg_type=AggType.SUM)
def anzahl_mehrlinge_jüngstes_kind_fg(
    jüngstes_kind_oder_mehrling: bool,
    fg_id: int,
) -> int:
    pass


@policy_function(
    start_date="2011-01-01",
    rounding_spec=RoundingSpec(base=0.01, direction="down"),
)
def betrag_m(
    grundsätzlich_anspruchsberechtigt: bool,
    anspruchshöhe_m: float,
) -> float:
    """Parental leave benefit (Elterngeld) received by the parent."""
    if grundsätzlich_anspruchsberechtigt:
        out = anspruchshöhe_m
    else:
        out = 0.0
    return out


@policy_function(start_date="2007-01-01")
def basisbetrag_m(
    nettoeinkommen_vorjahr_m: float,
    lohnersatzanteil: float,
    anzurechnendes_nettoeinkommen_m: float,
    max_zu_berücksichtigendes_einkommen: float,
) -> float:
    """Base parental leave benefit without accounting for floor and ceiling.

    Basisbetrag is calculated on the parental level.

    """
    berücksichtigtes_einkommen = min(
        nettoeinkommen_vorjahr_m,
        max_zu_berücksichtigendes_einkommen,
    )
    return (
        berücksichtigtes_einkommen - anzurechnendes_nettoeinkommen_m
    ) * lohnersatzanteil


@policy_function(
    start_date="2007-01-01",
    end_date="2010-12-31",
    leaf_name="betrag_m",
    rounding_spec=RoundingSpec(base=0.01, direction="down"),
)
def elterngeld_not_implemented() -> float:
    raise NotImplementedError("Elterngeld is not implemented prior to 2011.")


@policy_function(start_date="2007-01-01")
def anspruchshöhe_m(
    basisbetrag_m: float,
    geschwisterbonus_m: float,
    mehrlingsbonus_m: float,
    mindestbetrag: float,
    höchstbetrag: float,
) -> float:
    """Elterngeld before checking eligibility.

    Anspruchshöhe is calculated on the parental level.

    """
    return (
        min(
            max(basisbetrag_m, mindestbetrag),
            höchstbetrag,
        )
        + geschwisterbonus_m
        + mehrlingsbonus_m
    )


@policy_function(
    start_date="2007-01-01",
    end_date="2010-12-31",
    leaf_name="grundsätzlich_anspruchsberechtigt",
)
def grundsätzlich_anspruchsberechtigt_ohne_maximales_vorjahreseinkommen(
    claimed: bool,
    arbeitsstunden_w: float,
    kind_grundsätzlich_anspruchsberechtigt_fg: bool,
    bezugsmonate_unter_grenze_fg: bool,
    max_arbeitsstunden_w: int,
) -> bool:
    """Parent is eligible to receive Elterngeld."""
    return (
        claimed
        and arbeitsstunden_w <= max_arbeitsstunden_w
        and kind_grundsätzlich_anspruchsberechtigt_fg
        and bezugsmonate_unter_grenze_fg
    )


@policy_function(start_date="2011-01-01", leaf_name="grundsätzlich_anspruchsberechtigt")
def grundsätzlich_anspruchsberechtigt_mit_maximales_vorjahreseinkommen(
    claimed: bool,
    arbeitsstunden_w: float,
    kind_grundsätzlich_anspruchsberechtigt_fg: bool,
    einkommen_vorjahr_unter_bezugsgrenze: bool,
    bezugsmonate_unter_grenze_fg: bool,
    max_arbeitsstunden_w: int,
) -> bool:
    """Parent is eligible to receive Elterngeld.

    Maximum income in the previous year introduced via § 1 (8) BEEG.
    """
    return (
        claimed
        and arbeitsstunden_w <= max_arbeitsstunden_w
        and einkommen_vorjahr_unter_bezugsgrenze
        and kind_grundsätzlich_anspruchsberechtigt_fg
        and bezugsmonate_unter_grenze_fg
    )


@policy_function(start_date="2007-01-01")
def bezugsmonate_unter_grenze_fg(
    bisherige_bezugsmonate_fg: int,
    bezugsmonate_partner: int,
    familie__alleinerziehend: bool,
    anzahl_anträge_fg: int,
    max_bezugsmonate: dict[str, int],
) -> bool:
    """Elterngeld claimed for less than the maximum number of months in the past by the
    parent.

    """
    if familie__alleinerziehend or bezugsmonate_partner >= 2:
        out = (
            bisherige_bezugsmonate_fg
            < max_bezugsmonate["basismonate"] + max_bezugsmonate["partnermonate"]
        )
    elif anzahl_anträge_fg > 1:
        out = (
            bisherige_bezugsmonate_fg + 1
            < max_bezugsmonate["basismonate"] + max_bezugsmonate["partnermonate"]
        )
    else:
        out = bisherige_bezugsmonate_fg < max_bezugsmonate["basismonate"]
    return out


@policy_function(start_date="2007-01-01")
def kind_grundsätzlich_anspruchsberechtigt(
    alter: int,
    max_bezugsmonate: dict[str, int],
) -> bool:
    """Child is young enough to give rise to Elterngeld claim."""
    return alter <= max_bezugsmonate["basismonate"] + max_bezugsmonate["partnermonate"]


@policy_function(start_date="2011-01-01")
def lohnersatzanteil(
    nettoeinkommen_vorjahr_m: float,
    lohnersatzanteil_einkommen_untere_grenze: float,
    lohnersatzanteil_einkommen_obere_grenze: float,
    einkommensschritte_korrektur: float,
    satz: float,
    prozent_korrektur: float,
    prozent_minimum: float,
    nettoeinkommensstufen_für_lohnersatzrate: dict[str, float],
) -> float:
    """Replacement rate of Elterngeld (before applying floor and ceiling rules).

    According to § 2 (2) BEEG the percentage increases below the first step and
    decreases above the second step until prozent_minimum.

    """
    # Higher replacement rate if considered income is below a threshold
    if (
        nettoeinkommen_vorjahr_m
        < nettoeinkommensstufen_für_lohnersatzrate["lower_threshold"]
        and nettoeinkommen_vorjahr_m > 0
    ):
        out = satz + (
            lohnersatzanteil_einkommen_untere_grenze
            / einkommensschritte_korrektur
            * prozent_korrektur
        )
    # Lower replacement rate if considered income is above a threshold
    elif (
        nettoeinkommen_vorjahr_m
        > nettoeinkommensstufen_für_lohnersatzrate["upper_threshold"]
    ):
        # Replacement rate is only lowered up to a specific value
        out = max(
            satz
            - (
                lohnersatzanteil_einkommen_obere_grenze
                / einkommensschritte_korrektur
                * prozent_korrektur
            ),
            prozent_minimum,
        )
    else:
        out = satz

    return out


# TODO(@MImmesberger): Elterngeld is considered as SGB II income since 2011. Also, there
# is a 300€ Freibetrag under some conditions since 2011.
# https://github.com/iza-institute-of-labor-economics/gettsim/issues/549
@policy_function(start_date="2007-01-01")
def anrechenbarer_betrag_m(
    betrag_m: float,
    anzahl_mehrlinge_fg: int,
    mindestbetrag: float,
) -> float:
    """Elterngeld that can be considered as income for other transfers.

    Relevant for Wohngeld and Grundsicherung im Alter.

    For Arbeitslosengeld II / Bürgergeld as well as Kinderzuschlag the whole amount of
    Elterngeld is considered as income, except for the case in which the parents still
    worked right before they had children. See:
    https://www.kindergeld.org/elterngeld-einkommen/


    """
    return max(
        betrag_m - ((1 + anzahl_mehrlinge_fg) * mindestbetrag),
        0,
    )


@policy_function()
def jüngstes_kind_oder_mehrling(
    alter_monate: int,
    alter_monate_jüngstes_mitglied_fg: float,
    familie__kind: bool,
) -> bool:
    """Check if person is the youngest child in the household or a twin, triplet, etc.
    of the youngest child.

    # ToDo: replace familie__kind by some age restriction
    # ToDo: Check definition as relevant for Elterngeld. Currently, it is calculated as
    # ToDo: age not being larger than 0.1 of a month

    """
    return ((alter_monate - alter_monate_jüngstes_mitglied_fg) < 0.1) and familie__kind
