"""Tax allowances for special expenses."""

from ttsim import (
    AggType,
    RoundingSpec,
    agg_by_p_id_function,
    policy_function,
)


@agg_by_p_id_function(agg_type=AggType.SUM)
def betreuungskosten_elternteil_m(
    betreuungskosten_m: float, p_id_betreuungskostenträger: int, p_id: int
) -> float:
    pass


@policy_function(end_date="2011-12-31", leaf_name="sonderausgaben_y_sn")
def sonderausgaben_y_sn_nur_pauschale(
    einkommensteuer__anzahl_personen_sn: int,
    sonderausgabenpauschbetrag: float,
) -> float:
    """Sonderausgaben on Steuernummer level until 2011.

    Only a lump sum payment is implemented.


    """

    return sonderausgabenpauschbetrag * einkommensteuer__anzahl_personen_sn


@policy_function(start_date="2012-01-01", leaf_name="sonderausgaben_y_sn")
def sonderausgaben_y_sn_mit_betreuung(
    absetzbare_betreuungskosten_y_sn: float,
    einkommensteuer__anzahl_personen_sn: int,
    sonderausgabenpauschbetrag: float,
) -> float:
    """Sonderausgaben on Steuernummer level since 2012.

    We follow 10 Abs.1 Nr. 5 EStG. You can find
    details here https://www.buzer.de/s1.htm?a=10&g=estg.

    """

    return max(
        absetzbare_betreuungskosten_y_sn,
        sonderausgabenpauschbetrag * einkommensteuer__anzahl_personen_sn,
    )


@policy_function()
def ausgaben_für_betreuung_y(
    betreuungskosten_elternteil_y: float,
    eink_st_abzuege_params: dict,
) -> float:
    """Individual deductable childcare cost for each individual child under 14."""
    out = min(
        betreuungskosten_elternteil_y,
        eink_st_abzuege_params["maximal_absetzbare_kinderbetreuungskosten"],
    )
    return out


@policy_function(rounding_spec=RoundingSpec(base=1, direction="up"))
def absetzbare_betreuungskosten_y_sn(
    ausgaben_für_betreuung_y_sn: float,
    eink_st_abzuege_params: dict,
) -> float:
    """Sonderausgaben for childcare on Steuernummer level.

    We follow 10 Abs.1 Nr. 5 EStG. You can
    details here https://www.buzer.de/s1.htm?a=10&g=estg.



    """

    out = (
        ausgaben_für_betreuung_y_sn
        * eink_st_abzuege_params["anteil_absetzbare_kinderbetreuungskosten"]
    )

    return out
