"""Functions to compute parental leave benefits (Erziehungsgeld, -2007)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ttsim import (
    AggType,
    RoundingSpec,
    agg_by_p_id_function,
    params_function,
    policy_function,
)


@dataclass(frozen=True)
class EinkommensgrenzeErziehungsgeldGegebenFamilienstandNachSatzArt:
    regelsatz: float
    budgetsatz: float


@dataclass(frozen=True)
class EinkommensgrenzeErziehungsgeldNachFamilienstand:
    alleinerziehend: EinkommensgrenzeErziehungsgeldGegebenFamilienstandNachSatzArt
    paar: EinkommensgrenzeErziehungsgeldGegebenFamilienstandNachSatzArt


@dataclass(frozen=True)
class EinkommensgrenzeErziehungsgeld:
    regulär: EinkommensgrenzeErziehungsgeldNachFamilienstand
    reduziert: EinkommensgrenzeErziehungsgeldNachFamilienstand
    maximalalter_reguläres_limit_monate: int


@params_function(
    start_date="2004-02-09",
    end_date="2008-12-31",
)
def einkommensgrenze(
    parameter_einkommensgrenze: dict[str, Any],
) -> EinkommensgrenzeErziehungsgeld:
    """Parameter der Einkommensgrenze des Erziehungsgelds."""
    regulär = EinkommensgrenzeErziehungsgeldNachFamilienstand(
        alleinerziehend=EinkommensgrenzeErziehungsgeldGegebenFamilienstandNachSatzArt(
            regelsatz=parameter_einkommensgrenze["regulär"]["alleinerziehend"][
                "regelsatz"
            ],
            budgetsatz=parameter_einkommensgrenze["regulär"]["alleinerziehend"][
                "budgetsatz"
            ],
        ),
        paar=EinkommensgrenzeErziehungsgeldGegebenFamilienstandNachSatzArt(
            regelsatz=parameter_einkommensgrenze["regulär"]["paar"]["regelsatz"],
            budgetsatz=parameter_einkommensgrenze["regulär"]["paar"]["budgetsatz"],
        ),
    )
    reduziert = EinkommensgrenzeErziehungsgeldNachFamilienstand(
        alleinerziehend=EinkommensgrenzeErziehungsgeldGegebenFamilienstandNachSatzArt(
            regelsatz=parameter_einkommensgrenze["reduziert"]["alleinerziehend"][
                "regelsatz"
            ],
            budgetsatz=parameter_einkommensgrenze["reduziert"]["alleinerziehend"][
                "budgetsatz"
            ],
        ),
        paar=EinkommensgrenzeErziehungsgeldGegebenFamilienstandNachSatzArt(
            regelsatz=parameter_einkommensgrenze["reduziert"]["paar"]["regelsatz"],
            budgetsatz=parameter_einkommensgrenze["reduziert"]["paar"]["budgetsatz"],
        ),
    )
    return EinkommensgrenzeErziehungsgeld(
        regulär=regulär,
        reduziert=reduziert,
        maximalalter_reguläres_limit_monate=parameter_einkommensgrenze[
            "start_age_m_reduced_income_limit"
        ],
    )


@agg_by_p_id_function(agg_type=AggType.SUM)
def anspruchshöhe_m(
    anspruchshöhe_kind_m: float, p_id_empfänger: int, p_id: int
) -> float:
    pass


@policy_function(
    start_date="2004-01-01", end_date="2008-12-31", vectorization_strategy="loop"
)
def betrag_m(
    anspruchshöhe_m: int,
    grundsätzlich_anspruchsberechtigt: bool,
) -> float:
    """Total parental leave benefits (Erziehungsgeld) received by the parent.

    Legal reference: BErzGG (BGBl. I 1985 S. 2154; BGBl. I 2004 S. 206)
    """
    if grundsätzlich_anspruchsberechtigt:
        out: float = anspruchshöhe_m
    else:
        out = 0.0

    return out


@policy_function(
    end_date="2003-12-31",
    leaf_name="anspruchshöhe_kind_m",
    rounding_spec=RoundingSpec(base=0.01, direction="nearest"),
)
def erziehungsgeld_kind_ohne_budgetsatz_m() -> NotImplementedError:
    raise NotImplementedError(
        """
    Erziehungsgeld is not implemented yet prior to 2004, see
    https://github.com/iza-institute-of-labor-economics/gettsim/issues/673
        """
    )


@policy_function(
    start_date="2004-01-01",
    end_date="2008-12-31",
    leaf_name="anspruchshöhe_kind_m",
    rounding_spec=RoundingSpec(base=0.01, direction="nearest"),
)
def anspruchshöhe_kind_mit_budgetsatz_m(
    kind_grundsätzlich_anspruchsberechtigt: bool,
    abzug_durch_einkommen_m: float,
    basisbetrag_m: float,
) -> float:
    """Parental leave benefit (Erziehungsgeld) on child level.

    For the calculation, the relevant income, the age of the youngest child, the income
    threshold and the eligibility for erziehungsgeld is needed.

    Legal reference: Bundesgesetzblatt Jahrgang 2004 Teil I Nr. 6
    """
    if kind_grundsätzlich_anspruchsberechtigt:
        out = max(
            basisbetrag_m - abzug_durch_einkommen_m,
            0.0,
        )
    else:
        out = 0.0

    return out


@policy_function(start_date="2004-01-01", end_date="2008-12-31")
def basisbetrag_m(
    budgetsatz: bool,
    anzurechnendes_einkommen_y: float,
    einkommensgrenze_y: float,
    alter_monate: int,
    satz: dict[str, float],
    einkommensgrenze: EinkommensgrenzeErziehungsgeld,
) -> float:
    """Parental leave benefit (Erziehungsgeld) without means-test on child level."""
    # no benefit if income is above threshold and child is younger than threshold
    if (
        anzurechnendes_einkommen_y > einkommensgrenze_y
        and alter_monate < einkommensgrenze.maximalalter_reguläres_limit_monate
    ):
        out = 0.0
    elif budgetsatz:
        out = satz["budgetsatz"]
    else:
        out = satz["regelsatz"]

    return out


@policy_function(start_date="2004-01-01", end_date="2008-12-31")
def abzug_durch_einkommen_m(
    anzurechnendes_einkommen_m: float,
    einkommensgrenze_m: float,
    alter_monate: int,
    abschlagsfaktor: float,
    einkommensgrenze: EinkommensgrenzeErziehungsgeld,
) -> float:
    """Reduction of parental leave benefits (means-test).

    Legal reference: Bundesgesetzblatt Jahrgang 2004 Teil I Nr. 6 (p.209)
    """
    if (
        anzurechnendes_einkommen_m > einkommensgrenze_m
        and alter_monate >= einkommensgrenze.maximalalter_reguläres_limit_monate
    ):
        out = anzurechnendes_einkommen_m * abschlagsfaktor
    else:
        out = 0.0
    return out


@policy_function(
    start_date="2004-01-01",
    end_date="2006-12-10",
    leaf_name="kind_grundsätzlich_anspruchsberechtigt",
)
def _kind_grundsätzlich_anspruchsberechtigt_vor_abschaffung(
    familie__kind: bool,
    alter_monate: int,
    budgetsatz: bool,
    maximales_kindsalter_budgetsatz: float,
    maximales_kindsalter_regelsatz: float,
) -> bool:
    """Eligibility for parental leave benefit (Erziehungsgeld) on child level.

    Legal reference: Bundesgesetzblatt Jahrgang 2004 Teil I Nr. 6 (pp.207)
    """
    if budgetsatz:
        out = familie__kind and alter_monate <= maximales_kindsalter_budgetsatz

    else:
        out = familie__kind and alter_monate <= maximales_kindsalter_regelsatz

    return out


@policy_function(
    start_date="2006-12-11",
    end_date="2008-12-31",
    leaf_name="kind_grundsätzlich_anspruchsberechtigt",
)
def _kind_grundsätzlich_anspruchsberechtigt_nach_abschaffung(
    familie__kind: bool,
    geburtsjahr: int,
    alter_monate: int,
    budgetsatz: bool,
    abolishment_cohort: int,
    maximales_kindsalter_budgetsatz: float,
    maximales_kindsalter_regelsatz: float,
) -> bool:
    """Eligibility for parental leave benefit (Erziehungsgeld) on child level. Abolished
    for children born after the cut-off date.

    Legal reference: Bundesgesetzblatt Jahrgang 2004 Teil I Nr. 6 (pp.207)
    """
    if budgetsatz and geburtsjahr <= abolishment_cohort:
        out = familie__kind and alter_monate <= maximales_kindsalter_budgetsatz

    elif geburtsjahr <= abolishment_cohort:
        out = familie__kind and alter_monate <= maximales_kindsalter_regelsatz

    else:
        out = False

    return out


@policy_function(start_date="2004-01-01", end_date="2008-12-31")
def grundsätzlich_anspruchsberechtigt(
    arbeitsstunden_w: float,
    kind_grundsätzlich_anspruchsberechtigt_fg: bool,
    maximale_wochenarbeitszeit: float,
) -> bool:
    """Eligibility for parental leave benefit (Erziehungsgeld) on parental level.

    Legal reference: Bundesgesetzblatt Jahrgang 2004 Teil I Nr. 6 (p.207)
    """
    return kind_grundsätzlich_anspruchsberechtigt_fg and (
        arbeitsstunden_w <= maximale_wochenarbeitszeit
    )


@policy_function(start_date="2004-01-01", end_date="2008-12-31")
def anzurechnendes_einkommen_y(
    einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_vorjahr_y_fg: float,
    arbeitslosengeld_2__anzahl_erwachsene_fg: int,
    kind_grundsätzlich_anspruchsberechtigt: bool,
    pauschaler_abzug_vom_einkommen: float,
    einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__werbungskostenpauschale: float,
) -> float:
    """Income relevant for means testing for parental leave benefit (Erziehungsgeld).

    Legal reference: Bundesgesetzblatt Jahrgang 2004 Teil I Nr. 6 (p.209)

    There is special rule for "Beamte, Soldaten und Richter" which is not
    implemented yet.
    """

    if kind_grundsätzlich_anspruchsberechtigt:
        out = (
            einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__bruttolohn_vorjahr_y_fg
            - einkommensteuer__einkünfte__aus_nichtselbstständiger_arbeit__werbungskostenpauschale
            * arbeitslosengeld_2__anzahl_erwachsene_fg
        ) * pauschaler_abzug_vom_einkommen
    else:
        out = 0.0
    return out


@policy_function(
    start_date="2004-01-01", end_date="2008-12-31", vectorization_strategy="loop"
)
def einkommensgrenze_y(
    einkommensgrenze_ohne_geschwisterbonus: float,
    arbeitslosengeld_2__anzahl_kinder_fg: float,
    kind_grundsätzlich_anspruchsberechtigt: bool,
    aufschlag_einkommen: float,
) -> float:
    """Income threshold for parental leave benefit (Erziehungsgeld).

    Legal reference: Bundesgesetzblatt Jahrgang 2004 Teil I Nr. 6 (pp.208)
    """

    out = (
        einkommensgrenze_ohne_geschwisterbonus
        + (arbeitslosengeld_2__anzahl_kinder_fg - 1) * aufschlag_einkommen
    )
    if not kind_grundsätzlich_anspruchsberechtigt:
        out = 0.0
    return out


@policy_function(
    start_date="2004-01-01", end_date="2008-12-31", vectorization_strategy="loop"
)
def einkommensgrenze_ohne_geschwisterbonus(
    alter_monate: int,
    einkommensgrenze_ohne_geschwisterbonus_kind_jünger_als_reduzierungsgrenze: float,
    einkommensgrenze_ohne_geschwisterbonus_kind_älter_als_reduzierungsgrenze: float,
    einkommensgrenze: EinkommensgrenzeErziehungsgeld,
) -> float:
    """Income threshold for parental leave benefit (Erziehungsgeld) before adding the
    bonus for additional children.

    Legal reference: Bundesgesetzblatt Jahrgang 2004 Teil I Nr. 6 (pp.208)
    """
    if alter_monate < einkommensgrenze.maximalalter_reguläres_limit_monate:
        return einkommensgrenze_ohne_geschwisterbonus_kind_jünger_als_reduzierungsgrenze
    else:
        return einkommensgrenze_ohne_geschwisterbonus_kind_älter_als_reduzierungsgrenze


@policy_function(
    start_date="2004-01-01", end_date="2008-12-31", vectorization_strategy="loop"
)
def einkommensgrenze_ohne_geschwisterbonus_kind_jünger_als_reduzierungsgrenze(
    familie__alleinerziehend_fg: bool,
    budgetsatz: bool,
    einkommensgrenze: EinkommensgrenzeErziehungsgeld,
) -> float:
    """Base income threshold for parents of children younger than the age threshold.

    Legal reference: Bundesgesetzblatt Jahrgang 2004 Teil I Nr. 6 (pp.208)
    """
    if budgetsatz and familie__alleinerziehend_fg:
        return einkommensgrenze.regulär.alleinerziehend.budgetsatz
    elif budgetsatz and not familie__alleinerziehend_fg:
        return einkommensgrenze.regulär.paar.budgetsatz
    elif not budgetsatz and familie__alleinerziehend_fg:
        return einkommensgrenze.regulär.alleinerziehend.regelsatz
    else:
        return einkommensgrenze.regulär.paar.regelsatz


@policy_function(
    start_date="2004-01-01", end_date="2008-12-31", vectorization_strategy="loop"
)
def einkommensgrenze_ohne_geschwisterbonus_kind_älter_als_reduzierungsgrenze(
    familie__alleinerziehend_fg: bool,
    budgetsatz: bool,
    einkommensgrenze: EinkommensgrenzeErziehungsgeld,
) -> float:
    """Base income threshold for parents of children older than age threshold.

    Legal reference: Bundesgesetzblatt Jahrgang 2004 Teil I Nr. 6 (pp.208)
    """
    if budgetsatz and familie__alleinerziehend_fg:
        return einkommensgrenze.reduziert.alleinerziehend.budgetsatz
    elif budgetsatz and not familie__alleinerziehend_fg:
        return einkommensgrenze.reduziert.paar.budgetsatz
    elif not budgetsatz and familie__alleinerziehend_fg:
        return einkommensgrenze.reduziert.alleinerziehend.regelsatz
    else:
        return einkommensgrenze.reduziert.paar.regelsatz


@agg_by_p_id_function(agg_type=AggType.SUM)
def erziehungsgeld_spec_target(
    erziehungsgeld_source_field: bool, p_id_field: int, p_id: int
) -> int:
    pass
