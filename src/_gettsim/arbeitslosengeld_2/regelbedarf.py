"""Regelbedarf for Arbeitslosengeld II."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ttsim import params_function, policy_function

if TYPE_CHECKING:
    from ttsim.typing import RawParamsRequiringConversion


@dataclass(frozen=True)
class RegelsatzAnteilErwachsen:
    je_erwachsener_bei_zwei_erwachsenen: float
    je_erwachsener_ab_drei_erwachsene: float


@dataclass(frozen=True)
class RegelsatzAnteilKind:
    anteil: float
    min_alter: int
    max_alter: int


@dataclass(frozen=True)
class RegelsatzAnteilKindNachAlter:
    kleinkind: RegelsatzAnteilKind
    schulkind: RegelsatzAnteilKind
    jugendliche_und_junge_erwachsene: RegelsatzAnteilKind


@dataclass(frozen=True)
class RegelsatzAnteilsbasiert:
    basissatz: float
    erwachsen: RegelsatzAnteilErwachsen
    kind: RegelsatzAnteilKindNachAlter


@params_function(start_date="2005-01-01", end_date="2010-12-31")
def regelsatz_anteilsbasiert(
    parameter_regelsatz_anteilsbasiert: RawParamsRequiringConversion,
) -> RegelsatzAnteilsbasiert:
    """Regelsatz as a fraction of the Basissatz."""
    anteilssätze_kinder = parameter_regelsatz_anteilsbasiert[
        "anteil_vom_basissatz_für_kinder"
    ]
    kind_kleinkind = RegelsatzAnteilKind(
        anteil=anteilssätze_kinder["kleinkind"]["anteil"],
        min_alter=anteilssätze_kinder["kleinkind"]["min_alter"],
        max_alter=anteilssätze_kinder["kleinkind"]["max_alter"],
    )
    kind_schulkind = RegelsatzAnteilKind(
        anteil=anteilssätze_kinder["schulkind"]["anteil"],
        min_alter=anteilssätze_kinder["schulkind"]["min_alter"],
        max_alter=anteilssätze_kinder["schulkind"]["max_alter"],
    )
    kind_jugendliche_und_junge_erwachsene = RegelsatzAnteilKind(
        anteil=anteilssätze_kinder["jugendliche_und_junge_erwachsene"]["anteil"],
        min_alter=anteilssätze_kinder["jugendliche_und_junge_erwachsene"]["min_alter"],
        max_alter=anteilssätze_kinder["jugendliche_und_junge_erwachsene"]["max_alter"],
    )
    erwachsen = RegelsatzAnteilErwachsen(
        je_erwachsener_bei_zwei_erwachsenen=parameter_regelsatz_anteilsbasiert[
            "anteil_vom_basissatz_bei_zwei_erwachsenen"
        ],
        je_erwachsener_ab_drei_erwachsene=parameter_regelsatz_anteilsbasiert[
            "anteil_vom_basissatz_bei_weiteren_erwachsenen"
        ],
    )
    return RegelsatzAnteilsbasiert(
        basissatz=parameter_regelsatz_anteilsbasiert["basissatz"],
        erwachsen=erwachsen,
        kind=RegelsatzAnteilKindNachAlter(
            kleinkind=kind_kleinkind,
            schulkind=kind_schulkind,
            jugendliche_und_junge_erwachsene=kind_jugendliche_und_junge_erwachsene,
        ),
    )


@dataclass(frozen=True)
class RegelbedarfsstufeErwachsener:
    regelsatz: float


@dataclass(frozen=True)
class RegelbedarfsstufeKind:
    regelsatz: float
    min_alter: int
    max_alter: int


@dataclass(frozen=True)
class RegelsatzNachRegelbedarfsstufen:
    """Regelsatz as a fraction of the Basissatz."""

    rbs_1: RegelbedarfsstufeErwachsener
    rbs_2: RegelbedarfsstufeErwachsener
    rbs_3: RegelbedarfsstufeErwachsener
    rbs_4: RegelbedarfsstufeKind
    rbs_5: RegelbedarfsstufeKind
    rbs_6: RegelbedarfsstufeKind


@params_function(start_date="2011-01-01")
def regelsatz_nach_regelbedarfsstufen(
    parameter_regelsatz_nach_regelbedarfsstufen: RawParamsRequiringConversion,
) -> RegelsatzNachRegelbedarfsstufen:
    """Regelsatz nach Regelbedarfsstufen."""
    rbs_4 = RegelbedarfsstufeKind(
        regelsatz=parameter_regelsatz_nach_regelbedarfsstufen[4]["betrag"],
        min_alter=parameter_regelsatz_nach_regelbedarfsstufen[4]["min_alter"],
        max_alter=parameter_regelsatz_nach_regelbedarfsstufen[4]["max_alter"],
    )
    rbs_5 = RegelbedarfsstufeKind(
        regelsatz=parameter_regelsatz_nach_regelbedarfsstufen[5]["betrag"],
        min_alter=parameter_regelsatz_nach_regelbedarfsstufen[5]["min_alter"],
        max_alter=parameter_regelsatz_nach_regelbedarfsstufen[5]["max_alter"],
    )
    rbs_6 = RegelbedarfsstufeKind(
        regelsatz=parameter_regelsatz_nach_regelbedarfsstufen[6]["betrag"],
        min_alter=parameter_regelsatz_nach_regelbedarfsstufen[6]["min_alter"],
        max_alter=parameter_regelsatz_nach_regelbedarfsstufen[6]["max_alter"],
    )
    return RegelsatzNachRegelbedarfsstufen(
        rbs_1=RegelbedarfsstufeErwachsener(
            parameter_regelsatz_nach_regelbedarfsstufen[1]
        ),
        rbs_2=RegelbedarfsstufeErwachsener(
            parameter_regelsatz_nach_regelbedarfsstufen[2]
        ),
        rbs_3=RegelbedarfsstufeErwachsener(
            parameter_regelsatz_nach_regelbedarfsstufen[3]
        ),
        rbs_4=rbs_4,
        rbs_5=rbs_5,
        rbs_6=rbs_6,
    )


@dataclass(frozen=True)
class BerechtigteWohnflächeEigentum:
    anzahl_personen_zu_fläche: dict[int, float]
    je_weitere_person: float
    max_anzahl_direkt: int


@params_function(start_date="2005-01-01")
def berechtigte_wohnfläche_eigentum(
    parameter_berechtigte_wohnfläche_eigentum: RawParamsRequiringConversion,
) -> BerechtigteWohnflächeEigentum:
    """Berechtigte Wohnfläche für Eigenheim."""
    return BerechtigteWohnflächeEigentum(
        anzahl_personen_zu_fläche={
            1: parameter_berechtigte_wohnfläche_eigentum[1],
            2: parameter_berechtigte_wohnfläche_eigentum[2],
            3: parameter_berechtigte_wohnfläche_eigentum[3],
            4: parameter_berechtigte_wohnfläche_eigentum[4],
        },
        je_weitere_person=parameter_berechtigte_wohnfläche_eigentum[
            "je_weitere_person"
        ],
        max_anzahl_direkt=parameter_berechtigte_wohnfläche_eigentum[
            "max_anzahl_direkt"
        ],
    )


@policy_function(start_date="2005-01-01")
def regelbedarf_m(
    regelsatz_m: float,
    kosten_der_unterkunft_m: float,
) -> float:
    """Basic monthly subsistence level on individual level.

    This includes cost of dwelling.

    Note: Since 2023, Arbeitslosengeld 2 is referred to as Bürgergeld.:
    """
    return regelsatz_m + kosten_der_unterkunft_m


@policy_function(start_date="2005-01-01", vectorization_strategy="loop")
def mehrbedarf_alleinerziehend_m(
    familie__alleinerziehend: bool,
    anzahl_kinder_fg: int,
    anzahl_kinder_bis_6_fg: int,
    anzahl_kinder_bis_15_fg: int,
    parameter_mehrbedarf_alleinerziehend: dict[str, float],
) -> float:
    """Compute additional SGB II need for single parents.

    Additional need for single parents. Maximum 60% of the standard amount on top if
    you have at least one kid below 6 or two or three below 15, you get 36%
    on top alternatively, you get 12% per kid, depending on what's higher.

    Note: Since 2023, Arbeitslosengeld 2 is referred to as Bürgergeld.
    """
    if familie__alleinerziehend:
        # Clip value at calculated minimal share and given upper share
        # Note that upper limit is applied last (for many children lower
        # could be greater than upper)
        out = min(
            max(
                # Minimal Mehrbedarf share. Minimal rate times number of children
                parameter_mehrbedarf_alleinerziehend["min_1_kind"] * anzahl_kinder_fg,
                # Increased rated if children up to 6 and/or 2-3 up to 15 are present.
                (
                    parameter_mehrbedarf_alleinerziehend[
                        "kind_bis_6_oder_mehrere_bis_15"
                    ]
                    if (anzahl_kinder_bis_6_fg >= 1)
                    or (2 <= anzahl_kinder_bis_15_fg <= 3)
                    else 0.0
                ),
            ),
            parameter_mehrbedarf_alleinerziehend["max"],
        )
    else:
        out = 0.0
    return out


@policy_function(
    start_date="2005-01-01", end_date="2010-12-31", leaf_name="kindersatz_m"
)
def kindersatz_m_anteilsbasiert(
    alter: int,
    kindergeld__gleiche_fg_wie_empfänger: bool,
    regelsatz_anteilsbasiert: RegelsatzAnteilsbasiert,
) -> float:
    """Basic monthly subsistence / SGB II needs of children until 2010."""
    basissatz = regelsatz_anteilsbasiert.basissatz

    if (
        alter
        >= regelsatz_anteilsbasiert.kind.jugendliche_und_junge_erwachsene.min_alter
        and alter
        <= regelsatz_anteilsbasiert.kind.jugendliche_und_junge_erwachsene.max_alter
        and kindergeld__gleiche_fg_wie_empfänger
    ):
        out = (
            basissatz
            * regelsatz_anteilsbasiert.kind.jugendliche_und_junge_erwachsene.anteil
        )
    elif (
        alter >= regelsatz_anteilsbasiert.kind.schulkind.min_alter
        and alter <= regelsatz_anteilsbasiert.kind.schulkind.max_alter
        and kindergeld__gleiche_fg_wie_empfänger
    ):
        out = basissatz * regelsatz_anteilsbasiert.kind.schulkind.anteil
    elif (
        alter >= regelsatz_anteilsbasiert.kind.kleinkind.min_alter
        and alter <= regelsatz_anteilsbasiert.kind.kleinkind.max_alter
        and kindergeld__gleiche_fg_wie_empfänger
    ):
        out = basissatz * regelsatz_anteilsbasiert.kind.kleinkind.anteil
    else:
        out = 0.0

    return out


@policy_function(
    start_date="2011-01-01",
    end_date="2022-06-30",
    leaf_name="kindersatz_m",
)
def kindersatz_m_nach_regelbedarfsstufen_ohne_sofortzuschlag(
    alter: int,
    kindergeld__gleiche_fg_wie_empfänger: bool,
    regelsatz_nach_regelbedarfsstufen: RegelsatzNachRegelbedarfsstufen,
) -> float:
    """Basic monthly subsistence / SGB II needs of children since 2011.

    Note: Since 2023, Arbeitslosengeld 2 is referred to as Bürgergeld.
    """
    if (
        alter >= regelsatz_nach_regelbedarfsstufen.rbs_6.min_alter
        and alter <= regelsatz_nach_regelbedarfsstufen.rbs_6.max_alter
        and kindergeld__gleiche_fg_wie_empfänger
    ):
        out = regelsatz_nach_regelbedarfsstufen.rbs_6.regelsatz
    elif (
        alter >= regelsatz_nach_regelbedarfsstufen.rbs_5.min_alter
        and alter <= regelsatz_nach_regelbedarfsstufen.rbs_5.max_alter
        and kindergeld__gleiche_fg_wie_empfänger
    ):
        out = regelsatz_nach_regelbedarfsstufen.rbs_5.regelsatz
    elif (
        alter >= regelsatz_nach_regelbedarfsstufen.rbs_4.min_alter
        and alter <= regelsatz_nach_regelbedarfsstufen.rbs_4.max_alter
        and kindergeld__gleiche_fg_wie_empfänger
    ):
        out = regelsatz_nach_regelbedarfsstufen.rbs_4.regelsatz
    elif kindergeld__gleiche_fg_wie_empfänger:  # adult children with parents in FG
        out = regelsatz_nach_regelbedarfsstufen.rbs_3.regelsatz
    else:
        out = 0.0

    return out


@policy_function(
    start_date="2022-07-01",
    leaf_name="kindersatz_m",
)
def kindersatz_m_nach_regelbedarfsstufen_mit_sofortzuschlag(
    alter: int,
    kindergeld__gleiche_fg_wie_empfänger: bool,
    regelsatz_nach_regelbedarfsstufen: RegelsatzNachRegelbedarfsstufen,
    kindersofortzuschlag: float,
) -> float:
    """Basic monthly subsistence / SGB II needs of children since 2011.

    Note: Since 2023, Arbeitslosengeld 2 is referred to as Bürgergeld.
    """
    if (
        alter >= regelsatz_nach_regelbedarfsstufen.rbs_6.min_alter
        and alter <= regelsatz_nach_regelbedarfsstufen.rbs_6.max_alter
        and kindergeld__gleiche_fg_wie_empfänger
    ):
        out = kindersofortzuschlag + regelsatz_nach_regelbedarfsstufen.rbs_6.regelsatz
    elif (
        alter >= regelsatz_nach_regelbedarfsstufen.rbs_5.min_alter
        and alter <= regelsatz_nach_regelbedarfsstufen.rbs_5.max_alter
        and kindergeld__gleiche_fg_wie_empfänger
    ):
        out = kindersofortzuschlag + regelsatz_nach_regelbedarfsstufen.rbs_5.regelsatz
    elif (
        alter >= regelsatz_nach_regelbedarfsstufen.rbs_4.min_alter
        and alter <= regelsatz_nach_regelbedarfsstufen.rbs_4.max_alter
        and kindergeld__gleiche_fg_wie_empfänger
    ):
        out = kindersofortzuschlag + regelsatz_nach_regelbedarfsstufen.rbs_4.regelsatz
    elif kindergeld__gleiche_fg_wie_empfänger:  # adult children with parents in FG
        out = kindersofortzuschlag + regelsatz_nach_regelbedarfsstufen.rbs_3.regelsatz
    else:
        out = 0.0

    return out


@policy_function(
    start_date="2005-01-01",
    end_date="2010-12-31",
    leaf_name="erwachsenensatz_m",
)
def arbeitsl_geld_2_erwachsenensatz_m_bis_2010(
    mehrbedarf_alleinerziehend_m: float,
    kindersatz_m: float,
    p_id_einstandspartner: int,
    regelsatz_anteilsbasiert: RegelsatzAnteilsbasiert,
) -> float:
    """Basic monthly subsistence / SGB II needs for adults without dwelling."""
    # BG has 2 adults
    if p_id_einstandspartner >= 0:
        out = regelsatz_anteilsbasiert.basissatz * (
            regelsatz_anteilsbasiert.erwachsen.je_erwachsener_bei_zwei_erwachsenen
        )
    # This observation is not a child, so BG has 1 adult
    elif kindersatz_m == 0.0:
        out = regelsatz_anteilsbasiert.basissatz
    else:
        out = 0.0

    return out * (1 + mehrbedarf_alleinerziehend_m)


@policy_function(
    start_date="2011-01-01",
    leaf_name="erwachsenensatz_m",
)
def arbeitsl_geld_2_erwachsenensatz_m_ab_2011(
    mehrbedarf_alleinerziehend_m: float,
    kindersatz_m: float,
    p_id_einstandspartner: int,
    regelsatz_nach_regelbedarfsstufen: RegelsatzNachRegelbedarfsstufen,
) -> float:
    """Basic monthly subsistence / SGB II needs for adults without dwelling since 2011.

    Note: Since 2023, Arbeitslosengeld 2 is referred to as Bürgergeld.
    """
    # BG has 2 adults
    if p_id_einstandspartner >= 0:
        out = regelsatz_nach_regelbedarfsstufen.rbs_2.regelsatz
    # This observation is not a child, so BG has 1 adult
    elif kindersatz_m == 0.0:
        out = regelsatz_nach_regelbedarfsstufen.rbs_1.regelsatz
    else:
        out = 0.0

    return out * (1 + mehrbedarf_alleinerziehend_m)


@policy_function(start_date="2005-01-01")
def regelsatz_m(
    erwachsenensatz_m: float,
    kindersatz_m: float,
) -> float:
    """Calculate basic monthly subsistence without dwelling until 2010."""
    return erwachsenensatz_m + kindersatz_m


@policy_function(
    start_date="2005-01-01",
    end_date="2022-12-31",
    leaf_name="kosten_der_unterkunft_m",
)
def kosten_der_unterkunft_m_bis_2022(
    berechtigte_wohnfläche: float,
    anerkannte_warmmiete_je_qm_m: float,
) -> float:
    """Calculate costs of living eligible to claim until 2022.

    Note: Since 2023, Arbeitslosengeld 2 is referred to as Bürgergeld.
    """
    return berechtigte_wohnfläche * anerkannte_warmmiete_je_qm_m


@policy_function(
    start_date="2023-01-01",
    leaf_name="kosten_der_unterkunft_m",
)
def kosten_der_unterkunft_m_ab_2023(
    bruttokaltmiete_m: float,
    heizkosten_m: float,
    arbeitslosengeld_2_bezug_im_vorjahr: bool,
    berechtigte_wohnfläche: float,
    anerkannte_warmmiete_je_qm_m: float,
) -> float:
    """Calculate costs of living eligible to claim since 2023. During the first year,
    the waiting period (Karenzzeit), only the appropriateness of the heating costs is
    tested, while the living costs are fully considered in Bürgergeld.

    Note: Since 2023, Arbeitslosengeld 2 is referred to as Bürgergeld.
    """
    if arbeitslosengeld_2_bezug_im_vorjahr:
        out = berechtigte_wohnfläche * anerkannte_warmmiete_je_qm_m
    else:
        out = bruttokaltmiete_m + heizkosten_m

    return out


@policy_function(start_date="2005-01-01")
def anerkannte_warmmiete_je_qm_m(
    bruttokaltmiete_m: float,
    heizkosten_m: float,
    wohnfläche: float,
    mietobergrenze_pro_qm: float,
) -> float:
    """Calculate rent per square meter.

    Note: Since 2023, Arbeitslosengeld 2 is referred to as Bürgergeld.
    """
    out = (bruttokaltmiete_m + heizkosten_m) / wohnfläche
    return min(out, mietobergrenze_pro_qm)


@policy_function(vectorization_strategy="loop", start_date="2005-01-01")
def berechtigte_wohnfläche(
    wohnfläche: float,
    wohnen__bewohnt_eigentum_hh: bool,
    anzahl_personen_hh: int,
    berechtigte_wohnfläche_miete: dict[str, float],
    berechtigte_wohnfläche_eigentum: BerechtigteWohnflächeEigentum,
) -> float:
    """Calculate size of dwelling eligible to claim.

    Note: Since 2023, Arbeitslosengeld 2 is referred to as Bürgergeld.
    """
    if wohnen__bewohnt_eigentum_hh:
        if anzahl_personen_hh <= berechtigte_wohnfläche_eigentum.max_anzahl_direkt:
            maximum = berechtigte_wohnfläche_eigentum.anzahl_personen_zu_fläche[
                anzahl_personen_hh
            ]
        else:
            maximum = (
                berechtigte_wohnfläche_eigentum.max_anzahl_direkt
                + (
                    anzahl_personen_hh
                    - berechtigte_wohnfläche_eigentum.max_anzahl_direkt
                )
                * berechtigte_wohnfläche_eigentum.je_weitere_person
            )
    else:
        maximum = (
            berechtigte_wohnfläche_miete["single"]
            + max(anzahl_personen_hh - 1, 0)
            * berechtigte_wohnfläche_miete["je_weitere_person"]
        )
    return min(wohnfläche, maximum / anzahl_personen_hh)


@policy_function(start_date="2005-01-01")
def bruttokaltmiete_m(
    wohnen__bruttokaltmiete_m_hh: float,
    anzahl_personen_hh: int,
) -> float:
    """Monthly rent attributed to a single person.

    Reference:
    BSG Urteil v. 09.03.2016 - B 14 KG 1/15 R.
    BSG Urteil vom 15.04.2008 - B 14/7b AS 58/06 R.
    """
    return wohnen__bruttokaltmiete_m_hh / anzahl_personen_hh


@policy_function(start_date="2005-01-01")
def heizkosten_m(
    wohnen__heizkosten_m_hh: float,
    anzahl_personen_hh: int,
) -> float:
    """Monthly heating expenses attributed to a single person.

    Reference:
    BSG Urteil v. 09.03.2016 - B 14 KG 1/15 R.
    BSG Urteil vom 15.04.2008 - B 14/7b AS 58/06 R.
    """
    return wohnen__heizkosten_m_hh / anzahl_personen_hh


@policy_function(start_date="2005-01-01")
def wohnfläche(
    wohnen__wohnfläche_hh: float,
    anzahl_personen_hh: int,
) -> float:
    """Share of household's dwelling size attributed to a single person."""
    return wohnen__wohnfläche_hh / anzahl_personen_hh
