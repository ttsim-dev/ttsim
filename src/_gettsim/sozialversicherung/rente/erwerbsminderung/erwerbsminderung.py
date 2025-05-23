"""Public pension benefits for retirement due to reduced earnings potential."""

from __future__ import annotations

from ttsim import policy_function


@policy_function(start_date="2001-01-01", end_date="2023-06-30", leaf_name="betrag_m")
def betrag_m_nach_wohnort(
    zugangsfaktor: float,
    entgeltpunkte_west: float,
    entgeltpunkte_ost: float,
    rentenartfaktor: float,
    grundsätzlich_anspruchsberechtigt: bool,
    sozialversicherung__rente__altersrente__parameter_rentenwert_nach_wohnort: dict[
        str, float
    ],
) -> float:
    """Erwerbsminderungsrente (public disability insurance).

    Legal reference: SGB VI § 64: Rentenformel für Monatsbetrag der Rente
    """

    if grundsätzlich_anspruchsberechtigt:
        out = (
            (
                entgeltpunkte_west
                * sozialversicherung__rente__altersrente__parameter_rentenwert_nach_wohnort[
                    "west"
                ]
                + entgeltpunkte_ost
                * sozialversicherung__rente__altersrente__parameter_rentenwert_nach_wohnort[
                    "ost"
                ]
            )
            * zugangsfaktor
            * rentenartfaktor
        )
    else:
        out = 0.0
    return out


@policy_function(start_date="2023-07-01", leaf_name="betrag_m")
def betrag_m_einheitlich(
    zugangsfaktor: float,
    entgeltpunkte_west: float,
    entgeltpunkte_ost: float,
    rentenartfaktor: float,
    grundsätzlich_anspruchsberechtigt: bool,
    sozialversicherung__rente__altersrente__rentenwert: float,
) -> float:
    """Erwerbsminderungsrente (public disability insurance).

    Legal reference: SGB VI § 64: Rentenformel für Monatsbetrag der Rente
    """

    if grundsätzlich_anspruchsberechtigt:
        out = (
            (entgeltpunkte_ost + entgeltpunkte_west)
            * zugangsfaktor
            * sozialversicherung__rente__altersrente__rentenwert
            * rentenartfaktor
        )
    else:
        out = 0.0
    return out


@policy_function(start_date="2001-01-01")
def grundsätzlich_anspruchsberechtigt(
    voll_erwerbsgemindert: bool,
    teilweise_erwerbsgemindert: bool,
    sozialversicherung__rente__pflichtbeitragsmonate: float,
    sozialversicherung__rente__mindestwartezeit_erfüllt: bool,
) -> bool:
    """
    Eligibility for Erwerbsminderungsrente.

    Legal reference: § 43 Abs. 1  SGB VI.
    """

    anspruch_erwerbsm_rente = (
        (voll_erwerbsgemindert or teilweise_erwerbsgemindert)
        and sozialversicherung__rente__mindestwartezeit_erfüllt
        and sozialversicherung__rente__pflichtbeitragsmonate >= 36
    )

    return anspruch_erwerbsm_rente


@policy_function(start_date="2001-01-01")
def entgeltpunkte_west(
    sozialversicherung__rente__entgeltpunkte_west: float,
    zurechnungszeit: float,
    anteil_entgeltpunkte_ost: float,
) -> float:
    """Entgeltpunkte from West German Beitrags- and Zurechnungszeiten.

    In the case of the public disability insurance, pensioners are credited with
    additional earning points. They receive their average earned income points for each
    year between their age of retirement and the "zurechnungszeitgrenze".
    """

    return sozialversicherung__rente__entgeltpunkte_west + (
        zurechnungszeit * (1 - anteil_entgeltpunkte_ost)
    )


@policy_function(start_date="2001-01-01")
def entgeltpunkte_ost(
    sozialversicherung__rente__entgeltpunkte_ost: float,
    zurechnungszeit: float,
    anteil_entgeltpunkte_ost: float,
) -> float:
    """Entgeltpunkte from East German Beitrags- and Zurechnungszeiten.

    Provides the Entgeltpunkt-basis for calculation of the Erwerbsminderungsrente.

    In the case of the public disability insurance, pensioners are credited with
    additional earning points. They receive their average earned income points for each
    year between their age of retirement and the "zurechnungszeitgrenze".
    """

    return sozialversicherung__rente__entgeltpunkte_ost + (
        zurechnungszeit * anteil_entgeltpunkte_ost
    )


@policy_function(start_date="2001-01-01")
def zurechnungszeit(
    mean_entgeltpunkte_pro_bewertungsmonat: float,
    sozialversicherung__rente__alter_bei_renteneintritt: float,
    erwerbsm_rente_params: dict,
) -> float:
    """Additional Entgeltpunkte accumulated through "Zurechnungszeit".

    Provides the Entgeltpunkt-basis for calculation of the Erwerbsminderungsrente.

    In the case of the public disability insurance, pensioners are credited with
    additional earning points. They receive their average earned income points for each
    year between their age of retirement and the "zurechnungszeitgrenze".
    """
    zurechnungszeitgrenze = erwerbsm_rente_params["zurechnungszeitgrenze"]

    return (
        zurechnungszeitgrenze - (sozialversicherung__rente__alter_bei_renteneintritt)
    ) * mean_entgeltpunkte_pro_bewertungsmonat


@policy_function(start_date="2001-01-01")
def rentenartfaktor(
    teilweise_erwerbsgemindert: bool,
    erwerbsm_rente_params: dict,
) -> float:
    """Rentenartfaktor.

    Legal reference: SGB VI § 67: rentenartfaktor
    """

    if teilweise_erwerbsgemindert:
        out = erwerbsm_rente_params["rentenartfaktor"]["teilw"]

    else:
        out = erwerbsm_rente_params["rentenartfaktor"]["voll"]

    return out


@policy_function(start_date="2001-01-01")
def zugangsfaktor(
    sozialversicherung__rente__alter_bei_renteneintritt: float,
    wartezeit_langjährig_versichert_erfüllt: bool,
    erwerbsm_rente_params: dict,
    sozialversicherung__rente__altersrente__zugangsfaktor_veränderung_pro_jahr: dict[
        str, float
    ],
) -> float:
    """Zugangsfaktor.

    For each month that a pensioner retires before the age limit, a fraction of the
    pension is deducted. The maximum deduction is capped. This max deduction is the norm
    for the public disability insurance.

    Legal reference: § 77 Abs. 2-4  SGB VI

    Paragraph 4 regulates an exceptional case in which pensioners can already retire at
    63 without deductions if they can prove 40 years of (Pflichtbeiträge,
    Berücksichtigungszeiten and certain Anrechnungszeiten or Ersatzzeiten).
    """

    if wartezeit_langjährig_versichert_erfüllt:
        altersgrenze = erwerbsm_rente_params[
            "altersgrenze_langjährig_versicherte_abschlagsfrei"
        ]
    else:
        altersgrenze = erwerbsm_rente_params["parameter_altersgrenze"]

    zugangsfaktor = (
        1
        + (sozialversicherung__rente__alter_bei_renteneintritt - altersgrenze)
        * (
            sozialversicherung__rente__altersrente__zugangsfaktor_veränderung_pro_jahr[
                "vorzeitiger_renteneintritt"
            ]
        )
    )
    return max(zugangsfaktor, erwerbsm_rente_params["min_zugangsfaktor"])


# TODO(@MImmesberger): Reuse Altersrente Wartezeiten for Erwerbsminderungsrente
# https://github.com/iza-institute-of-labor-economics/gettsim/issues/838
@policy_function(start_date="2001-01-01")
def wartezeit_langjährig_versichert_erfüllt(
    sozialversicherung__rente__pflichtbeitragsmonate: float,
    sozialversicherung__rente__freiwillige_beitragsmonate: float,
    sozialversicherung__rente__anrechnungsmonate_45_jahre_wartezeit: float,
    sozialversicherung__rente__ersatzzeiten_monate: float,
    sozialversicherung__rente__kinderberücksichtigungszeiten_monate: float,
    sozialversicherung__rente__pflegeberücksichtigungszeiten_monate: float,
    sozialversicherung__rente__mindestpflichtbeitragsjahre_für_anrechenbarkeit_freiwilliger_beitragszeiten: float,
    erwerbsm_rente_params: dict,
) -> bool:
    """Wartezeit for Rente für langjährige Versicherte (Erwerbsminderung) is fulfilled.

    Eligibility criteria differ in comparison to Altersrente für langjährige
    Versicherte. In particular, freiwillige Beitragszeiten are not always considered (§
    51 Abs. 3a SGB VI).

    This pathway makes it possible to claim pension benefits without deductions at the
    age of 63.
    """
    if (
        sozialversicherung__rente__pflichtbeitragsmonate / 12
        >= sozialversicherung__rente__mindestpflichtbeitragsjahre_für_anrechenbarkeit_freiwilliger_beitragszeiten
    ):
        freiwillige_beitragszeiten = (
            sozialversicherung__rente__freiwillige_beitragsmonate
        )
    else:
        freiwillige_beitragszeiten = 0

    return (
        sozialversicherung__rente__pflichtbeitragsmonate
        + freiwillige_beitragszeiten
        + sozialversicherung__rente__anrechnungsmonate_45_jahre_wartezeit
        + sozialversicherung__rente__ersatzzeiten_monate
        + sozialversicherung__rente__pflegeberücksichtigungszeiten_monate
        + sozialversicherung__rente__kinderberücksichtigungszeiten_monate
    ) / 12 >= erwerbsm_rente_params[
        "wartezeitgrenze_langjährig_versicherte_abschlagsfrei"
    ]


@policy_function()
def anteil_entgeltpunkte_ost(
    sozialversicherung__rente__entgeltpunkte_west: float,
    sozialversicherung__rente__entgeltpunkte_ost: float,
) -> float:
    """Proportion of Entgeltpunkte accumulated in East Germany."""
    if (
        sozialversicherung__rente__entgeltpunkte_west
        == sozialversicherung__rente__entgeltpunkte_ost
        == 0.0
    ):
        out = 0.0
    else:
        out = sozialversicherung__rente__entgeltpunkte_ost / (
            sozialversicherung__rente__entgeltpunkte_west
            + sozialversicherung__rente__entgeltpunkte_ost
        )

    return out


@policy_function()
def mean_entgeltpunkte_pro_bewertungsmonat(
    sozialversicherung__rente__entgeltpunkte_west: float,
    sozialversicherung__rente__entgeltpunkte_ost: float,
    sozialversicherung__rente__alter_bei_renteneintritt: float,
    erwerbsm_rente_params: dict,
) -> float:
    """Average earning points per Bewertungsmonat (as part of the "Grundbewertung").

    Earnings points are divided by "belegungsfähige Gesamtzeitraum" which is the period
    from the age of 17 until the start of the pension.

    Legal reference: SGB VI § 72: Grundbewertung
    """

    belegungsfähiger_gesamtzeitraum = (
        sozialversicherung__rente__alter_bei_renteneintritt
        - erwerbsm_rente_params["altersgrenze_grundbewertung"]
    )

    mean_entgeltpunkte_pro_bewertungsmonat = (
        sozialversicherung__rente__entgeltpunkte_west
        + sozialversicherung__rente__entgeltpunkte_ost
    ) / belegungsfähiger_gesamtzeitraum

    return mean_entgeltpunkte_pro_bewertungsmonat
