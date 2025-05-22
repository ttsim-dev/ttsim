"""Pathway for unemployed individuals.

Revoked for birth cohorts after 1951.
"""

from __future__ import annotations

from ttsim import policy_function


@policy_function(
    start_date="1989-12-18",
    end_date="1996-07-28",
    leaf_name="altersgrenze",
    vectorization_strategy="loop",
)
def altersgrenze_ohne_vertrauensschutzprüfung_bis_1996(
    altersgrenze_ohne_vertrauensschutzprüfung: float,
) -> float:
    """Full retirement age for unemployed without Vertrauensschutz.

    Does not check for eligibility for this pathway into retirement.
    """
    return altersgrenze_ohne_vertrauensschutzprüfung


@policy_function(
    start_date="1996-07-29",
    end_date="2009-12-31",
    leaf_name="altersgrenze",
    vectorization_strategy="loop",
)
def altersgrenze_mit_vertrauensschutzprüfung(
    geburtsjahr: int,
    geburtsmonat: int,
    vertrauensschutz_1997: bool,
    altersgrenze_ohne_vertrauensschutzprüfung: float,
    ges_rente_params: dict,
) -> float:
    """Full retirement age for unemployed with Vertrauensschutz.

    Full retirement age depends on birth year and month. Policy becomes inactive in 2010
    because then all potential beneficiaries have reached the normal retirement age.

    Does not check for eligibility for this pathway into retirement.
    """
    if (
        vertrauensschutz_1997
        and geburtsjahr
        <= ges_rente_params["altersgrenze_rente_wegen_arbeitslosigkeit_abschlagsfrei"][
            "vertrauensschutz"
        ]["max_birthyear_old_regime"]
    ):
        out = ges_rente_params[
            "altersgrenze_rente_wegen_arbeitslosigkeit_abschlagsfrei"
        ]["vertrauensschutz"]["entry_age_old_regime"]

    elif vertrauensschutz_1997:
        out = ges_rente_params[
            "altersgrenze_rente_wegen_arbeitslosigkeit_abschlagsfrei"
        ]["vertrauensschutz"][geburtsjahr][geburtsmonat]
    else:
        out = altersgrenze_ohne_vertrauensschutzprüfung

    return out


@policy_function(
    start_date="2010-01-01",
    end_date="2017-12-31",
    leaf_name="altersgrenze",
    vectorization_strategy="loop",
)
def altersgrenze_ohne_vertrauensschutzprüfung_ab_2010(
    altersgrenze_ohne_vertrauensschutzprüfung: float,
) -> float:
    """Full retirement age for unemployed without Vertrauensschutz.

    Full retirement age depends on birth year and month. Policy becomes inactive in 2017
    because then all potential beneficiaries have reached the normal retirement age.

    Does not check for eligibility for this pathway into retirement.
    """
    return altersgrenze_ohne_vertrauensschutzprüfung


@policy_function(
    end_date="1989-12-17",
    leaf_name="altersgrenze_vorzeitig",
    vectorization_strategy="not_required",
)
def altersgrenze_vorzeitig_ohne_staffelung(ges_rente_params: dict) -> float:
    """Early retirement age of pension for unemployed.

    Early retirement age does not depend on birth year and month.

    Does not check for eligibility for this pathway into retirement.
    """

    return ges_rente_params["altersgrenze_rente_wegen_arbeitslosigkeit_vorzeitig"]


@policy_function(
    start_date="1989-12-18",
    end_date="1996-07-28",
    leaf_name="altersgrenze_vorzeitig",
)
def altersgrenze_vorzeitig_ohne_vertrauensschutz_bis_1996_07(
    altersgrenze_vorzeitig_ohne_vertrauensschutzprüfung: float,
) -> float:
    """Early retirement age of pension for unemployed.

    Does not check for eligibility for this pathway into retirement.
    """

    return altersgrenze_vorzeitig_ohne_vertrauensschutzprüfung


@policy_function(
    start_date="1996-07-29",
    end_date="1996-09-26",
    leaf_name="altersgrenze_vorzeitig",
)
def altersgrenze_vorzeitig_mit_vertrauensschutz_ab_1996_07_bis_1996_09(
    vertrauensschutz_1997: bool,
    altersgrenze_vorzeitig_ohne_vertrauensschutzprüfung: float,
    ges_rente_params: dict,
) -> float:
    """Early retirement age of pension for unemployed.

    Includes Vertrauensschutz rules implemented from July to September 1996.

    Does not check for eligibility for this pathway into retirement.
    """

    if vertrauensschutz_1997:
        arbeitsl_vorzeitig = ges_rente_params[
            "altersgrenze_rente_wegen_arbeitslosigkeit_vorzeitig"
        ]["vertrauensschutz"]
    else:
        arbeitsl_vorzeitig = altersgrenze_vorzeitig_ohne_vertrauensschutzprüfung

    return arbeitsl_vorzeitig


@policy_function(
    start_date="1996-09-27",
    end_date="2004-07-25",
    leaf_name="altersgrenze_vorzeitig",
    vectorization_strategy="not_required",
)
def altersgrenze_vorzeitig_ohne_staffelung_ab_1996_09(ges_rente_params: dict) -> float:
    """Early retirement age of pension for unemployed.

    Early retirement age does not depend on birth year and month.

    Does not check for eligibility for this pathway into retirement.
    """

    return ges_rente_params["altersgrenze_rente_wegen_arbeitslosigkeit_vorzeitig"]


@policy_function(
    start_date="2004-07-26",
    end_date="2017-12-31",
    leaf_name="altersgrenze_vorzeitig",
)
def ges_rente_arbeitsl_vorzeitig_mit_vertrauenss_ab_2004_07(
    vertrauensschutz_2004: bool,
    altersgrenze_vorzeitig_ohne_vertrauensschutzprüfung: float,
    ges_rente_params: dict,
) -> float:
    """Early retirement age of pension for unemployed.

    Includes Vertrauensschutz rules implemented in July 2004. Policy becomes inactive in
    2018 because then all potential beneficiaries have reached the normal retirement
    age.

    Does not check for eligibility for this pathway into retirement.
    """

    if vertrauensschutz_2004:
        arbeitsl_vorzeitig = ges_rente_params[
            "altersgrenze_rente_wegen_arbeitslosigkeit_vorzeitig"
        ]["vertrauensschutz"]
    else:
        arbeitsl_vorzeitig = altersgrenze_vorzeitig_ohne_vertrauensschutzprüfung

    return arbeitsl_vorzeitig


@policy_function(end_date="2017-12-31", vectorization_strategy="loop")
def altersgrenze_ohne_vertrauensschutzprüfung(
    geburtsjahr: int,
    geburtsmonat: int,
    ges_rente_params: dict,
) -> float:
    """Full retirement age for unemployed without Vertrauensschutz.

    Full retirement age depends on birth year and month.

    Does not check for eligibility for this pathway into retirement.
    """
    if (
        geburtsjahr
        <= ges_rente_params["altersgrenze_rente_wegen_arbeitslosigkeit_abschlagsfrei"][
            "max_birthyear_old_regime"
        ]
    ):
        out = ges_rente_params[
            "altersgrenze_rente_wegen_arbeitslosigkeit_abschlagsfrei"
        ]["entry_age_old_regime"]
    elif (
        geburtsjahr
        >= ges_rente_params["altersgrenze_rente_wegen_arbeitslosigkeit_abschlagsfrei"][
            "min_birthyear_new_regime"
        ]
    ):
        out = ges_rente_params[
            "altersgrenze_rente_wegen_arbeitslosigkeit_abschlagsfrei"
        ]["entry_age_new_regime"]
    else:
        out = ges_rente_params[
            "altersgrenze_rente_wegen_arbeitslosigkeit_abschlagsfrei"
        ][geburtsjahr][geburtsmonat]

    return out


@policy_function(end_date="2017-12-31", vectorization_strategy="loop")
def altersgrenze_vorzeitig_ohne_vertrauensschutzprüfung(
    geburtsjahr: int,
    geburtsmonat: int,
    ges_rente_params: dict,
) -> float:
    """Early retirement age of pension for unemployed without Vertrauensschutz.

    Relevant if the early retirement age depends on birth year and month.

    Does not check for eligibility for this pathway into retirement.
    """

    if (
        geburtsjahr
        <= ges_rente_params["altersgrenze_rente_wegen_arbeitslosigkeit_vorzeitig"][
            "max_birthyear_old_regime"
        ]
    ):
        arbeitsl_vorzeitig = ges_rente_params[
            "altersgrenze_rente_wegen_arbeitslosigkeit_vorzeitig"
        ]["entry_age_old_regime"]
    elif (
        geburtsjahr
        >= ges_rente_params["altersgrenze_rente_wegen_arbeitslosigkeit_vorzeitig"][
            "min_birthyear_new_regime"
        ]
    ):
        arbeitsl_vorzeitig = ges_rente_params[
            "altersgrenze_rente_wegen_arbeitslosigkeit_vorzeitig"
        ]["entry_age_new_regime"]
    else:
        arbeitsl_vorzeitig = ges_rente_params[
            "altersgrenze_rente_wegen_arbeitslosigkeit_vorzeitig"
        ][geburtsjahr][geburtsmonat]

    return arbeitsl_vorzeitig


@policy_function(end_date="2007-04-29", leaf_name="grundsätzlich_anspruchsberechtigt")
def grundsätzlich_anspruchsberechtigt_ohne_prüfung_geburtsjahr(
    arbeitslos_für_1_jahr_nach_alter_58_ein_halb: bool,
    sozialversicherung__rente__wartezeit_15_jahre_erfüllt: bool,
    pflichtbeitragsjahre_8_von_10: bool,
) -> bool:
    """Eligibility for Altersrente für Arbeitslose (pension for unemployed).

    Wartezeit 15 years, 8 contribution years past 10 years, being unemployed for at
    least 1 year after age 58 and 6 months. The person is also required to be
    unemployed at the time of claiming the pension. As there are no restrictions
    regarding voluntary unemployment this requirement may be viewed as always satisfied
    and is therefore not included when checking for eligibility.
    """

    return (
        arbeitslos_für_1_jahr_nach_alter_58_ein_halb
        and sozialversicherung__rente__wartezeit_15_jahre_erfüllt
        and pflichtbeitragsjahre_8_von_10
    )


@policy_function(
    start_date="2007-04-30",
    end_date="2017-12-31",
    leaf_name="grundsätzlich_anspruchsberechtigt",
)
def grundsätzlich_anspruchsberechtigt_mit_prüfung_geburtsjahr(
    arbeitslos_für_1_jahr_nach_alter_58_ein_halb: bool,
    sozialversicherung__rente__wartezeit_15_jahre_erfüllt: bool,
    pflichtbeitragsjahre_8_von_10: bool,
    geburtsjahr: int,
    kohorte_abschaffung: int,
) -> bool:
    """Eligibility for Altersrente für Arbeitslose (pension for unemployed).

    Wartezeit 15 years, 8 contributionyears past 10 years, being at least 1 year
    unemployed after age 58 and 6 months and being born before 1952. The person is also
    required to be unemployed at the time of claiming the pension. As there are no
    restrictions regarding voluntary unemployment this requirement may be viewed as
    always satisfied and is therefore not included when checking for eligibility. Policy
    becomes inactive in 2018 because then all potential beneficiaries have reached the
    normal retirement age.
    """

    return (
        arbeitslos_für_1_jahr_nach_alter_58_ein_halb
        and sozialversicherung__rente__wartezeit_15_jahre_erfüllt
        and pflichtbeitragsjahre_8_von_10
        and geburtsjahr < kohorte_abschaffung
    )
