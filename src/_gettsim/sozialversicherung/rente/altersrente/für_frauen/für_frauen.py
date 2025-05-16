"""Pathway for women.

Revoked for birth cohorts after 1951.
"""

from __future__ import annotations

from ttsim import policy_function


@policy_function(
    end_date="1989-12-17",
    leaf_name="altersgrenze",
    vectorization_strategy="not_required",
)
def altersgrenze_ohne_staffelung(ges_rente_params: dict) -> float:
    """Full retirement age (FRA) for women.

    FRA is the same for each birth cohort.

    Does not check for eligibility for this pathway into retirement.

    Parameters
    ----------
    geburtsjahr
        See basic input variable :ref:`geburtsjahr <geburtsjahr>`.
    ges_rente_params
        See params documentation :ref:`ges_rente_params <ges_rente_params>`.

    Returns
    -------
    Full retirement age for women.

    """

    return ges_rente_params["altersgrenze_für_frauen_abschlagsfrei"]


@policy_function(
    start_date="1989-12-18", leaf_name="altersgrenze", vectorization_strategy="loop"
)
def altersgrenze_mit_staffelung(
    geburtsjahr: int,
    geburtsmonat: int,
    ges_rente_params: dict,
) -> float:
    """Full retirement age (FRA) for women.

    FRA differs by birth cohort.

    Does not check for eligibility for this pathway into retirement.

    Parameters
    ----------
    geburtsjahr
        See basic input variable :ref:`geburtsjahr <geburtsjahr>`.
    geburtsmonat
        See basic input variable :ref:`geburtsmonat <geburtsmonat>`.
    ges_rente_params
        See params documentation :ref:`ges_rente_params <ges_rente_params>`.

    Returns
    -------
    Full retirement age for women.

    """
    if (
        geburtsjahr
        <= ges_rente_params["altersgrenze_für_frauen_abschlagsfrei"][
            "max_birthyear_old_regime"
        ]
    ):
        out = ges_rente_params["altersgrenze_für_frauen_abschlagsfrei"][
            "entry_age_old_regime"
        ]
    elif (
        geburtsjahr
        >= ges_rente_params["altersgrenze_für_frauen_abschlagsfrei"][
            "min_birthyear_new_regime"
        ]
    ):
        out = ges_rente_params["altersgrenze_für_frauen_abschlagsfrei"][
            "entry_age_new_regime"
        ]
    else:
        out = ges_rente_params["altersgrenze_für_frauen_abschlagsfrei"][geburtsjahr][
            geburtsmonat
        ]

    return out


@policy_function(
    end_date="1989-12-17",
    leaf_name="altersgrenze_vorzeitig",
    vectorization_strategy="not_required",
)
def altersgrenze_vorzeitig_ohne_staffelung(ges_rente_params: dict) -> float:
    """Early retirement age (ERA) for Renten für Frauen.

    ERA does not depend on birth year and month.

    Does not check for eligibility for this pathway into retirement.

    Parameters
    ----------
    geburtsjahr
        See basic input variable :ref:`geburtsjahr <geburtsjahr>`.
    ges_rente_params
        See params documentation :ref:`ges_rente_params <ges_rente_params>`.

    Returns
    -------
    Early retirement age

    """

    return ges_rente_params["altersgrenze_für_frauen_vorzeitig"]


@policy_function(
    start_date="1989-12-18",
    end_date="1996-09-26",
    leaf_name="altersgrenze_vorzeitig",
    vectorization_strategy="loop",
)
def altersgrenze_vorzeitig_mit_staffelung(
    geburtsjahr: int,
    geburtsmonat: int,
    ges_rente_params: dict,
) -> float:
    """Early retirement age (ERA) for Renten für Frauen.

    ERA depends on birth year and month.

    Does not check for eligibility for this pathway into retirement.

    Parameters
    ----------
    geburtsjahr
        See basic input variable :ref:`geburtsjahr <geburtsjahr>`.
    geburtsmonat
        See basic input variable :ref:`geburtsmonat <geburtsmonat>`.
    ges_rente_params
        See params documentation :ref:`ges_rente_params <ges_rente_params>`.

    Returns
    -------
    Early retirement age

    """
    if (
        geburtsjahr
        <= ges_rente_params["altersgrenze_für_frauen_vorzeitig"][
            "max_birthyear_old_regime"
        ]
    ):
        out = ges_rente_params["altersgrenze_für_frauen_vorzeitig"][
            "entry_age_old_regime"
        ]
    elif (
        geburtsjahr
        >= ges_rente_params["altersgrenze_für_frauen_vorzeitig"][
            "min_birthyear_new_regime"
        ]
    ):
        out = ges_rente_params["altersgrenze_für_frauen_vorzeitig"][
            "entry_age_new_regime"
        ]
    else:
        out = ges_rente_params["altersgrenze_für_frauen_vorzeitig"][geburtsjahr][
            geburtsmonat
        ]

    return out


@policy_function(
    start_date="1996-09-27",
    leaf_name="altersgrenze_vorzeitig",
    vectorization_strategy="not_required",
)
def altersgrenze_vorzeitig_ohne_staffelung_nach_1996(ges_rente_params: dict) -> float:
    """Early retirement age (ERA) for Renten für Frauen.

    ERA does not depend on birth year and month.

    Does not check for eligibility for this pathway into retirement.

    Parameters
    ----------
    geburtsjahr
        See basic input variable :ref:`geburtsjahr <geburtsjahr>`.
    ges_rente_params
        See params documentation :ref:`ges_rente_params <ges_rente_params>`.

    Returns
    -------
    Early retirement age

    """

    return ges_rente_params["altersgrenze_für_frauen_vorzeitig"]


@policy_function(end_date="1997-12-15", leaf_name="grundsätzlich_anspruchsberechtigt")
def grundsätzlich_anspruchsberechtigt_ohne_prüfung_geburtsjahr(
    weiblich: bool,
    sozialversicherung__rente__wartezeit_15_jahre_erfüllt: bool,
    pflichtsbeitragsjahre_ab_alter_40: float,
    ges_rente_params: dict,
) -> bool:
    """Eligibility for Altersrente für Frauen (pension for women).

    Eligibility does not depend on birth year.

    Policy becomes inactive in 2018 because then all potential beneficiaries have
    reached the normal retirement age.

    Parameters
    ----------
    weiblich
        See basic input variable :ref:`weiblich <weiblich>`.
    sozialversicherung__rente__wartezeit_15_jahre_erfüllt
        See :func:`sozialversicherung__rente__wartezeit_15_jahre_erfüllt`
    pflichtsbeitragsjahre_ab_alter_40
        See basic input variable :ref:`pflichtsbeitragsjahre_ab_alter_40
        <pflichtsbeitragsjahre_ab_alter_40>`.
    ges_rente_params
        See params documentation :ref:`ges_rente_params <ges_rente_params>`.

    Returns
    -------
    Eligibility as bool.

    """

    out = (
        weiblich
        and sozialversicherung__rente__wartezeit_15_jahre_erfüllt
        and pflichtsbeitragsjahre_ab_alter_40
        > ges_rente_params["rente_für_frauen_mindestpflichtbeitragsjahre_ab_alter_40"]
    )

    return out


@policy_function(
    start_date="1997-12-16",
    end_date="2017-12-31",
    leaf_name="grundsätzlich_anspruchsberechtigt",
)
def grundsätzlich_anspruchsberechtigt_mit_geburtsjahr_prüfung(
    weiblich: bool,
    sozialversicherung__rente__wartezeit_15_jahre_erfüllt: bool,
    pflichtsbeitragsjahre_ab_alter_40: float,
    geburtsjahr: int,
    ges_rente_params: dict,
) -> bool:
    """Eligibility for Altersrente für Frauen (pension for women).

    Only individuals born before a certain year are eligible.

    Wartezeit 15 years, contributions for 10 years after age 40, being a woman. Policy
    becomes inactive in 2018 because then all potential beneficiaries have reached the
    normal retirement age.

    Parameters
    ----------
    weiblich
        See basic input variable :ref:`weiblich <weiblich>`.
    sozialversicherung__rente__wartezeit_15_jahre_erfüllt
        See :func:`sozialversicherung__rente__wartezeit_15_jahre_erfüllt`
    pflichtsbeitragsjahre_ab_alter_40
        See basic input variable :ref:`pflichtsbeitragsjahre_ab_alter_40 <pflichtsbeitragsjahre_ab_alter_40>`.
    geburtsjahr
        See basic input variable :ref:`geburtsjahr <geburtsjahr>`.
    ges_rente_params
        See params documentation :ref:`ges_rente_params <ges_rente_params>`.

    Returns
    -------
    Eligibility as bool.

    """

    out = (
        weiblich
        and sozialversicherung__rente__wartezeit_15_jahre_erfüllt
        and pflichtsbeitragsjahre_ab_alter_40
        > ges_rente_params["rente_für_frauen_mindestpflichtbeitragsjahre_ab_alter_40"]
        and geburtsjahr < ges_rente_params["first_birthyear_without_rente_für_frauen"]
    )

    return out
