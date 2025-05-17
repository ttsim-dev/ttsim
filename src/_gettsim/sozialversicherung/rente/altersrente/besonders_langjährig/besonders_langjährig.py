"""Pathway for the very long-term insured."""

from ttsim import policy_function


@policy_function(
    start_date="2012-01-01",
    end_date="2014-06-22",
    leaf_name="altersgrenze",
    vectorization_strategy="not_required",
)
def altersgrenze_ohne_staffelung(ges_rente_params: dict) -> float:
    """
    Full retirement age (FRA) for very long term insured.

    FRA is the same for each birth year.

    Calculate the threshold from which very long term insured people (at least 45
    years) can claim their full pension without deductions.

    Does not check for eligibility for this pathway into retirement.
    """
    return ges_rente_params["altersgrenze_besonders_langjährig_versicherte"]


@policy_function(
    start_date="2014-06-23", leaf_name="altersgrenze", vectorization_strategy="loop"
)
def altersgrenze_mit_staffelung(
    geburtsjahr: int,
    ges_rente_params: dict,
) -> float:
    """
    Full retirement age (FRA) for very long term insured.

    FRA depends on birth year and month.

    Calculate the threshold from which very long term insured people (at least 45
    years) can claim their full pension without deductions.

    Does not check for eligibility for this pathway into retirement.
    """
    if (
        geburtsjahr
        <= ges_rente_params["altersgrenze_besonders_langjährig_versicherte"][
            "max_birthyear_old_regime"
        ]
    ):
        out = ges_rente_params["altersgrenze_besonders_langjährig_versicherte"][
            "entry_age_old_regime"
        ]
    elif (
        geburtsjahr
        >= ges_rente_params["altersgrenze_besonders_langjährig_versicherte"][
            "min_birthyear_new_regime"
        ]
    ):
        out = ges_rente_params["altersgrenze_besonders_langjährig_versicherte"][
            "entry_age_new_regime"
        ]
    else:
        out = ges_rente_params["altersgrenze_besonders_langjährig_versicherte"][
            geburtsjahr
        ]

    return out


@policy_function(start_date="2012-01-01")
def grundsätzlich_anspruchsberechtigt(
    sozialversicherung__rente__wartezeit_45_jahre_erfüllt: bool,
) -> bool:
    """Determining the eligibility for Altersrente für besonders langjährig Versicherte
    (pension for very long-term insured). Wartezeit 45 years. aka "Rente mit 63".
    """

    return sozialversicherung__rente__wartezeit_45_jahre_erfüllt
