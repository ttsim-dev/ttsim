"""Contribution rate to public long-term care insurance."""

from ttsim import AggType, agg_by_p_id_function, policy_function


@agg_by_p_id_function(agg_type=AggType.SUM)
def anzahl_kinder_bis_24_elternteil_1(
    alter_bis_24: bool,
    einkommensteuer__p_id_kinderfreibetragsempfänger_1: int,
    p_id: int,
) -> int:
    pass


@agg_by_p_id_function(agg_type=AggType.SUM)
def anzahl_kinder_bis_24_elternteil_2(
    alter_bis_24: bool,
    einkommensteuer__p_id_kinderfreibetragsempfänger_2: int,
    p_id: int,
) -> int:
    pass


@policy_function(
    start_date="1995-01-01",
    end_date="2004-12-31",
    leaf_name="beitragssatz",
    vectorization_strategy="loop",
)
def beitragssatz_ohne_zusatz_für_kinderlose(
    ges_pflegev_params: dict,
) -> float:
    """Employee's long-term care insurance contribution rate.

    Before 2005, the contribution rate was independent of the number of children.

    Parameters
    ----------
    sozialv_beitr_params
        See params documentation :ref:`sozialv_beitr_params <sozialv_beitr_params>`.

    Returns
    -------

    """

    return ges_pflegev_params["beitragssatz"]


@policy_function(
    start_date="2005-01-01",
    end_date="2023-06-30",
    leaf_name="beitragssatz",
    vectorization_strategy="loop",
)
def beitragssatz_zusatz_kinderlos_dummy(
    zusatzbetrag_kinderlos: bool,
    ges_pflegev_params: dict,
) -> float:
    """Employee's long-term care insurance contribution rate.

    Since 2005, the contribution rate is increased for childless individuals.

    Parameters
    ----------
    zusatzbetrag_kinderlos
        See :func:`zusatzbetrag_kinderlos`.
    sozialv_beitr_params
        See params documentation :ref:`sozialv_beitr_params <sozialv_beitr_params>`.

    Returns
    -------

    """
    out = ges_pflegev_params["beitragssatz"]["standard"]

    # Add additional contribution for childless individuals
    if zusatzbetrag_kinderlos:
        out += ges_pflegev_params["beitragssatz"]["zusatz_kinderlos"]

    return out


@policy_function(
    start_date="2023-07-01", leaf_name="beitragssatz", vectorization_strategy="loop"
)
def beitragssatz_mit_kinder_abschlag(
    anzahl_kinder_bis_24: int,
    zusatzbetrag_kinderlos: bool,
    ges_pflegev_params: dict,
) -> float:
    """Employee's long-term care insurance contribution rate.

    Since July 2023, the contribution rate is reduced for individuals with children
    younger than 25.

    Parameters
    ----------
    anzahl_kinder_bis_24: int,
        See :func:`anzahl_kinder_bis_24`.
    zusatzbetrag_kinderlos
        See :func:`zusatzbetrag_kinderlos`.
    sozialv_beitr_params
        See params documentation :ref:`sozialv_beitr_params <sozialv_beitr_params>`.

    Returns
    -------

    """
    out = ges_pflegev_params["beitragssatz"]["standard"]

    # Add additional contribution for childless individuals
    if zusatzbetrag_kinderlos:
        out += ges_pflegev_params["beitragssatz"]["zusatz_kinderlos"]

    # Reduced contribution for individuals with two or more children under 25
    if anzahl_kinder_bis_24 >= 2:
        out -= ges_pflegev_params["beitragssatz"]["abschlag_kinder"] * min(
            anzahl_kinder_bis_24 - 1, 4
        )

    return out


@policy_function(start_date="2005-01-01", vectorization_strategy="loop")
def zusatzbetrag_kinderlos(
    hat_kinder: bool,
    alter: int,
    ges_pflegev_params: dict,
) -> bool:
    """Whether additional care insurance contribution for childless individuals applies.

    Not relevant before 2005 because the contribution rate was independent of the number
    of children.

    Parameters
    ----------
    hat_kinder
        See basic input variable :ref:`hat_kinder <hat_kinder>`.
    alter
        See basic input variable :ref:`alter <alter>`.
    ges_pflegev_params: dict,
        See params documentation :ref:`sozialv_beitr_params <sozialv_beitr_params>`.

    Returns
    -------

    """
    mindestalter = ges_pflegev_params["zusatz_kinderlos_mindestalter"]
    return (not hat_kinder) and alter >= mindestalter


@policy_function()
def anzahl_kinder_bis_24(
    anzahl_kinder_bis_24_elternteil_1: int,
    anzahl_kinder_bis_24_elternteil_2: int,
) -> int:
    """Number of children under 25 years of age.
    Parameters
    ----------
    anzahl_kinder_bis_24_elternteil_1
        See :func:`anzahl_kinder_bis_24_elternteil_1`.
    anzahl_kinder_bis_24_elternteil_2
        See :func:`anzahl_kinder_bis_24_elternteil_2`.

    Returns
    -------
    """
    return anzahl_kinder_bis_24_elternteil_1 + anzahl_kinder_bis_24_elternteil_2
