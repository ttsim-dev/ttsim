from __future__ import annotations

import datetime

import numpy

from ttsim import policy_function


@policy_function(vectorization_strategy="loop")
def geburtsdatum(
    geburtsjahr: int,
    geburtsmonat: int,
    geburtstag: int,
) -> numpy.datetime64:
    """Create date of birth datetime variable.

    Parameters
    ----------
    geburtsjahr
        See basic input variable :ref:`geburtsjahr <geburtsjahr>`.
    geburtsmonat
        See basic input variable :ref:`geburtsmonat <geburtsmonat>`.
    geburtstag
        See basic input variable :ref:`geburtstag <geburtstag>`.

    Returns
    -------

    """
    return numpy.datetime64(
        datetime.datetime(
            geburtsjahr,
            geburtsmonat,
            geburtstag,
        )
    ).astype("datetime64[D]")


@policy_function(vectorization_strategy="loop")
def alter_monate(geburtsdatum: numpy.datetime64, elterngeld_params: dict) -> float:
    """Calculate age of youngest child in months.

    Parameters
    ----------
    hh_id
        See basic input variable :ref:`hh_id <hh_id>`.
    geburtsdatum
        See :func:`geburtsdatum`.
    elterngeld_params
        See params documentation :ref:`elterngeld_params <elterngeld_params>`.
    Returns
    -------

    """

    # TODO(@hmgaudecker): Remove explicit cast when vectorisation is enabled.
    # https://github.com/iza-institute-of-labor-economics/gettsim/issues/515
    age_in_days = elterngeld_params["datum"] - numpy.datetime64(geburtsdatum)

    out = age_in_days / 30.436875
    return out.astype(float)


@policy_function()
def alter_bis_24(alter: int) -> bool:
    """Age is 24 years at most.

    Trivial, but necessary in order to use the target for aggregation.

    Parameters
    ----------
    alter
        See basic input variable :ref:`alter <alter>`.

    Returns
    -------
    """
    return alter <= 24
