"""This module computes demographic variables directly on the data.

These information are used throughout modules of gettsim.

"""

from __future__ import annotations

from ttsim import AggType, agg_by_group_function, policy_function


@agg_by_group_function(agg_type=AggType.COUNT)
def anzahl_personen_ehe(ehe_id: int) -> int:
    pass


@policy_function()
def kind_bis_2(alter: int, kind: bool) -> bool:
    """Calculate if child under the age of 3.

    Parameters
    ----------
    alter
        See basic input variable :ref:`alter <alter>`.
    kind
        See basic input variable :ref:`kind <kind>`.

    Returns
    -------

    """
    return kind and (alter <= 2)


@policy_function()
def kind_bis_5(alter: int, kind: bool) -> bool:
    """Calculate if child under the age of 6.

    Parameters
    ----------
    alter
        See basic input variable :ref:`alter <alter>`.
    kind
        See basic input variable :ref:`kind <kind>`.

    Returns
    -------

    """
    return kind and (alter <= 5)


@policy_function()
def kind_bis_6(alter: int, kind: bool) -> bool:
    """Calculate if child under the age of 7.

    Parameters
    ----------
    alter
        See basic input variable :ref:`alter <alter>`.
    kind
        See basic input variable :ref:`kind <kind>`.

    Returns
    -------

    """
    return kind and (alter <= 6)


@policy_function()
def kind_bis_15(alter: int, kind: bool) -> bool:
    """Calculate if child under the age of 16.

    Parameters
    ----------
    alter
        See basic input variable :ref:`alter <alter>`.
    kind
        See basic input variable :ref:`kind <kind>`.

    Returns
    -------

    """
    return kind and (alter <= 15)


@policy_function()
def kind_bis_17(alter: int, kind: bool) -> bool:
    """Calculate if underage person.

    Parameters
    ----------
    alter
        See basic input variable :ref:`alter <alter>`.
    kind
        See basic input variable :ref:`kind <kind>`.

    Returns
    -------

    """
    return kind and (alter <= 17)


@policy_function()
def erwachsen(kind: bool) -> bool:
    """Calculate if adult.

    Parameters
    ----------
    alter
        See basic input variable :ref:`alter <alter>`.
    kind
        See basic input variable :ref:`kind <kind>`.

    Returns
    -------

    """
    return not kind
