"""This module computes demographic variables directly on the data.

These information are used throughout modules of gettsim.

"""

import numpy

from ttsim import AggregateByGroupSpec, group_by_function, policy_function

aggregation_specs = {
    "anzahl_personen_ehe": AggregateByGroupSpec(
        aggr=AggregationType.COUNT,
    ),
}


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
    out = kind and (alter <= 2)
    return out


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
    out = kind and (alter <= 5)
    return out


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
    out = kind and (alter <= 6)
    return out


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
    out = kind and (alter <= 15)
    return out


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
    out = kind and (alter <= 17)
    return out


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
    out = not kind
    return out


@group_by_function()
def ehe_id(
    p_id: numpy.ndarray[int],
    p_id_ehepartner: numpy.ndarray[int],
) -> numpy.ndarray[int]:
    """
    Compute the ID of the Ehe for each person.
    """
    p_id_to_ehe_id = {}
    next_ehe_id = 0
    result = []

    for index, current_p_id in enumerate(p_id):
        current_p_id_ehepartner = p_id_ehepartner[index]

        if current_p_id_ehepartner >= 0 and current_p_id_ehepartner in p_id_to_ehe_id:
            result.append(p_id_to_ehe_id[current_p_id_ehepartner])
            continue

        # New married couple
        result.append(next_ehe_id)
        p_id_to_ehe_id[current_p_id] = next_ehe_id
        next_ehe_id += 1

    return numpy.asarray(result)
