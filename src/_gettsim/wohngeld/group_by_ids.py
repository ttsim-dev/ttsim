"""Wohngeldrechtlicher Teilhaushalt ID."""

import numpy

from ttsim import AggType, agg_by_group_function, group_creation_function


@agg_by_group_function(agg_type=AggType.COUNT)
def anzahl_personen_wthh(wthh_id: int) -> int:
    pass


@group_creation_function()
def wthh_id(
    hh_id: numpy.ndarray[int],
    vorrangprüfungen__wohngeld_vorrang_vor_arbeitslosengeld_2_bg: numpy.ndarray[bool],
    vorrangprüfungen__wohngeld_und_kinderzuschlag_vorrang_vor_arbeitslosengeld_2_bg: numpy.ndarray[
        bool
    ],
) -> numpy.ndarray[int]:
    """
    Compute the ID of the wohngeldrechtlicher Teilhaushalt.
    """
    result = []
    for index, current_hh_id in enumerate(hh_id):
        if (
            vorrangprüfungen__wohngeld_vorrang_vor_arbeitslosengeld_2_bg[index]
            or vorrangprüfungen__wohngeld_und_kinderzuschlag_vorrang_vor_arbeitslosengeld_2_bg[
                index
            ]
        ):
            result.append(current_hh_id * 100 + 1)
        else:
            result.append(current_hh_id * 100)

    return numpy.asarray(result)
