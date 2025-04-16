"""Steuernummer ID."""

import numpy

from ttsim import AggType, agg_by_group_function, group_creation_function


@agg_by_group_function(agg_type=AggType.COUNT)
def anzahl_personen_sn(sn_id: int) -> int:
    pass


@agg_by_group_function(agg_type=AggType.ANY)
def alleinerziehend_sn(familie__alleinerziehend: bool, sn_id: int) -> bool:
    pass


@group_creation_function()
def sn_id(
    p_id: numpy.ndarray[int],
    familie__p_id_ehepartner: numpy.ndarray[int],
    gemeinsam_veranlagt: numpy.ndarray[bool],
) -> numpy.ndarray[int]:
    """
    Compute a Steuernummer (ID) for each person / couple.
    """
    p_id_to_sn_id = {}
    p_id_to_gemeinsam_veranlagt = {}
    next_sn_id = 0
    result = []

    for index, current_p_id in enumerate(p_id):
        current_p_id_ehepartner = familie__p_id_ehepartner[index]
        current_gemeinsam_veranlagt = gemeinsam_veranlagt[index]

        if current_p_id_ehepartner >= 0 and current_p_id_ehepartner in p_id_to_sn_id:
            gemeinsam_veranlagt_ehepartner = p_id_to_gemeinsam_veranlagt[
                current_p_id_ehepartner
            ]

            if current_gemeinsam_veranlagt != gemeinsam_veranlagt_ehepartner:
                message = (
                    f"{current_p_id_ehepartner} and {current_p_id} are "
                    "married, but have different values for "
                    "gemeinsam_veranlagt."
                )
                raise ValueError(message)

            if current_gemeinsam_veranlagt:
                result.append(p_id_to_sn_id[current_p_id_ehepartner])
                continue

        # New Steuersubjekt
        result.append(next_sn_id)
        p_id_to_sn_id[current_p_id] = next_sn_id
        p_id_to_gemeinsam_veranlagt[current_p_id] = current_gemeinsam_veranlagt
        next_sn_id += 1

    return numpy.asarray(result)
