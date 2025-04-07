import numpy

from ttsim import group_by_function


@group_by_function()
def sp_id(
    p_id: numpy.ndarray[int],
    p_id_spouse: numpy.ndarray[int],
) -> numpy.ndarray[int]:
    """
    Compute the spouse (sp) group ID for each person.
    """
    p_id_to_sp_id = {}
    next_sp_id = 0
    result = []

    for index, current_p_id in enumerate(p_id):
        current_p_id_spouse = p_id_spouse[index]

        if current_p_id_spouse >= 0 and current_p_id_spouse in p_id_to_sp_id:
            result.append(p_id_to_sp_id[current_p_id_spouse])
            continue

        # New married couple
        result.append(next_sp_id)
        p_id_to_sp_id[current_p_id] = next_sp_id
        next_sp_id += 1

    return numpy.asarray(result)
