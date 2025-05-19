from ttsim import group_creation_function
from ttsim.config import numpy_or_jax as np


@group_creation_function()
def sp_id(
    p_id: np.ndarray,
    p_id_spouse: np.ndarray,
) -> np.ndarray:
    """
    Compute the spouse (sp) group ID for each person.
    """
    p_id_to_sp_id: dict[int, int] = {}
    next_sp_id: int = 0
    result: list[int] = []

    for index, current_p_id in enumerate(map(int, p_id)):
        current_p_id_spouse = int(p_id_spouse[index])

        if current_p_id_spouse >= 0 and current_p_id_spouse in p_id_to_sp_id:
            result.append(p_id_to_sp_id[current_p_id_spouse])
            continue

        # New married couple
        result.append(next_sp_id)
        p_id_to_sp_id[current_p_id] = next_sp_id
        next_sp_id += 1

    return np.array(result)


@group_creation_function()
def fam_id(
    p_id_spouse: np.ndarray,
    p_id: np.ndarray,
    age: np.ndarray,
    p_id_parent_1: np.ndarray,
    p_id_parent_2: np.ndarray,
) -> np.ndarray:
    """
    Compute the family ID for each person.
    """
    # Build indexes
    p_id_to_index: dict[int, int] = {}
    p_id_to_p_ids_children: dict[int, list[int]] = {}

    for index, current_p_id in enumerate(map(int, p_id)):
        p_id_to_index[current_p_id] = index
        current_p_id_parent_1 = int(p_id_parent_1[index])
        current_p_id_parent_2 = int(p_id_parent_2[index])

        if current_p_id_parent_1 >= 0:
            if current_p_id_parent_1 not in p_id_to_p_ids_children:
                p_id_to_p_ids_children[current_p_id_parent_1] = []
            p_id_to_p_ids_children[current_p_id_parent_1].append(current_p_id)

        if current_p_id_parent_2 >= 0:
            if current_p_id_parent_2 not in p_id_to_p_ids_children:
                p_id_to_p_ids_children[current_p_id_parent_2] = []
            p_id_to_p_ids_children[current_p_id_parent_2].append(current_p_id)

    p_id_to_fam_id = {}
    next_fam_id = 0

    for index, current_p_id in enumerate(map(int, p_id)):
        # Already assigned a fam_id to this p_id via spouse / parent
        if current_p_id in p_id_to_fam_id:
            continue

        p_id_to_fam_id[current_p_id] = next_fam_id

        current_p_id_spouse = int(p_id_spouse[index])
        current_p_id_children = p_id_to_p_ids_children.get(current_p_id, [])

        # Assign fam_id to spouse
        if current_p_id_spouse >= 0:
            p_id_to_fam_id[current_p_id_spouse] = next_fam_id

        # Assign fam_id to children
        for current_p_id_child in current_p_id_children:
            child_index = p_id_to_index[current_p_id_child]
            child_age = age[child_index]
            child_p_id_children = p_id_to_p_ids_children.get(current_p_id_child, [])

            if child_age < 25 and len(child_p_id_children) == 0:
                p_id_to_fam_id[current_p_id_child] = next_fam_id

        next_fam_id += 1

    # Compute result vector
    result = [p_id_to_fam_id[current_p_id] for current_p_id in map(int, p_id)]
    return np.array(result)
