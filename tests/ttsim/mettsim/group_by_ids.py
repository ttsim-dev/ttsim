from __future__ import annotations

from ttsim.config import numpy_or_jax as np
from ttsim.tt_dag_elements import group_creation_function


@group_creation_function()
def sp_id(
    p_id: np.ndarray,
    p_id_spouse: np.ndarray,
) -> np.ndarray:
    """
    Compute the spouse (sp) group ID for each person.
    """
    n = np.max(p_id)
    p_id_spouse = np.where(p_id_spouse < 0, p_id, p_id_spouse)
    sp_id = np.maximum(p_id, p_id_spouse) + np.minimum(p_id, p_id_spouse) * n

    return sp_id


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
    n = np.max(p_id)

    p_id_parent_1_loc = p_id_parent_1
    p_id_parent_2_loc = p_id_parent_2
    for i in range(p_id.shape[0]):
        p_id_parent_1_loc = np.where(p_id_parent_1_loc == p_id[i], i, p_id_parent_1_loc)
        p_id_parent_2_loc = np.where(p_id_parent_2_loc == p_id[i], i, p_id_parent_2_loc)

    children = np.isin(p_id, p_id_parent_1) + np.isin(p_id, p_id_parent_2)
    fam_id = np.where(
        p_id_spouse < 0,
        p_id + p_id * n,
        np.maximum(p_id, p_id_spouse) + np.minimum(p_id, p_id_spouse) * n,
    )
    fam_id = np.where(
        (fam_id == p_id + p_id * n)
        * (p_id_parent_1_loc >= 0)
        * (age < 25)
        * (1 - children),
        fam_id[p_id_parent_1_loc],
        fam_id,
    )
    fam_id = np.where(
        (fam_id == p_id + p_id * n)
        * (p_id_parent_2_loc >= 0)
        * (age < 25)
        * (1 - children),
        fam_id[p_id_parent_2_loc],
        fam_id,
    )

    return fam_id
