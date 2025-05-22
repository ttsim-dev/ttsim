from __future__ import annotations

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
    n = np.max(p_id)
    p_id_spouse = np.where(p_id_spouse < 0, p_id, p_id_spouse)
    sp_id = np.maximum(p_id, p_id_spouse) + np.minimum(p_id, p_id_spouse) * n

    return reorder_ids(sp_id)


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

    return reorder_ids(fam_id)


def reorder_ids(ids: np.ndarray) -> np.ndarray:
    """Make ID's consecutively numbered."""
    sorting = np.argsort(ids)
    ids_sorted = ids[sorting]
    index_after_sort = np.arange(ids.shape[0])[sorting]
    diff_to_prev = np.where(np.diff(ids_sorted) >= 1, 1, 0)
    cons_ids = np.concatenate((np.asarray([0]), np.cumsum(diff_to_prev)))
    return cons_ids[np.argsort(index_after_sort)]
