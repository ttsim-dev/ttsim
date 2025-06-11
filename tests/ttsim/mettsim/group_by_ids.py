from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

    import numpy
from ttsim.tt_dag_elements import group_creation_function


@group_creation_function()
def sp_id(
    p_id: numpy.ndarray, p_id_spouse: numpy.ndarray, xnp: ModuleType
) -> numpy.ndarray:
    """
    Compute the spouse (sp) group ID for each person.
    """
    n = xnp.max(p_id)
    p_id_spouse = xnp.where(p_id_spouse < 0, p_id, p_id_spouse)
    return xnp.maximum(p_id, p_id_spouse) + xnp.minimum(p_id, p_id_spouse) * n


@group_creation_function()
def fam_id(
    p_id_spouse: numpy.ndarray,
    p_id: numpy.ndarray,
    age: numpy.ndarray,
    p_id_parent_1: numpy.ndarray,
    p_id_parent_2: numpy.ndarray,
    xnp: ModuleType,
) -> numpy.ndarray:
    """
    Compute the family ID for each person.
    """
    n = xnp.max(p_id)

    p_id_parent_1_loc = p_id_parent_1
    p_id_parent_2_loc = p_id_parent_2
    for i in range(p_id.shape[0]):
        p_id_parent_1_loc = xnp.where(
            p_id_parent_1_loc == p_id[i], i, p_id_parent_1_loc
        )
        p_id_parent_2_loc = xnp.where(
            p_id_parent_2_loc == p_id[i], i, p_id_parent_2_loc
        )

    children = xnp.isin(p_id, p_id_parent_1) + xnp.isin(p_id, p_id_parent_2)
    out = xnp.where(
        p_id_spouse < 0,
        p_id + p_id * n,
        xnp.maximum(p_id, p_id_spouse) + xnp.minimum(p_id, p_id_spouse) * n,
    )
    out = xnp.where(
        (out == p_id + p_id * n)
        * (p_id_parent_1_loc >= 0)
        * (age < 25)
        * (1 - children),
        out[p_id_parent_1_loc],
        out,
    )
    out = xnp.where(
        (out == p_id + p_id * n)
        * (p_id_parent_2_loc >= 0)
        * (age < 25)
        * (1 - children),
        out[p_id_parent_2_loc],
        out,
    )

    return out
