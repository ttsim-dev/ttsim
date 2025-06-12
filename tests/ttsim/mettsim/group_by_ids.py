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
    n = xnp.max(p_id) + 1

    # Get the array index for all p_ids of parents
    p_id_parent_1_loc = p_id_parent_1
    p_id_parent_2_loc = p_id_parent_2
    for i in range(p_id.shape[0]):
        p_id_parent_1_loc = xnp.where(p_id_parent_1 == p_id[i], i, p_id_parent_1_loc)
        p_id_parent_2_loc = xnp.where(p_id_parent_2 == p_id[i], i, p_id_parent_2_loc)

    children = xnp.isin(p_id, p_id_parent_1) | xnp.isin(p_id, p_id_parent_2)

    # Assign the same fam_id to everybody who has a spouse,
    # otherwise create a new one from p_id
    out = xnp.where(
        p_id_spouse < 0,
        p_id + p_id * n,
        xnp.maximum(p_id, p_id_spouse) + xnp.minimum(p_id, p_id_spouse) * n,
    )

    out = _assign_parents_fam_id(
        fam_id=out,
        p_id=p_id,
        p_id_parent_loc=p_id_parent_1_loc,
        age=age,
        children=children,
        n=n,
        xnp=xnp,
    )
    out = _assign_parents_fam_id(
        fam_id=out,
        p_id=p_id,
        p_id_parent_loc=p_id_parent_2_loc,
        age=age,
        children=children,
        n=n,
        xnp=xnp,
    )

    return out


def _assign_parents_fam_id(
    fam_id: numpy.ndarray,
    p_id: numpy.ndarray,
    p_id_parent_loc: numpy.ndarray,
    age: numpy.ndarray,
    children: numpy.ndarray,
    n: numpy.ndarray,
    xnp: ModuleType,
) -> numpy.ndarray:
    """Return the fam_id of the child's parents."""

    return xnp.where(
        (fam_id == p_id + p_id * n)
        * (p_id_parent_loc >= 0)
        * (age < 25)
        * (1 - children),
        fam_id[p_id_parent_loc],
        fam_id,
    )
