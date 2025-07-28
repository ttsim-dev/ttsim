from __future__ import annotations

from typing import TYPE_CHECKING

from ttsim.tt import group_creation_function

if TYPE_CHECKING:
    from types import ModuleType

    from ttsim.tt import IntColumn


@group_creation_function()
def sp_id(p_id: IntColumn, p_id_spouse: IntColumn, xnp: ModuleType) -> IntColumn:
    """
    Compute the spouse (sp) group ID for each person.
    """
    n = xnp.max(p_id)
    p_id_spouse = xnp.where(p_id_spouse < 0, p_id, p_id_spouse)
    return xnp.maximum(p_id, p_id_spouse) + xnp.minimum(p_id, p_id_spouse) * n


@group_creation_function()
def fam_id(
    p_id_spouse: IntColumn,
    p_id: IntColumn,
    age: IntColumn,
    p_id_parent_1: IntColumn,
    p_id_parent_2: IntColumn,
    xnp: ModuleType,
) -> IntColumn:
    """
    Compute the family ID for each person.
    """
    n = xnp.max(p_id) + 1

    # Sort all arrays according to p_id to make the id equal location in array
    sorting = xnp.argsort(p_id)
    index_after_sort = xnp.argsort(xnp.arange(p_id.shape[0])[sorting])
    sorted_p_id = p_id[sorting]
    sorted_age = age[sorting]
    sorted_p_id_parent_1 = p_id_parent_1[sorting]
    sorted_p_id_parent_2 = p_id_parent_2[sorting]
    sorted_p_id_spouse = p_id_spouse[sorting]

    children = xnp.isin(sorted_p_id, sorted_p_id_parent_1) | xnp.isin(
        sorted_p_id, sorted_p_id_parent_2
    )

    # Assign the same fam_id to everybody who has a spouse,
    # otherwise create a new one from p_id
    out = xnp.where(
        sorted_p_id_spouse < 0,
        sorted_p_id + sorted_p_id * n,
        xnp.maximum(sorted_p_id, sorted_p_id_spouse)
        + xnp.minimum(sorted_p_id, sorted_p_id_spouse) * n,
    )

    out = _assign_parents_fam_id(
        fam_id=out,
        p_id=sorted_p_id,
        p_id_parent_loc=sorted_p_id_parent_1,
        age=sorted_age,
        children=children,
        n=n,
        xnp=xnp,
    )
    out = _assign_parents_fam_id(
        fam_id=out,
        p_id=sorted_p_id,
        p_id_parent_loc=sorted_p_id_parent_2,
        age=sorted_age,
        children=children,
        n=n,
        xnp=xnp,
    )

    return out[index_after_sort]


def _assign_parents_fam_id(
    fam_id: IntColumn,
    p_id: IntColumn,
    p_id_parent_loc: IntColumn,
    age: IntColumn,
    children: IntColumn,
    n: int,
    xnp: ModuleType,
) -> IntColumn:
    """Return the fam_id of the child's parents."""
    return xnp.where(
        (fam_id == p_id + p_id * n)
        * (p_id_parent_loc >= 0)
        * (age < 25)
        * (1 - children),
        fam_id[p_id_parent_loc],
        fam_id,
    )
