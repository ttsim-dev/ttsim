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
    n = numpy.max(p_id)
    p_id_spouse = numpy.where(p_id_spouse < 0, p_id, p_id_spouse)
    sp_id = numpy.maximum(p_id, p_id_spouse) + numpy.minimum(p_id, p_id_spouse) * n

    return __reorder_ids(sp_id, xnp)


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
    n = numpy.max(p_id)

    p_id_parent_1_loc = p_id_parent_1
    p_id_parent_2_loc = p_id_parent_2
    for i in range(p_id.shape[0]):
        p_id_parent_1_loc = numpy.where(
            p_id_parent_1_loc == p_id[i], i, p_id_parent_1_loc
        )
        p_id_parent_2_loc = numpy.where(
            p_id_parent_2_loc == p_id[i], i, p_id_parent_2_loc
        )

    children = numpy.isin(p_id, p_id_parent_1) + numpy.isin(p_id, p_id_parent_2)
    fam_id = numpy.where(
        p_id_spouse < 0,
        p_id + p_id * n,
        numpy.maximum(p_id, p_id_spouse) + numpy.minimum(p_id, p_id_spouse) * n,
    )
    fam_id = numpy.where(
        (fam_id == p_id + p_id * n)
        * (p_id_parent_1_loc >= 0)
        * (age < 25)
        * (1 - children),
        fam_id[p_id_parent_1_loc],
        fam_id,
    )
    fam_id = numpy.where(
        (fam_id == p_id + p_id * n)
        * (p_id_parent_2_loc >= 0)
        * (age < 25)
        * (1 - children),
        fam_id[p_id_parent_2_loc],
        fam_id,
    )

    return __reorder_ids(fam_id, xnp)


def __reorder_ids(ids: numpy.ndarray, xnp: ModuleType) -> numpy.ndarray:
    """Make ID's consecutively numbered."""
    sorting = xnp.argsort(ids)
    ids_sorted = ids[sorting]
    index_after_sort = xnp.arange(ids.shape[0])[sorting]
    diff_to_prev = xnp.where(xnp.diff(ids_sorted) >= 1, 1, 0)
    cons_ids = xnp.concatenate((xnp.asarray([0]), xnp.cumsum(diff_to_prev)))
    return cons_ids[xnp.argsort(index_after_sort)]
