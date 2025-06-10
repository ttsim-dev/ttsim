"""Input columns."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

    import numpy
from ttsim.tt_dag_elements import group_creation_function, policy_input


@policy_input()
def p_id() -> int:
    """Unique identifier for each person. Always required, must be unique."""


@policy_input()
def hh_id() -> int:
    """Individuals living together in a household in the Wohngeld sense (§5 WoGG)."""


@group_creation_function()
def ehe_id(
    p_id: numpy.ndarray, familie__p_id_ehepartner: numpy.ndarray, xnp: ModuleType
) -> numpy.ndarray:
    """Couples that are either married or in a civil union."""
    n = numpy.max(p_id) + 1
    p_id_ehepartner_or_own_p_id = numpy.where(
        familie__p_id_ehepartner < 0, p_id, familie__p_id_ehepartner
    )
    result = (
        numpy.maximum(p_id, p_id_ehepartner_or_own_p_id)
        + numpy.minimum(p_id, p_id_ehepartner_or_own_p_id) * n
    )

    return _reorder_ids(result, xnp)


@group_creation_function()
def fg_id(
    arbeitslosengeld_2__p_id_einstandspartner: numpy.ndarray,
    p_id: numpy.ndarray,
    hh_id: numpy.ndarray,
    alter: numpy.ndarray,
    familie__p_id_elternteil_1: numpy.ndarray,
    familie__p_id_elternteil_2: numpy.ndarray,
    xnp: ModuleType,
) -> numpy.ndarray:
    """Familiengemeinschaft. Base unit for some transfers.

    Maximum of two generations, the relevant base unit for Bürgergeld / Arbeitslosengeld
    2, before excluding children who have enough income fend for themselves.
    """
    n = numpy.max(p_id) + 1

    # Get the array index for all p_ids of parents
    p_id_elternteil_1_loc = familie__p_id_elternteil_1
    p_id_elternteil_2_loc = familie__p_id_elternteil_2
    for i in range(p_id.shape[0]):
        p_id_elternteil_1_loc = numpy.where(
            familie__p_id_elternteil_1 == p_id[i], i, p_id_elternteil_1_loc
        )
        p_id_elternteil_2_loc = numpy.where(
            familie__p_id_elternteil_2 == p_id[i], i, p_id_elternteil_2_loc
        )

    children = numpy.isin(p_id, familie__p_id_elternteil_1) | numpy.isin(
        p_id, familie__p_id_elternteil_2
    )

    # Assign the same fg_id to everybody who has an Einstandspartner,
    # otherwise create a new one from p_id
    fg_id = numpy.where(
        arbeitslosengeld_2__p_id_einstandspartner < 0,
        p_id + p_id * n,
        numpy.maximum(p_id, arbeitslosengeld_2__p_id_einstandspartner)
        + numpy.minimum(p_id, arbeitslosengeld_2__p_id_einstandspartner) * n,
    )

    fg_id = _assign_parents_fg_id(
        fg_id, p_id, p_id_elternteil_1_loc, hh_id, alter, children, n, xnp
    )
    fg_id = _assign_parents_fg_id(
        fg_id, p_id, p_id_elternteil_2_loc, hh_id, alter, children, n, xnp
    )

    return _reorder_ids(fg_id, xnp)


def _assign_parents_fg_id(
    fg_id: numpy.ndarray,
    p_id: numpy.ndarray,
    p_id_elternteil_loc: numpy.ndarray,
    hh_id: numpy.ndarray,
    alter: numpy.ndarray,
    children: numpy.ndarray,
    n: numpy.ndarray,
    xnp: ModuleType,
) -> numpy.ndarray:
    """Get the fg_id of the childs parents.

    If the child is not married, has no children, is under 25 and in the same household,
    assign the fg_id of its parents."""

    # TODO(@MImmesberger): Remove input variable eigenbedarf_gedeckt
    # once Bedarfsgemeinschaften are fully endogenous
    # https://github.com/iza-institute-of-labor-economics/gettsim/issues/763
    # TODO(@MImmesberger): Remove hard-coded number
    # https://github.com/iza-institute-of-labor-economics/gettsim/issues/668

    return numpy.where(
        (p_id_elternteil_loc >= 0)
        * (fg_id == p_id + p_id * n)
        * (hh_id == hh_id[p_id_elternteil_loc])
        * (alter < 25)
        * (1 - children),
        fg_id[p_id_elternteil_loc],
        fg_id,
    )


@group_creation_function()
def bg_id(
    fg_id: numpy.ndarray,
    p_id: numpy.ndarray,
    arbeitslosengeld_2__eigenbedarf_gedeckt: numpy.ndarray,
    alter: numpy.ndarray,
    xnp: ModuleType,
) -> numpy.ndarray:
    """Bedarfsgemeinschaft

    Familiengemeinschaft except for children who have enough income to fend for
    themselves. Relevant unit for Bürgergeld / Arbeitslosengeld 2
    """
    # TODO(@MImmesberger): Remove input variable eigenbedarf_gedeckt
    # once Bedarfsgemeinschaften are fully endogenous
    # https://github.com/iza-institute-of-labor-economics/gettsim/issues/763

    # TODO(@MImmesberger): Remove hard-coded number
    # https://github.com/iza-institute-of-labor-economics/gettsim/issues/668
    offset = numpy.max(fg_id) + 1
    # Create new id for everyone who is not part of the Bedarfsgemeinschaft
    bg_id = numpy.where(
        (arbeitslosengeld_2__eigenbedarf_gedeckt) * (alter < 25),
        offset + p_id,
        fg_id,
    )

    return _reorder_ids(bg_id, xnp)


@group_creation_function()
def eg_id(
    arbeitslosengeld_2__p_id_einstandspartner: numpy.ndarray,
    p_id: numpy.ndarray,
    xnp: ModuleType,
) -> numpy.ndarray:
    """Einstandsgemeinschaft / Einstandspartner according to SGB II.

    A couple whose members are deemed to be responsible for each other.
    """
    n = numpy.max(p_id) + 1
    p_id_einstandspartner__or_own_p_id = numpy.where(
        arbeitslosengeld_2__p_id_einstandspartner < 0,
        p_id,
        arbeitslosengeld_2__p_id_einstandspartner,
    )
    result = (
        numpy.maximum(p_id, p_id_einstandspartner__or_own_p_id)
        + numpy.minimum(p_id, p_id_einstandspartner__or_own_p_id) * n
    )

    return _reorder_ids(result, xnp)


@group_creation_function()
def wthh_id(
    hh_id: numpy.ndarray,
    vorrangprüfungen__wohngeld_vorrang_vor_arbeitslosengeld_2_bg: numpy.ndarray,
    vorrangprüfungen__wohngeld_und_kinderzuschlag_vorrang_vor_arbeitslosengeld_2_bg: numpy.ndarray,
    xnp: ModuleType,
) -> numpy.ndarray:
    """Wohngeldrechtlicher Teilhaushalt.

    The relevant unit for Wohngeld. Members of a household for whom the Wohngeld
    priority check compared to Bürgergeld yields the same result ∈ {True, False}.
    """
    offset = numpy.max(hh_id) + 1
    wthh_id = numpy.where(
        vorrangprüfungen__wohngeld_vorrang_vor_arbeitslosengeld_2_bg
        | vorrangprüfungen__wohngeld_und_kinderzuschlag_vorrang_vor_arbeitslosengeld_2_bg,
        hh_id + offset,
        hh_id,
    )
    return _reorder_ids(wthh_id, xnp)


@group_creation_function()
def sn_id(
    p_id: numpy.ndarray,
    familie__p_id_ehepartner: numpy.ndarray,
    einkommensteuer__gemeinsam_veranlagt: numpy.ndarray,
    xnp: ModuleType,
) -> numpy.ndarray:
    """Steuernummer.

    Spouses filing taxes jointly or individuals.
    """

    n = numpy.max(p_id) + 1

    p_id_ehepartner_or_own_p_id = numpy.where(
        (familie__p_id_ehepartner >= 0) * (einkommensteuer__gemeinsam_veranlagt),
        familie__p_id_ehepartner,
        p_id,
    )

    result = (
        numpy.maximum(p_id, p_id_ehepartner_or_own_p_id)
        + numpy.minimum(p_id, p_id_ehepartner_or_own_p_id) * n
    )

    return _reorder_ids(result, xnp)


def _reorder_ids(ids: numpy.ndarray, xnp: ModuleType) -> numpy.ndarray:
    """Make ID's consecutively numbered."""
    sorting = xnp.argsort(ids)
    ids_sorted = ids[sorting]
    index_after_sort = xnp.arange(ids.shape[0])[sorting]
    # Look for difference from previous entry in sorted array
    diff_to_prev = xnp.where(xnp.diff(ids_sorted) >= 1, 1, 0)
    # Sum up all differences to get new id
    cons_ids = xnp.concatenate((xnp.asarray([0]), xnp.cumsum(diff_to_prev)))
    return cons_ids[xnp.argsort(index_after_sort)]
