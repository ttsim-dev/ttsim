"""Input columns."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ttsim.tt_dag_elements import group_creation_function, policy_input

if TYPE_CHECKING:
    from types import ModuleType

    from ttsim.interface_dag_elements.typing import BoolColumn, IntColumn


@policy_input()
def p_id() -> int:
    """Unique identifier for each person. Always required, must be unique."""


@policy_input()
def hh_id() -> int:
    """Individuals living together in a household in the Wohngeld sense (§5 WoGG)."""


@group_creation_function()
def ehe_id(
    p_id: IntColumn,
    familie__p_id_ehepartner: IntColumn,
    xnp: ModuleType,
) -> IntColumn:
    """Couples that are either married or in a civil union."""
    n = xnp.max(p_id) + 1
    p_id_ehepartner_or_own_p_id = xnp.where(
        familie__p_id_ehepartner < 0,
        p_id,
        familie__p_id_ehepartner,
    )
    result = (
        xnp.maximum(p_id, p_id_ehepartner_or_own_p_id)
        + xnp.minimum(p_id, p_id_ehepartner_or_own_p_id) * n
    )

    return result


@group_creation_function()
def fg_id(
    arbeitslosengeld_2__p_id_einstandspartner: IntColumn,
    p_id: IntColumn,
    hh_id: IntColumn,
    alter: IntColumn,
    familie__p_id_elternteil_1: IntColumn,
    familie__p_id_elternteil_2: IntColumn,
    xnp: ModuleType,
) -> IntColumn:
    """Familiengemeinschaft. Base unit for some transfers.

    Maximum of two generations, the relevant base unit for Bürgergeld / Arbeitslosengeld
    2, before excluding children who have enough income fend for themselves.
    """
    n = xnp.max(p_id) + 1

    # Sort all arrays according to p_id to make the id equal location in array
    sorting = xnp.argsort(p_id)
    index_after_sort = xnp.argsort(xnp.arange(p_id.shape[0])[sorting])
    sorted_p_id = p_id[sorting]
    sorted_hh_id = hh_id[sorting]
    sorted_alter = alter[sorting]
    sorted_familie__p_id_elternteil_1 = familie__p_id_elternteil_1[sorting]
    sorted_familie__p_id_elternteil_2 = familie__p_id_elternteil_2[sorting]
    sorted_arbeitslosengeld_2__p_id_einstandspartner = (
        arbeitslosengeld_2__p_id_einstandspartner[sorting]
    )

    children = xnp.isin(sorted_p_id, sorted_familie__p_id_elternteil_1) | xnp.isin(
        sorted_p_id,
        sorted_familie__p_id_elternteil_2,
    )

    # Assign the same fg_id to everybody who has an Einstandspartner,
    # otherwise create a new one from p_id
    out = xnp.where(
        sorted_arbeitslosengeld_2__p_id_einstandspartner < 0,
        sorted_p_id + sorted_p_id * n,
        xnp.maximum(sorted_p_id, sorted_arbeitslosengeld_2__p_id_einstandspartner)
        + xnp.minimum(sorted_p_id, sorted_arbeitslosengeld_2__p_id_einstandspartner)
        * n,
    )

    out = _assign_parents_fg_id(
        fg_id=out,
        p_id=sorted_p_id,
        p_id_elternteil_loc=sorted_familie__p_id_elternteil_1,
        hh_id=sorted_hh_id,
        alter=sorted_alter,
        children=children,
        n=n,
        xnp=xnp,
    )
    out = _assign_parents_fg_id(
        fg_id=out,
        p_id=sorted_p_id,
        p_id_elternteil_loc=sorted_familie__p_id_elternteil_2,
        hh_id=sorted_hh_id,
        alter=sorted_alter,
        children=children,
        n=n,
        xnp=xnp,
    )

    return out[index_after_sort]


def _assign_parents_fg_id(
    fg_id: IntColumn,
    p_id: IntColumn,
    p_id_elternteil_loc: IntColumn,
    hh_id: IntColumn,
    alter: IntColumn,
    children: IntColumn,
    n: IntColumn,
    xnp: ModuleType,
) -> IntColumn:
    """Return the fg_id of the child's parents."""
    # TODO(@MImmesberger): Remove hard-coded number
    # https://github.com/iza-institute-of-labor-economics/gettsim/issues/668

    return xnp.where(
        (p_id_elternteil_loc >= 0)
        * (fg_id == p_id + p_id * n)
        * (hh_id == hh_id[p_id_elternteil_loc])
        * (alter < 25)  # noqa: PLR2004
        * (1 - children),
        fg_id[p_id_elternteil_loc],
        fg_id,
    )


@group_creation_function()
def bg_id(
    fg_id: IntColumn,
    p_id: IntColumn,
    arbeitslosengeld_2__eigenbedarf_gedeckt: BoolColumn,
    alter: IntColumn,
    xnp: ModuleType,
) -> IntColumn:
    """Bedarfsgemeinschaft. Relevant unit for Bürgergeld / Arbeitslosengeld 2.

    Familiengemeinschaft except for children who have enough income to fend for
    themselves.
    """
    offset = xnp.max(fg_id) + 1
    # TODO(@MImmesberger): Remove input variable eigenbedarf_gedeckt
    # once Bedarfsgemeinschaften are fully endogenous
    # https://github.com/iza-institute-of-labor-economics/gettsim/issues/763

    # TODO(@MImmesberger): Remove hard-coded number
    # https://github.com/iza-institute-of-labor-economics/gettsim/issues/668
    return xnp.where(
        (arbeitslosengeld_2__eigenbedarf_gedeckt) * (alter < 25),  # noqa: PLR2004
        offset + p_id,
        fg_id,
    )


@group_creation_function()
def eg_id(
    arbeitslosengeld_2__p_id_einstandspartner: IntColumn,
    p_id: IntColumn,
    xnp: ModuleType,
) -> IntColumn:
    """Einstandsgemeinschaft / Einstandspartner according to SGB II.

    A couple whose members are deemed to be responsible for each other.
    """
    n = xnp.max(p_id) + 1
    p_id_einstandspartner__or_own_p_id = xnp.where(
        arbeitslosengeld_2__p_id_einstandspartner < 0,
        p_id,
        arbeitslosengeld_2__p_id_einstandspartner,
    )

    return (
        xnp.maximum(p_id, p_id_einstandspartner__or_own_p_id)
        + xnp.minimum(p_id, p_id_einstandspartner__or_own_p_id) * n
    )


@group_creation_function()
def wthh_id(
    hh_id: IntColumn,
    vorrangprüfungen__wohngeld_vorrang_vor_arbeitslosengeld_2_bg: BoolColumn,
    vorrangprüfungen__wohngeld_und_kinderzuschlag_vorrang_vor_arbeitslosengeld_2_bg: BoolColumn,
    xnp: ModuleType,
) -> IntColumn:
    """Wohngeldrechtlicher Teilhaushalt.

    The relevant unit for Wohngeld. Members of a household for whom the Wohngeld
    priority check compared to Bürgergeld yields the same result ∈ {True, False}.
    """
    offset = xnp.max(hh_id) + 1

    return xnp.where(
        vorrangprüfungen__wohngeld_vorrang_vor_arbeitslosengeld_2_bg
        | vorrangprüfungen__wohngeld_und_kinderzuschlag_vorrang_vor_arbeitslosengeld_2_bg,
        hh_id + offset,
        hh_id,
    )


@group_creation_function()
def sn_id(
    p_id: IntColumn,
    familie__p_id_ehepartner: IntColumn,
    einkommensteuer__gemeinsam_veranlagt: BoolColumn,
    xnp: ModuleType,
) -> IntColumn:
    """Steuernummer. Spouses filing taxes jointly or individuals."""
    n = xnp.max(p_id) + 1

    p_id_ehepartner_or_own_p_id = xnp.where(
        (familie__p_id_ehepartner >= 0) * (einkommensteuer__gemeinsam_veranlagt),
        familie__p_id_ehepartner,
        p_id,
    )

    return (
        xnp.maximum(p_id, p_id_ehepartner_or_own_p_id)
        + xnp.minimum(p_id, p_id_ehepartner_or_own_p_id) * n
    )
