"""Input columns."""

from __future__ import annotations

from ttsim.config import numpy_or_jax as np
from ttsim.tt_dag_elements import group_creation_function, policy_input


@policy_input()
def p_id() -> int:
    """Unique identifier for each person. Always required, must be unique."""


@policy_input()
def hh_id() -> int:
    """Individuals living together in a household in the Wohngeld sense (§5 WoGG)."""


@group_creation_function()
def ehe_id(
    p_id: np.ndarray,
    familie__p_id_ehepartner: np.ndarray,
) -> np.ndarray:
    """Couples that are either married or in a civil union."""
    n = np.max(p_id) + 1
    p_id_ehepartner_or_own_p_id = np.where(
        familie__p_id_ehepartner < 0, p_id, familie__p_id_ehepartner
    )
    result = (
        np.maximum(p_id, p_id_ehepartner_or_own_p_id)
        + np.minimum(p_id, p_id_ehepartner_or_own_p_id) * n
    )

    return result


@group_creation_function()
def fg_id(
    arbeitslosengeld_2__p_id_einstandspartner: np.ndarray,
    p_id: np.ndarray,
    hh_id: np.ndarray,
    alter: np.ndarray,
    familie__p_id_elternteil_1: np.ndarray,
    familie__p_id_elternteil_2: np.ndarray,
) -> np.ndarray:
    """Familiengemeinschaft. Base unit for some transfers.

    Maximum of two generations, the relevant base unit for Bürgergeld / Arbeitslosengeld
    2, before excluding children who have enough income fend for themselves.
    """
    n = np.max(p_id) + 1

    # Get the array index for all p_ids of parents
    p_id_elternteil_1_loc = familie__p_id_elternteil_1
    p_id_elternteil_2_loc = familie__p_id_elternteil_2
    for i in range(p_id.shape[0]):
        p_id_elternteil_1_loc = np.where(
            familie__p_id_elternteil_1 == p_id[i], i, p_id_elternteil_1_loc
        )
        p_id_elternteil_2_loc = np.where(
            familie__p_id_elternteil_2 == p_id[i], i, p_id_elternteil_2_loc
        )

    children = np.isin(p_id, familie__p_id_elternteil_1) | np.isin(
        p_id, familie__p_id_elternteil_2
    )

    # Assign the same fg_id to everybody who has an Einstandspartner,
    # otherwise create a new one from p_id
    out = np.where(
        arbeitslosengeld_2__p_id_einstandspartner < 0,
        p_id + p_id * n,
        np.maximum(p_id, arbeitslosengeld_2__p_id_einstandspartner)
        + np.minimum(p_id, arbeitslosengeld_2__p_id_einstandspartner) * n,
    )

    out = _assign_parents_fg_id(
        out, p_id, p_id_elternteil_1_loc, hh_id, alter, children, n
    )
    out = _assign_parents_fg_id(
        out, p_id, p_id_elternteil_2_loc, hh_id, alter, children, n
    )

    return out


def _assign_parents_fg_id(
    fg_id: np.ndarray,
    p_id: np.ndarray,
    p_id_elternteil_loc: np.ndarray,
    hh_id: np.ndarray,
    alter: np.ndarray,
    children: np.ndarray,
    n: np.ndarray,
) -> np.ndarray:
    """Get the fg_id of the childs parents.

    If the child is not married, has no children, is under 25 and in the same household,
    assign the fg_id of its parents."""

    # TODO(@MImmesberger): Remove input variable eigenbedarf_gedeckt
    # once Bedarfsgemeinschaften are fully endogenous
    # https://github.com/iza-institute-of-labor-economics/gettsim/issues/763
    # TODO(@MImmesberger): Remove hard-coded number
    # https://github.com/iza-institute-of-labor-economics/gettsim/issues/668

    return np.where(
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
    fg_id: np.ndarray,
    p_id: np.ndarray,
    arbeitslosengeld_2__eigenbedarf_gedeckt: np.ndarray,
    alter: np.ndarray,
) -> np.ndarray:
    """Bedarfsgemeinschaft

    Familiengemeinschaft except for children who have enough income to fend for
    themselves. Relevant unit for Bürgergeld / Arbeitslosengeld 2
    """
    # TODO(@MImmesberger): Remove input variable eigenbedarf_gedeckt
    # once Bedarfsgemeinschaften are fully endogenous
    # https://github.com/iza-institute-of-labor-economics/gettsim/issues/763

    # TODO(@MImmesberger): Remove hard-coded number
    # https://github.com/iza-institute-of-labor-economics/gettsim/issues/668
    offset = np.max(fg_id) + 1
    # Create new id for everyone who is not part of the Bedarfsgemeinschaft

    return np.where(
        (arbeitslosengeld_2__eigenbedarf_gedeckt) * (alter < 25),
        offset + p_id,
        fg_id,
    )


@group_creation_function()
def eg_id(
    arbeitslosengeld_2__p_id_einstandspartner: np.ndarray,
    p_id: np.ndarray,
) -> np.ndarray:
    """Einstandsgemeinschaft / Einstandspartner according to SGB II.

    A couple whose members are deemed to be responsible for each other.
    """
    n = np.max(p_id) + 1
    p_id_einstandspartner__or_own_p_id = np.where(
        arbeitslosengeld_2__p_id_einstandspartner < 0,
        p_id,
        arbeitslosengeld_2__p_id_einstandspartner,
    )

    return (
        np.maximum(p_id, p_id_einstandspartner__or_own_p_id)
        + np.minimum(p_id, p_id_einstandspartner__or_own_p_id) * n
    )


@group_creation_function()
def wthh_id(
    hh_id: np.ndarray,
    vorrangprüfungen__wohngeld_vorrang_vor_arbeitslosengeld_2_bg: np.ndarray,
    vorrangprüfungen__wohngeld_und_kinderzuschlag_vorrang_vor_arbeitslosengeld_2_bg: np.ndarray,
) -> np.ndarray:
    """Wohngeldrechtlicher Teilhaushalt.

    The relevant unit for Wohngeld. Members of a household for whom the Wohngeld
    priority check compared to Bürgergeld yields the same result ∈ {True, False}.
    """
    offset = np.max(hh_id) + 1

    return np.where(
        vorrangprüfungen__wohngeld_vorrang_vor_arbeitslosengeld_2_bg
        | vorrangprüfungen__wohngeld_und_kinderzuschlag_vorrang_vor_arbeitslosengeld_2_bg,
        hh_id + offset,
        hh_id,
    )


@group_creation_function()
def sn_id(
    p_id: np.ndarray,
    familie__p_id_ehepartner: np.ndarray,
    einkommensteuer__gemeinsam_veranlagt: np.ndarray,
) -> np.ndarray:
    """Steuernummer.

    Spouses filing taxes jointly or individuals.
    """

    n = np.max(p_id) + 1

    p_id_ehepartner_or_own_p_id = np.where(
        (familie__p_id_ehepartner >= 0) * (einkommensteuer__gemeinsam_veranlagt),
        familie__p_id_ehepartner,
        p_id,
    )

    return (
        np.maximum(p_id, p_id_ehepartner_or_own_p_id)
        + np.minimum(p_id, p_id_ehepartner_or_own_p_id) * n
    )
