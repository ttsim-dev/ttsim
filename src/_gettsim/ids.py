"""Input columns."""

from ttsim import group_creation_function, policy_input
from ttsim.config import numpy_or_jax as np


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
    n = 1000000
    familie__p_id_ehepartner = np.where(
        familie__p_id_ehepartner < 0, p_id, familie__p_id_ehepartner
    )
    result = (
        np.maximum(p_id, familie__p_id_ehepartner)
        + np.minimum(p_id, familie__p_id_ehepartner) * n
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
    n = 1000000
    children = np.isin(p_id, familie__p_id_elternteil_1) + np.isin(
        p_id, familie__p_id_elternteil_2
    )
    fg_id = np.where(
        arbeitslosengeld_2__p_id_einstandspartner < 0,
        p_id + p_id * n,
        np.maximum(p_id, arbeitslosengeld_2__p_id_einstandspartner)
        + np.minimum(p_id, arbeitslosengeld_2__p_id_einstandspartner) * n,
    )
    fg_id = np.where(
        (familie__p_id_elternteil_1 >= 0)
        * (hh_id == hh_id[familie__p_id_elternteil_1])
        * alter
        < 25 * (1 - children),
        fg_id[familie__p_id_elternteil_1],
        fg_id,
    )
    fg_id = np.where(
        (familie__p_id_elternteil_2 >= 0)
        * (hh_id == hh_id[familie__p_id_elternteil_2])
        * alter
        < 25 * (1 - children),
        fg_id[familie__p_id_elternteil_1],
        fg_id,
    )

    return fg_id


@group_creation_function()
def bg_id(
    fg_id: np.ndarray,
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
    n = 1000000
    p_id = np.arange(fg_id.shape[0])
    hh_id = np.where(
        np.logical_and(arbeitslosengeld_2__eigenbedarf_gedeckt, alter < 25),
        p_id + p_id * n,
        fg_id,
    )
    return hh_id


@group_creation_function()
def eg_id(
    arbeitslosengeld_2__p_id_einstandspartner: np.ndarray,
    p_id: np.ndarray,
) -> np.ndarray:
    """Einstandsgemeinschaft / Einstandspartner according to SGB II.

    A couple whose members are deemed to be responsible for each other.
    """
    n = 1000000
    arbeitslosengeld_2__p_id_einstandspartner = np.where(
        arbeitslosengeld_2__p_id_einstandspartner < 0,
        p_id,
        arbeitslosengeld_2__p_id_einstandspartner,
    )
    result = (
        np.maximum(p_id, arbeitslosengeld_2__p_id_einstandspartner)
        + np.minimum(p_id, arbeitslosengeld_2__p_id_einstandspartner) * n
    )

    return np.array(result)


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
    p_id = np.arange(hh_id.shape[0])
    hh_id = np.where(
        np.logical_or(
            vorrangprüfungen__wohngeld_vorrang_vor_arbeitslosengeld_2_bg,
            vorrangprüfungen__wohngeld_und_kinderzuschlag_vorrang_vor_arbeitslosengeld_2_bg,
        ),
        hh_id * hh_id.shape[0] + p_id,
        hh_id,
    )
    return hh_id


@group_creation_function()
def sn_id(
    p_id: np.ndarray,
    familie__p_id_ehepartner: np.ndarray,
    einkommensteuer__gemeinsam_veranlagt: np.ndarray,
) -> np.ndarray:
    """Steuernummer.

    Spouses filing taxes jointly or individuals.
    """

    n = 1000000
    familie__p_id_ehepartner = np.where(
        np.logical_and(
            familie__p_id_ehepartner >= 0, einkommensteuer__gemeinsam_veranlagt
        ),
        familie__p_id_ehepartner,
        p_id,
    )
    result = (
        np.maximum(p_id, familie__p_id_ehepartner)
        + np.minimum(p_id, familie__p_id_ehepartner) * n
    )

    return result
