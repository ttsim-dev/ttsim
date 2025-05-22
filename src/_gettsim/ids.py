"""Input columns."""

from __future__ import annotations

from collections import Counter

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
    p_id_to_ehe_id: dict[int, int] = {}
    next_ehe_id = 0
    result: list[int] = []

    for index, current_p_id in enumerate(map(int, p_id)):
        current_p_id_ehepartner = int(familie__p_id_ehepartner[index])

        if current_p_id_ehepartner >= 0 and current_p_id_ehepartner in p_id_to_ehe_id:
            result.append(p_id_to_ehe_id[current_p_id_ehepartner])
            continue

        # New married couple
        result.append(next_ehe_id)
        p_id_to_ehe_id[current_p_id] = next_ehe_id
        next_ehe_id += 1

    return np.array(result)


@group_creation_function()
def fg_id(  # noqa: PLR0912
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
    # Build indexes
    p_id_to_index: dict[int, int] = {}
    p_id_to_p_ids_children: dict[int, list[int]] = {}

    for index, current_p_id in enumerate(map(int, p_id)):
        # Fast access from p_id to index
        p_id_to_index[current_p_id] = index

        # Fast access from p_id to p_ids of children
        current_familie__p_id_elternteil_1 = int(familie__p_id_elternteil_1[index])
        current_familie__p_id_elternteil_2 = int(familie__p_id_elternteil_2[index])

        if current_familie__p_id_elternteil_1 >= 0:
            if current_familie__p_id_elternteil_1 not in p_id_to_p_ids_children:
                p_id_to_p_ids_children[current_familie__p_id_elternteil_1] = []
            p_id_to_p_ids_children[current_familie__p_id_elternteil_1].append(
                current_p_id
            )

        if current_familie__p_id_elternteil_2 >= 0:
            if current_familie__p_id_elternteil_2 not in p_id_to_p_ids_children:
                p_id_to_p_ids_children[current_familie__p_id_elternteil_2] = []
            p_id_to_p_ids_children[current_familie__p_id_elternteil_2].append(
                current_p_id
            )

    p_id_to_fg_id = {}
    next_fg_id = 0

    for index, current_p_id in enumerate(map(int, p_id)):
        # Already assigned a fg_id to this p_id via einstandspartner /
        # parent
        if current_p_id in p_id_to_fg_id:
            continue

        p_id_to_fg_id[current_p_id] = next_fg_id

        current_hh_id = int(hh_id[index])
        current_p_id_einstandspartner = int(
            arbeitslosengeld_2__p_id_einstandspartner[index]
        )
        current_p_id_children = p_id_to_p_ids_children.get(current_p_id, [])

        # Assign fg to children
        for current_p_id_child in current_p_id_children:
            child_index = p_id_to_index[current_p_id_child]
            child_hh_id = int(hh_id[child_index])
            child_alter = int(alter[child_index])
            child_p_id_children = p_id_to_p_ids_children.get(current_p_id_child, [])

            if (
                child_hh_id == current_hh_id
                # TODO (@MImmesberger): Check correct conditions for grown up children
                # https://github.com/iza-institute-of-labor-economics/gettsim/pull/509
                # TODO(@MImmesberger): Remove hard-coded number
                # https://github.com/iza-institute-of-labor-economics/gettsim/issues/668
                and child_alter < 25
                and len(child_p_id_children) == 0
            ):
                p_id_to_fg_id[current_p_id_child] = next_fg_id

        # Assign fg to einstandspartner
        if current_p_id_einstandspartner >= 0:
            p_id_to_fg_id[current_p_id_einstandspartner] = next_fg_id
            current_p_id_einstandspartner_children = p_id_to_p_ids_children.get(
                current_p_id_einstandspartner, []
            )
            # Assign fg to children of einstandspartner
            for current_p_id_child in current_p_id_einstandspartner_children:
                if current_p_id_child in p_id_to_fg_id:
                    continue
                child_index = p_id_to_index[current_p_id_child]
                child_hh_id = int(hh_id[child_index])
                child_alter = int(alter[child_index])
                child_p_id_children = p_id_to_p_ids_children.get(current_p_id_child, [])

                if (
                    child_hh_id == current_hh_id
                    # TODO (@MImmesberger): Check correct conditions for grown up children
                    # https://github.com/iza-institute-of-labor-economics/gettsim/pull/509
                    # TODO(@MImmesberger): Remove hard-coded number
                    # https://github.com/iza-institute-of-labor-economics/gettsim/issues/668
                    and child_alter < 25
                    and len(child_p_id_children) == 0
                ):
                    p_id_to_fg_id[current_p_id_child] = next_fg_id

        next_fg_id += 1

    # Compute result vector
    result = [p_id_to_fg_id[current_p_id] for current_p_id in map(int, p_id)]
    return np.array(result)


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
    counter: Counter[int] = Counter()
    result: list[int] = []

    for index, current_fg_id in enumerate(map(int, fg_id)):
        current_alter = int(alter[index])
        current_eigenbedarf_gedeckt = bool(
            arbeitslosengeld_2__eigenbedarf_gedeckt[index]
        )
        # TODO(@MImmesberger): Remove hard-coded number
        # https://github.com/iza-institute-of-labor-economics/gettsim/issues/668
        if current_alter < 25 and current_eigenbedarf_gedeckt:
            counter[current_fg_id] += 1
            result.append(current_fg_id * 100 + counter[current_fg_id])
        else:
            result.append(current_fg_id * 100)

    return np.array(result)


@group_creation_function()
def eg_id(
    arbeitslosengeld_2__p_id_einstandspartner: np.ndarray,
    p_id: np.ndarray,
) -> np.ndarray:
    """Einstandsgemeinschaft / Einstandspartner according to SGB II.

    A couple whose members are deemed to be responsible for each other.
    """
    p_id_to_eg_id: dict[int, int] = {}
    next_eg_id = 0
    result: list[int] = []

    for index, current_p_id in enumerate(map(int, p_id)):
        current_p_id_einstandspartner = int(
            arbeitslosengeld_2__p_id_einstandspartner[index]
        )

        if (
            current_p_id_einstandspartner >= 0
            and current_p_id_einstandspartner in p_id_to_eg_id
        ):
            result.append(p_id_to_eg_id[current_p_id_einstandspartner])
            continue

        # New Einstandsgemeinschaft
        result.append(next_eg_id)
        p_id_to_eg_id[current_p_id] = next_eg_id
        next_eg_id += 1

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
    result: list[int] = []
    for index, current_hh_id in enumerate(map(int, hh_id)):
        if bool(
            vorrangprüfungen__wohngeld_vorrang_vor_arbeitslosengeld_2_bg[index]
        ) or bool(
            vorrangprüfungen__wohngeld_und_kinderzuschlag_vorrang_vor_arbeitslosengeld_2_bg[
                index
            ]
        ):
            result.append(current_hh_id * 100 + 1)
        else:
            result.append(current_hh_id * 100)

    return np.array(result)


@group_creation_function()
def sn_id(
    p_id: np.ndarray,
    familie__p_id_ehepartner: np.ndarray,
    einkommensteuer__gemeinsam_veranlagt: np.ndarray,
) -> np.ndarray:
    """Steuernummer.

    Spouses filing taxes jointly or individuals.
    """
    p_id_to_sn_id: dict[int, int] = {}
    p_id_to_gemeinsam_veranlagt: dict[int, bool] = {}
    next_sn_id = 0
    result: list[int] = []

    for index, current_p_id in enumerate(map(int, p_id)):
        current_p_id_ehepartner = int(familie__p_id_ehepartner[index])
        current_gemeinsam_veranlagt = bool(einkommensteuer__gemeinsam_veranlagt[index])

        if current_p_id_ehepartner >= 0 and current_p_id_ehepartner in p_id_to_sn_id:
            gemeinsam_veranlagt_ehepartner = p_id_to_gemeinsam_veranlagt[
                current_p_id_ehepartner
            ]

            if current_gemeinsam_veranlagt != gemeinsam_veranlagt_ehepartner:
                message = (
                    f"{current_p_id_ehepartner} and {current_p_id} are "
                    "married, but have different values for "
                    "gemeinsam_veranlagt."
                )
                raise ValueError(message)

            if current_gemeinsam_veranlagt:
                result.append(p_id_to_sn_id[current_p_id_ehepartner])
                continue

        # New Steuersubjekt
        result.append(next_sn_id)
        p_id_to_sn_id[current_p_id] = next_sn_id
        p_id_to_gemeinsam_veranlagt[current_p_id] = current_gemeinsam_veranlagt
        next_sn_id += 1

    return np.array(result)
