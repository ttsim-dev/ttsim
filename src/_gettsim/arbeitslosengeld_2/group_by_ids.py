from collections import Counter

import numpy

from ttsim import AggregateByGroupSpec, AggregationType, group_creation_function

# TODO(@MImmesberger): Many of these keys can go once we have `_eg` for SGB XII.
# https://github.com/iza-institute-of-labor-economics/gettsim/issues/738
aggregation_specs = (
    AggregateByGroupSpec(
        target="anzahl_erwachsene_fg",
        source="familie__erwachsen",
        agg=AggregationType.SUM,
    ),
    AggregateByGroupSpec(
        target="anzahl_kinder_fg",
        source="familie__kind",
        agg=AggregationType.SUM,
    ),
    AggregateByGroupSpec(
        target="anzahl_kinder_bis_6_fg",
        source="familie__kind_bis_6",
        agg=AggregationType.SUM,
    ),
    AggregateByGroupSpec(
        target="anzahl_kinder_bis_15_fg",
        source="familie__kind_bis_15",
        agg=AggregationType.SUM,
    ),
    AggregateByGroupSpec(
        target="anzahl_erwachsene_bg",
        source="familie__erwachsen",
        agg=AggregationType.SUM,
    ),
    AggregateByGroupSpec(
        target="anzahl_kinder_bg",
        source="familie__kind",
        agg=AggregationType.SUM,
    ),
    AggregateByGroupSpec(
        target="anzahl_personen_bg", source=None, agg=AggregationType.COUNT
    ),
    AggregateByGroupSpec(
        target="anzahl_kinder_bis_17_bg",
        source="familie__kind_bis_17",
        agg=AggregationType.SUM,
    ),
    AggregateByGroupSpec(
        target="alleinerziehend_bg",
        source="familie__alleinerziehend",
        agg=AggregationType.ANY,
    ),
    AggregateByGroupSpec(
        target="anzahl_erwachsene_eg",
        source="familie__erwachsen",
        agg=AggregationType.SUM,
    ),
    AggregateByGroupSpec(
        target="anzahl_kinder_eg",
        source="familie__kind",
        agg=AggregationType.SUM,
    ),
    AggregateByGroupSpec(
        target="anzahl_personen_eg", source=None, agg=AggregationType.COUNT
    ),
)


@group_creation_function()
def fg_id(  # noqa: PLR0912
    p_id_einstandspartner: numpy.ndarray[int],
    p_id: numpy.ndarray[int],
    hh_id: numpy.ndarray[int],
    alter: numpy.ndarray[int],
    familie__p_id_elternteil_1: numpy.ndarray[int],
    familie__p_id_elternteil_2: numpy.ndarray[int],
) -> numpy.ndarray[int]:
    """
    Compute the ID of the Familiengemeinschaft for each person.
    """
    # Build indexes
    p_id_to_index = {}
    p_id_to_p_ids_children = {}

    for index, current_p_id in enumerate(p_id):
        # Fast access from p_id to index
        p_id_to_index[current_p_id] = index

        # Fast access from p_id to p_ids of children
        current_familie__p_id_elternteil_1 = familie__p_id_elternteil_1[index]
        current_familie__p_id_elternteil_2 = familie__p_id_elternteil_2[index]

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

    for index, current_p_id in enumerate(p_id):
        # Already assigned a fg_id to this p_id via einstandspartner /
        # parent
        if current_p_id in p_id_to_fg_id:
            continue

        p_id_to_fg_id[current_p_id] = next_fg_id

        current_hh_id = hh_id[index]
        current_p_id_einstandspartner = p_id_einstandspartner[index]
        current_p_id_children = p_id_to_p_ids_children.get(current_p_id, [])

        # Assign fg to children
        for current_p_id_child in current_p_id_children:
            child_index = p_id_to_index[current_p_id_child]
            child_hh_id = hh_id[child_index]
            child_alter = alter[child_index]
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
                child_hh_id = hh_id[child_index]
                child_alter = alter[child_index]
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
    result = [p_id_to_fg_id[current_p_id] for current_p_id in p_id]
    return numpy.asarray(result)


@group_creation_function()
def bg_id(
    fg_id: numpy.ndarray[int],
    eigenbedarf_gedeckt: numpy.ndarray[bool],
    alter: numpy.ndarray[int],
) -> numpy.ndarray[int]:
    """
    Compute the ID of the Bedarfsgemeinschaft for each person.
    """
    # TODO(@MImmesberger): Remove input variable eigenbedarf_gedeckt
    # once Bedarfsgemeinschaften are fully endogenous
    # https://github.com/iza-institute-of-labor-economics/gettsim/issues/763
    counter = Counter()
    result = []

    for index, current_fg_id in enumerate(fg_id):
        current_alter = alter[index]
        current_eigenbedarf_gedeckt = eigenbedarf_gedeckt[index]
        # TODO(@MImmesberger): Remove hard-coded number
        # https://github.com/iza-institute-of-labor-economics/gettsim/issues/668
        if current_alter < 25 and current_eigenbedarf_gedeckt:
            counter[current_fg_id] += 1
            result.append(current_fg_id * 100 + counter[current_fg_id])
        else:
            result.append(current_fg_id * 100)

    return numpy.asarray(result)


@group_creation_function()
def eg_id(
    p_id_einstandspartner: numpy.ndarray[int],
    p_id: numpy.ndarray[int],
) -> numpy.ndarray[int]:
    """
    Compute the ID of the Einstandsgemeinschaft for each person.
    """
    p_id_to_eg_id = {}
    next_eg_id = 0
    result = []

    for index, current_p_id in enumerate(p_id):
        current_p_id_einstandspartner = p_id_einstandspartner[index]

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

    return numpy.asarray(result)
