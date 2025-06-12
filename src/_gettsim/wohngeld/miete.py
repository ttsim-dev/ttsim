"""Renting costs relevant for housing benefit calculation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ttsim.tt_dag_elements import (
    ConsecutiveInt1dLookupTableParamValue,
    ConsecutiveInt2dLookupTableParamValue,
    get_consecutive_int_1d_lookup_table_param_value,
    get_consecutive_int_2d_lookup_table_param_value,
    param_function,
    policy_function,
)

if TYPE_CHECKING:
    from types import ModuleType

    import numpy


@dataclass(frozen=True)
class LookupTableBaujahr:
    baujahre: numpy.ndarray
    lookup_table: numpy.ndarray
    lookup_base_to_subtract_cols: numpy.ndarray
    lookup_base_to_subtract_rows: numpy.ndarray


@param_function(
    start_date="1984-01-01", end_date="2008-12-31", leaf_name="max_miete_m_lookup"
)
def max_miete_m_lookup_mit_baujahr(
    raw_max_miete_m_nach_baujahr: dict[int | str, dict[int, dict[int, float]]],
    max_anzahl_personen: dict[str, int],
    xnp: ModuleType,
) -> LookupTableBaujahr:
    """Maximum rent considered in Wohngeld calculation."""
    tmp = raw_max_miete_m_nach_baujahr.copy()
    per_additional_person = tmp.pop("jede_weitere_person")
    max_n_p_defined = max(tmp.keys())
    assert all(isinstance(i, int) for i in tmp)
    baujahre = sorted(tmp[1].keys())
    values = []
    subtract_cols = []
    subtract_rows = []
    for baujahr in baujahre:
        this_dict = {n_p: tmp[n_p][baujahr] for n_p in tmp}
        for n_p in range(max_n_p_defined + 1, max_anzahl_personen["indizierung"] + 1):  # type: ignore[operator]
            this_dict[n_p] = {
                ms: this_dict[max_n_p_defined][ms]
                + (n_p - max_n_p_defined) * per_additional_person[baujahr][ms]  # type: ignore[operator]
                for ms in this_dict[max_n_p_defined]
            }
        lookup_table = get_consecutive_int_2d_lookup_table_param_value(
            raw=this_dict, xnp=xnp
        )
        values.append(lookup_table.values_to_look_up)
        subtract_cols.append(lookup_table.base_to_subtract_cols)
        subtract_rows.append(lookup_table.base_to_subtract_rows)

    full_lookup_table = xnp.stack(values, axis=0)
    full_lookup_base_to_subtract_cols = xnp.asarray(subtract_cols)
    full_lookup_base_to_subtract_rows = xnp.asarray(subtract_rows)

    return LookupTableBaujahr(
        baujahre=xnp.asarray(baujahre),
        lookup_table=full_lookup_table,
        lookup_base_to_subtract_cols=full_lookup_base_to_subtract_cols,
        lookup_base_to_subtract_rows=full_lookup_base_to_subtract_rows,
    )


@param_function(start_date="2009-01-01", leaf_name="max_miete_m_lookup")
def max_miete_m_lookup_ohne_baujahr(
    raw_max_miete_m: dict[int | str, dict[int, float]],
    max_anzahl_personen: dict[str, int],
    xnp: ModuleType,
) -> ConsecutiveInt2dLookupTableParamValue:
    """Maximum rent considered in Wohngeld calculation."""
    expanded = raw_max_miete_m.copy()
    per_additional_person = expanded.pop("jede_weitere_person")
    max_n_p_defined = max(expanded.keys())
    assert all(isinstance(i, int) for i in expanded)
    for n_p in range(max_n_p_defined + 1, max_anzahl_personen["indizierung"] + 1):  # type: ignore[operator]
        expanded[n_p] = {
            ms: expanded[max_n_p_defined][ms]
            + (n_p - max_n_p_defined) * per_additional_person[ms]  # type: ignore[operator]
            for ms in expanded[max_n_p_defined]
        }
    return get_consecutive_int_2d_lookup_table_param_value(raw=expanded, xnp=xnp)


@param_function(start_date="1984-01-01")
def min_miete_lookup(
    raw_min_miete_m: dict[int, float],
    max_anzahl_personen: dict[str, int],
    xnp: ModuleType,
) -> ConsecutiveInt1dLookupTableParamValue:
    """Minimum rent considered in Wohngeld calculation."""
    max_n_p_normal = max_anzahl_personen["normale_berechnung"]
    assert max(raw_min_miete_m.keys()) == max_n_p_normal, (
        "The maximum number of persons for the normal calculation of the basic"
        "Wohngeld formula `max_anzahl_personen['normale_berechnung'] "
        f"(got: {max_n_p_normal}) must be the same as the maximum number of household "
        "members in `koeffizienten_berechnungsformel` "
        f"(got: {max(raw_min_miete_m.keys())})"
    )
    expanded = raw_min_miete_m.copy()
    for n_p in range(max_n_p_normal + 1, max_anzahl_personen["indizierung"] + 1):
        expanded[n_p] = raw_min_miete_m[max_n_p_normal]
    return get_consecutive_int_1d_lookup_table_param_value(raw=expanded, xnp=xnp)


@param_function(start_date="2021-01-01")
def heizkostenentlastung_m_lookup(
    raw_heizkostenentlastung_m: dict[int | str, float],
    max_anzahl_personen: dict[str, int],
    xnp: ModuleType,
) -> ConsecutiveInt1dLookupTableParamValue:
    """Heizkostenentlastung as a lookup table."""
    expanded = raw_heizkostenentlastung_m.copy()
    per_additional_person = expanded.pop("jede_weitere_person")
    max_n_p_defined = max(expanded.keys())
    assert all(isinstance(i, int) for i in expanded)
    for n_p in range(max_n_p_defined + 1, max_anzahl_personen["indizierung"] + 1):  # type: ignore[operator]
        expanded[n_p] = (
            expanded[max_n_p_defined] + (n_p - max_n_p_defined) * per_additional_person  # type: ignore[operator]
        )
    return get_consecutive_int_1d_lookup_table_param_value(raw=expanded, xnp=xnp)


@param_function(start_date="2023-01-01")
def dauerhafte_heizkostenkomponente_m_lookup(
    raw_dauerhafte_heizkostenkomponente_m: dict[int | str, float],
    max_anzahl_personen: dict[str, int],
    xnp: ModuleType,
) -> ConsecutiveInt1dLookupTableParamValue:
    """Dauerhafte Heizkostenenkomponente as a lookup table."""
    expanded = raw_dauerhafte_heizkostenkomponente_m.copy()
    per_additional_person = expanded.pop("jede_weitere_person")
    max_n_p_defined = max(expanded.keys())
    assert all(isinstance(i, int) for i in expanded)
    for n_p in range(max_n_p_defined + 1, max_anzahl_personen["indizierung"] + 1):  # type: ignore[operator]
        expanded[n_p] = (
            expanded[max_n_p_defined] + (n_p - max_n_p_defined) * per_additional_person  # type: ignore[operator]
        )
    return get_consecutive_int_1d_lookup_table_param_value(raw=expanded, xnp=xnp)


@param_function(start_date="2023-01-01")
def klimakomponente_m_lookup(
    raw_klimakomponente_m: dict[int | str, float],
    max_anzahl_personen: dict[str, int],
    xnp: ModuleType,
) -> ConsecutiveInt1dLookupTableParamValue:
    """Klimakomponente as a lookup table."""
    expanded = raw_klimakomponente_m.copy()
    per_additional_person = expanded.pop("jede_weitere_person")
    max_n_p_defined = max(expanded.keys())
    assert all(isinstance(i, int) for i in expanded)
    for n_p in range(max_n_p_defined + 1, max_anzahl_personen["indizierung"] + 1):  # type: ignore[operator]
        expanded[n_p] = (
            expanded[max_n_p_defined] + (n_p - max_n_p_defined) * per_additional_person  # type: ignore[operator]
        )
    return get_consecutive_int_1d_lookup_table_param_value(raw=expanded, xnp=xnp)


@policy_function()
def miete_m_wthh(
    miete_m_hh: float,
    anzahl_personen_wthh: int,
    anzahl_personen_hh: int,
) -> float:
    """Rent considered in housing benefit calculation on wohngeldrechtlicher
    Teilhaushalt level.

    This target is used to calculate the actual Wohngeld of all Bedarfsgemeinschaften
    that passed the priority check against Arbeitslosengeld II / Bürgergeld.
    """
    return miete_m_hh * (anzahl_personen_wthh / anzahl_personen_hh)


@policy_function()
def miete_m_bg(
    miete_m_hh: float,
    arbeitslosengeld_2__anzahl_personen_bg: int,
    anzahl_personen_hh: int,
) -> float:
    """Rent considered in housing benefit calculation on BG level.

    This target is used for the priority check calculation against Arbeitslosengeld II /
    Bürgergeld on the Bedarfsgemeinschaft level.
    """
    return miete_m_hh * (arbeitslosengeld_2__anzahl_personen_bg / anzahl_personen_hh)


@policy_function()
def min_miete_m_hh(
    anzahl_personen_hh: int, min_miete_lookup: ConsecutiveInt1dLookupTableParamValue
) -> float:
    """Minimum rent considered in Wohngeld calculation."""
    return min_miete_lookup.values_to_look_up[
        anzahl_personen_hh - min_miete_lookup.base_to_subtract
    ]


@policy_function(
    start_date="1984-01-01",
    end_date="2008-12-31",
    leaf_name="miete_m_hh",
)
def miete_m_hh_mit_baujahr(
    mietstufe: int,
    wohnen__baujahr_immobilie_hh: int,
    anzahl_personen_hh: int,
    wohnen__bruttokaltmiete_m_hh: float,
    min_miete_m_hh: float,
    max_miete_m_lookup: LookupTableBaujahr,
    xnp: ModuleType,
) -> float:
    """Rent considered in housing benefit calculation on household level until 2008."""

    selected_bin_index = xnp.searchsorted(
        max_miete_m_lookup.baujahre,
        wohnen__baujahr_immobilie_hh,
        side="left",
    )
    max_miete_m = max_miete_m_lookup.lookup_table[
        selected_bin_index,
        anzahl_personen_hh - max_miete_m_lookup.lookup_base_to_subtract_rows[selected_bin_index],
        mietstufe - max_miete_m_lookup.lookup_base_to_subtract_cols[selected_bin_index],
    ]  # fmt: skip
    return max(min(wohnen__bruttokaltmiete_m_hh, max_miete_m), min_miete_m_hh)


@policy_function(
    start_date="2009-01-01",
    end_date="2020-12-31",
    leaf_name="miete_m_hh",
)
def miete_m_hh_ohne_baujahr_ohne_heizkostenentlastung(
    mietstufe: int,
    anzahl_personen_hh: int,
    wohnen__bruttokaltmiete_m_hh: float,
    min_miete_m_hh: float,
    max_miete_m_lookup: ConsecutiveInt2dLookupTableParamValue,
) -> float:
    """Rent considered in housing benefit since 2009."""

    max_miete_m = max_miete_m_lookup.values_to_look_up[
        anzahl_personen_hh - max_miete_m_lookup.base_to_subtract_rows,
        mietstufe - max_miete_m_lookup.base_to_subtract_cols,
    ]

    return max(min(wohnen__bruttokaltmiete_m_hh, max_miete_m), min_miete_m_hh)


@policy_function(
    start_date="2021-01-01",
    end_date="2022-12-31",
    leaf_name="miete_m_hh",
)
def miete_m_hh_mit_heizkostenentlastung(
    mietstufe: int,
    anzahl_personen_hh: int,
    wohnen__bruttokaltmiete_m_hh: float,
    min_miete_m_hh: float,
    max_miete_m_lookup: ConsecutiveInt2dLookupTableParamValue,
    heizkostenentlastung_m_lookup: ConsecutiveInt1dLookupTableParamValue,
) -> float:
    """Rent considered in housing benefit since 2009."""
    max_miete_m = max_miete_m_lookup.values_to_look_up[
        anzahl_personen_hh - max_miete_m_lookup.base_to_subtract_rows,
        mietstufe - max_miete_m_lookup.base_to_subtract_cols,
    ]

    heating_allowance_m = heizkostenentlastung_m_lookup.values_to_look_up[
        anzahl_personen_hh - heizkostenentlastung_m_lookup.base_to_subtract
    ]

    return (
        max(min(wohnen__bruttokaltmiete_m_hh, max_miete_m), min_miete_m_hh)
        + heating_allowance_m
    )


@policy_function(
    start_date="2023-01-01",
    leaf_name="miete_m_hh",
)
def miete_m_hh_mit_heizkostenentlastung_dauerhafte_heizkostenkomponente_klimakomponente(
    mietstufe: int,
    anzahl_personen_hh: int,
    wohnen__bruttokaltmiete_m_hh: float,
    min_miete_m_hh: float,
    max_miete_m_lookup: ConsecutiveInt2dLookupTableParamValue,
    heizkostenentlastung_m_lookup: ConsecutiveInt1dLookupTableParamValue,
    dauerhafte_heizkostenkomponente_m_lookup: ConsecutiveInt1dLookupTableParamValue,
    klimakomponente_m_lookup: ConsecutiveInt1dLookupTableParamValue,
) -> float:
    """Rent considered in housing benefit since 2009."""
    max_miete_m = max_miete_m_lookup.values_to_look_up[
        anzahl_personen_hh - max_miete_m_lookup.base_to_subtract_rows,
        mietstufe - max_miete_m_lookup.base_to_subtract_cols,
    ]

    heizkostenentlastung = heizkostenentlastung_m_lookup.values_to_look_up[
        anzahl_personen_hh - heizkostenentlastung_m_lookup.base_to_subtract
    ]
    dauerhafte_heizkostenkomponente = (
        dauerhafte_heizkostenkomponente_m_lookup.values_to_look_up[
            anzahl_personen_hh
            - dauerhafte_heizkostenkomponente_m_lookup.base_to_subtract
        ]
    )
    klimakomponente = klimakomponente_m_lookup.values_to_look_up[
        anzahl_personen_hh - klimakomponente_m_lookup.base_to_subtract
    ]
    return (
        max(
            min(wohnen__bruttokaltmiete_m_hh, max_miete_m + klimakomponente),
            min_miete_m_hh,
        )
        + heizkostenentlastung
        + dauerhafte_heizkostenkomponente
    )
