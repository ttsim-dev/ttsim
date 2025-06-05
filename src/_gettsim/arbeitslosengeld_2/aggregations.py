"""Aggregations for Arbeitslosengeld II."""

from __future__ import annotations

from ttsim import AggType, agg_by_group_function


# TODO(@MImmesberger): Many of these keys can go once we have `_eg` for SGB XII.
# https://github.com/iza-institute-of-labor-economics/gettsim/issues/738
@agg_by_group_function(start_date="2005-01-01", agg_type=AggType.SUM)
def anzahl_erwachsene_fg(familie__erwachsen: bool, fg_id: int) -> int:
    pass


@agg_by_group_function(start_date="2005-01-01", agg_type=AggType.SUM)
def anzahl_kinder_fg(familie__kind: bool, fg_id: int) -> int:
    pass


@agg_by_group_function(start_date="2005-01-01", agg_type=AggType.SUM)
def anzahl_kinder_bis_6_fg(familie__kind_bis_6: bool, fg_id: int) -> int:
    pass


@agg_by_group_function(start_date="2005-01-01", agg_type=AggType.SUM)
def anzahl_kinder_bis_15_fg(familie__kind_bis_15: bool, fg_id: int) -> int:
    pass


@agg_by_group_function(start_date="2005-01-01", agg_type=AggType.SUM)
def anzahl_kinder_bis_17_fg(familie__kind_bis_17: bool, fg_id: int) -> int:
    pass


@agg_by_group_function(start_date="2005-01-01", agg_type=AggType.SUM)
def anzahl_erwachsene_bg(familie__erwachsen: bool, bg_id: int) -> int:
    pass


@agg_by_group_function(start_date="2005-01-01", agg_type=AggType.SUM)
def anzahl_kinder_bg(familie__kind: bool, bg_id: int) -> int:
    pass


@agg_by_group_function(start_date="2005-01-01", agg_type=AggType.COUNT)
def anzahl_personen_bg(bg_id: int) -> int:
    pass


@agg_by_group_function(start_date="2005-01-01", agg_type=AggType.SUM)
def anzahl_kinder_bis_17_bg(familie__kind_bis_17: bool, bg_id: int) -> int:
    pass


@agg_by_group_function(start_date="2005-01-01", agg_type=AggType.ANY)
def alleinerziehend_bg(familie__alleinerziehend: bool, bg_id: int) -> bool:
    pass


@agg_by_group_function(start_date="2005-01-01", agg_type=AggType.SUM)
def anzahl_erwachsene_eg(familie__erwachsen: bool, eg_id: int) -> int:
    pass


@agg_by_group_function(start_date="2005-01-01", agg_type=AggType.SUM)
def anzahl_kinder_eg(familie__kind: bool, eg_id: int) -> int:
    pass


@agg_by_group_function(start_date="2005-01-01", agg_type=AggType.COUNT)
def anzahl_personen_eg(eg_id: int) -> int:
    pass
