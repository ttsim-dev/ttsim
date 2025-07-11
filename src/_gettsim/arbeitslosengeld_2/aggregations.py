"""Aggregations for Arbeitslosengeld II."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ttsim.tt_dag_elements import AggType, agg_by_group_function, join, policy_function

if TYPE_CHECKING:
    from ttsim.tt_dag_elements.typing import BoolColumn, IntColumn, ModuleType


# TODO(@MImmesberger): Many of these keys can go once we have `_eg` for SGB XII.
# https://github.com/iza-institute-of-labor-economics/gettsim/issues/738
@agg_by_group_function(start_date="2005-01-01", agg_type=AggType.SUM)
def anzahl_kinder_bg(ist_kind_in_bedarfsgemeinschaft: bool, bg_id: int) -> int:
    pass


@agg_by_group_function(start_date="2005-01-01", agg_type=AggType.COUNT)
def anzahl_personen_bg(bg_id: int) -> int:
    pass


@agg_by_group_function(start_date="2005-01-01", agg_type=AggType.SUM)
def anzahl_kinder_bis_17_bg(familie__kind_in_fg_bis_17: bool, bg_id: int) -> int:
    pass


@agg_by_group_function(start_date="2005-01-01", agg_type=AggType.ANY)
def alleinerziehend_bg(familie__alleinerziehend: bool, bg_id: int) -> bool:
    pass


@policy_function()
def kind_in_bg_bis_17(alter: int, ist_kind_in_bedarfsgemeinschaft: bool) -> bool:
    """Child under the age of 18 in Bedarfsgemeinschaft."""
    return ist_kind_in_bedarfsgemeinschaft and (alter <= 17)


@policy_function()
def ist_kind_in_bedarfsgemeinschaft(
    familie__p_id_elternteil_1: IntColumn,
    familie__p_id_elternteil_2: IntColumn,
    p_id: IntColumn,
    bg_id: IntColumn,
    xnp: ModuleType,
) -> BoolColumn:
    """
    Determines whether the given person is a child in a family group.
    """
    bg_id_elternteil_1 = join(
        foreign_key=familie__p_id_elternteil_1,
        primary_key=p_id,
        target=bg_id,
        value_if_foreign_key_is_missing=-1,
        xnp=xnp,
    )
    bg_id_elternteil_2 = join(
        foreign_key=familie__p_id_elternteil_2,
        primary_key=p_id,
        target=bg_id,
        value_if_foreign_key_is_missing=-1,
        xnp=xnp,
    )
    in_gleicher_fg_wie_elternteil_1 = bg_id_elternteil_1 == bg_id
    in_gleicher_fg_wie_elternteil_2 = bg_id_elternteil_2 == bg_id
    return in_gleicher_fg_wie_elternteil_1 | in_gleicher_fg_wie_elternteil_2
