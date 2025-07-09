"""This module computes demographic variables directly on the data.

These information are used throughout modules of gettsim.

"""

from __future__ import annotations

from ttsim.tt_dag_elements import AggType, agg_by_group_function, policy_function


@agg_by_group_function(agg_type=AggType.COUNT)
def anzahl_personen_ehe(ehe_id: int) -> int:
    pass


@policy_function()
def kind_bis_2(alter: int) -> bool:
    """Child under the age of 3."""
    return alter <= 2


@policy_function()
def kind_bis_5(alter: int) -> bool:
    """Child under the age of 6."""
    return alter <= 5


@policy_function()
def kind_bis_6(alter: int) -> bool:
    """Child under the age of 7."""
    return alter <= 6


@policy_function()
def kind_bis_15(alter: int) -> bool:
    """Child under the age of 16."""
    return alter <= 15


@policy_function()
def kind_bis_17(alter: int) -> bool:
    """Underage person."""
    return alter <= 17


@policy_function()
def person_ab_18(alter: int) -> bool:
    """Older than 18."""
    return alter >= 18


@policy_function()
def person_ab_25(alter: int) -> bool:
    """Older than 25."""
    return alter >= 25
