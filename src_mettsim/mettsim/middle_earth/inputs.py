from __future__ import annotations

from ttsim.tt import AggType, FKType, TTSIMUnit, agg_by_group_function, policy_input


@policy_input(unit=TTSIMUnit.DIMENSIONLESS)
def p_id() -> int:
    """Person ID, always required by TTSIM."""


@policy_input(unit=TTSIMUnit.DIMENSIONLESS)
def kin_id() -> int:
    """Kinstead ID."""


@policy_input(
    foreign_key_type=FKType.MUST_NOT_POINT_TO_SELF, unit=TTSIMUnit.DIMENSIONLESS
)
def p_id_parent_1() -> int:
    """Identifier of the first parent."""


@policy_input(
    foreign_key_type=FKType.MUST_NOT_POINT_TO_SELF, unit=TTSIMUnit.DIMENSIONLESS
)
def p_id_parent_2() -> int:
    """Identifier of the second parent."""


@policy_input(
    foreign_key_type=FKType.MUST_NOT_POINT_TO_SELF, unit=TTSIMUnit.DIMENSIONLESS
)
def p_id_spouse() -> int:
    """Identifier of married partner."""


@policy_input(unit=TTSIMUnit.CALENDAR_YEAR)
def birth_year() -> int:
    """Calendar year the person was born in."""


@policy_input(unit=TTSIMUnit.CALENDAR_MONTH)
def birth_month() -> int:
    """Month of birth (1-12)."""


@policy_input(unit=TTSIMUnit.DIMENSIONLESS)
def parent_is_noble() -> bool:
    """Whether at least one parent is noble."""


@agg_by_group_function(agg_type=AggType.ANY, unit=TTSIMUnit.DIMENSIONLESS.PER_FAM)
def parent_is_noble_fam(parent_is_noble: bool, fam_id: int) -> bool:
    pass


@policy_input(unit=TTSIMUnit.CURRENCY)
def wealth() -> float:
    """Wealth of the person."""
