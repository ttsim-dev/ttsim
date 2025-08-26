from __future__ import annotations

from ttsim.tt import AggType, FKType, agg_by_group_function, policy_input


@policy_input()
def p_id() -> int:
    """Person ID, always required by TTSIM."""


@policy_input()
def kin_id() -> int:
    """Kinstead ID."""


@policy_input(foreign_key_type=FKType.MUST_NOT_POINT_TO_SELF)
def p_id_parent_1() -> int:
    """Identifier of the first parent."""


@policy_input(foreign_key_type=FKType.MUST_NOT_POINT_TO_SELF)
def p_id_parent_2() -> int:
    """Identifier of the second parent."""


@policy_input(foreign_key_type=FKType.MUST_NOT_POINT_TO_SELF)
def p_id_spouse() -> int:
    """Identifier of married partner."""


@policy_input()
def age() -> int:
    """Age of the person."""


@policy_input()
def parent_is_noble() -> bool:
    """Whether at least one parent is noble."""


@agg_by_group_function(agg_type=AggType.ANY)
def parent_is_noble_fam(parent_is_noble: bool, fam_id: int) -> bool:
    pass


@policy_input()
def wealth() -> float:
    """Wealth of the person."""
