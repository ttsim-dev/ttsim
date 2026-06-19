from __future__ import annotations

from ttsim.tt import AggType, FKType, Unit, agg_by_group_function, policy_input


@policy_input(unit=Unit.DIMENSIONLESS)
def p_id() -> int:
    """Person ID, always required by TTSIM."""


@policy_input(unit=Unit.DIMENSIONLESS)
def kin_id() -> int:
    """Kinstead ID."""


@policy_input(foreign_key_type=FKType.MUST_NOT_POINT_TO_SELF, unit=Unit.DIMENSIONLESS)
def p_id_parent_1() -> int:
    """Identifier of the first parent."""


@policy_input(foreign_key_type=FKType.MUST_NOT_POINT_TO_SELF, unit=Unit.DIMENSIONLESS)
def p_id_parent_2() -> int:
    """Identifier of the second parent."""


@policy_input(foreign_key_type=FKType.MUST_NOT_POINT_TO_SELF, unit=Unit.DIMENSIONLESS)
def p_id_spouse() -> int:
    """Identifier of married partner."""


@policy_input(unit=Unit.CALENDAR_YEAR)
def geburtsjahr() -> int:
    """Year of birth: a point on the calendar, not a duration (GEP 10).

    The person's age is computed from it (``demographics.age``): subtracting two
    calendar years yields a duration in years.
    """


@policy_input(unit=Unit.DIMENSIONLESS)
def geburtsmonat() -> int:
    """Month of birth (1-12): a cyclic ordinal, hence ``DIMENSIONLESS`` (GEP 10).

    Unlike :func:`geburtsjahr`, a month-of-year is not a point on a running
    calendar but a recurring label, so it carries no calendar-month unit.
    """


@policy_input(unit=Unit.DIMENSIONLESS)
def parent_is_noble() -> bool:
    """Whether at least one parent is noble."""


@agg_by_group_function(agg_type=AggType.ANY)
def parent_is_noble_fam(parent_is_noble: bool, fam_id: int) -> bool:
    pass


@policy_input(unit=Unit.CURRENCY)
def wealth() -> float:
    """Wealth of the person."""
