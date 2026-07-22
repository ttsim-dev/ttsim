from __future__ import annotations

from ttsim.tt import AggType, TTSIMUnit, agg_by_group_function, policy_function


@agg_by_group_function(agg_type=AggType.COUNT, unit=TTSIMUnit.DIMENSIONLESS.PER_KIN)
def number_of_individuals_kin(
    kin_id: int,  # noqa: ARG001
) -> int:
    return 1


@agg_by_group_function(agg_type=AggType.MIN, unit=TTSIMUnit.YEARS.PER_KIN)
def age_of_youngest_member_kin(kin_id: int, age: int) -> int:
    """The age of the kinstead's youngest member.

    An extreme is a property of the group whatever the source's base (GEP 10):
    the level-less duration ``age`` acquires the ``[kin]`` level here, so the
    declaration spells it.
    """


@agg_by_group_function(agg_type=AggType.MIN, unit=TTSIMUnit.CALENDAR_YEAR.PER_KIN)
def birth_year_of_oldest_member_kin(kin_id: int, birth_year: int) -> int:
    """The birth year of the kinstead's oldest member.

    A calendar *point* that is the kinstead's property — a leveled calendar
    point (GEP 10): the level is index bookkeeping, exempt from the
    offset-arithmetic rules that govern the point itself.
    """


@policy_function(vectorization_strategy="vectorize", unit=TTSIMUnit.YEARS)
def age(policy_year: int, birth_year: int) -> int:
    """Age in completed years on the first day of the policy year.

    The worked example for the calendar-point model (GEP 10): ``policy_year`` and
    ``birth_year`` are calendar *points* (``CALENDAR_YEAR``), and the difference
    of two calendar years is a *duration* in years (``TTSIMUnit.YEARS``). Tagging
    either side as a duration, or adding two calendar years, is rejected by the
    build-time unit check.
    """
    return policy_year - birth_year


@policy_function(vectorization_strategy="vectorize", unit=TTSIMUnit.DIMENSIONLESS)
def had_birthday_this_year(policy_month: int, birth_month: int) -> bool:
    """Whether the person has already had their birthday in the policy year.

    Exercises the month model (GEP 10): both ``policy_month`` (the framework
    date node) and ``birth_month`` carry a month-of-year (1-12) — cyclic
    ordinals, hence ``DIMENSIONLESS``, not calendar points — so the comparison
    screens as plain dimensionless arithmetic.
    """
    return policy_month >= birth_month


@policy_function(vectorization_strategy="vectorize", unit=TTSIMUnit.DIMENSIONLESS)
def coming_of_age_celebration(age: int) -> bool:
    """Whether the person reaches a coming-of-age milestone this year.

    Hobbits come of age at 33 and famously celebrate their eleventy-first
    (111th) birthday.
    """
    return age == 33 or age == 111
