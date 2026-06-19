from __future__ import annotations

from ttsim.tt import AggType, Unit, agg_by_group_function, policy_function


@agg_by_group_function(agg_type=AggType.COUNT)
def number_of_individuals_kin(
    kin_id: int,  # noqa: ARG001
) -> int:
    return 1


@policy_function(vectorization_strategy="vectorize", unit=Unit.YEARS)
def age(policy_year: int, geburtsjahr: int) -> int:
    """Age in completed years on the first day of the policy year.

    The worked example for the calendar-point model (GEP 10): ``policy_year`` and
    ``geburtsjahr`` are calendar *points* (``CALENDAR_YEAR``), and the difference
    of two calendar years is a *duration* in years (``Unit.YEARS``). Tagging
    either side as a duration, or adding two calendar years, is rejected by the
    build-time unit check.
    """
    return policy_year - geburtsjahr


@policy_function(vectorization_strategy="vectorize", unit=Unit.DIMENSIONLESS)
def had_birthday_this_year(policy_month: int, geburtsmonat: int) -> bool:
    """Whether the person has already had their birthday in the policy year.

    Exercises the month axis (GEP 10): ``policy_month`` is a ``CALENDAR_MONTH``
    framework node, while ``geburtsmonat`` is a cyclic month-of-year ordinal
    (``DIMENSIONLESS``). Reading the calendar-month node as an ordinal in a
    comparison is the documented cyclic use — the unit check does not screen it.
    """
    return policy_month >= geburtsmonat


@policy_function(vectorization_strategy="vectorize", unit=Unit.DIMENSIONLESS)
def coming_of_age_celebration(age: int) -> bool:
    """Whether the person reaches a coming-of-age milestone this year.

    Hobbits come of age at 33 and famously celebrate their eleventy-first
    (111th) birthday.
    """
    return age == 33 or age == 111
