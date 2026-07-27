from __future__ import annotations

from ttsim.tt import (
    AggType,
    TTSIMUnit,
    agg_by_group_function,
    cast_ttsim_unit,
    policy_function,
)


@agg_by_group_function(agg_type=AggType.COUNT, unit=TTSIMUnit.DIMENSIONLESS.PER_KIN)
def number_of_individuals_kin(
    kin_id: int,  # noqa: ARG001
) -> int:
    return 1


@agg_by_group_function(agg_type=AggType.MIN, unit=TTSIMUnit.YEARS.PER_KIN)
def age_of_youngest_member_kin(kin_id: int, age: int) -> int:
    """The age of the kinstead's youngest member."""


@agg_by_group_function(agg_type=AggType.MIN, unit=TTSIMUnit.CALENDAR_YEAR.PER_KIN)
def birth_year_of_oldest_member_kin(kin_id: int, birth_year: int) -> int:
    """The birth year of the kinstead's oldest member."""


@policy_function(unit=TTSIMUnit.YEARS)
def age(birth_year: int, evaluation_year: int) -> int:
    """Age in completed years at the evaluation date."""
    return evaluation_year - birth_year


@policy_function(vectorization_strategy="vectorize", unit=TTSIMUnit.DIMENSIONLESS)
def had_birthday_this_year(policy_month: int, birth_month: int) -> bool:
    """Whether the person has already had their birthday in the policy year."""
    return policy_month >= birth_month


@policy_function(vectorization_strategy="vectorize", unit=TTSIMUnit.DIMENSIONLESS)
def coming_of_age_celebration(age: int) -> bool:
    """Whether the person reaches a coming-of-age milestone this year.

    Hobbits come of age at 33 and famously celebrate their eleventy-first
    (111th) birthday.
    """
    return age == cast_ttsim_unit(
        value=33, unit=TTSIMUnit.YEARS
    ) or age == cast_ttsim_unit(value=111, unit=TTSIMUnit.YEARS)


@policy_function(vectorization_strategy="vectorize", unit=TTSIMUnit.DIMENSIONLESS)
def of_age(age: int) -> bool:
    """Whether the person has reached the age of majority."""
    # The bound is stated in years, so it is written that way rather than as a
    # bare number.
    return age >= cast_ttsim_unit(value=33, unit=TTSIMUnit.YEARS)


@policy_function(
    vectorization_strategy="vectorize", unit=TTSIMUnit.DIMENSIONLESS.PER_KIN
)
def number_of_dependants_kin(number_of_individuals_kin: int) -> int:
    """The kinstead's members other than its head."""
    # The one head subtracted here is a head count of the kinstead, like the
    # count it is subtracted from.
    return number_of_individuals_kin - cast_ttsim_unit(
        value=1, unit=TTSIMUnit.DIMENSIONLESS.PER_KIN
    )
