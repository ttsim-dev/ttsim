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


@policy_function(unit=TTSIMUnit.YEARS)
def age(birth_year: int, evaluation_year: int) -> int:
    """Age in completed years at the evaluation date."""
    return evaluation_year - birth_year


@policy_function(vectorization_strategy="vectorize", unit=TTSIMUnit.DIMENSIONLESS)
def coming_of_age_celebration(age: int) -> bool:
    """Whether the person reaches a coming-of-age milestone this year.

    Hobbits come of age at 33 and famously celebrate their eleventy-first
    (111th) birthday.
    """
    return age == cast_ttsim_unit(
        value=33, unit=TTSIMUnit.YEARS
    ) or age == cast_ttsim_unit(value=111, unit=TTSIMUnit.YEARS)
