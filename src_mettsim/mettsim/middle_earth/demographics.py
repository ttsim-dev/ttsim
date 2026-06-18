from __future__ import annotations

from ttsim.tt import AggType, Unit, agg_by_group_function, policy_function


@agg_by_group_function(agg_type=AggType.COUNT)
def number_of_individuals_kin(
    kin_id: int,  # noqa: ARG001
) -> int:
    return 1


@policy_function(vectorization_strategy="vectorize", unit=Unit.DIMENSIONLESS)
def coming_of_age_celebration(age: int) -> bool:
    """Whether the person reaches a coming-of-age milestone this year.

    Hobbits come of age at 33 and famously celebrate their eleventy-first
    (111th) birthday.
    """
    return age == 33 or age == 111
