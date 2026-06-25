"""Eligibility for housing benefits.

Policy regime until 2019:
    - Requirement is fulfilled if income of spouses is below subsistence income
    - Subsistence income is calculated per spouse

Policy regime starting in 2020:
    - Requirement is fulfilled if income of family is below subsistence income
    - Subsistence income is calculated per spouse and child
"""

from __future__ import annotations

from ttsim.tt import (
    AggType,
    Unit,
    agg_by_group_function,
    policy_function,
)


@agg_by_group_function(
    agg_type=AggType.SUM, end_date="2019-12-31", unit=Unit.DIMENSIONLESS
)
def number_of_adults_fam(fam_id: int, adult: bool) -> int:
    """The number of adults in the family."""


@agg_by_group_function(agg_type=AggType.COUNT)
def number_of_individuals_fam(fam_id: int) -> int:
    """The number of individuals in the family."""


@policy_function(
    end_date="2019-12-31",
    leaf_name="requirement_fulfilled_fam",
    unit=Unit.DIMENSIONLESS,
)
def requirement_fulfilled_fam_not_considering_children(
    housing_benefits__income__amount_m_fam: float,
    number_of_adults_fam: int,
    subsistence_income_level: dict[str, float],
) -> bool:
    return (
        housing_benefits__income__amount_m_fam
        < subsistence_income_level["per_spouse"] * number_of_adults_fam
    )


@policy_function(
    start_date="2020-01-01",
    leaf_name="requirement_fulfilled_fam",
    unit=Unit.DIMENSIONLESS,
    verify_units=False,
)
def requirement_fulfilled_fam_considering_children(
    housing_benefits__income__amount_m_fam: float,
    number_of_family_members_considered_fam: int,
    subsistence_income_level: dict[str, float],
) -> bool:
    # ``verify_units=False``: the per-family subsistence threshold is the
    # per-person amount times the (capped) head count of considered family
    # members. ``number_of_family_members_considered_fam`` is a head count
    # (``[person] / [fam]``), but a hand-written policy function cannot *declare*
    # a head-count unit (only aggregations mint it, and there is no head-count
    # token). So the dry-run cannot see that
    # ``CURRENCY / [person] * [person] / [fam] = CURRENCY / [fam]`` matches
    # ``amount_m_fam``; we opt the body out. See GEP 10 (known limitation).
    return housing_benefits__income__amount_m_fam < (
        subsistence_income_level["per_individual"]
        * number_of_family_members_considered_fam
    )


@policy_function(
    start_date="2020-01-01",
    vectorization_strategy="vectorize",
    unit=Unit.DIMENSIONLESS,
    verify_units=False,
)
def number_of_family_members_considered_fam(
    number_of_individuals_fam: int,
    max_number_of_family_members: int,
) -> int:
    # ``verify_units=False``: ``number_of_individuals_fam`` is a COUNT
    # aggregation and so carries the head-count unit ``[person] / [fam]``, while
    # the cap ``max_number_of_family_members`` is a plain dimensionless scalar.
    # The capped result is still a head count, but there is no head-count token a
    # hand-written function can declare, so the ``min`` would trip the dry-run on
    # mixing ``[person] / [fam]`` with ``dimensionless``. Opt the body out and
    # keep the declared unit as the edge contract. See GEP 10 (known limitation).
    return min(number_of_individuals_fam, max_number_of_family_members)


@policy_function(vectorization_strategy="vectorize", unit=Unit.DIMENSIONLESS)
def child(
    age: int,
    max_age_children: int,
) -> bool:
    return age <= max_age_children


@policy_function(vectorization_strategy="vectorize", unit=Unit.DIMENSIONLESS)
def adult(
    age: int,
    max_age_children: int,
) -> bool:
    return age > max_age_children


@policy_function(vectorization_strategy="vectorize", unit=Unit.DIMENSIONLESS)
def young_adult(
    age: int,
    max_age_children: int,
    age_of_majority: int,
) -> bool:
    """Whether the person is a young adult: past child age but not yet of age."""
    return max_age_children < age and age < age_of_majority
