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
    TTSIMUnit,
    agg_by_group_function,
    policy_function,
)


@agg_by_group_function(
    agg_type=AggType.SUM, end_date="2019-12-31", unit=TTSIMUnit.DIMENSIONLESS.PER_FAM
)
def number_of_adults_fam(fam_id: int, adult: bool) -> int:
    """The number of adults in the family — a head count (SUM over a boolean)."""


@agg_by_group_function(agg_type=AggType.COUNT, unit=TTSIMUnit.DIMENSIONLESS.PER_FAM)
def number_of_individuals_fam(fam_id: int) -> int:
    """The number of individuals in the family."""


@policy_function(
    end_date="2019-12-31",
    leaf_name="requirement_fulfilled_fam",
    unit=TTSIMUnit.DIMENSIONLESS.PER_FAM,
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
    unit=TTSIMUnit.DIMENSIONLESS.PER_FAM,
)
def requirement_fulfilled_fam_considering_children(
    housing_benefits__income__amount_m_fam: float,
    number_of_family_members_considered_fam: int,
    subsistence_income_level: dict[str, float],
) -> bool:
    return housing_benefits__income__amount_m_fam < (
        subsistence_income_level["per_individual"]
        * number_of_family_members_considered_fam
    )


@policy_function(
    start_date="2020-01-01",
    vectorization_strategy="vectorize",
    unit=TTSIMUnit.DIMENSIONLESS.PER_FAM,
)
def number_of_family_members_considered_fam(
    number_of_individuals_fam: int,
    max_number_of_family_members: int,
) -> int:
    return min(number_of_individuals_fam, max_number_of_family_members)


@policy_function(vectorization_strategy="vectorize", unit=TTSIMUnit.DIMENSIONLESS)
def child(
    age: int,
    max_age_children: int,
) -> bool:
    return age <= max_age_children


@policy_function(vectorization_strategy="vectorize", unit=TTSIMUnit.DIMENSIONLESS)
def adult(
    age: int,
    max_age_children: int,
) -> bool:
    return age > max_age_children


@policy_function(vectorization_strategy="vectorize", unit=TTSIMUnit.DIMENSIONLESS)
def young_adult(
    age: int,
    max_age_children: int,
    age_of_maturity: int,
) -> bool:
    """Whether the person is a young adult: past child age but not yet of age."""
    return max_age_children < age and age < age_of_maturity
