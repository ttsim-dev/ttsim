"""Eligibility for housing benefits.

Policy regime until 2019:
    - Requirement is fulfilled if income of spouses is below subsistence income
    - Subsistence income is calculated per spouse

Policy regime starting in 2020:
    - Requirement is fulfilled if income of family is below subsistence income
    - Subsistence income is calculated per spouse and child
"""

from ttsim import (
    AggType,
    DictTTSIMParam,
    ScalarTTSIMParam,
    agg_by_group_function,
    policy_function,
)


@agg_by_group_function(agg_type=AggType.SUM, end_date="2019-12-31")
def number_of_adults_fam(fam_id: int, adult: bool) -> int:
    """The number of adults in the family."""


@agg_by_group_function(agg_type=AggType.COUNT)
def number_of_individuals_fam(fam_id: int) -> int:
    """The number of individuals in the family."""


@policy_function(
    end_date="2019-12-31",
    leaf_name="requirement_fulfilled_fam",
)
def requirement_fulfilled_fam_not_considering_children(
    housing_benefits__income__amount_m_fam: float,
    number_of_adults_fam: int,
    requirements: DictTTSIMParam,
) -> bool:
    return (
        housing_benefits__income__amount_m_fam
        < requirements.value["subsistence_income_per_spouse_m"] * number_of_adults_fam
    )


@policy_function(
    start_date="2020-01-01",
    leaf_name="requirement_fulfilled_fam",
)
def requirement_fulfilled_fam_considering_children(
    housing_benefits__income__amount_m_fam: float,
    number_of_family_members_considered_fam: int,
    requirements: DictTTSIMParam,
) -> bool:
    return housing_benefits__income__amount_m_fam < (
        requirements.value["subsistence_income_per_individual_m"]
        * number_of_family_members_considered_fam
    )


@policy_function(start_date="2020-01-01", vectorization_strategy="vectorize")
def number_of_family_members_considered_fam(
    number_of_individuals_fam: int,
    requirements: DictTTSIMParam,
) -> int:
    return min(
        number_of_individuals_fam,
        requirements.value["max_number_of_family_members"],
    )


@policy_function(vectorization_strategy="vectorize")
def child(
    age: int,
    max_age_children: ScalarTTSIMParam,
) -> bool:
    return age <= max_age_children.value


@policy_function(vectorization_strategy="vectorize")
def adult(
    age: int,
    max_age_children: ScalarTTSIMParam,
) -> bool:
    return age > max_age_children.value
