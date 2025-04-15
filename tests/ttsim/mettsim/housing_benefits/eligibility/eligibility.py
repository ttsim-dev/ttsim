"""Eligibility for housing benefits.

Policy regime until 2019:
    - Requirement is fulfilled if income of spouses is below subsistence income
    - Subsistence income is calculated per spouse

Policy regime starting in 2020:
    - Requirement is fulfilled if income of family is below subsistence income
    - Subsistence income is calculated per spouse and child
"""

from ttsim import AggregateByGroupSpec, AggType, agg_by_group_function, policy_function

aggregation_specs = (
    AggregateByGroupSpec(
        target="number_of_children_fam",
        source="child",
        agg=AggType.SUM,
    ),
)


@agg_by_group_function(agg_type=AggType.SUM)
def number_of_children_fam(fam_id, child):
    """The number of children in the family."""


@policy_function(end_date="2019-12-31", leaf_name="requirement_fulfilled_fam")
def requirement_fulfilled_fam_not_considering_children(
    housing_benefits__income__amount_m_sp: float,
    number_of_individuals_sp: int,
    housing_benefits_params: dict,
) -> bool:
    return (
        housing_benefits__income__amount_m_sp
        < housing_benefits_params["subsistence_income_per_spouse_m"]
        * number_of_individuals_sp
    )


@policy_function(start_date="2020-01-01", leaf_name="requirement_fulfilled_fam")
def requirement_fulfilled_fam_considering_children(
    housing_benefits__income__amount_m_fam: float,
    housing_benefits_params: dict,
    number_of_children_considered: int,
    number_of_individuals_sp: int,
) -> bool:
    return housing_benefits__income__amount_m_fam < (
        housing_benefits_params["subsistence_income_per_spouse"]
        * number_of_individuals_sp
        + housing_benefits_params["subsistence_income_per_child"]
        * number_of_children_considered
    )


@policy_function(start_date="2020-01-01")
def number_of_children_considered(
    number_of_children_fam: int,
    housing_benefits_params: dict,
) -> int:
    return min(
        number_of_children_fam, housing_benefits_params["max_number_of_children"]
    )
