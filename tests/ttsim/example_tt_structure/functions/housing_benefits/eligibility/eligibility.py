"""Eligibility for housing benefits.

Policy regime until 2019:
    - Requirement is fulfilled if income of spouses is below subsitence income
    - Subsistence income is calculated per spouse

Policy regime starting in 2020:
    - Requirement is fulfilled if income of family is below subsistence income
    - Subsistence income is calculated per spouse and child
"""

from ttsim import AggregateByGroupSpec, policy_function

aggregation_specs = {
    "number_children_fam": AggregateByGroupSpec(
        source="child",
        aggr="sum",
    ),
}


@policy_function(end_date="2019-12-31", leaf_name="requirement_fulfilled_fam")
def requirement_fulfilled_fam_not_considering_children(
    housing_benefits__income__amount_m_sp: float,
    subsistence_income_per_spouse_m: float,
    number_individuals_sp: int,
) -> bool:
    return (
        housing_benefits__income__amount_m_sp
        < subsistence_income_per_spouse_m * number_individuals_sp
    )


@policy_function(start_date="2020-01-01", leaf_name="requirement_fulfilled_fam")
def requirement_fulfilled_fam_considering_children(
    housing_benefits__income__amount_m_fam: float,
    subsistence_income_per_spouse: float,
    subsistence_income_per_child: float,
    number_of_children_considered: int,
    number_individuals_sp: int,
) -> bool:
    return housing_benefits__income__amount_m_fam < (
        subsistence_income_per_spouse * number_individuals_sp
        + subsistence_income_per_child * number_of_children_considered
    )


@policy_function(start_date="2020-01-01")
def number_of_children_considered(
    number_children_fam: int,
    max_number_of_children: int,
) -> int:
    return min(number_children_fam, max_number_of_children)
