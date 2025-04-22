from ttsim import (
    AggType,
    agg_by_p_id_function,
    join,
    policy_function,
)


@agg_by_p_id_function(agg_type=AggType.SUM)
def amount_y(
    p_id: int,
    p_id_recipient: int,
    claim_of_child_y: float,
) -> float:
    """The amount of child tax credit at the recipient level."""


@policy_function(vectorization_strategy="vectorize")
def claim_of_child_y(
    child_eligible: bool,
    payroll_tax_params: dict,
) -> float:
    if child_eligible:
        return payroll_tax_params["child_tax_credit"]["child_amount_y"]
    else:
        return 0


@policy_function(vectorization_strategy="vectorize")
def child_eligible(
    age: int,
    payroll_tax_params: dict,
    in_same_household_as_recipient: float,
) -> bool:
    return (
        age <= payroll_tax_params["child_tax_credit"]["max_age"]
        and in_same_household_as_recipient
    )


@policy_function(vectorization_strategy="not_required")
def in_same_household_as_recipient(
    p_id: int,
    hh_id: int,
    p_id_recipient: int,
) -> bool:
    return (
        join(
            foreign_key=p_id_recipient,
            primary_key=p_id,
            target=hh_id,
            value_if_foreign_key_is_missing=-1,
        )
        == hh_id
    )
