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
    schedule: dict[str, float],
) -> float:
    if child_eligible:
        return schedule["child_amount_y"]
    else:
        return 0


@policy_function(vectorization_strategy="vectorize")
def child_eligible(
    age: int,
    schedule: dict[str, float],
    in_same_household_as_recipient: bool,
) -> bool:
    return age <= schedule["max_age"] and in_same_household_as_recipient


@policy_function(vectorization_strategy="not_required")
def in_same_household_as_recipient(
    p_id: int,
    kin_id: int,
    p_id_recipient: int,
) -> bool:
    return (
        join(
            foreign_key=p_id_recipient,
            primary_key=p_id,
            target=kin_id,
            value_if_foreign_key_is_missing=-1,
        )
        == kin_id
    )
