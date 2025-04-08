from ttsim import AggregateByPIDSpec, join_numpy, policy_function

aggregation_specs = {
    "amount_y": AggregateByPIDSpec(
        p_id_to_aggregate_by="recipient_id",
        source="claim_of_child_y",
        aggr="sum",
    ),
}


@policy_function()
def claim_of_child_y(
    child_eligible: bool,
    child_amount_y: float,
) -> float:
    if child_eligible:
        return child_amount_y
    else:
        return 0


@policy_function()
def child_eligible(
    age: int,
    max_age: int,
    child_in_same_household_as_recipient: float,
) -> bool:
    return age <= max_age and child_in_same_household_as_recipient


@policy_function(skip_vectorization=True)
def child_in_same_household_as_recipient(
    p_id: int,
    hh_id: int,
    recipient_id: int,
) -> bool:
    return (
        join_numpy(
            foreign_key=recipient_id,
            primary_key=p_id,
            target=hh_id,
            value_if_foreign_key_is_missing=-1,
        )
        == hh_id
    )
