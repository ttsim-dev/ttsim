from ttsim import (
    AggregateByPIDSpec,
    AggType,
    join_numpy,
    policy_function,
    policy_input,
)
from ttsim.ttsim_objects import FKType

aggregation_specs = (
    AggregateByPIDSpec(
        target="amount_y",
        source="claim_of_child_y",
        p_id_to_aggregate_by="p_id_recipient",
        agg=AggType.SUM,
    ),
)


@policy_input(foreign_key_type=FKType.MAY_POINT_TO_SELF)
def p_id_recipient() -> int:
    """Identifier of the recipient of the child tax credit."""


@policy_function()
def claim_of_child_y(
    child_eligible: bool,
    payroll_tax_params: dict,
) -> float:
    if child_eligible:
        return payroll_tax_params["child_tax_credit"]
    else:
        return 0


@policy_function()
def child_eligible(
    age: int,
    payroll_tax_params: dict,
    child_in_same_household_as_recipient: float,
) -> bool:
    return age <= payroll_tax_params["max_age"] and child_in_same_household_as_recipient


@policy_function(skip_vectorization=True)
def child_in_same_household_as_recipient(
    p_id: int,
    hh_id: int,
    p_id_recipient: int,
) -> bool:
    return (
        join_numpy(
            foreign_key=p_id_recipient,
            primary_key=p_id,
            target=hh_id,
            value_if_foreign_key_is_missing=-1,
        )
        == hh_id
    )
