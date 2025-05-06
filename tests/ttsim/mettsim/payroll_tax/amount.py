from ttsim import (
    AggType,
    agg_by_group_function,
    policy_function,
)


@policy_function(vectorization_strategy="vectorize")
def amount_y(
    income__amount_y: float,
    parent_is_noble_fam: bool,
    wealth_fam: float,
    wealth_threshold_for_reduced_tax_rate: float,
    income__schedule: dict[str, float],
) -> float:
    if parent_is_noble_fam:
        return 0.0
    elif wealth_fam >= wealth_threshold_for_reduced_tax_rate:
        return income__amount_y * income__schedule["reduced_rate"]
    else:
        return income__amount_y * income__schedule["rate"]


@agg_by_group_function(agg_type=AggType.ANY)
def parent_is_noble_fam(
    parent_is_noble: bool,
    fam_id: int,
) -> bool:
    pass
