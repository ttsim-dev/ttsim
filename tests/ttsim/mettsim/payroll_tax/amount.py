from ttsim import (
    AggType,
    DictTTSIMParam,
    ScalarTTSIMParam,
    agg_by_group_function,
    policy_function,
)


@policy_function(vectorization_strategy="vectorize")
def amount_y(
    income__amount_y: float,
    parent_is_noble_fam: bool,
    wealth_fam: float,
    wealth_threshold_for_reduced_tax_rate: ScalarTTSIMParam,
    income__schedule: DictTTSIMParam,
) -> float:
    if parent_is_noble_fam:
        return 0.0
    elif wealth_fam >= wealth_threshold_for_reduced_tax_rate.value:
        return income__amount_y * income__schedule.value["reduced_rate"]
    else:
        return income__amount_y * income__schedule.value["rate"]


@agg_by_group_function(agg_type=AggType.ANY)
def parent_is_noble_fam(
    parent_is_noble: bool,
    fam_id: int,
) -> bool:
    pass
