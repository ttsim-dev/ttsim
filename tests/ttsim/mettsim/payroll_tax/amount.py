from ttsim import AggType, agg_by_group_function, policy_function


@policy_function()
def amount_y(
    income__amount_y: float,
    payroll_tax_params: dict,
    parent_is_noble_fam: bool,
) -> float:
    if parent_is_noble_fam:
        return 0.0
    else:
        return income__amount_y * payroll_tax_params["income"]["rate"]


@agg_by_group_function(agg_type=AggType.ANY)
def parent_is_noble_fam(
    parent_is_noble: bool,
    fam_id: int,
) -> bool:
    pass
