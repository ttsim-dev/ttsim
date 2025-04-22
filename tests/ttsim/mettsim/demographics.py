from ttsim import AggType, agg_by_group_function


@agg_by_group_function(agg_type=AggType.COUNT)
def number_of_individuals_hh(
    hh_id: int,  # noqa: ARG001
) -> int:
    return 1
