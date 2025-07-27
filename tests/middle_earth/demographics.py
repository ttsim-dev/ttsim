from __future__ import annotations

from ttsim.tt import AggType, agg_by_group_function


@agg_by_group_function(agg_type=AggType.COUNT)
def number_of_individuals_kin(
    kin_id: int,  # noqa: ARG001
) -> int:
    return 1
