from __future__ import annotations

from enum import StrEnum

from ttsim.config import IS_JAX_INSTALLED
from ttsim.tt_dag_elements import aggregation_jax, aggregation_numpy


class AggType(StrEnum):
    """
    Enum for aggregation types.
    """

    COUNT = "count"
    SUM = "sum"
    MEAN = "mean"
    MAX = "max"
    MIN = "min"
    ANY = "any"
    ALL = "all"


aggregation_module = aggregation_jax if IS_JAX_INSTALLED else aggregation_numpy

# The signature of the functions must be the same in both modules, except that all JAX
# functions have the additional `num_segments` argument.
grouped_count = aggregation_module.grouped_count
grouped_sum = aggregation_module.grouped_sum
grouped_mean = aggregation_module.grouped_mean
grouped_max = aggregation_module.grouped_max
grouped_min = aggregation_module.grouped_min
grouped_any = aggregation_module.grouped_any
grouped_all = aggregation_module.grouped_all
count_by_p_id = aggregation_module.count_by_p_id
sum_by_p_id = aggregation_module.sum_by_p_id
mean_by_p_id = aggregation_module.mean_by_p_id
max_by_p_id = aggregation_module.max_by_p_id
min_by_p_id = aggregation_module.min_by_p_id
any_by_p_id = aggregation_module.any_by_p_id
all_by_p_id = aggregation_module.all_by_p_id
