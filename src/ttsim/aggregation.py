from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

from ttsim.aggregation_jax import all_by_p_id as all_by_p_id_jax
from ttsim.aggregation_jax import any_by_p_id as any_by_p_id_jax
from ttsim.aggregation_jax import count_by_p_id as count_by_p_id_jax
from ttsim.aggregation_jax import grouped_all as grouped_all_jax
from ttsim.aggregation_jax import grouped_any as grouped_any_jax
from ttsim.aggregation_jax import grouped_count as grouped_count_jax
from ttsim.aggregation_jax import grouped_max as grouped_max_jax
from ttsim.aggregation_jax import grouped_mean as grouped_mean_jax
from ttsim.aggregation_jax import grouped_min as grouped_min_jax
from ttsim.aggregation_jax import grouped_sum as grouped_sum_jax
from ttsim.aggregation_jax import max_by_p_id as max_by_p_id_jax
from ttsim.aggregation_jax import mean_by_p_id as mean_by_p_id_jax
from ttsim.aggregation_jax import min_by_p_id as min_by_p_id_jax
from ttsim.aggregation_jax import sum_by_p_id as sum_by_p_id_jax
from ttsim.aggregation_numpy import all_by_p_id as all_by_p_id_numpy
from ttsim.aggregation_numpy import any_by_p_id as any_by_p_id_numpy
from ttsim.aggregation_numpy import count_by_p_id as count_by_p_id_numpy
from ttsim.aggregation_numpy import grouped_all as grouped_all_numpy
from ttsim.aggregation_numpy import grouped_any as grouped_any_numpy
from ttsim.aggregation_numpy import grouped_count as grouped_count_numpy
from ttsim.aggregation_numpy import grouped_max as grouped_max_numpy
from ttsim.aggregation_numpy import grouped_mean as grouped_mean_numpy
from ttsim.aggregation_numpy import grouped_min as grouped_min_numpy
from ttsim.aggregation_numpy import grouped_sum as grouped_sum_numpy
from ttsim.aggregation_numpy import max_by_p_id as max_by_p_id_numpy
from ttsim.aggregation_numpy import mean_by_p_id as mean_by_p_id_numpy
from ttsim.aggregation_numpy import min_by_p_id as min_by_p_id_numpy
from ttsim.aggregation_numpy import sum_by_p_id as sum_by_p_id_numpy
from ttsim.config import IS_JAX_INSTALLED


class AggregationType(StrEnum):
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


@dataclass
class AggregateByGroupSpec:
    """
    A container for aggregate by group specifications.
    """

    aggr: AggregationType
    source: str | None = None
    _agg_func: Callable = field(init=False)

    def __post_init__(self):
        if not isinstance(self.aggr, AggregationType):
            raise TypeError(
                f"aggr must be of type AggregationType, not {type(self.aggr)}"
            )

        if self.aggr == AggregationType.COUNT and self.source is not None:
            raise ValueError("COUNT aggregation cannot use a source.")

        aggregation_registry = {
            AggregationType.SUM: grouped_sum,
            AggregationType.MEAN: grouped_mean,
            AggregationType.MAX: grouped_max,
            AggregationType.MIN: grouped_min,
            AggregationType.ANY: grouped_any,
            AggregationType.ALL: grouped_all,
            AggregationType.COUNT: grouped_count,
        }

        func = aggregation_registry.get(self.aggr)
        if func is None:
            raise ValueError(f"Aggregation type {self.aggr} not implemented")

        self._agg_func = func

    def agg_func(self, source, group_by_id):
        return self._agg_func(source, group_by_id)

    def mapper(self, group_by_id):
        if self.aggr == AggregationType.COUNT:
            return {"group_by_id": group_by_id}
        return {"source": self.source, "group_by_id": group_by_id}


@dataclass
class AggregateByPIDSpec:
    """
    A container for aggregate by p_id specifications.
    """

    p_id_to_aggregate_by: str
    source: str
    aggr: AggregationType


def grouped_count(group_id):
    if IS_JAX_INSTALLED:
        return grouped_count_jax(group_id)
    else:
        return grouped_count_numpy(group_id)


def grouped_sum(column, group_id):
    if IS_JAX_INSTALLED:
        return grouped_sum_jax(column, group_id)
    else:
        return grouped_sum_numpy(column, group_id)


def grouped_mean(column, group_id):
    if IS_JAX_INSTALLED:
        return grouped_mean_jax(column, group_id)
    else:
        return grouped_mean_numpy(column, group_id)


def grouped_max(column, group_id):
    if IS_JAX_INSTALLED:
        return grouped_max_jax(column, group_id)
    else:
        return grouped_max_numpy(column, group_id)


def grouped_min(column, group_id):
    if IS_JAX_INSTALLED:
        return grouped_min_jax(column, group_id)
    else:
        return grouped_min_numpy(column, group_id)


def grouped_any(column, group_id):
    if IS_JAX_INSTALLED:
        return grouped_any_jax(column, group_id)
    else:
        return grouped_any_numpy(column, group_id)


def grouped_all(column, group_id):
    if IS_JAX_INSTALLED:
        return grouped_all_jax(column, group_id)
    else:
        return grouped_all_numpy(column, group_id)


def count_by_p_id(p_id_to_aggregate_by, p_id_to_store_by):
    if IS_JAX_INSTALLED:
        return count_by_p_id_jax(p_id_to_aggregate_by, p_id_to_store_by)
    else:
        return count_by_p_id_numpy(p_id_to_aggregate_by, p_id_to_store_by)


def sum_by_p_id(column, p_id_to_aggregate_by, p_id_to_store_by):
    if IS_JAX_INSTALLED:
        return sum_by_p_id_jax(column, p_id_to_aggregate_by, p_id_to_store_by)
    else:
        return sum_by_p_id_numpy(column, p_id_to_aggregate_by, p_id_to_store_by)


def mean_by_p_id(column, p_id_to_aggregate_by, p_id_to_store_by):
    if IS_JAX_INSTALLED:
        return mean_by_p_id_jax(column, p_id_to_aggregate_by, p_id_to_store_by)
    else:
        return mean_by_p_id_numpy(column, p_id_to_aggregate_by, p_id_to_store_by)


def max_by_p_id(column, p_id_to_aggregate_by, p_id_to_store_by):
    if IS_JAX_INSTALLED:
        return max_by_p_id_jax(column, p_id_to_aggregate_by, p_id_to_store_by)
    else:
        return max_by_p_id_numpy(column, p_id_to_aggregate_by, p_id_to_store_by)


def min_by_p_id(column, p_id_to_aggregate_by, p_id_to_store_by):
    if IS_JAX_INSTALLED:
        return min_by_p_id_jax(column, p_id_to_aggregate_by, p_id_to_store_by)
    else:
        return min_by_p_id_numpy(column, p_id_to_aggregate_by, p_id_to_store_by)


def any_by_p_id(column, p_id_to_aggregate_by, p_id_to_store_by):
    if IS_JAX_INSTALLED:
        return any_by_p_id_jax(column, p_id_to_aggregate_by, p_id_to_store_by)
    else:
        return any_by_p_id_numpy(column, p_id_to_aggregate_by, p_id_to_store_by)


def all_by_p_id(column, p_id_to_aggregate_by, p_id_to_store_by):
    if IS_JAX_INSTALLED:
        return all_by_p_id_jax(column, p_id_to_aggregate_by, p_id_to_store_by)
    else:
        return all_by_p_id_numpy(column, p_id_to_aggregate_by, p_id_to_store_by)
