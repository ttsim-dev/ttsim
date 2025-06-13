from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Literal

from ttsim.tt_dag_elements import aggregation_jax, aggregation_numpy

if TYPE_CHECKING:
    from ttsim.tt_dag_elements.typing import BoolColumn, FloatColumn, IntColumn


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


# The signature of the functions must be the same in both modules, except that all JAX
# functions have the additional `num_segments` argument.
def grouped_count(
    group_id: IntColumn, num_segments: int, backend: Literal["numpy", "jax"]
) -> IntColumn:
    if backend == "numpy":
        return aggregation_numpy.grouped_count(group_id)
    else:
        return aggregation_jax.grouped_count(group_id, num_segments)


def grouped_sum(
    column: FloatColumn | IntColumn | BoolColumn,
    group_id: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> FloatColumn | IntColumn:
    if backend == "numpy":
        return aggregation_numpy.grouped_sum(column, group_id)
    else:
        return aggregation_jax.grouped_sum(column, group_id, num_segments)


def grouped_mean(
    column: FloatColumn | IntColumn | BoolColumn,
    group_id: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> FloatColumn:
    if backend == "numpy":
        return aggregation_numpy.grouped_mean(column, group_id)
    else:
        return aggregation_jax.grouped_mean(column, group_id, num_segments)


def grouped_max(
    column: FloatColumn | IntColumn,
    group_id: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> FloatColumn | IntColumn:
    if backend == "numpy":
        return aggregation_numpy.grouped_max(column, group_id)
    else:
        return aggregation_jax.grouped_max(column, group_id, num_segments)


def grouped_min(
    column: FloatColumn | IntColumn,
    group_id: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> FloatColumn | IntColumn:
    if backend == "numpy":
        return aggregation_numpy.grouped_min(column, group_id)
    else:
        return aggregation_jax.grouped_min(column, group_id, num_segments)


def grouped_any(
    column: BoolColumn | IntColumn,
    group_id: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> BoolColumn:
    if backend == "numpy":
        return aggregation_numpy.grouped_any(column, group_id)
    else:
        return aggregation_jax.grouped_any(column, group_id, num_segments)


def grouped_all(
    column: BoolColumn | IntColumn,
    group_id: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> BoolColumn:
    if backend == "numpy":
        return aggregation_numpy.grouped_all(column, group_id)
    else:
        return aggregation_jax.grouped_all(column, group_id, num_segments)


def count_by_p_id(
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> IntColumn:
    if backend == "numpy":
        return aggregation_numpy.count_by_p_id(p_id_to_aggregate_by, p_id_to_store_by)
    else:
        return aggregation_jax.count_by_p_id(
            p_id_to_aggregate_by, p_id_to_store_by, num_segments
        )


def sum_by_p_id(
    column: FloatColumn | IntColumn | BoolColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> FloatColumn | IntColumn:
    if backend == "numpy":
        return aggregation_numpy.sum_by_p_id(
            column, p_id_to_aggregate_by, p_id_to_store_by
        )
    else:
        return aggregation_jax.sum_by_p_id(
            column, p_id_to_aggregate_by, p_id_to_store_by, num_segments
        )


def mean_by_p_id(
    column: FloatColumn | IntColumn | BoolColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> FloatColumn:
    if backend == "numpy":
        return aggregation_numpy.mean_by_p_id(
            column, p_id_to_aggregate_by, p_id_to_store_by
        )
    else:
        return aggregation_jax.mean_by_p_id(
            column, p_id_to_aggregate_by, p_id_to_store_by, num_segments
        )


def max_by_p_id(
    column: FloatColumn | IntColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> FloatColumn | IntColumn:
    if backend == "numpy":
        return aggregation_numpy.max_by_p_id(
            column, p_id_to_aggregate_by, p_id_to_store_by
        )
    else:
        return aggregation_jax.max_by_p_id(
            column, p_id_to_aggregate_by, p_id_to_store_by, num_segments
        )


def min_by_p_id(
    column: FloatColumn | IntColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> FloatColumn | IntColumn:
    if backend == "numpy":
        return aggregation_numpy.min_by_p_id(
            column, p_id_to_aggregate_by, p_id_to_store_by
        )
    else:
        return aggregation_jax.min_by_p_id(
            column, p_id_to_aggregate_by, p_id_to_store_by, num_segments
        )


def any_by_p_id(
    column: BoolColumn | IntColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> BoolColumn:
    if backend == "numpy":
        return aggregation_numpy.any_by_p_id(
            column, p_id_to_aggregate_by, p_id_to_store_by
        )
    else:
        return aggregation_jax.any_by_p_id(
            column, p_id_to_aggregate_by, p_id_to_store_by, num_segments
        )


def all_by_p_id(
    column: BoolColumn | IntColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> BoolColumn:
    if backend == "numpy":
        return aggregation_numpy.all_by_p_id(
            column, p_id_to_aggregate_by, p_id_to_store_by
        )
    else:
        return aggregation_jax.all_by_p_id(
            column, p_id_to_aggregate_by, p_id_to_store_by, num_segments
        )
