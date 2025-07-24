from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Literal, overload

from ttsim.tt import aggregation_jax, aggregation_numpy

if TYPE_CHECKING:
    from ttsim.tt import BoolColumn, FloatColumn, IntColumn


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
    group_id: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> IntColumn:
    if backend == "numpy":
        return aggregation_numpy.grouped_count(group_id)
    return aggregation_jax.grouped_count(group_id, num_segments)


@overload
def grouped_sum(
    column: FloatColumn,
    group_id: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> FloatColumn: ...


@overload
def grouped_sum(
    column: IntColumn,
    group_id: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> IntColumn: ...


@overload
def grouped_sum(
    column: BoolColumn,
    group_id: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> IntColumn: ...


def grouped_sum(
    column: FloatColumn | IntColumn | BoolColumn,
    group_id: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> FloatColumn | IntColumn:
    if backend == "numpy":
        return aggregation_numpy.grouped_sum(column, group_id)
    return aggregation_jax.grouped_sum(column, group_id, num_segments)


@overload
def grouped_mean(
    column: FloatColumn,
    group_id: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> FloatColumn: ...
@overload
def grouped_mean(
    column: IntColumn,
    group_id: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> FloatColumn: ...
@overload
def grouped_mean(
    column: BoolColumn,
    group_id: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> FloatColumn: ...
def grouped_mean(
    column: FloatColumn | IntColumn | BoolColumn,
    group_id: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> FloatColumn:
    if backend == "numpy":
        return aggregation_numpy.grouped_mean(column, group_id)
    return aggregation_jax.grouped_mean(column, group_id, num_segments)


@overload
def grouped_max(
    column: FloatColumn,
    group_id: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> FloatColumn: ...


@overload
def grouped_max(
    column: IntColumn,
    group_id: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> IntColumn: ...


def grouped_max(
    column: FloatColumn | IntColumn,
    group_id: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> FloatColumn | IntColumn:
    if backend == "numpy":
        return aggregation_numpy.grouped_max(column, group_id)
    return aggregation_jax.grouped_max(column, group_id, num_segments)


@overload
def grouped_min(
    column: FloatColumn,
    group_id: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> FloatColumn: ...


@overload
def grouped_min(
    column: IntColumn,
    group_id: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> IntColumn: ...


def grouped_min(
    column: FloatColumn | IntColumn,
    group_id: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> FloatColumn | IntColumn:
    if backend == "numpy":
        return aggregation_numpy.grouped_min(column, group_id)
    return aggregation_jax.grouped_min(column, group_id, num_segments)


@overload
def grouped_any(
    column: IntColumn,
    group_id: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> BoolColumn: ...


@overload
def grouped_any(
    column: BoolColumn,
    group_id: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> BoolColumn: ...


def grouped_any(
    column: IntColumn | BoolColumn,
    group_id: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> BoolColumn:
    if backend == "numpy":
        return aggregation_numpy.grouped_any(column, group_id)
    return aggregation_jax.grouped_any(column, group_id, num_segments)


@overload
def grouped_all(
    column: IntColumn,
    group_id: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> BoolColumn: ...


@overload
def grouped_all(
    column: BoolColumn,
    group_id: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> BoolColumn: ...


def grouped_all(
    column: IntColumn | BoolColumn,
    group_id: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> BoolColumn:
    if backend == "numpy":
        return aggregation_numpy.grouped_all(column, group_id)
    return aggregation_jax.grouped_all(column, group_id, num_segments)


def count_by_p_id(
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> IntColumn:
    if backend == "numpy":
        return aggregation_numpy.count_by_p_id(p_id_to_aggregate_by, p_id_to_store_by)
    return aggregation_jax.count_by_p_id(
        p_id_to_aggregate_by,
        p_id_to_store_by,
        num_segments,
    )


@overload
def sum_by_p_id(
    column: FloatColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> FloatColumn: ...
@overload
def sum_by_p_id(
    column: IntColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> IntColumn: ...


@overload
def sum_by_p_id(
    column: BoolColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> IntColumn: ...


def sum_by_p_id(
    column: FloatColumn | IntColumn | BoolColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> FloatColumn | IntColumn:
    if backend == "numpy":
        return aggregation_numpy.sum_by_p_id(
            column,
            p_id_to_aggregate_by,
            p_id_to_store_by,
        )
    return aggregation_jax.sum_by_p_id(
        column,
        p_id_to_aggregate_by,
        p_id_to_store_by,
        num_segments,
    )


@overload
def mean_by_p_id(
    column: FloatColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> FloatColumn: ...


@overload
def mean_by_p_id(
    column: IntColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> FloatColumn: ...


@overload
def mean_by_p_id(
    column: BoolColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> FloatColumn: ...


def mean_by_p_id(
    column: FloatColumn | IntColumn | BoolColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> FloatColumn:
    if backend == "numpy":
        return aggregation_numpy.mean_by_p_id(
            column,
            p_id_to_aggregate_by,
            p_id_to_store_by,
        )
    return aggregation_jax.mean_by_p_id(
        column,
        p_id_to_aggregate_by,
        p_id_to_store_by,
        num_segments,
    )


@overload
def max_by_p_id(
    column: FloatColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> FloatColumn: ...


@overload
def max_by_p_id(
    column: IntColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> IntColumn: ...


def max_by_p_id(
    column: FloatColumn | IntColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> FloatColumn | IntColumn:
    if backend == "numpy":
        return aggregation_numpy.max_by_p_id(
            column,
            p_id_to_aggregate_by,
            p_id_to_store_by,
        )
    return aggregation_jax.max_by_p_id(
        column,
        p_id_to_aggregate_by,
        p_id_to_store_by,
        num_segments,
    )


@overload
def min_by_p_id(
    column: FloatColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> FloatColumn: ...


@overload
def min_by_p_id(
    column: IntColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> IntColumn: ...


def min_by_p_id(
    column: FloatColumn | IntColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> FloatColumn | IntColumn:
    if backend == "numpy":
        return aggregation_numpy.min_by_p_id(
            column,
            p_id_to_aggregate_by,
            p_id_to_store_by,
        )
    return aggregation_jax.min_by_p_id(
        column,
        p_id_to_aggregate_by,
        p_id_to_store_by,
        num_segments,
    )


@overload
def any_by_p_id(
    column: IntColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> IntColumn: ...


@overload
def any_by_p_id(
    column: BoolColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> BoolColumn: ...


def any_by_p_id(
    column: IntColumn | BoolColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> BoolColumn:
    if backend == "numpy":
        return aggregation_numpy.any_by_p_id(
            column,
            p_id_to_aggregate_by,
            p_id_to_store_by,
        )
    return aggregation_jax.any_by_p_id(
        column,
        p_id_to_aggregate_by,
        p_id_to_store_by,
        num_segments,
    )


@overload
def all_by_p_id(
    column: IntColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> IntColumn: ...


@overload
def all_by_p_id(
    column: BoolColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> BoolColumn: ...


def all_by_p_id(
    column: IntColumn | BoolColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> BoolColumn:
    if backend == "numpy":
        return aggregation_numpy.all_by_p_id(
            column,
            p_id_to_aggregate_by,
            p_id_to_store_by,
        )
    return aggregation_jax.all_by_p_id(
        column,
        p_id_to_aggregate_by,
        p_id_to_store_by,
        num_segments,
    )
