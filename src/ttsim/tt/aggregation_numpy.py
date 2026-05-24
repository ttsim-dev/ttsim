from __future__ import annotations

import numpy
import numpy_groupies as npg

from ttsim.typing import BoolColumn, DatetimeColumn, FloatColumn, IntColumn


def grouped_count(group_id: IntColumn) -> IntColumn:
    out_grouped = npg.aggregate(
        group_id,
        numpy.ones(len(group_id), dtype=int),
        func="sum",
        fill_value=0,
    )

    return out_grouped[group_id]


def grouped_sum(
    column: FloatColumn | IntColumn | BoolColumn,
    group_id: IntColumn,
) -> FloatColumn | IntColumn:
    if column.dtype == bool:
        column = column.astype(int)
    out_grouped = npg.aggregate(group_id, column, func="sum", fill_value=0)

    # Expand to individual level
    return out_grouped[group_id]


def grouped_mean(
    column: FloatColumn | IntColumn | BoolColumn,
    group_id: IntColumn,
) -> FloatColumn:
    out_grouped = npg.aggregate(group_id, column, func="mean", fill_value=0)

    # Expand to individual level
    return out_grouped[group_id]


def grouped_max(
    column: FloatColumn | IntColumn | DatetimeColumn,
    group_id: IntColumn,
) -> FloatColumn | IntColumn | DatetimeColumn:
    # For datetime, convert to integer (as numpy_groupies can handle datetime only if
    # numba is installed)
    if numpy.issubdtype(column.dtype, numpy.datetime64):
        dtype = column.dtype
        float_col = column.astype("datetime64[D]").astype(int)

        out_grouped_float = npg.aggregate(group_id, float_col, func="max")

        out_grouped = out_grouped_float.astype("datetime64[D]").astype(dtype)

        # Expand to individual level
        out = out_grouped[group_id]

    else:
        out_grouped = npg.aggregate(group_id, column, func="max")

        # Expand to individual level
        out = out_grouped[group_id]
    return out


def grouped_min(
    column: FloatColumn | IntColumn | DatetimeColumn,
    group_id: IntColumn,
) -> FloatColumn | IntColumn | DatetimeColumn:
    # For datetime, convert to integer (as numpy_groupies can handle datetime only if
    # numba is installed)
    if numpy.issubdtype(column.dtype, numpy.datetime64):
        dtype = column.dtype
        float_col = column.astype("datetime64[D]").astype(int)

        out_grouped_float = npg.aggregate(group_id, float_col, func="min")

        out_grouped = out_grouped_float.astype("datetime64[D]").astype(dtype)

        # Expand to individual level
        out = out_grouped[group_id]

    else:
        out_grouped = npg.aggregate(group_id, column, func="min")

        # Expand to individual level
        out = out_grouped[group_id]
    return out


def grouped_any(column: BoolColumn | IntColumn, group_id: IntColumn) -> BoolColumn:
    out_grouped = npg.aggregate(group_id, column, func="any", fill_value=0)

    # Expand to individual level
    return out_grouped[group_id]


def grouped_all(column: BoolColumn | IntColumn, group_id: IntColumn) -> BoolColumn:
    out_grouped = npg.aggregate(group_id, column, func="all", fill_value=0)

    # Expand to individual level
    return out_grouped[group_id]


def count_by_p_id(
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
) -> IntColumn:
    raise NotImplementedError


def sum_by_p_id(
    column: FloatColumn | IntColumn | BoolColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
) -> FloatColumn | IntColumn:
    if column.dtype == bool:
        column = column.astype(int)

    # Vectorized implementation using numpy_groupies
    valid_mask = p_id_to_aggregate_by >= 0
    valid_p_ids = p_id_to_aggregate_by[valid_mask]
    valid_column = column[valid_mask]

    if len(valid_p_ids) > 0:
        max_p_id = int(max(numpy.max(valid_p_ids), numpy.max(p_id_to_store_by)))
        grouped_sums = npg.aggregate(
            valid_p_ids,
            valid_column,
            func="sum",
            size=max_p_id + 1,
            fill_value=0,
        )
        out = grouped_sums[p_id_to_store_by]
    else:
        out = numpy.zeros_like(p_id_to_store_by, dtype=column.dtype)

    return out


def mean_by_p_id(
    column: FloatColumn | IntColumn | BoolColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
) -> FloatColumn:
    raise NotImplementedError


def max_by_p_id(
    column: FloatColumn | IntColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
) -> FloatColumn | IntColumn:
    raise NotImplementedError


def min_by_p_id(
    column: FloatColumn | IntColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
) -> FloatColumn | IntColumn:
    raise NotImplementedError


def any_by_p_id(
    column: BoolColumn | IntColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
) -> BoolColumn:
    raise NotImplementedError


def all_by_p_id(
    column: BoolColumn | IntColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
) -> BoolColumn:
    raise NotImplementedError
