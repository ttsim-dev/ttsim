from __future__ import annotations

from typing import TYPE_CHECKING

import numpy
import numpy_groupies as npg

if TYPE_CHECKING:
    from ttsim.tt import BoolColumn, FloatColumn, IntColumn


def grouped_count(group_id: IntColumn) -> IntColumn:
    fail_if_dtype_not_int(group_id, agg_func="grouped_count")
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
    fail_if_dtype_not_numeric_or_boolean(column, agg_func="grouped_sum")
    fail_if_dtype_not_int(group_id, agg_func="grouped_sum")
    if column.dtype == bool:
        column = column.astype(int)
    out_grouped = npg.aggregate(group_id, column, func="sum", fill_value=0)

    # Expand to individual level
    return out_grouped[group_id]


def grouped_mean(
    column: FloatColumn | IntColumn | BoolColumn,
    group_id: IntColumn,
) -> FloatColumn:
    fail_if_dtype_not_numeric_or_boolean(column, agg_func="grouped_mean")
    fail_if_dtype_not_int(group_id, agg_func="grouped_mean")

    out_grouped = npg.aggregate(group_id, column, func="mean", fill_value=0)

    # Expand to individual level
    return out_grouped[group_id]


def grouped_max(
    column: FloatColumn | IntColumn | BoolColumn,
    group_id: IntColumn,
) -> FloatColumn | IntColumn:
    fail_if_dtype_not_numeric_or_datetime(column, agg_func="grouped_max")
    fail_if_dtype_not_int(group_id, agg_func="grouped_max")

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
    column: FloatColumn | IntColumn,
    group_id: IntColumn,
) -> FloatColumn | IntColumn:
    fail_if_dtype_not_numeric_or_datetime(column, agg_func="grouped_min")
    fail_if_dtype_not_int(group_id, agg_func="grouped_min")

    # For datetime, convert to integer (as numpy_groupies can handle datetime only if
    # numba is installed)

    if numpy.issubdtype(column.dtype, numpy.datetime64) or numpy.issubdtype(
        column.dtype,
        numpy.timedelta64,
    ):
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
    fail_if_dtype_not_boolean_or_int(column, agg_func="grouped_any")
    fail_if_dtype_not_int(group_id, agg_func="grouped_any")

    out_grouped = npg.aggregate(group_id, column, func="any", fill_value=0)

    # Expand to individual level
    return out_grouped[group_id]


def grouped_all(column: BoolColumn | IntColumn, group_id: IntColumn) -> BoolColumn:
    fail_if_dtype_not_boolean_or_int(column, agg_func="grouped_all")
    fail_if_dtype_not_int(group_id, agg_func="grouped_all")

    out_grouped = npg.aggregate(group_id, column, func="all", fill_value=0)

    # Expand to individual level
    return out_grouped[group_id]


def count_by_p_id(
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
) -> IntColumn:
    fail_if_dtype_not_int(p_id_to_aggregate_by, agg_func="count_by_p_id")
    fail_if_dtype_not_int(p_id_to_store_by, agg_func="count_by_p_id")

    raise NotImplementedError


def sum_by_p_id(
    column: FloatColumn | IntColumn | BoolColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
) -> FloatColumn | IntColumn:
    fail_if_dtype_not_numeric_or_boolean(column, agg_func="sum_by_p_id")
    fail_if_dtype_not_int(p_id_to_aggregate_by, agg_func="sum_by_p_id")
    fail_if_dtype_not_int(p_id_to_store_by, agg_func="sum_by_p_id")

    if column.dtype in ["bool"]:
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
    fail_if_dtype_not_numeric_or_boolean(column, agg_func="mean_by_p_id")
    fail_if_dtype_not_int(p_id_to_aggregate_by, agg_func="mean_by_p_id")
    fail_if_dtype_not_int(p_id_to_store_by, agg_func="mean_by_p_id")
    raise NotImplementedError


def max_by_p_id(
    column: FloatColumn | IntColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
) -> FloatColumn | IntColumn:
    fail_if_dtype_not_numeric_or_datetime(column, agg_func="max_by_p_id")
    fail_if_dtype_not_int(p_id_to_aggregate_by, agg_func="max_by_p_id")
    fail_if_dtype_not_int(p_id_to_store_by, agg_func="max_by_p_id")
    raise NotImplementedError


def min_by_p_id(
    column: FloatColumn | IntColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
) -> FloatColumn | IntColumn:
    fail_if_dtype_not_numeric_or_datetime(column, agg_func="min_by_p_id")
    fail_if_dtype_not_int(p_id_to_aggregate_by, agg_func="min_by_p_id")
    fail_if_dtype_not_int(p_id_to_store_by, agg_func="min_by_p_id")
    raise NotImplementedError


def any_by_p_id(
    column: BoolColumn | IntColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
) -> BoolColumn:
    fail_if_dtype_not_boolean_or_int(column, agg_func="any_by_p_id")
    fail_if_dtype_not_int(p_id_to_aggregate_by, agg_func="any_by_p_id")
    fail_if_dtype_not_int(p_id_to_store_by, agg_func="any_by_p_id")
    raise NotImplementedError


def all_by_p_id(
    column: BoolColumn | IntColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
) -> BoolColumn:
    fail_if_dtype_not_boolean_or_int(column, agg_func="all_by_p_id")
    fail_if_dtype_not_int(p_id_to_store_by, agg_func="all_by_p_id")
    fail_if_dtype_not_int(p_id_to_aggregate_by, agg_func="all_by_p_id")
    raise NotImplementedError


def fail_if_dtype_not_numeric(
    column: FloatColumn | IntColumn | BoolColumn,
    agg_func: str,
) -> None:
    if not numpy.issubdtype(column.dtype, numpy.number):
        raise TypeError(
            f"Aggregation function {agg_func} was applied to a column "
            f"with dtype {column.dtype}. Allowed are only numerical dtypes.",
        )


def fail_if_dtype_not_float(
    column: FloatColumn | IntColumn | BoolColumn,
    agg_func: str,
) -> None:
    if not numpy.issubdtype(column.dtype, numpy.floating):
        raise TypeError(
            f"Aggregation function {agg_func} was applied to a column "
            f"with dtype {column.dtype}. Allowed is only float.",
        )


def fail_if_dtype_not_int(p_id_to_aggregate_by: IntColumn, agg_func: str) -> None:
    if not numpy.issubdtype(p_id_to_aggregate_by.dtype, numpy.integer):
        raise TypeError(
            f"The dtype of id columns must be integer. Aggregation function {agg_func} "
            f"was applied to a id columns that has dtype {p_id_to_aggregate_by.dtype}.",
        )


def fail_if_dtype_not_numeric_or_boolean(
    column: FloatColumn | IntColumn | BoolColumn,
    agg_func: str,
) -> None:
    if not (numpy.issubdtype(column.dtype, numpy.number) or column.dtype == "bool"):
        raise TypeError(
            f"Aggregation function {agg_func} was applied to a column with dtype "
            f"{column.dtype}. Allowed are only numerical or Boolean dtypes.",
        )


def fail_if_dtype_not_numeric_or_datetime(
    column: FloatColumn | IntColumn | BoolColumn,
    agg_func: str,
) -> None:
    if not (
        numpy.issubdtype(column.dtype, numpy.number)
        or numpy.issubdtype(column.dtype, numpy.datetime64)
    ):
        raise TypeError(
            f"Aggregation function {agg_func} was applied to a column with dtype "
            f"{column.dtype}. Allowed are only numerical or datetime dtypes.",
        )


def fail_if_dtype_not_boolean_or_int(
    column: BoolColumn | IntColumn,
    agg_func: str,
) -> None:
    if not (
        numpy.issubdtype(column.dtype, numpy.integer)
        or numpy.issubdtype(column.dtype, numpy.bool_)
    ):
        raise TypeError(
            f"Aggregation function {agg_func} was applied to a column with dtype "
            f"{column.dtype}. Allowed are only Boolean and int dtypes.",
        )
