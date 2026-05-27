from __future__ import annotations

from ttsim.typing import BoolColumn, FloatColumn, IntColumn

try:
    import jax.numpy as jnp
    from jax.ops import (
        segment_max,
        segment_min,
        segment_sum,
    )
except ImportError:
    pass


def grouped_count(group_id: IntColumn, num_segments: int) -> jnp.ndarray:
    out_grouped = segment_sum(
        data=jnp.ones(len(group_id), dtype=jnp.int32),
        segment_ids=group_id,
        num_segments=num_segments,
    )
    return out_grouped[group_id]


def grouped_sum(
    column: FloatColumn | IntColumn | BoolColumn,
    group_id: IntColumn,
    num_segments: int,
) -> FloatColumn | IntColumn:
    if column.dtype == "bool":
        column = column.astype(int)

    out_grouped = segment_sum(
        data=column,
        segment_ids=group_id,
        num_segments=num_segments,
    )
    return out_grouped[group_id]


def grouped_mean(
    column: FloatColumn | IntColumn | BoolColumn,
    group_id: IntColumn,
    num_segments: int,
) -> FloatColumn:
    if column.dtype == "bool":
        column = column.astype(int)
    sum_grouped = segment_sum(
        data=column,
        segment_ids=group_id,
        num_segments=num_segments,
    )
    sizes = segment_sum(
        data=jnp.ones(len(column)),
        segment_ids=group_id,
        num_segments=num_segments,
    )
    mean_grouped = sum_grouped / sizes
    return mean_grouped[group_id]


def grouped_max(
    column: FloatColumn | IntColumn,
    group_id: IntColumn,
    num_segments: int,
) -> FloatColumn | IntColumn:
    out_grouped = segment_max(
        data=column,
        segment_ids=group_id,
        num_segments=num_segments,
    )
    return out_grouped[group_id]


def grouped_min(
    column: FloatColumn | IntColumn,
    group_id: IntColumn,
    num_segments: int,
) -> FloatColumn | IntColumn:
    out_grouped = segment_min(
        data=column,
        segment_ids=group_id,
        num_segments=num_segments,
    )
    return out_grouped[group_id]


def grouped_any(
    column: BoolColumn | IntColumn,
    group_id: IntColumn,
    num_segments: int,
) -> BoolColumn:
    # Convert to boolean if necessary
    if jnp.issubdtype(column.dtype, jnp.integer):
        my_col = column.astype("bool")
    else:
        my_col = column

    out_grouped = segment_max(
        data=my_col,
        segment_ids=group_id,
        num_segments=num_segments,
    )
    return out_grouped[group_id]


def grouped_all(
    column: BoolColumn | IntColumn,
    group_id: IntColumn,
    num_segments: int,
) -> BoolColumn:
    # Convert to boolean if necessary
    if jnp.issubdtype(column.dtype, jnp.integer):
        column = column.astype("bool")

    out_grouped = segment_min(
        data=column,
        segment_ids=group_id,
        num_segments=num_segments,
    )
    return out_grouped[group_id]


def count_by_p_id(
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
    num_segments: int,  # noqa: ARG001
) -> IntColumn:
    mapped_index, valid = _by_p_id_index(
        p_id_to_aggregate_by=p_id_to_aggregate_by, p_id_to_store_by=p_id_to_store_by
    )
    contributions = jnp.where(valid, jnp.int32(1), jnp.int32(0))
    out = jnp.zeros_like(p_id_to_store_by, dtype=jnp.int32)
    return out.at[mapped_index].add(contributions)


def sum_by_p_id(
    column: FloatColumn | IntColumn | BoolColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
    num_segments: int,  # noqa: ARG001
) -> FloatColumn | IntColumn:
    if column.dtype == bool:
        column = column.astype(int)

    mapped_index, valid = _by_p_id_index(
        p_id_to_aggregate_by=p_id_to_aggregate_by, p_id_to_store_by=p_id_to_store_by
    )
    contributions = jnp.where(valid, column, jnp.asarray(0, dtype=column.dtype))
    out = jnp.zeros_like(p_id_to_store_by, dtype=column.dtype)
    return out.at[mapped_index].add(contributions)


def mean_by_p_id(
    column: FloatColumn | IntColumn | BoolColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
    num_segments: int,  # noqa: ARG001
) -> FloatColumn:
    # Promote to at least float32 so integer / bool sources have a sensible
    # mean, but keep float64 columns at float64 — hard-coding `jnp.float32`
    # silently downcasts when JAX is running with `jax_enable_x64=True`.
    out_dtype = jnp.result_type(column, jnp.float32)
    column = column.astype(out_dtype)
    mapped_index, valid = _by_p_id_index(
        p_id_to_aggregate_by=p_id_to_aggregate_by, p_id_to_store_by=p_id_to_store_by
    )
    contributions = jnp.where(valid, column, jnp.asarray(0, dtype=out_dtype))
    counts = jnp.where(valid, jnp.int32(1), jnp.int32(0))
    sum_out = jnp.zeros_like(p_id_to_store_by, dtype=out_dtype)
    sum_out = sum_out.at[mapped_index].add(contributions)
    count_out = jnp.zeros_like(p_id_to_store_by, dtype=jnp.int32)
    count_out = count_out.at[mapped_index].add(counts)
    # Empty groups: counts==0; divide by 1 instead and the numerator is also 0,
    # so the result is the desired fill_value=0.
    safe_count = jnp.where(count_out == 0, jnp.int32(1), count_out)
    return sum_out / safe_count


def max_by_p_id(
    column: FloatColumn | IntColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
    num_segments: int,  # noqa: ARG001
) -> FloatColumn | IntColumn:
    return _scatter_reduce_by_p_id(
        column=column,
        p_id_to_aggregate_by=p_id_to_aggregate_by,
        p_id_to_store_by=p_id_to_store_by,
        reducer="max",
        sentinel=_minimum_of_dtype(column.dtype),
        empty_fill=jnp.asarray(0, dtype=column.dtype),
    )


def min_by_p_id(
    column: FloatColumn | IntColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
    num_segments: int,  # noqa: ARG001
) -> FloatColumn | IntColumn:
    return _scatter_reduce_by_p_id(
        column=column,
        p_id_to_aggregate_by=p_id_to_aggregate_by,
        p_id_to_store_by=p_id_to_store_by,
        reducer="min",
        sentinel=_maximum_of_dtype(column.dtype),
        empty_fill=jnp.asarray(0, dtype=column.dtype),
    )


def any_by_p_id(
    column: BoolColumn | IntColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
    num_segments: int,  # noqa: ARG001
) -> BoolColumn:
    column = column.astype(bool)
    mapped_index, valid = _by_p_id_index(
        p_id_to_aggregate_by=p_id_to_aggregate_by, p_id_to_store_by=p_id_to_store_by
    )
    false_ = jnp.asarray(False)  # noqa: FBT003
    contributions = jnp.where(valid, column, false_)
    out = jnp.zeros_like(p_id_to_store_by, dtype=jnp.bool_)
    return out.at[mapped_index].max(contributions)


def all_by_p_id(
    column: BoolColumn | IntColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
    num_segments: int,  # noqa: ARG001
) -> BoolColumn:
    column = column.astype(bool)
    mapped_index, valid = _by_p_id_index(
        p_id_to_aggregate_by=p_id_to_aggregate_by, p_id_to_store_by=p_id_to_store_by
    )
    true_ = jnp.asarray(True)  # noqa: FBT003
    contributions = jnp.where(valid, column, true_)
    out = jnp.ones_like(p_id_to_store_by, dtype=jnp.bool_)
    return out.at[mapped_index].min(contributions)


def _by_p_id_index(
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
) -> tuple[IntColumn, BoolColumn]:
    """Build the scatter index from `p_id_to_aggregate_by` keys to positions
    in `p_id_to_store_by`, plus a mask flagging the valid (non-negative,
    actually-present) source entries.

    `jnp.searchsorted` returns a *clipped* insertion index when the query
    isn't in the array — there is no "not found" signal. So a non-negative
    `p_id_to_aggregate_by` value that doesn't exist in `p_id_to_store_by`
    would silently scatter into whichever bucket the clipped index points
    at. Gate this by comparing the gathered store value against the query
    and folding misses into the validity mask alongside the negative-source
    check.
    """
    sorted_idx = jnp.argsort(p_id_to_store_by)
    sorted_store = p_id_to_store_by[sorted_idx]
    candidate_pos = jnp.clip(
        jnp.searchsorted(sorted_store, p_id_to_aggregate_by),
        0,
        sorted_store.shape[0] - 1,
    )
    candidate_idx = sorted_idx[candidate_pos]
    hit = p_id_to_store_by[candidate_idx] == p_id_to_aggregate_by
    valid = (p_id_to_aggregate_by >= 0) & hit
    mapped_index = jnp.where(valid, candidate_idx, 0)
    return mapped_index, valid


def _scatter_reduce_by_p_id(
    column: FloatColumn | IntColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
    reducer: str,
    sentinel: jnp.ndarray,
    empty_fill: jnp.ndarray,
) -> FloatColumn | IntColumn:
    """Scatter-reduce `column` from `p_id_to_aggregate_by` keys to
    `p_id_to_store_by` keys using `reducer` (`"min"` or `"max"`). Invalid
    source entries become `sentinel` so they do not influence the result;
    destinations that no valid source actually wrote to are rewritten to
    `empty_fill`.

    "Was this destination written?" is tracked with a separate boolean
    accumulator (`touched`) rather than `reduced == sentinel`, because a
    legitimate column value can collide with the sentinel (e.g. an int
    column carrying `iinfo.min`, or a float column carrying `finfo.min`);
    sentinel-equality would silently rewrite that real result to zero.
    """
    mapped_index, valid = _by_p_id_index(
        p_id_to_aggregate_by=p_id_to_aggregate_by, p_id_to_store_by=p_id_to_store_by
    )
    contributions = jnp.where(valid, column, sentinel)
    out = jnp.full_like(p_id_to_store_by, sentinel, dtype=column.dtype)
    reduced = getattr(out.at[mapped_index], reducer)(contributions)
    touched = jnp.zeros_like(p_id_to_store_by, dtype=jnp.bool_)
    touched = touched.at[mapped_index].max(valid)
    return jnp.where(touched, reduced, empty_fill)


def _minimum_of_dtype(dtype: jnp.dtype) -> jnp.ndarray:
    return (
        jnp.asarray(jnp.finfo(dtype).min)
        if jnp.issubdtype(dtype, jnp.floating)
        else jnp.asarray(jnp.iinfo(dtype).min)
    )


def _maximum_of_dtype(dtype: jnp.dtype) -> jnp.ndarray:
    return (
        jnp.asarray(jnp.finfo(dtype).max)
        if jnp.issubdtype(dtype, jnp.floating)
        else jnp.asarray(jnp.iinfo(dtype).max)
    )
