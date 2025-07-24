from __future__ import annotations

from typing import TYPE_CHECKING

try:
    import jax.numpy as jnp
    from jax.ops import segment_max, segment_min, segment_sum
except ImportError:
    pass

if TYPE_CHECKING:
    from ttsim.tt import BoolColumn, FloatColumn, IntColumn


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
    if column.dtype in ["bool"]:
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
    if column.dtype in ["bool"]:
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
    num_segments: int,
) -> IntColumn:
    raise NotImplementedError


def sum_by_p_id(
    column: FloatColumn | IntColumn | BoolColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
    num_segments: int,  # noqa: ARG001
) -> FloatColumn | IntColumn:
    if column.dtype == bool:
        column = column.astype(int)

    # Build an index mapping from p_id values to positions in p_id_to_store_by.
    sorted_idx = jnp.argsort(p_id_to_store_by)
    sorted_store = p_id_to_store_by[sorted_idx]

    # For every element in p_id_to_aggregate_by (even negatives),
    # use searchsorted to get its candidate index in sorted_store.
    candidate = jnp.searchsorted(sorted_store, p_id_to_aggregate_by)
    candidate_idx = sorted_idx[candidate]

    # For invalid (negative) IDs, force a dummy index (0) that will be masked out.
    mapped_index = jnp.where(p_id_to_aggregate_by >= 0, candidate_idx, 0)

    # Only valid entries contribute to the sum.
    contributions = jnp.where(p_id_to_aggregate_by >= 0, column, 0)

    # Scatter-add the contributions to the output array.
    out = jnp.zeros_like(p_id_to_store_by, dtype=column.dtype)
    return out.at[mapped_index].add(contributions)


def mean_by_p_id(
    column: FloatColumn | IntColumn | BoolColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
    num_segments: int,
) -> FloatColumn:
    raise NotImplementedError


def max_by_p_id(
    column: FloatColumn | IntColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
    num_segments: int,
) -> FloatColumn | IntColumn:
    raise NotImplementedError


def min_by_p_id(
    column: FloatColumn | IntColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
    num_segments: int,
) -> FloatColumn | IntColumn:
    raise NotImplementedError


def any_by_p_id(
    column: BoolColumn | IntColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
    num_segments: int,
) -> BoolColumn:
    raise NotImplementedError


def all_by_p_id(
    column: BoolColumn | IntColumn,
    p_id_to_aggregate_by: IntColumn,
    p_id_to_store_by: IntColumn,
    num_segments: int,
) -> BoolColumn:
    raise NotImplementedError
