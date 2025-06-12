from __future__ import annotations

from typing import TYPE_CHECKING

try:
    import jax.numpy as jnp
    from jax.ops import segment_max, segment_min, segment_sum
except ImportError:
    pass

if TYPE_CHECKING:
    try:
        import jax.numpy as jnp
    except ImportError:
        import numpy as jnp  # noqa: TC004


def grouped_count(group_id: jnp.ndarray, num_segments: int) -> jnp.ndarray:
    out_on_hh = segment_sum(
        data=jnp.ones(len(group_id)), segment_ids=group_id, num_segments=num_segments
    )
    return out_on_hh[group_id]


def grouped_sum(
    column: jnp.ndarray, group_id: jnp.ndarray, num_segments: int
) -> jnp.ndarray:
    if column.dtype in ["bool"]:
        column = column.astype(int)

    out_on_hh = segment_sum(
        data=column, segment_ids=group_id, num_segments=num_segments
    )
    return out_on_hh[group_id]


def grouped_mean(
    column: jnp.ndarray, group_id: jnp.ndarray, num_segments: int
) -> jnp.ndarray:
    sum_on_hh = segment_sum(
        data=column, segment_ids=group_id, num_segments=num_segments
    )
    sizes = segment_sum(
        data=jnp.ones(len(column)), segment_ids=group_id, num_segments=num_segments
    )
    mean_on_hh = sum_on_hh / sizes
    return mean_on_hh[group_id]


def grouped_max(
    column: jnp.ndarray, group_id: jnp.ndarray, num_segments: int
) -> jnp.ndarray:
    out_on_hh = segment_max(
        data=column, segment_ids=group_id, num_segments=num_segments
    )
    return out_on_hh[group_id]


def grouped_min(
    column: jnp.ndarray, group_id: jnp.ndarray, num_segments: int
) -> jnp.ndarray:
    out_on_hh = segment_min(
        data=column, segment_ids=group_id, num_segments=num_segments
    )
    return out_on_hh[group_id]


def grouped_any(
    column: jnp.ndarray, group_id: jnp.ndarray, num_segments: int
) -> jnp.ndarray:
    # Convert to boolean if necessary
    if jnp.issubdtype(column.dtype, jnp.integer):
        my_col = column.astype("bool")
    else:
        my_col = column

    out_on_hh = segment_max(
        data=my_col, segment_ids=group_id, num_segments=num_segments
    )
    return out_on_hh[group_id]


def grouped_all(
    column: jnp.ndarray, group_id: jnp.ndarray, num_segments: int
) -> jnp.ndarray:
    # Convert to boolean if necessary
    if jnp.issubdtype(column.dtype, jnp.integer):
        column = column.astype("bool")

    out_on_hh = segment_min(
        data=column, segment_ids=group_id, num_segments=num_segments
    )
    return out_on_hh[group_id]


def count_by_p_id(
    p_id_to_aggregate_by: jnp.ndarray,
    p_id_to_store_by: jnp.ndarray,
    num_segments: int,
) -> jnp.ndarray:
    raise NotImplementedError


def sum_by_p_id(
    column: jnp.ndarray,
    p_id_to_aggregate_by: jnp.ndarray,
    p_id_to_store_by: jnp.ndarray,
    num_segments: int,  # noqa: ARG001
) -> jnp.ndarray:
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
    column: jnp.ndarray,
    p_id_to_aggregate_by: jnp.ndarray,
    p_id_to_store_by: jnp.ndarray,
    num_segments: int,
) -> jnp.ndarray:
    raise NotImplementedError


def max_by_p_id(
    column: jnp.ndarray,
    p_id_to_aggregate_by: jnp.ndarray,
    p_id_to_store_by: jnp.ndarray,
    num_segments: int,
) -> jnp.ndarray:
    raise NotImplementedError


def min_by_p_id(
    column: jnp.ndarray,
    p_id_to_aggregate_by: jnp.ndarray,
    p_id_to_store_by: jnp.ndarray,
    num_segments: int,
) -> jnp.ndarray:
    raise NotImplementedError


def any_by_p_id(
    column: jnp.ndarray,
    p_id_to_aggregate_by: jnp.ndarray,
    p_id_to_store_by: jnp.ndarray,
    num_segments: int,
) -> jnp.ndarray:
    raise NotImplementedError


def all_by_p_id(
    column: jnp.ndarray,
    p_id_to_aggregate_by: jnp.ndarray,
    p_id_to_store_by: jnp.ndarray,
    num_segments: int,
) -> jnp.ndarray:
    raise NotImplementedError
