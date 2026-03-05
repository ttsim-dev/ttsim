"""Utilities for validating interval notation in piecewise specs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy
import portion

if TYPE_CHECKING:
    from types import ModuleType

    from jaxtyping import Array, Float


def validate_intervals(intervals: list[portion.Interval], leaf_name: str) -> None:
    """Validate that intervals are ascending, non-overlapping, and contiguous."""
    if not intervals:
        raise ValueError(f"No intervals provided for {leaf_name}.")

    for i in range(1, len(intervals)):
        prev, curr = intervals[i - 1], intervals[i]
        if curr.lower <= prev.lower:
            raise ValueError(
                f"Intervals for {leaf_name} are not in ascending order: "
                f"interval {i - 1} has lower bound {prev.lower}, "
                f"interval {i} has lower bound {curr.lower}."
            )
        if not (prev & curr).empty:
            raise ValueError(
                f"Overlapping intervals for {leaf_name}: "
                f"interval {i - 1} = {prev} and interval {i} = {curr}."
            )
        if prev.upper != curr.lower:
            raise ValueError(
                f"Gap between intervals for {leaf_name}: "
                f"interval {i - 1} upper = {prev.upper}, "
                f"interval {i} lower = {curr.lower}."
            )


def _bound_to_float(v: object) -> float:
    """Convert a portion bound (including portion.inf) to a Python float."""
    if v == -portion.inf:
        return float("-inf")
    if v == portion.inf:
        return float("inf")
    return float(v)


def intervals_to_thresholds(
    intervals: list[portion.Interval], xnp: ModuleType
) -> tuple[
    Float[Array, " n"],
    Float[Array, " n"],
    Float[Array, " n_plus_1"],
]:
    """Extract threshold arrays from parsed intervals."""
    lower = numpy.array([_bound_to_float(iv.lower) for iv in intervals])
    upper = numpy.array([_bound_to_float(iv.upper) for iv in intervals])
    all_bounds = sorted(set(lower) | set(upper))
    return xnp.array(lower), xnp.array(upper), xnp.array(all_bounds)


def merge_piecewise_intervals(
    base: list[dict[str, Any]],
    update: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge updated piecewise intervals into a base set.

    Updated intervals take precedence. Previous intervals overlapping with the
    updated range are trimmed. Coefficients not specified in an updated interval
    are left unspecified in the result.
    """
    if not update:
        return base

    # Parse all intervals
    parsed_base = [
        (portion.from_string(item["interval"], conv=float), item) for item in base
    ]
    parsed_update = [
        (portion.from_string(item["interval"], conv=float), item) for item in update
    ]

    # Compute the total domain covered by the update
    update_domain = portion.empty()
    for iv, _ in parsed_update:
        update_domain = update_domain | iv

    # Collect base intervals outside the update domain
    kept_before = []
    kept_after = []
    for b_iv, b_item in parsed_base:
        remaining = b_iv - update_domain
        if remaining.empty:
            continue
        # remaining could be split into multiple parts; handle each
        for atomic in remaining:
            trimmed = {
                **b_item,
                "interval": portion.to_string(atomic),
            }
            if atomic.upper <= update_domain.lower:
                kept_before.append(trimmed)
            else:
                kept_after.append(trimmed)

    result = kept_before + update + kept_after
    return extend_intervals_to_real_line(result)


def extend_intervals_to_real_line(
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extend intervals so adjacent intervals are contiguous.

    After merging specs (e.g., via updates_previous), changing one interval's
    bounds can leave adjacent intervals with stale boundaries. This propagates
    each interval's upper bound as the next interval's lower bound.

    Example
    -------
    >>> extend_intervals_to_real_line([
    ...     {"interval": "(-inf, 50)", "slope": 0.1},
    ...     {"interval": "[100, inf)", "slope": 0.3},
    ... ])
    [
        {"interval": "(-inf, 50)", "slope": 0.1},
        {"interval": "[50, inf)", "slope": 0.3},
    ]
    """
    if not items or not any("interval" in item for item in items):
        return items

    result = [item.copy() for item in items]
    for i in range(len(result) - 1):
        if "interval" not in result[i] or "interval" not in result[i + 1]:
            continue
        curr = portion.from_string(result[i]["interval"], conv=float)
        next_ = portion.from_string(result[i + 1]["interval"], conv=float)
        if curr.upper != next_.lower:
            complement = portion.CLOSED if curr.right == portion.OPEN else portion.OPEN
            fixed = portion.Interval.from_atomic(
                complement, curr.upper, next_.upper, next_.right
            )
            result[i + 1] = {
                **result[i + 1],
                "interval": portion.to_string(fixed),
            }

    return result
