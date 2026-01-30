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


def extend_intervals_to_real_line(
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extend intervals so adjacent intervals are contiguous.

    After merging specs (e.g., via updates_previous), changing one interval's
    bounds can leave adjacent intervals with stale boundaries. This propagates
    each interval's upper bound as the next interval's lower bound.
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
