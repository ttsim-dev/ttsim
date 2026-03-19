"""Utilities for validating interval notation in piecewise specs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

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
        if prev.right == portion.OPEN and curr.left == portion.OPEN:
            raise ValueError(
                f"Gap at boundary {prev.upper} for {leaf_name}: "
                f"interval {i - 1} = {prev} is open-right and "
                f"interval {i} = {curr} is open-left."
            )


def merge_piecewise_intervals(
    base: list[dict[str, Any]],
    update: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge update intervals into base intervals by exact bound matching.

    For each interval in `update`, find the base interval with identical parsed bounds
    and replace it. Intervals in `base` that are not matched are kept as-is.
    """
    base_parsed = [portion.from_string(item["interval"], conv=float) for item in base]
    result = list(base)

    for upd_item in update:
        upd_interval = portion.from_string(upd_item["interval"], conv=float)
        matched = False
        for i, base_interval in enumerate(base_parsed):
            if base_interval == upd_interval:
                result[i] = upd_item
                matched = True
                break
        if not matched:
            msg = (
                f"Update interval {upd_item['interval']!r} "
                f"does not match any base interval."
            )
            raise ValueError(msg)

    return result


def _bound_to_float(v: float) -> float:
    """Convert a portion bound (including portion.inf) to a Python float."""
    if v == -portion.inf:
        return float("-inf")
    if v == portion.inf:
        return float("inf")
    return float(v)


def intervals_to_thresholds(
    intervals: list[portion.Interval] | list[dict[str, Any]],
    xnp: ModuleType,
    leaf_name: str = "",
) -> tuple[
    Float[Array, " n"],
    Float[Array, " n"],
    Float[Array, " n_plus_1"],
]:
    """Extract threshold arrays from parsed or raw intervals.

    Parameters
    ----------
    intervals
        Either a list of ``portion.Interval`` objects (already parsed) or a list
        of raw interval dicts containing an ``"interval"`` key with string
        notation (e.g. ``"[0, 100)"``).
    xnp
        The backend module to use for array creation.
    leaf_name
        Name used in error messages during validation.

    """
    if intervals and isinstance(intervals[0], dict):
        parsed = [
            portion.from_string(item["interval"], conv=float) for item in intervals
        ]
        if leaf_name:
            validate_intervals(parsed, leaf_name)
    else:
        parsed = cast("list[portion.Interval]", intervals)
    lower = numpy.array([_bound_to_float(iv.lower) for iv in parsed])
    upper = numpy.array([_bound_to_float(iv.upper) for iv in parsed])
    all_bounds = numpy.array(sorted(set(lower) | set(upper)))
    return xnp.array(lower), xnp.array(upper), xnp.array(all_bounds)
