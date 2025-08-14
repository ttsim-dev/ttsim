from __future__ import annotations

from typing import TYPE_CHECKING, overload

if TYPE_CHECKING:
    from types import ModuleType

    from ttsim.typing import BoolColumn, FloatColumn, IntColumn


@overload
def join(
    foreign_key: IntColumn,
    primary_key: IntColumn,
    target: FloatColumn,
    value_if_foreign_key_is_missing: float | bool,
    xnp: ModuleType,
) -> FloatColumn: ...


@overload
def join(
    foreign_key: IntColumn,
    primary_key: IntColumn,
    target: IntColumn,
    value_if_foreign_key_is_missing: float | bool,
    xnp: ModuleType,
) -> IntColumn: ...


@overload
def join(
    foreign_key: IntColumn,
    primary_key: IntColumn,
    target: BoolColumn,
    value_if_foreign_key_is_missing: float | bool,
    xnp: ModuleType,
) -> BoolColumn: ...


def join(
    foreign_key: IntColumn,
    primary_key: IntColumn,
    target: FloatColumn | IntColumn | BoolColumn,
    value_if_foreign_key_is_missing: float | bool,
    xnp: ModuleType,
) -> FloatColumn | IntColumn | BoolColumn:
    """
    Given a foreign key, find the corresponding primary key, and return the target at
    the same index as the primary key. When using Jax, does not work on String Arrays.

    Parameters
    ----------
    foreign_key:
        The foreign keys.
    primary_key:
        The primary keys.
    target:
        The targets, in the same order as the primary keys.
    value_if_foreign_key_is_missing:
        The value to return if no matching primary key is found.
    xnp:
        The numpy module to use for calculations.

    Returns
    -------
    The joined array.
    """
    # First, get the sort order of primary_key to enable efficient lookup
    sort_indices = xnp.argsort(primary_key)
    sorted_primary_key = primary_key[sort_indices]
    sorted_target = target[sort_indices]

    # Find where each foreign_key would be inserted in the sorted primary_key array
    positions = xnp.searchsorted(sorted_primary_key, foreign_key, side="left")

    # Check if the foreign keys actually match the primary keys at those positions
    # Handle out-of-bounds positions
    valid_positions = positions < len(sorted_primary_key)
    matches = valid_positions & (
        sorted_primary_key[xnp.minimum(positions, len(sorted_primary_key) - 1)]
        == foreign_key
    )

    # Create result array initialized with the missing value
    result = xnp.full_like(
        foreign_key, value_if_foreign_key_is_missing, dtype=target.dtype
    )

    # Get the corresponding target values for valid matches, use 0 for invalid indices
    valid_indices = xnp.where(matches, positions, 0)
    return xnp.where(matches, sorted_target[valid_indices], result)
