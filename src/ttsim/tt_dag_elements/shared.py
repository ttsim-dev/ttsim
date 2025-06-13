from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

    from ttsim.interface_dag_elements.typing import BoolColumn, FloatColumn, IntColumn


def join(
    foreign_key: IntColumn,
    primary_key: IntColumn,
    target: BoolColumn | IntColumn | FloatColumn,
    value_if_foreign_key_is_missing: float | bool,
    xnp: ModuleType,
) -> BoolColumn | IntColumn | FloatColumn:
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
    # For each foreign key and for each primary key, check if they match
    matches_foreign_key = foreign_key[:, None] == primary_key

    # For each foreign key, add a column with True at the end, to later fall back to
    # the value for unresolved foreign keys
    padded_matches_foreign_key = xnp.pad(
        matches_foreign_key,
        ((0, 0), (0, 1)),
        "constant",
        constant_values=True,
    )

    # For each foreign key, compute the index of the first matching primary key
    indices = xnp.argmax(padded_matches_foreign_key, axis=1)

    # Add the value for unresolved foreign keys at the end of the target array
    padded_targets = xnp.pad(
        target,
        (0, 1),
        "constant",
        constant_values=value_if_foreign_key_is_missing,
    )

    # Return the target at the index of the first matching primary key
    return padded_targets.take(indices)
