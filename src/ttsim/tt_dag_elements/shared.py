from __future__ import annotations

from ttsim.config import numpy_or_jax as np


def join(
    foreign_key: np.ndarray,
    primary_key: np.ndarray,
    target: np.ndarray,
    value_if_foreign_key_is_missing: float | bool,
) -> np.ndarray:
    """
    Given a foreign key, find the corresponding primary key, and return the target at
    the same index as the primary key. When using Jax, does not work on String Arrays.

    Parameters
    ----------
    foreign_key : np.ndarray[Key]
        The foreign keys.
    primary_key : np.ndarray[Key]
        The primary keys.
    target : np.ndarray[Out]
        The targets in the same order as the primary keys.
    value_if_foreign_key_is_missing : Out
        The value to return if no matching primary key is found.

    Returns
    -------
    The joined array.
    """
    # For each foreign key and for each primary key, check if they match
    matches_foreign_key = foreign_key[:, None] == primary_key

    # For each foreign key, add a column with True at the end, to later fall back to
    # the value for unresolved foreign keys
    padded_matches_foreign_key = np.pad(
        matches_foreign_key, ((0, 0), (0, 1)), "constant", constant_values=True
    )

    # For each foreign key, compute the index of the first matching primary key
    indices = np.argmax(padded_matches_foreign_key, axis=1)

    # Add the value for unresolved foreign keys at the end of the target array
    padded_targets = np.pad(
        target, (0, 1), "constant", constant_values=value_if_foreign_key_is_missing
    )

    # Return the target at the index of the first matching primary key
    return padded_targets.take(indices)
