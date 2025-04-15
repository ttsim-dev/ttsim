import pytest

from ttsim.config import numpy_or_jax

from ttsim.shared import join


@pytest.mark.parametrize(
    "foreign_key, primary_key, target, value_if_foreign_key_is_missing, expected",
    [
        (
            numpy_or_jax.array([1, 2, 3]),
            numpy_or_jax.array([1, 2, 3]),
            numpy_or_jax.array([1, 2, 3]),
            4,
            numpy_or_jax.array([1, 2, 3]),
        ),
        (
            numpy_or_jax.array([3, 2, 1]),
            numpy_or_jax.array([1, 2, 3]),
            numpy_or_jax.array([1, 2, 3]),
            4,
            numpy_or_jax.array([3, 2, 1]),
        ),
        (
            numpy_or_jax.array([1, 1, 1]),
            numpy_or_jax.array([1, 2, 3]),
            numpy_or_jax.array([1, 2, 3]),
            4,
            numpy_or_jax.array([1, 1, 1]),
        ),
        (
            numpy_or_jax.array([-1]),
            numpy_or_jax.array([1]),
            numpy_or_jax.array([1]),
            4,
            numpy_or_jax.array([4]),
        ),
    ],
)
def test_join(
    foreign_key: numpy_or_jax.ndarray,
    primary_key: numpy_or_jax.ndarray,
    target: numpy_or_jax.ndarray,
    value_if_foreign_key_is_missing: int,
    expected: numpy_or_jax.ndarray,
):
    assert numpy_or_jax.array_equal(
        join(foreign_key, primary_key, target, value_if_foreign_key_is_missing),
        expected,
    )
