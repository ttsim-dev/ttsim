import numpy
import pytest
from jax import numpy as jnp

from ttsim.shared import join_jax, join_numpy


@pytest.mark.parametrize(
    "foreign_key, primary_key, target, value_if_foreign_key_is_missing, expected",
    [
        (
            numpy.array([1, 2, 3]),
            numpy.array([1, 2, 3]),
            numpy.array(["a", "b", "c"]),
            "d",
            numpy.array(["a", "b", "c"]),
        ),
        (
            numpy.array([3, 2, 1]),
            numpy.array([1, 2, 3]),
            numpy.array(["a", "b", "c"]),
            "d",
            numpy.array(["c", "b", "a"]),
        ),
        (
            numpy.array([1, 1, 1]),
            numpy.array([1, 2, 3]),
            numpy.array(["a", "b", "c"]),
            "d",
            numpy.array(["a", "a", "a"]),
        ),
        (
            numpy.array([-1]),
            numpy.array([1]),
            numpy.array(["a"]),
            "d",
            numpy.array(["d"]),
        ),
    ],
)
def test_join_numpy(
    foreign_key: numpy.ndarray[int],
    primary_key: numpy.ndarray[int],
    target: numpy.ndarray[str],
    value_if_foreign_key_is_missing: str,
    expected: numpy.ndarray[str],
):
    assert numpy.array_equal(
        join_numpy(foreign_key, primary_key, target, value_if_foreign_key_is_missing),
        expected,
    )


@pytest.mark.parametrize(
    "foreign_key, primary_key, target, value_if_foreign_key_is_missing, expected",
    [
        (
            numpy.array([1, 2, 3]),
            numpy.array([1, 2, 3]),
            numpy.array([1, 2, 3]),
            4,
            numpy.array([1, 2, 3]),
        ),
        (
            numpy.array([3, 2, 1]),
            numpy.array([1, 2, 3]),
            numpy.array([1, 2, 3]),
            4,
            numpy.array([3, 2, 1]),
        ),
        (
            numpy.array([1, 1, 1]),
            numpy.array([1, 2, 3]),
            numpy.array([1, 2, 3]),
            4,
            numpy.array([1, 1, 1]),
        ),
        (
            numpy.array([-1]),
            numpy.array([1]),
            numpy.array([1]),
            4,
            numpy.array([4]),
        ),
    ],
)
def test_join_jax(
    foreign_key: numpy.ndarray[int],
    primary_key: numpy.ndarray[int],
    target: numpy.ndarray[str],
    value_if_foreign_key_is_missing: str,
    expected: numpy.ndarray[str],
):
    assert numpy.array_equal(
        join_jax(
            jnp.array(foreign_key),
            jnp.array(primary_key),
            jnp.array(target),
            jnp.array(value_if_foreign_key_is_missing),
        ),
        jnp.array(expected),
    )


def test_join_numpy_raises_duplicate_primary_key():
    with pytest.raises(ValueError, match="Duplicate primary keys:"):
        join_numpy(
            numpy.array([1, 1, 1]),
            numpy.array([1, 1, 1]),
            numpy.array(["a", "b", "c"]),
            "default",
        )


def test_join_numpy_raises_invalid_foreign_key():
    with pytest.raises(ValueError, match="Invalid foreign keys:"):
        join_numpy(numpy.array([2]), numpy.array([1]), numpy.array(["a"]), "d")
