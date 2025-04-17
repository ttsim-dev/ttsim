import pytest

from ttsim.config import numpy_or_jax as np
from ttsim.shared import join


@pytest.mark.parametrize(
    "foreign_key, primary_key, target, value_if_foreign_key_is_missing, expected",
    [
        (
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            4,
            np.array([1, 2, 3]),
        ),
        (
            np.array([3, 2, 1]),
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            4,
            np.array([3, 2, 1]),
        ),
        (
            np.array([1, 1, 1]),
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            4,
            np.array([1, 1, 1]),
        ),
        (
            np.array([-1]),
            np.array([1]),
            np.array([1]),
            4,
            np.array([4]),
        ),
    ],
)
def test_join(
    foreign_key: np.ndarray,
    primary_key: np.ndarray,
    target: np.ndarray,
    value_if_foreign_key_is_missing: int,
    expected: np.ndarray,
):
    assert np.array_equal(
        join(foreign_key, primary_key, target, value_if_foreign_key_is_missing),
        expected,
    )
