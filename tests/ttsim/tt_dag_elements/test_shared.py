from __future__ import annotations

import numpy
import pytest

from ttsim.tt_dag_elements import join


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
def test_join(
    foreign_key: numpy.ndarray,
    primary_key: numpy.ndarray,
    target: numpy.ndarray,
    value_if_foreign_key_is_missing: int,
    expected: numpy.ndarray,
):
    assert numpy.array_equal(
        join(foreign_key, primary_key, target, value_if_foreign_key_is_missing),
        expected,
    )
