from __future__ import annotations

from typing import TYPE_CHECKING

import numpy
import pytest

from ttsim.tt import join

if TYPE_CHECKING:
    from types import ModuleType

    from ttsim.typing.interface_dag_elements import IntColumn


@pytest.mark.parametrize(
    (
        "foreign_key",
        "primary_key",
        "target",
        "value_if_foreign_key_is_missing",
        "expected",
    ),
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
    foreign_key: IntColumn,
    primary_key: IntColumn,
    target: IntColumn,
    value_if_foreign_key_is_missing: int,
    expected: IntColumn,
    xnp: ModuleType,
):
    assert numpy.array_equal(
        join(
            foreign_key=xnp.asarray(foreign_key),
            primary_key=xnp.asarray(primary_key),
            target=xnp.asarray(target),
            value_if_foreign_key_is_missing=value_if_foreign_key_is_missing,
            xnp=xnp,
        ),
        expected,
    )
