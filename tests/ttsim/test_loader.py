from __future__ import annotations

from typing import TYPE_CHECKING

import numpy
import pytest
from mettsim.config import RESOURCE_DIR

from ttsim import policy_function
from ttsim.loader import (
    _find_files_recursively,
    _load_module,
)
from ttsim.ttsim_objects import _vectorize_func

if TYPE_CHECKING:
    from collections.abc import Callable


def test_load_path():
    assert _load_module(
        path=RESOURCE_DIR / "payroll_tax" / "amount.py",
        root_path=RESOURCE_DIR,
    )


def test_dont_load_init_py():
    """Don't load __init__.py files as sources for PolicyFunctions and
    AggregationSpecs."""
    all_files = _find_files_recursively(root=RESOURCE_DIR, suffix=".py")
    assert "__init__.py" not in [file.name for file in all_files]


def scalar_func(x: int) -> int:
    if x < 0:
        return 0
    else:
        return x * 2


@policy_function(vectorization_strategy="not_required")
def already_vectorized_func(x: numpy.ndarray) -> numpy.ndarray:
    return numpy.where(x < 0, 0, x * 2)


@pytest.mark.parametrize(
    "vectorized_function",
    [
        _vectorize_func(scalar_func, vectorization_strategy="loop"),
        already_vectorized_func,
    ],
)
def test_vectorize_func(vectorized_function: Callable) -> None:
    assert numpy.array_equal(
        vectorized_function(numpy.array([-1, 0, 2, 3])), numpy.array([0, 0, 4, 6])
    )
