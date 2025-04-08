from __future__ import annotations

from typing import TYPE_CHECKING

import numpy
import pytest

from tests.ttsim.mettsim.config import METTSIM_RESSOURCE_DIR
from ttsim.function_types import _vectorize_func, policy_function
from ttsim.loader import (
    _convert_path_to_tree_path,
    _find_python_files_recursively,
    _load_module,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def test_load_path():
    assert _load_module(
        METTSIM_RESSOURCE_DIR / "payroll_tax" / "amount.py",
        METTSIM_RESSOURCE_DIR,
    )


def test_dont_load_init_py():
    """Don't load __init__.py files as sources for PolicyFunctions and
    AggregationSpecs."""
    all_files = _find_python_files_recursively(METTSIM_RESSOURCE_DIR)
    assert "__init__.py" not in [file.name for file in all_files]


def scalar_func(x: int) -> int:
    return x * 2


@policy_function(skip_vectorization=True)
def already_vectorized_func(x: numpy.ndarray) -> numpy.ndarray:
    return numpy.asarray([xi * 2 for xi in x])


@pytest.mark.parametrize(
    "vectorized_function",
    [
        _vectorize_func(scalar_func),
        already_vectorized_func,
    ],
)
def test_vectorize_func(vectorized_function: Callable) -> None:
    assert numpy.array_equal(
        vectorized_function(numpy.array([1, 2, 3])), numpy.array([2, 4, 6])
    )


@pytest.mark.parametrize(
    (
        "path",
        "root_path",
        "expected_tree_path",
    ),
    [
        (
            METTSIM_RESSOURCE_DIR
            / "payroll_tax"
            / "child_tax_credit"
            / "child_tax_credit.py",
            METTSIM_RESSOURCE_DIR,
            ("payroll_tax", "child_tax_credit"),
        ),
        (METTSIM_RESSOURCE_DIR / "foo" / "bar.py", METTSIM_RESSOURCE_DIR, ("foo",)),
        (METTSIM_RESSOURCE_DIR / "foo.py", METTSIM_RESSOURCE_DIR, tuple()),  # noqa: C408
    ],
)
def test_convert_path_to_tree_path(
    path: Path, root_path: Path, expected_tree_path: tuple[str, ...]
) -> None:
    assert (
        _convert_path_to_tree_path(path=path, root_path=root_path) == expected_tree_path
    )
