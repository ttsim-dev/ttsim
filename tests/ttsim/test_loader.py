from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

import numpy
import optree
import pytest
from mettsim.config import METTSIM_ROOT

from ttsim import policy_function
from ttsim.loader import (
    _find_files_recursively,
    _load_module,
    active_ttsim_objects_tree,
    orig_ttsim_objects_tree,
)
from ttsim.ttsim_objects import _vectorize_func

if TYPE_CHECKING:
    from collections.abc import Callable

    from ttsim.typing import NestedTTSIMObjectDict


def test_load_path():
    assert _load_module(
        path=METTSIM_ROOT / "payroll_tax" / "amount.py",
        root=METTSIM_ROOT,
    )


def test_dont_load_init_py():
    """Don't load __init__.py files as sources for PolicyFunctions and
    AggregationSpecs."""
    all_files = _find_files_recursively(root=METTSIM_ROOT, suffix=".py")
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


@pytest.mark.parametrize(
    "tree, last_day, function_name_last_day, function_name_next_day",
    [
        (
            {"housing_benefits": {"eligibility": {"requirement_fulfilled_fam": None}}},
            datetime.date(2019, 12, 31),
            "requirement_fulfilled_fam_not_considering_children",
            "requirement_fulfilled_fam_considering_children",
        ),
    ],
)
def test_load_functions_tree_for_date(
    tree: NestedTTSIMObjectDict,
    last_day: datetime.date,
    function_name_last_day: str,
    function_name_next_day: str,
):
    _orig_ttsim_objects_tree = orig_ttsim_objects_tree(root=METTSIM_ROOT)
    functions_last_day = active_ttsim_objects_tree(
        orig_ttsim_objects_tree=_orig_ttsim_objects_tree, date=last_day
    )
    functions_next_day = active_ttsim_objects_tree(
        orig_ttsim_objects_tree=_orig_ttsim_objects_tree,
        date=last_day + datetime.timedelta(days=1),
    )

    accessor = optree.tree_accessors(tree, none_is_leaf=True)[0]

    assert accessor(functions_last_day).__name__ == function_name_last_day
    assert accessor(functions_next_day).__name__ == function_name_next_day
