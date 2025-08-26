from __future__ import annotations

import pytest
from mettsim import middle_earth

from ttsim.interface_dag_elements.orig_policy_objects import (
    _find_files_recursively,
    load_module,
)
from ttsim.tt.param_objects import (
    ConsecutiveIntLookupTableParam,
    DictParam,
    PiecewisePolynomialParam,
    RawParam,
    ScalarParam,
)


def test_load_path():
    assert load_module(
        path=middle_earth.ROOT_PATH / "payroll_tax" / "amount.py",
        root=middle_earth.ROOT_PATH,
    )


def test_dont_load_init_py():
    """Don't load __init__.py files as sources for PolicyFunctions and
    AggregationSpecs.
    """
    all_files = _find_files_recursively(root=middle_earth.ROOT_PATH, suffix=".py")
    assert "__init__.py" not in [file.name for file in all_files]


@pytest.mark.parametrize(
    "param_object",
    [
        ScalarParam,
        DictParam,
        PiecewisePolynomialParam,
        ConsecutiveIntLookupTableParam,
        RawParam,
    ],
)
def test_param_object_requires_value(param_object):
    with pytest.raises(
        ValueError,
        match="'value' field must be specified for any type of 'ParamObject'",
    ):
        param_object()
