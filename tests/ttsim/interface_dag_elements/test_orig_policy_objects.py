from __future__ import annotations

from pathlib import Path

import pytest

from ttsim.interface_dag_elements.orig_policy_objects import (
    _find_files_recursively,
    load_module,
)
from ttsim.tt_dag_elements.param_objects import (
    ConsecutiveInt1dLookupTableParam,
    ConsecutiveInt2dLookupTableParam,
    DictParam,
    PiecewisePolynomialParam,
    RawParam,
    ScalarParam,
)

METTSIM_ROOT = Path(__file__).parent.parent / "mettsim"


def test_load_path():
    assert load_module(
        path=METTSIM_ROOT / "payroll_tax" / "amount.py",
        root=METTSIM_ROOT,
    )


def test_dont_load_init_py():
    """Don't load __init__.py files as sources for PolicyFunctions and
    AggregationSpecs.
    """
    all_files = _find_files_recursively(root=METTSIM_ROOT, suffix=".py")
    assert "__init__.py" not in [file.name for file in all_files]


def test_scalar_param_requires_value():
    with pytest.raises(
        ValueError,
        match="'value' field must be specified for ScalarParam",
    ):
        ScalarParam()


def test_dict_param_requires_value():
    with pytest.raises(
        ValueError,
        match="'value' field must be specified for DictParam",
    ):
        DictParam()


def test_piecewise_polynomial_param_requires_value():
    with pytest.raises(
        ValueError,
        match="'value' field must be specified for PiecewisePolynomialParam",
    ):
        PiecewisePolynomialParam()


def test_consecutive_int_1d_lookup_table_param_requires_value():
    with pytest.raises(
        ValueError,
        match="'value' field must be specified for ConsecutiveInt1dLookupTableParam",
    ):
        ConsecutiveInt1dLookupTableParam()


def test_consecutive_int_2d_lookup_table_param_requires_value():
    with pytest.raises(
        ValueError,
        match="'value' field must be specified for ConsecutiveInt2dLookupTableParam",
    ):
        ConsecutiveInt2dLookupTableParam()


def test_raw_param_requires_value():
    with pytest.raises(
        ValueError,
        match="'value' field must be specified for RawParam",
    ):
        RawParam()
