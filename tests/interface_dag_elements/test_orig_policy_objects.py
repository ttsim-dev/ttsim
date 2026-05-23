from __future__ import annotations

import pytest
from mettsim import middle_earth

from ttsim.interface_dag_elements.orig_policy_objects import (
    _find_canonical_module_name,
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


def test_load_module_uses_canonical_name_when_part_of_python_package():
    """When the policy module lives under an `__init__.py` chain reaching
    `sys.path`, `load_module` returns the canonical-imported module so that
    objects defined in it carry an importable `__module__` (required for
    `cloudpickle.dumps` to round-trip).
    """
    module = load_module(
        path=middle_earth.ROOT_PATH / "payroll_tax" / "amount.py",
        root=middle_earth.ROOT_PATH,
    )
    assert module.__name__ == "mettsim.middle_earth.payroll_tax.amount"


def test_find_canonical_module_name_walks_init_chain():
    canonical = _find_canonical_module_name(
        middle_earth.ROOT_PATH / "payroll_tax" / "amount.py"
    )
    assert canonical == "mettsim.middle_earth.payroll_tax.amount"


def test_find_canonical_module_name_returns_none_for_bare_directory(tmp_path):
    (tmp_path / "foo.py").write_text("x = 1\n")
    assert _find_canonical_module_name(tmp_path / "foo.py") is None


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
