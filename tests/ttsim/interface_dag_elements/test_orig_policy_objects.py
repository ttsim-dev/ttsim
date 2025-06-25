from __future__ import annotations

from pathlib import Path

from ttsim.interface_dag_elements.orig_policy_objects import (
    _find_files_recursively,
    load_module,
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
