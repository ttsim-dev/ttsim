"""Unit tests for the Layer-2 unit-annotated input/output boundary (GEP 10).

Currency-free tests only — they use the always-available agnostic ``CURRENCY``
token plus time/dimensionless units, so they do not depend on a concrete
currency being registered. The concrete currency conversion / substitution and
the end-to-end ``main`` runs are exercised against mettsim's registered
currencies in ``src_mettsim`` (where the example is fully unit-annotated).
"""

from __future__ import annotations

import numpy
import pint
import pytest

from ttsim.exceptions import UnitConsistencyError, UnitDefinitionError
from ttsim.interface_dag_elements.input_data import (
    flat_from_tree_with_unit_annotations,
    units_from_tree_with_unit_annotations,
)
from ttsim.interface_dag_elements.unit_checks import (
    fail_if_input_units_are_inconsistent,
    fail_if_not_all_leaves_are_quantities,
)
from ttsim.tt.units import UNIT_REGISTRY, output_unit_in_run_currency


def test_output_unit_in_run_currency_without_run_currency_raises():
    unit = UNIT_REGISTRY.parse_units("CURRENCY / month")
    with pytest.raises(UnitDefinitionError, match="without a run currency"):
        output_unit_in_run_currency(units=unit, run_currency=None)


def test_output_unit_in_run_currency_leaves_non_currency_units_untouched():
    for spelling in ("year", "hectare", "dimensionless"):
        unit = UNIT_REGISTRY.parse_units(spelling)
        assert output_unit_in_run_currency(units=unit, run_currency="CURRENCY") == unit


def test_units_from_tree_with_unit_annotations_extracts_each_tag():
    tree = {
        "wage_m": UNIT_REGISTRY.Quantity(numpy.array([1.0, 2.0]), "CURRENCY / month"),
        "nested": {"alter": UNIT_REGISTRY.Quantity(numpy.array([30, 40]), "year")},
        "p_id": UNIT_REGISTRY.Quantity(numpy.array([0, 1]), "dimensionless"),
    }
    units = units_from_tree_with_unit_annotations(tree_with_unit_annotations=tree)
    assert {k: str(v) for k, v in units.items()} == {
        "wage_m": "CURRENCY / month",
        "nested__alter": "year",
        "p_id": "dimensionless",
    }


def test_fail_if_not_all_leaves_are_quantities_rejects_bare_leaf():
    flat: dict[tuple[str, ...], object] = {
        ("wage_m",): UNIT_REGISTRY.Quantity(numpy.array([1.0]), "CURRENCY / month"),
        ("alter",): numpy.array([30]),  # bare — not tagged
    }
    with pytest.raises(UnitConsistencyError, match="requires every leaf"):
        fail_if_not_all_leaves_are_quantities(flat=flat)


def test_flat_from_tree_with_unit_annotations_strips_to_bare_arrays():
    tree = {
        "wage_m": UNIT_REGISTRY.Quantity(numpy.array([1.0, 2.0]), "CURRENCY / month"),
        "p_id": UNIT_REGISTRY.Quantity(numpy.array([0, 1]), "dimensionless"),
    }
    flat = flat_from_tree_with_unit_annotations(
        tree_with_unit_annotations=tree, currency=None
    )
    assert not isinstance(flat[("wage_m",)], pint.Quantity)
    assert not isinstance(flat[("p_id",)], pint.Quantity)
    # A dimensionless integer id must keep its integer dtype (no .to() upcast).
    assert flat[("p_id",)].dtype.kind == "i"
    assert list(flat[("wage_m",)]) == [1.0, 2.0]


def test_fail_if_not_all_leaves_are_quantities_passes_when_all_tagged():
    flat: dict[tuple[str, ...], object] = {
        ("wage_m",): UNIT_REGISTRY.Quantity(numpy.array([1.0]), "CURRENCY / month"),
        ("p_id",): UNIT_REGISTRY.Quantity(numpy.array([0]), "dimensionless"),
    }
    fail_if_not_all_leaves_are_quantities(flat=flat)


def test_flat_from_tree_with_unit_annotations_fails_on_period_mismatch():
    # A `_m` column tagged without a period (a stock tag) is a boundary error.
    tree = {"wage_m": UNIT_REGISTRY.Quantity(numpy.array([1.0]), "CURRENCY")}
    with pytest.raises(UnitConsistencyError):
        flat_from_tree_with_unit_annotations(
            tree_with_unit_annotations=tree, currency=None
        )


def test_input_units_consistent_passes():
    resolved = {
        "wage_m": UNIT_REGISTRY.parse_units("CURRENCY / month"),
        "alter": UNIT_REGISTRY.parse_units("year"),
    }
    fail_if_input_units_are_inconsistent(
        input_units={"wage_m": UNIT_REGISTRY.parse_units("CURRENCY / month")},
        resolved_units=resolved,
    )


def test_input_units_dimension_mismatch_raises():
    resolved = {"alter": UNIT_REGISTRY.parse_units("year")}
    with pytest.raises(UnitConsistencyError, match="inconsistent with the DAG"):
        fail_if_input_units_are_inconsistent(
            input_units={"alter": UNIT_REGISTRY.parse_units("CURRENCY")},
            resolved_units=resolved,
        )


def test_input_units_skips_columns_without_a_scalar_declared_unit():
    # A dict-parameter style entry (nested dict, not a single pint.Unit) and an
    # unknown column are both skipped rather than raising.
    resolved = {"some_dict_param": {"a": UNIT_REGISTRY.parse_units("CURRENCY")}}
    fail_if_input_units_are_inconsistent(
        input_units={
            "some_dict_param": UNIT_REGISTRY.parse_units("CURRENCY"),
            "not_in_env": UNIT_REGISTRY.parse_units("year"),
        },
        resolved_units=resolved,
    )
