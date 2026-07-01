"""Unit tests for the Layer-2 unit-annotated input/output boundary (GEP 10).

Currency-free tests mostly use the always-available agnostic ``CURRENCY`` token
plus time/dimensionless units, so they do not depend on a concrete currency being
registered. The concrete currency conversion / substitution and the end-to-end
``main`` runs are exercised against mettsim's registered currencies in
``src_mettsim`` (where the example is fully unit-annotated).
"""

from __future__ import annotations

import numpy
import pint
import pytest

from ttsim.exceptions import UnitConsistencyError, UnitDefinitionError
from ttsim.interface_dag_elements.fail_if import (
    input_currency_is_not_concrete,
    input_levels_disagree_with_declaration,
)
from ttsim.interface_dag_elements.input_data import (
    flat_from_tree_with_unit_annotations,
    units_from_tree_with_unit_annotations,
)
from ttsim.interface_dag_elements.unit_checks import (
    fail_if_input_units_are_inconsistent,
    fail_if_not_all_leaves_are_unit_annotated_columns,
)
from ttsim.tt.units import (
    UNIT_REGISTRY,
    Unit,
    UnitAnnotatedColumn,
    output_unit_in_run_currency,
    register_grouping_levels,
)


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
        "wage_m": UnitAnnotatedColumn(
            values=numpy.array([1.0, 2.0]), unit=Unit.CURRENCY.PER_MONTH
        ),
        "nested": {
            "alter": UnitAnnotatedColumn(values=numpy.array([30, 40]), unit=Unit.YEARS)
        },
        "p_id": UnitAnnotatedColumn(
            values=numpy.array([0, 1]), unit=Unit.DIMENSIONLESS
        ),
    }
    units = units_from_tree_with_unit_annotations(tree_with_unit_annotations=tree)
    assert {k: str(v) for k, v in units.items()} == {
        "wage_m": "CURRENCY / month",
        "nested__alter": "delta_calendar_year",
        "p_id": "dimensionless",
    }


def test_fail_if_not_all_leaves_are_unit_annotated_columns_rejects_bare_leaf():
    flat: dict[tuple[str, ...], object] = {
        ("wage_m",): UnitAnnotatedColumn(
            values=numpy.array([1.0]), unit=Unit.CURRENCY.PER_MONTH
        ),
        ("alter",): numpy.array([30]),  # bare — not tagged
    }
    with pytest.raises(UnitConsistencyError, match="requires every leaf"):
        fail_if_not_all_leaves_are_unit_annotated_columns(flat=flat)


def test_flat_from_tree_with_unit_annotations_strips_to_bare_arrays():
    tree = {
        "wage_m": UnitAnnotatedColumn(
            values=numpy.array([1.0, 2.0]), unit=Unit.CURRENCY.PER_MONTH
        ),
        "p_id": UnitAnnotatedColumn(
            values=numpy.array([0, 1]), unit=Unit.DIMENSIONLESS
        ),
    }
    flat = flat_from_tree_with_unit_annotations(
        tree_with_unit_annotations=tree, currency=None
    )
    assert not isinstance(flat[("wage_m",)], pint.Quantity)
    assert not isinstance(flat[("p_id",)], pint.Quantity)
    # A dimensionless integer id must keep its integer dtype (no .to() upcast).
    assert flat[("p_id",)].dtype.kind == "i"
    assert list(flat[("wage_m",)]) == [1.0, 2.0]


def test_fail_if_not_all_leaves_are_unit_annotated_columns_passes_when_all_tagged():
    flat: dict[tuple[str, ...], object] = {
        ("wage_m",): UnitAnnotatedColumn(
            values=numpy.array([1.0]), unit=Unit.CURRENCY.PER_MONTH
        ),
        ("p_id",): UnitAnnotatedColumn(
            values=numpy.array([0]), unit=Unit.DIMENSIONLESS
        ),
    }
    fail_if_not_all_leaves_are_unit_annotated_columns(flat=flat)


def test_flat_from_tree_with_unit_annotations_fails_on_period_mismatch():
    # A `_m` column tagged without a period (a stock tag) is a boundary error.
    tree = {
        "wage_m": UnitAnnotatedColumn(values=numpy.array([1.0]), unit=Unit.CURRENCY)
    }
    with pytest.raises(UnitConsistencyError):
        flat_from_tree_with_unit_annotations(
            tree_with_unit_annotations=tree, currency=None
        )


def test_input_currency_non_currency_columns_pass():
    # Non-currency columns are never flagged (the agnostic-rejection path only fires
    # for currency-dimensioned tags). Rejection-when-a-currency-is-registered is
    # covered end-to-end in the mettsim suite (a base currency is registered there).
    input_currency_is_not_concrete(
        input_data__tree_with_unit_annotations={
            "alter": UnitAnnotatedColumn(values=numpy.array([30]), unit=Unit.YEARS),
            "p_id": UnitAnnotatedColumn(
                values=numpy.array([0]), unit=Unit.DIMENSIONLESS
            ),
        }
    )


def test_input_level_must_match_declared():
    # A `_hh` column declares the hh level; a person-leaf (level-less) tag contradicts
    # it. Compared against the declared resolved unit, not the name suffix directly.
    register_grouping_levels(["hh"])
    with pytest.raises(UnitConsistencyError, match="disagrees with the column"):
        input_levels_disagree_with_declaration(
            input_data__tree_with_unit_annotations={
                "miete_m_hh": UnitAnnotatedColumn(
                    values=numpy.array([1.0]), unit=Unit.CURRENCY.PER_MONTH
                )
            },
            unit_checks__resolved_units={
                "miete_m_hh": UNIT_REGISTRY.parse_units(
                    "CURRENCY / month / grouping_level_hh"
                )
            },
        )


def test_input_level_matching_declared_passes():
    register_grouping_levels(["hh"])
    input_levels_disagree_with_declaration(
        input_data__tree_with_unit_annotations={
            "miete_m_hh": UnitAnnotatedColumn(
                values=numpy.array([1.0]), unit=Unit.CURRENCY.PER_MONTH.PER_LEVEL("hh")
            ),
            "wage_m": UnitAnnotatedColumn(
                values=numpy.array([1.0]), unit=Unit.CURRENCY.PER_MONTH
            ),
        },
        unit_checks__resolved_units={
            "miete_m_hh": UNIT_REGISTRY.parse_units(
                "CURRENCY / month / grouping_level_hh"
            ),
            "wage_m": UNIT_REGISTRY.parse_units(
                "CURRENCY / month / grouping_level_person"
            ),
        },
    )


def test_input_share_at_group_suffix_stays_level_less():
    # A group suffix must not force a level onto a level-less quantity.
    register_grouping_levels(["hh"])
    input_levels_disagree_with_declaration(
        input_data__tree_with_unit_annotations={
            "rate_hh": UnitAnnotatedColumn(
                values=numpy.array([0.5]), unit=Unit.DIMENSIONLESS
            )
        },
        unit_checks__resolved_units={
            "rate_hh": UNIT_REGISTRY.parse_units("dimensionless")
        },
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


def test_input_units_compatible_but_differently_scaled_area_raises():
    # A HECTARES column tagged in m² shares the area dimension but is a
    # 10,000-fold level error — rejected, not silently mis-stripped (S4, GEP 10).
    resolved = {"land": UNIT_REGISTRY.parse_units("hectare")}
    with pytest.raises(UnitConsistencyError, match="not equivalent"):
        fail_if_input_units_are_inconsistent(
            input_units={"land": UNIT_REGISTRY.parse_units("meter ** 2")},
            resolved_units=resolved,
        )


def test_input_units_compatible_but_differently_scaled_time_raises():
    # A YEARS age tagged in months is a 12-fold level error (S4, GEP 10).
    resolved = {"alter": UNIT_REGISTRY.parse_units("year")}
    with pytest.raises(UnitConsistencyError, match="not equivalent"):
        fail_if_input_units_are_inconsistent(
            input_units={"alter": UNIT_REGISTRY.parse_units("month")},
            resolved_units=resolved,
        )


def test_input_units_period_mismatch_is_left_to_the_suffix_guard():
    # The flow period is screened against the name suffix by the dedicated period
    # guard, not here: once currency and period are factored out, a `_m` flow
    # tagged per year has an equivalent (bare) residual, so this check passes and
    # the period guard owns the mismatch (S4, GEP 10).
    resolved = {"wage_m": UNIT_REGISTRY.parse_units("CURRENCY / month")}
    fail_if_input_units_are_inconsistent(
        input_units={"wage_m": UNIT_REGISTRY.parse_units("CURRENCY / year")},
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
