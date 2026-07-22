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

from ttsim.exceptions import UnitConsistencyError
from ttsim.interface_dag_elements.fail_if import (
    input_currency_is_not_concrete,
)
from ttsim.interface_dag_elements.input_data import (
    flat_from_tree_with_unit_annotations,
    units_from_tree_with_unit_annotations,
)
from ttsim.interface_dag_elements.unit_checks import (
    fail_if_input_units_are_inconsistent,
    fail_if_not_all_leaves_are_unit_annotated_columns,
)
from ttsim.tt.currencies import UnitSystem
from ttsim.tt.grouping_levels import register_grouping_levels
from ttsim.tt.units import (
    CompositeUnit,
    TTSIMUnit,
    UnitAnnotatedColumn,
    output_unit_in_data_currency,
    resolve_compositional_column_unit,
    resolve_compositional_unit,
)

# A representative system whose registry these boundary tests resolve against.
# They pass the agnostic ``CURRENCY`` as the data currency explicitly, so the
# base currency is never exercised; the grouping level ``hh`` is.
SYSTEM = UnitSystem(
    base_currency="EUR",
    statutory_currencies={"0001-01-01": "EUR"},
    grouping_levels=["hh"],
)
REGISTRY = SYSTEM.registry


def _resolved(unit: CompositeUnit) -> pint.Unit:
    """Resolve a CompositeUnit to its pint unit."""
    return resolve_compositional_unit(unit=unit, registry=REGISTRY)


def _column(
    unit: CompositeUnit,
    *,
    time_unit_id: str | None = None,
    grouping_level: str | None = None,
) -> pint.Unit:
    """A column's resolved unit exactly as the DAG builds it (bare when unsuffixed)."""
    return resolve_compositional_column_unit(
        unit=unit,
        time_unit_id=time_unit_id,
        grouping_level=grouping_level,
        where="test",
        registry=REGISTRY,
    )


def test_output_unit_in_data_currency_leaves_non_currency_units_untouched():
    for composite in (TTSIMUnit.YEARS, TTSIMUnit.HECTARE, TTSIMUnit.DIMENSIONLESS):
        unit = _resolved(composite)
        assert (
            output_unit_in_data_currency(
                units=unit, data_currency="CURRENCY", registry=REGISTRY
            )
            == unit
        )


def test_units_from_tree_with_unit_annotations_extracts_each_tag():
    tree = {
        "wage_m": UnitAnnotatedColumn(
            values=numpy.array([1.0, 2.0]), unit=TTSIMUnit.CURRENCY.PER_MONTH
        ),
        "nested": {
            "alter": UnitAnnotatedColumn(
                values=numpy.array([30, 40]), unit=TTSIMUnit.YEARS
            )
        },
        "p_id": UnitAnnotatedColumn(
            values=numpy.array([0, 1]), unit=TTSIMUnit.DIMENSIONLESS
        ),
    }
    units = units_from_tree_with_unit_annotations(
        tree_with_unit_annotations=tree, unit_system=SYSTEM
    )
    assert {k: str(v) for k, v in units.items()} == {
        "wage_m": "CURRENCY / month",
        "nested__alter": "delta_calendar_year",
        "p_id": "dimensionless",
    }


def test_fail_if_not_all_leaves_are_unit_annotated_columns_rejects_bare_leaf():
    flat: dict[tuple[str, ...], object] = {
        ("wage_m",): UnitAnnotatedColumn(
            values=numpy.array([1.0]), unit=TTSIMUnit.CURRENCY.PER_MONTH
        ),
        ("alter",): numpy.array([30]),  # bare — not tagged
    }
    with pytest.raises(UnitConsistencyError, match="requires every leaf"):
        fail_if_not_all_leaves_are_unit_annotated_columns(flat=flat)


def test_flat_from_tree_with_unit_annotations_strips_to_bare_arrays():
    tree = {
        "wage_m": UnitAnnotatedColumn(
            values=numpy.array([1.0, 2.0]), unit=TTSIMUnit.CURRENCY.PER_MONTH
        ),
        "p_id": UnitAnnotatedColumn(
            values=numpy.array([0, 1]), unit=TTSIMUnit.DIMENSIONLESS
        ),
    }
    flat = flat_from_tree_with_unit_annotations(
        tree_with_unit_annotations=tree, data_currency="CURRENCY", unit_system=SYSTEM
    )
    assert not isinstance(flat[("wage_m",)], pint.Quantity)
    assert not isinstance(flat[("p_id",)], pint.Quantity)
    # A dimensionless integer id must keep its integer dtype (no .to() upcast).
    assert flat[("p_id",)].dtype.kind == "i"
    assert list(flat[("wage_m",)]) == [1.0, 2.0]


def test_fail_if_not_all_leaves_are_unit_annotated_columns_passes_when_all_tagged():
    flat: dict[tuple[str, ...], object] = {
        ("wage_m",): UnitAnnotatedColumn(
            values=numpy.array([1.0]), unit=TTSIMUnit.CURRENCY.PER_MONTH
        ),
        ("p_id",): UnitAnnotatedColumn(
            values=numpy.array([0]), unit=TTSIMUnit.DIMENSIONLESS
        ),
    }
    fail_if_not_all_leaves_are_unit_annotated_columns(flat=flat)


def test_flat_from_tree_with_unit_annotations_fails_on_period_mismatch():
    # A `_m` column tagged without a period (a stock tag) is a boundary error.
    tree = {
        "wage_m": UnitAnnotatedColumn(
            values=numpy.array([1.0]), unit=TTSIMUnit.CURRENCY
        )
    }
    with pytest.raises(UnitConsistencyError):
        flat_from_tree_with_unit_annotations(
            tree_with_unit_annotations=tree,
            data_currency="CURRENCY",
            unit_system=SYSTEM,
        )


def test_input_currency_non_currency_columns_pass():
    # Non-currency columns are never flagged (the agnostic-rejection path only fires
    # for currency-dimensioned tags). Rejection-when-a-currency-is-registered is
    # covered end-to-end in the mettsim suite (a base currency is registered there).
    input_currency_is_not_concrete(
        input_data__tree_with_unit_annotations={
            "alter": UnitAnnotatedColumn(
                values=numpy.array([30]), unit=TTSIMUnit.YEARS
            ),
            "p_id": UnitAnnotatedColumn(
                values=numpy.array([0]), unit=TTSIMUnit.DIMENSIONLESS
            ),
        }
    )


def test_input_level_must_match_declared():
    # A `_hh` column declares the hh level; a bare tag contradicts it. Compared
    # against the declared resolved unit, not the name suffix directly.
    register_grouping_levels(["hh"], registry=REGISTRY)
    with pytest.raises(UnitConsistencyError, match="level"):
        fail_if_input_units_are_inconsistent(
            input_units={
                "miete_m_hh": _column(TTSIMUnit.CURRENCY.PER_MONTH, time_unit_id="m")
            },
            resolved_units={
                "miete_m_hh": _column(
                    TTSIMUnit.CURRENCY.PER_MONTH.PER_HH,
                    time_unit_id="m",
                    grouping_level="hh",
                )
            },
            unit_system=SYSTEM,
        )


def test_input_level_matching_declared_passes():
    register_grouping_levels(["hh"], registry=REGISTRY)
    miete = _column(
        TTSIMUnit.CURRENCY.PER_MONTH.PER_HH, time_unit_id="m", grouping_level="hh"
    )
    wage = _column(TTSIMUnit.CURRENCY.PER_MONTH, time_unit_id="m")
    fail_if_input_units_are_inconsistent(
        input_units={"miete_m_hh": miete, "wage_m": wage},
        resolved_units={"miete_m_hh": miete, "wage_m": wage},
        unit_system=SYSTEM,
    )


def test_bare_tag_matches_a_normalized_per_person_declaration():
    # A `_PER_PERSON` declaration normalizes to bare, so a bare tag matches it: a
    # per-person amount and a level-neutral amount are the same bare unit (GEP 10).
    register_grouping_levels(["hh"], registry=REGISTRY)
    betrag = _column(TTSIMUnit.CURRENCY.PER_MONTH.PER_PERSON, time_unit_id="m")
    fail_if_input_units_are_inconsistent(
        input_units={"betrag_m": _resolved(TTSIMUnit.CURRENCY.PER_MONTH)},
        resolved_units={"betrag_m": betrag},
        unit_system=SYSTEM,
    )


def test_group_level_tag_does_not_match_a_bare_declaration():
    # A group-level tag is distinct from a bare declaration (GEP 10): the level is
    # declared, not read off the suffix, so the mismatch is caught via the tokens.
    register_grouping_levels(["hh"], registry=REGISTRY)
    with pytest.raises(UnitConsistencyError, match="level"):
        fail_if_input_units_are_inconsistent(
            input_units={"betrag_m": _resolved(TTSIMUnit.CURRENCY.PER_MONTH.PER_HH)},
            resolved_units={"betrag_m": _resolved(TTSIMUnit.CURRENCY.PER_MONTH)},
            unit_system=SYSTEM,
            input_unit_tokens={"betrag_m": TTSIMUnit.CURRENCY.PER_MONTH.PER_HH},
            declared_unit_tokens={"betrag_m": TTSIMUnit.CURRENCY.PER_MONTH},
        )


def test_input_share_at_group_suffix_stays_level_less():
    # A group suffix must not force a level onto a level-less quantity.
    register_grouping_levels(["hh"], registry=REGISTRY)
    fail_if_input_units_are_inconsistent(
        input_units={"rate_hh": _resolved(TTSIMUnit.DIMENSIONLESS)},
        resolved_units={"rate_hh": _resolved(TTSIMUnit.DIMENSIONLESS)},
        unit_system=SYSTEM,
    )


def test_input_units_consistent_passes():
    resolved = {
        "wage_m": _resolved(TTSIMUnit.CURRENCY.PER_MONTH),
        "alter": _resolved(TTSIMUnit.YEARS),
    }
    fail_if_input_units_are_inconsistent(
        input_units={"wage_m": _resolved(TTSIMUnit.CURRENCY.PER_MONTH)},
        resolved_units=resolved,
        unit_system=SYSTEM,
    )


def test_input_units_dimension_mismatch_raises():
    resolved = {"alter": _resolved(TTSIMUnit.YEARS)}
    with pytest.raises(UnitConsistencyError, match="inconsistent with the DAG"):
        fail_if_input_units_are_inconsistent(
            input_units={"alter": _resolved(TTSIMUnit.CURRENCY)},
            resolved_units=resolved,
            unit_system=SYSTEM,
        )


def test_input_units_compatible_but_differently_scaled_area_raises():
    # A HECTARES column tagged in m² shares the area dimension but is a
    # 10,000-fold level error — rejected, not silently mis-stripped (S4, GEP 10).
    resolved = {"land": _resolved(TTSIMUnit.HECTARE)}
    with pytest.raises(UnitConsistencyError, match="not equivalent"):
        fail_if_input_units_are_inconsistent(
            input_units={"land": _resolved(TTSIMUnit.SQUARE_METER)},
            resolved_units=resolved,
            unit_system=SYSTEM,
        )


def test_input_units_compatible_but_differently_scaled_time_raises():
    # A YEARS age tagged in months is a 12-fold level error (S4, GEP 10).
    resolved = {"alter": _resolved(TTSIMUnit.YEARS)}
    with pytest.raises(UnitConsistencyError, match="not equivalent"):
        fail_if_input_units_are_inconsistent(
            input_units={"alter": _resolved(TTSIMUnit.MONTHS)},
            resolved_units=resolved,
            unit_system=SYSTEM,
        )


def test_input_units_period_mismatch_is_left_to_the_suffix_guard():
    # The flow period is screened against the name suffix by the dedicated period
    # guard, not here: once currency and period are factored out, a `_m` flow
    # tagged per year has an equivalent (bare) residual, so this check passes and
    # the period guard owns the mismatch (S4, GEP 10).
    resolved = {"wage_m": _resolved(TTSIMUnit.CURRENCY.PER_MONTH)}
    fail_if_input_units_are_inconsistent(
        input_units={"wage_m": _resolved(TTSIMUnit.CURRENCY.PER_YEAR)},
        resolved_units=resolved,
        unit_system=SYSTEM,
    )


def test_input_units_currency_tag_on_non_currency_column_raises():
    # A currency tag on a non-currency (dimensionless) column: dividing currency
    # out of both sides used to leave equal residuals and pass, while the boundary
    # would silently rescale the data by the currency factor on a non-base run.
    # The currency-presence mismatch is now rejected (defect #3, GEP 10).
    resolved = {"flag": _resolved(TTSIMUnit.DIMENSIONLESS)}
    with pytest.raises(UnitConsistencyError, match="one carries a currency"):
        fail_if_input_units_are_inconsistent(
            input_units={"flag": _resolved(TTSIMUnit.CURRENCY)},
            resolved_units=resolved,
            unit_system=SYSTEM,
        )


def test_input_units_non_currency_tag_on_currency_column_raises():
    # The converse: a dimensionless tag on a declared-currency column would skip
    # the currency conversion the column needs (defect #3, GEP 10).
    resolved = {"wage": _resolved(TTSIMUnit.CURRENCY)}
    with pytest.raises(UnitConsistencyError, match="one carries a currency"):
        fail_if_input_units_are_inconsistent(
            input_units={"wage": _resolved(TTSIMUnit.DIMENSIONLESS)},
            resolved_units=resolved,
            unit_system=SYSTEM,
        )


def test_input_units_skips_columns_without_a_scalar_declared_unit():
    # A dict-parameter style entry (nested dict, not a single pint.Unit) and an
    # unknown column are both skipped rather than raising.
    resolved = {"some_dict_param": {"a": _resolved(TTSIMUnit.CURRENCY)}}
    fail_if_input_units_are_inconsistent(
        input_units={
            "some_dict_param": _resolved(TTSIMUnit.CURRENCY),
            "not_in_env": _resolved(TTSIMUnit.YEARS),
        },
        resolved_units=resolved,
        unit_system=SYSTEM,
    )
