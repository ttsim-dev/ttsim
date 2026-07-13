"""Tests for the environment-level conservative unit checks (GEP 10, #121)."""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any

import mettsim.middle_earth  # noqa: F401
import numpy
import pint
import pytest

from ttsim.exceptions_and_warnings import (
    AggregationDefinitionError,
    UnitConsistencyError,
    UnitDefinitionError,
)
from ttsim.interface_dag_elements.automatically_added_functions import (
    create_agg_by_group_functions,
)
from ttsim.interface_dag_elements.unit_checks import (
    FRAMEWORK_DATE_NODE_UNITS,
    fail_if_environment_units_are_inconsistent,
    fail_if_environment_units_are_missing,
    node_is_boolean,
    resolve_environment_units,
    resolved_units,
)
from ttsim.tt import (
    UNSET_UNIT,
    AggType,
    FKType,
    RoundingSpec,
    Unit,
    agg_by_group_function,
    cast_unit,
    group_creation_function,
    join,
    param_function,
    piecewise_polynomial,
    policy_function,
    policy_input,
)
from ttsim.tt.param_objects import (
    ConsecutiveIntLookupTableParam,
    ConsecutiveIntLookupTableParamValue,
    DictParam,
    PiecewisePolynomialParam,
    PiecewisePolynomialParamValue,
    RawParam,
    ScalarParam,
)
from ttsim.tt.units import (
    PERSON_LEVEL,
    coerce_to_composite_unit,
    divide_by_grouping_level,
    parse_unit,
    units_are_equivalent,
)
from ttsim.typing import BoolColumn, FloatColumn, IntColumn, RawParamValue
from ttsim.unit_converters import m_to_y, per_m_to_per_y, y_to_m

if TYPE_CHECKING:
    from types import ModuleType

GROUPING_LEVELS = ("fam", "kin")

_START = datetime.date(2020, 1, 1)
_END = datetime.date(2030, 12, 31)

# Parameters must pin down the concrete currency their numbers are written in
# (GEP 10); these are mettsim's concrete (castar) compositional spellings.
CASTAR_PER_YEAR = coerce_to_composite_unit(value="CASTAR_PER_YEAR", where="test setup")
CASTAR_PER_MONTH = coerce_to_composite_unit(
    value="CASTAR_PER_MONTH", where="test setup"
)
CASTAR = coerce_to_composite_unit(value="CASTAR", where="test setup")


# ----------------------------------------------------------------------------
# Fixture objects
# ----------------------------------------------------------------------------


@policy_input(unit=Unit.DIMENSIONLESS)
def p_id() -> int:
    """Identifier; a dimensionless quantity (GEP 10)."""


@policy_input(foreign_key_type=FKType.MAY_POINT_TO_SELF, unit=Unit.DIMENSIONLESS)
def p_id_recipient() -> int:
    """Person pointer; a dimensionless quantity (GEP 10)."""


@policy_input(unit=Unit.CURRENCY)
def wealth() -> float:
    """A stock of currency."""


@policy_input(unit=Unit.DIMENSIONLESS)
def is_exempt() -> bool:
    """Boolean input; a dimensionless quantity (GEP 10)."""


@policy_input(unit=UNSET_UNIT)
def unannotated_income_y() -> float:
    """Carries the UNSET sentinel; the missing-units check must report it."""


@group_creation_function(unit=Unit.DIMENSIONLESS)
def fam_id(p_id: IntColumn, xnp: object) -> IntColumn:  # noqa: ARG001
    """Group creation (a dimensionless id)."""
    return p_id


def _scalar_unit(resolved: dict[str, Any], qname: str) -> pint.Unit:
    unit = resolved[qname]
    assert isinstance(unit, pint.Unit)
    return unit


def _unit_tree(resolved: dict[str, Any], qname: str) -> dict[str | int, Any]:
    unit = resolved[qname]
    assert isinstance(unit, dict)
    return unit


def make_flow_rate() -> ScalarParam:
    # A wealth-tax rate is a share per year: `DIMENSIONLESS_PER_YEAR`, used at the
    # `tax_rate_y` name (the spelled period agrees with the suffix, GEP 10).
    return ScalarParam(
        value=0.01,
        unit=Unit.DIMENSIONLESS.PER_YEAR,
        start_date=_START,
        end_date=_END,
    )


def make_dimensionless_rate() -> ScalarParam:
    # The bug the consistency check must catch: a plain dimensionless share with
    # no period, so `wealth * rate` is a stock where a flow node expects a flow.
    return ScalarParam(
        value=0.01,
        unit=Unit.DIMENSIONLESS,
        start_date=_START,
        end_date=_END,
    )


@policy_input(unit=Unit.CURRENCY)
def wealth_threshold() -> float:
    """A wealth threshold; a stock of currency. Comparing a quantity against a
    bare inline literal is rejected (GEP 10), so the bound is a named producer
    rather than a magic number."""


@policy_function(unit=Unit.CURRENCY.PER_YEAR)
def amount_y(wealth: float, tax_rate_y: float, is_exempt: bool) -> float:
    """The wealth-tax pattern: stock times a per-year rate, guarded by an
    exemption. ``tax_rate_y`` is a share per year, so the product is a flow."""
    if is_exempt:
        return 0.0
    return wealth * tax_rate_y


@policy_function(unit=Unit.CURRENCY.PER_YEAR)
def amount_buggy_y(wealth: float, tax_rate: float, is_exempt: bool) -> float:
    """The bug: ``tax_rate`` is a plain dimensionless share, so ``wealth *
    tax_rate`` is a stock, not the declared yearly flow."""
    if is_exempt:
        return 0.0
    return wealth * tax_rate


@policy_input(unit=Unit.CURRENCY.PER_MONTH)
def income_m() -> float:
    """A monthly flow of currency (CURRENCY / month)."""


@policy_input(unit=Unit.CURRENCY.PER_MONTH)
def other_income_m() -> float:
    """A second monthly flow of currency (CURRENCY / month)."""


@policy_input(unit=Unit.CURRENCY.PER_YEAR)
def bonus_y() -> float:
    """A yearly flow of currency (CURRENCY / year)."""


@policy_input(unit=Unit.CALENDAR_YEAR)
def geburtsjahr() -> int:
    """A birth year: a point on the calendar, not a duration (GEP 10)."""


@policy_input(unit=Unit.YEARS)
def statutory_age() -> int:
    """A duration in years (an age threshold)."""


@policy_input(unit=Unit.DIMENSIONLESS)
def geburtsmonat() -> int:
    """A month-of-year (1-12): a cyclic ordinal, not a calendar point (GEP 10)."""


@policy_input(unit=Unit.MONTHS)
def months_paid() -> int:
    """A duration in months."""


# ----------------------------------------------------------------------------
# Mandatory units, no exemptions: identifiers and booleans declare
# DIMENSIONLESS; group-creation group ids are auto-assigned DIMENSIONLESS;
# framework date nodes get their unit from the framework.
# ----------------------------------------------------------------------------


def test_boolean_nodes_are_detected():
    assert node_is_boolean(qname="is_exempt", obj=is_exempt)
    assert not node_is_boolean(qname="wealth", obj=wealth)


def test_missing_check_passes_for_declared_and_group_creation_nodes():
    fail_if_environment_units_are_missing(
        env={
            # Identifiers and the boolean declare DIMENSIONLESS (GEP 10);
            # the group-creation `fam_id` is auto-assigned DIMENSIONLESS.
            "p_id": p_id,
            "p_id_recipient": p_id_recipient,
            "fam_id": fam_id,
            "is_exempt": is_exempt,
            "wealth": wealth,
            "tax_rate_y": make_flow_rate(),
            "amount_y": amount_y,
        },
        grouping_levels=GROUPING_LEVELS,
    )


def test_missing_check_reports_unannotated_node():
    with pytest.raises(UnitDefinitionError, match="unannotated_income_y"):
        fail_if_environment_units_are_missing(
            env={"unannotated_income_y": unannotated_income_y},
            grouping_levels=GROUPING_LEVELS,
        )


def test_missing_check_reports_unannotated_identifier_and_boolean():
    # Identifiers and booleans are no longer structurally exempt (GEP 10): an
    # undeclared one is reported just like any other node.
    @policy_input(unit=UNSET_UNIT)
    def some_id() -> int:
        """An identifier carrying the UNSET sentinel."""

    @policy_input(unit=UNSET_UNIT)
    def some_flag() -> bool:
        """A boolean carrying the UNSET sentinel."""

    with pytest.raises(UnitDefinitionError, match="some_id"):
        fail_if_environment_units_are_missing(
            env={"some_id": some_id, "some_flag": some_flag},
            grouping_levels=GROUPING_LEVELS,
        )


# ----------------------------------------------------------------------------
# Currency-denominated rounding specs: mandatory on a currency-valued function,
# forbidden elsewhere, composite must equal the function's declared unit with
# the agnostic base swapped for the concrete currency (GEP 10)
# ----------------------------------------------------------------------------


def make_rounded_amount_y(rounding_spec: RoundingSpec):
    @policy_function(rounding_spec=rounding_spec, unit=Unit.CURRENCY.PER_YEAR)
    def rounded_amount_y(bonus_y: float) -> float:
        return bonus_y

    return rounded_amount_y


def test_missing_check_reports_currency_rounding_spec_without_unit():
    with pytest.raises(
        UnitDefinitionError, match=r"rounded_amount_y \(rounding_spec\)"
    ):
        fail_if_environment_units_are_missing(
            env={
                "rounded_amount_y": make_rounded_amount_y(
                    RoundingSpec(base=1, direction="down")
                ),
                "bonus_y": bonus_y,
            },
            grouping_levels=GROUPING_LEVELS,
        )


def test_missing_check_passes_for_currency_rounding_spec_with_unit():
    fail_if_environment_units_are_missing(
        env={
            "rounded_amount_y": make_rounded_amount_y(
                RoundingSpec(base=1, direction="down", unit=CASTAR_PER_YEAR)
            ),
            "bonus_y": bonus_y,
        },
        grouping_levels=GROUPING_LEVELS,
    )


def test_missing_check_passes_for_non_currency_rounding_spec_without_unit():
    @policy_function(
        rounding_spec=RoundingSpec(base=1, direction="down"), unit=Unit.YEARS
    )
    def rounded_age(statutory_age: int) -> int:
        return statutory_age

    fail_if_environment_units_are_missing(
        env={"rounded_age": rounded_age, "statutory_age": statutory_age},
        grouping_levels=GROUPING_LEVELS,
    )


def test_inconsistency_check_passes_for_matching_rounding_spec_unit():
    fail_if_environment_units_are_inconsistent(
        env={
            "rounded_amount_y": make_rounded_amount_y(
                RoundingSpec(base=1, direction="down", unit=CASTAR_PER_YEAR)
            ),
            "bonus_y": bonus_y,
        },
        grouping_levels=GROUPING_LEVELS,
    )


def test_inconsistency_check_reports_rounding_spec_composite_mismatch():
    with pytest.raises(UnitConsistencyError, match="same flow period"):
        fail_if_environment_units_are_inconsistent(
            env={
                "rounded_amount_y": make_rounded_amount_y(
                    RoundingSpec(base=1, direction="down", unit=CASTAR_PER_MONTH)
                ),
                "bonus_y": bonus_y,
            },
            grouping_levels=GROUPING_LEVELS,
        )


def test_inconsistency_check_reports_agnostic_rounding_spec_unit():
    with pytest.raises(UnitConsistencyError, match="concrete"):
        fail_if_environment_units_are_inconsistent(
            env={
                "rounded_amount_y": make_rounded_amount_y(
                    RoundingSpec(base=1, direction="down", unit=Unit.CURRENCY.PER_YEAR)
                ),
                "bonus_y": bonus_y,
            },
            grouping_levels=GROUPING_LEVELS,
        )


def test_inconsistency_check_reports_non_currency_rounding_spec_unit():
    with pytest.raises(UnitConsistencyError, match="registered currency"):
        fail_if_environment_units_are_inconsistent(
            env={
                "rounded_amount_y": make_rounded_amount_y(
                    RoundingSpec(base=1, direction="down", unit=Unit.YEARS)
                ),
                "bonus_y": bonus_y,
            },
            grouping_levels=GROUPING_LEVELS,
        )


def test_inconsistency_check_reports_rounding_spec_unit_on_non_currency_function():
    @policy_function(
        rounding_spec=RoundingSpec(base=1, direction="down", unit=CASTAR_PER_YEAR),
        unit=Unit.YEARS,
    )
    def rounded_age(statutory_age: int) -> int:
        return statutory_age

    with pytest.raises(UnitConsistencyError, match="nothing to convert"):
        fail_if_environment_units_are_inconsistent(
            env={"rounded_age": rounded_age, "statutory_age": statutory_age},
            grouping_levels=GROUPING_LEVELS,
        )


# ----------------------------------------------------------------------------
# Unit resolution
# ----------------------------------------------------------------------------


def test_resolution_combines_token_and_name_suffix():
    resolved = resolve_environment_units(
        env={
            "wealth": wealth,
            "tax_rate_y": make_flow_rate(),
            "amount_y": amount_y,
        },
        grouping_levels=GROUPING_LEVELS,
    )
    # `wealth` is a person-level currency stock, so it carries the individual
    # level as a denominator (GEP 10); `tax_rate_y` is dimensionless and stays
    # level-less.
    assert units_are_equivalent(
        left=_scalar_unit(resolved=resolved, qname="wealth"),
        right=parse_unit("CURRENCY / grouping_level_person"),
    )
    assert units_are_equivalent(
        left=_scalar_unit(resolved=resolved, qname="tax_rate_y"),
        right=parse_unit("1 / year"),
    )
    assert units_are_equivalent(
        left=_scalar_unit(resolved=resolved, qname="amount_y"),
        right=parse_unit("CURRENCY / year / grouping_level_person"),
    )


def test_resolved_units_node_wraps_the_resolver():
    # The interface node `unit_checks__resolved_units` is a thin wrapper whose
    # parameter names are the DAG dependencies it is wired to. Calling it with
    # those names must reproduce `resolve_environment_units` exactly — the three
    # unit consumers then share this single walk per build.
    env = {"wealth": wealth, "amount_y": amount_y}
    via_node = resolved_units(
        specialized_environment__without_tree_logic_and_with_derived_functions=env,
        labels__grouping_levels=GROUPING_LEVELS,
    )
    direct = resolve_environment_units(env=env, grouping_levels=GROUPING_LEVELS)
    assert via_node == direct


def test_resolution_includes_framework_date_nodes():
    # `policy_year` is a calendar *point*, not a duration (GEP 10): it is not
    # equivalent to a `year` duration.
    env = {
        "policy_year": ScalarParam(value=2020, start_date=_START, end_date=_END),
    }
    resolved = resolve_environment_units(env=env, grouping_levels=GROUPING_LEVELS)
    assert units_are_equivalent(
        left=_scalar_unit(resolved=resolved, qname="policy_year"),
        right=parse_unit("calendar_year"),
    )
    assert not units_are_equivalent(
        left=_scalar_unit(resolved=resolved, qname="policy_year"),
        right=parse_unit("year"),
    )
    assert "policy_year" in FRAMEWORK_DATE_NODE_UNITS


def test_dict_param_with_per_leaf_units_resolves_to_unit_tree():
    schedule = DictParam(
        value={"child_amount_y": 100.0, "max_age": 18},
        unit={"child_amount_y": "CASTAR_PER_YEAR", "max_age": "YEARS"},
        start_date=_START,
        end_date=_END,
    )
    resolved = resolve_environment_units(
        env={"schedule": schedule}, grouping_levels=GROUPING_LEVELS
    )
    unit_tree = _unit_tree(resolved=resolved, qname="schedule")
    assert units_are_equivalent(
        left=unit_tree["child_amount_y"],
        right=parse_unit("CURRENCY / year / grouping_level_person"),
    )
    assert units_are_equivalent(left=unit_tree["max_age"], right=parse_unit("year"))


def test_dict_param_leaf_key_suffix_must_agree_with_spelled_period():
    # A leaf key's time suffix must agree with the period spelled in its unit
    # (GEP 10): a `_y` key declaring a per-month unit is a contradiction.
    schedule = DictParam(
        value={"child_amount_y": 100.0},
        unit={"child_amount_y": "CASTAR_PER_MONTH"},
        start_date=_START,
        end_date=_END,
    )
    with pytest.raises(UnitDefinitionError, match="must agree"):
        resolve_environment_units(
            env={"schedule": schedule}, grouping_levels=GROUPING_LEVELS
        )


def test_dict_param_integer_keyed_flow_leaf_spells_its_period():
    # Integer keys cannot carry a suffix; the leaf's unit spells the period
    # directly (GEP 10).
    amount_by_rank = DictParam(
        value={1: 250.0, 2: 250.0},
        unit={1: "CASTAR_PER_MONTH", 2: "CASTAR_PER_MONTH"},
        start_date=_START,
        end_date=_END,
    )
    resolved = resolve_environment_units(
        env={"amount_by_rank": amount_by_rank}, grouping_levels=GROUPING_LEVELS
    )
    assert units_are_equivalent(
        left=_unit_tree(resolved=resolved, qname="amount_by_rank")[1],
        right=parse_unit("CURRENCY / month / grouping_level_person"),
    )


def test_dict_param_stock_token_on_suffixed_leaf_key_fails():
    schedule = DictParam(
        value={"amount_y": 100.0},
        unit={"amount_y": "CASTAR"},
        start_date=_START,
        end_date=_END,
    )
    with pytest.raises(UnitDefinitionError, match="must agree"):
        resolve_environment_units(
            env={"schedule": schedule}, grouping_levels=GROUPING_LEVELS
        )


def test_dict_param_mixed_periods_via_spelled_units_are_allowed():
    # Each flow leaf spells its own period: nothing implicit.
    schedule = DictParam(
        value={"base_amount_m": 100.0, "annual_bonus_y": 50.0},
        unit={"base_amount_m": "CASTAR_PER_MONTH", "annual_bonus_y": "CASTAR_PER_YEAR"},
        start_date=_START,
        end_date=_END,
    )
    resolved = resolve_environment_units(
        env={"schedule": schedule}, grouping_levels=GROUPING_LEVELS
    )
    unit_tree = _unit_tree(resolved=resolved, qname="schedule")
    assert units_are_equivalent(
        left=unit_tree["base_amount_m"],
        right=parse_unit("CURRENCY / month / grouping_level_person"),
    )
    assert units_are_equivalent(
        left=unit_tree["annual_bonus_y"],
        right=parse_unit("CURRENCY / year / grouping_level_person"),
    )


def test_dict_param_missing_leaf_unit_is_reported():
    schedule = DictParam(
        value={"child_amount_y": 100.0, "max_age": 18},
        unit={"child_amount_y": "CASTAR_PER_YEAR"},
        start_date=_START,
        end_date=_END,
    )
    with pytest.raises(UnitDefinitionError, match=r"schedule\[max_age\]"):
        fail_if_environment_units_are_missing(
            env={"schedule": schedule},
            grouping_levels=GROUPING_LEVELS,
        )


def test_scalar_flow_param_resolves_via_name_suffix():
    lump_sum = ScalarParam(
        value=100.0,
        unit=CASTAR_PER_YEAR,
        start_date=_START,
        end_date=_END,
    )
    resolved = resolve_environment_units(
        env={"lump_sum_deduction_y": lump_sum}, grouping_levels=GROUPING_LEVELS
    )
    assert units_are_equivalent(
        left=_scalar_unit(resolved=resolved, qname="lump_sum_deduction_y"),
        right=parse_unit("CURRENCY / year / grouping_level_person"),
    )


def test_scalar_param_spelled_period_must_agree_with_name_suffix():
    # The spelled period must agree with the name's time suffix (GEP 10): a
    # stock CASTAR on a `_y` name is a contradiction.
    threshold = ScalarParam(
        value=100.0,
        unit=CASTAR,
        start_date=_START,
        end_date=_END,
    )
    with pytest.raises(UnitDefinitionError, match="must agree"):
        resolve_environment_units(
            env={"some_amount_y": threshold}, grouping_levels=GROUPING_LEVELS
        )


# ----------------------------------------------------------------------------
# Conservative body verification
# ----------------------------------------------------------------------------


def _wealth_tax_env() -> dict:
    return {
        "wealth": wealth,
        "is_exempt": is_exempt,
        "tax_rate_y": make_flow_rate(),
        "amount_y": amount_y,
    }


def test_stock_times_rate_with_time_component_passes():
    fail_if_environment_units_are_inconsistent(
        env=_wealth_tax_env(),
        grouping_levels=GROUPING_LEVELS,
    )


def test_stock_times_rate_without_time_component_is_caught():
    """The motivating #121 case: wealth * tax_rate must resolve to a flow.

    ``tax_rate`` is a plain dimensionless share, so ``wealth * tax_rate`` is a
    stock while the node declares a yearly flow. The exemption branch returns
    0.0 (dimensionless -> fallback); the path explorer exercises the
    substantive branch with ``is_exempt=False`` and catches the missing
    ``/ year``.
    """
    with pytest.raises(UnitConsistencyError, match="amount_buggy_y"):
        fail_if_environment_units_are_inconsistent(
            env={
                "wealth": wealth,
                "is_exempt": is_exempt,
                "tax_rate": make_dimensionless_rate(),
                "amount_buggy_y": amount_buggy_y,
            },
            grouping_levels=GROUPING_LEVELS,
        )


def test_multi_boolean_guard_bug_only_on_mixed_assignment_is_caught():
    """#134: branch exploration covers multi-boolean guards.

    The bug branch is reached only when ``is_exempt=False`` AND
    ``use_wealth=True`` — a mixed assignment the former all-truthy / all-falsy
    two-run never produced (all-truthy hit the exempt return, all-falsy hit the
    correct flow branch). The path explorer walks every reachable combination.
    """

    @policy_input(unit=Unit.DIMENSIONLESS)
    def use_wealth() -> bool:
        """Boolean input selecting the stock branch."""

    @policy_function(unit=Unit.CURRENCY.PER_YEAR)
    def amount_mixed_guard_y(
        wealth: float,
        tax_rate_y: float,
        is_exempt: bool,
        use_wealth: bool,
    ) -> float:
        if is_exempt:
            return 0.0
        if use_wealth:
            return wealth  # bug: a stock where a yearly flow is declared
        return wealth * tax_rate_y

    with pytest.raises(UnitConsistencyError, match="amount_mixed_guard_y"):
        fail_if_environment_units_are_inconsistent(
            env={
                "wealth": wealth,
                "is_exempt": is_exempt,
                "use_wealth": use_wealth,
                "tax_rate_y": make_flow_rate(),
                "amount_mixed_guard_y": amount_mixed_guard_y,
            },
            grouping_levels=GROUPING_LEVELS,
        )


def test_multi_boolean_guard_all_paths_consistent_passes():
    """The enumeration adds no false positive when every path is consistent."""

    @policy_input(unit=Unit.DIMENSIONLESS)
    def use_wealth() -> bool:
        """Boolean input selecting between two flow branches."""

    @policy_function(unit=Unit.CURRENCY.PER_YEAR)
    def amount_two_flow_branches_y(
        wealth: float,
        tax_rate_y: float,
        is_exempt: bool,
        use_wealth: bool,
    ) -> float:
        if is_exempt:
            return 0.0
        if use_wealth:
            return wealth * tax_rate_y * 2.0
        return wealth * tax_rate_y

    fail_if_environment_units_are_inconsistent(
        env={
            "wealth": wealth,
            "is_exempt": is_exempt,
            "use_wealth": use_wealth,
            "tax_rate_y": make_flow_rate(),
            "amount_two_flow_branches_y": amount_two_flow_branches_y,
        },
        grouping_levels=GROUPING_LEVELS,
    )


def test_numeric_driven_branch_bug_is_caught():
    """Tier 2 of #134: numeric-driven branches are explored, not fixed.

    The bug lives on the high-wealth arm of a numeric comparison with no
    boolean input at all. A single representative magnitude would fix the
    comparison to one arm; the path explorer forces both arms and reaches the
    stock-returning branch. The threshold is a ``CURRENCY`` parameter, so the
    comparison itself is sound (equivalent units).
    """

    @policy_function(unit=Unit.CURRENCY.PER_YEAR)
    def amount_threshold_guard_y(
        wealth: float, tax_rate_y: float, wealth_threshold: float
    ) -> float:
        if wealth > wealth_threshold:
            return wealth  # bug: a stock on the high-wealth branch
        return wealth * tax_rate_y

    with pytest.raises(UnitConsistencyError, match="amount_threshold_guard_y"):
        fail_if_environment_units_are_inconsistent(
            env={
                "wealth": wealth,
                "tax_rate_y": make_flow_rate(),
                "wealth_threshold": wealth_threshold,
                "amount_threshold_guard_y": amount_threshold_guard_y,
            },
            grouping_levels=GROUPING_LEVELS,
        )


def test_numeric_driven_branch_with_dimensionless_arm_does_not_false_positive():
    """The literal-zero guard arm stays unit-polymorphic under exploration.

    Forcing the high-wealth arm infers a dimensionless ``0.0`` (the conservative
    fallback), so the ubiquitous ``if ...: return 0.0`` pattern is not flagged.
    """

    @policy_function(unit=Unit.CURRENCY.PER_YEAR)
    def amount_means_tested_y(
        wealth: float, tax_rate_y: float, wealth_threshold: float
    ) -> float:
        if wealth > wealth_threshold:
            return 0.0
        return wealth * tax_rate_y

    fail_if_environment_units_are_inconsistent(
        env={
            "wealth": wealth,
            "tax_rate_y": make_flow_rate(),
            "wealth_threshold": wealth_threshold,
            "amount_means_tested_y": amount_means_tested_y,
        },
        grouping_levels=GROUPING_LEVELS,
    )


# ----------------------------------------------------------------------------
# Calendar points vs durations (GEP 10, S1)
# ----------------------------------------------------------------------------


def _policy_year() -> ScalarParam:
    # A framework date node — resolved to `calendar_year` via
    # FRAMEWORK_DATE_NODE_UNITS, not from this object's own (absent) unit.
    return ScalarParam(value=2020, start_date=_START, end_date=_END)


def _policy_month() -> ScalarParam:
    # A framework date node carrying a month-of-year (1-12): a cyclic ordinal,
    # resolved to dimensionless via FRAMEWORK_DATE_NODE_UNITS (GEP 10).
    return ScalarParam(value=1, start_date=_START, end_date=_END)


def test_calendar_point_difference_is_a_duration_in_years():
    """The motivating S1 pattern: ``now - birth_year`` is a duration, declared
    ``YEARS``; the dry-run accepts it through pint's offset algebra."""

    @policy_function(unit=Unit.YEARS)
    def age(policy_year: int, geburtsjahr: int) -> int:
        return policy_year - geburtsjahr

    fail_if_environment_units_are_inconsistent(
        env={
            "policy_year": _policy_year(),
            "geburtsjahr": geburtsjahr,
            "age": age,
        },
        grouping_levels=GROUPING_LEVELS,
    )


def test_duration_shifts_a_calendar_point_to_a_calendar_point():
    """A ``YEARS`` duration added to a calendar year yields a calendar year
    (``geburtsjahr + statutory_age``), declared ``CALENDAR_YEAR``."""

    @policy_function(unit=Unit.CALENDAR_YEAR)
    def retirement_year(geburtsjahr: int, statutory_age: int) -> int:
        return geburtsjahr + statutory_age

    fail_if_environment_units_are_inconsistent(
        env={
            "geburtsjahr": geburtsjahr,
            "statutory_age": statutory_age,
            "retirement_year": retirement_year,
        },
        grouping_levels=GROUPING_LEVELS,
    )


def test_adding_two_calendar_points_is_caught():
    """``point + point`` has no affine meaning; pint refuses it and the dry-run
    reports a calendar misuse (GEP 10)."""

    @policy_function(unit=Unit.CALENDAR_YEAR)
    def nonsense(policy_year: int, geburtsjahr: int) -> int:
        return policy_year + geburtsjahr  # bug: two calendar points added

    with pytest.raises(UnitConsistencyError, match="combines a calendar point"):
        fail_if_environment_units_are_inconsistent(
            env={
                "policy_year": _policy_year(),
                "geburtsjahr": geburtsjahr,
                "nonsense": nonsense,
            },
            grouping_levels=GROUPING_LEVELS,
        )


def test_scaling_a_calendar_point_is_caught():
    """A calendar point cannot be scaled (it is affine, not multiplicative)."""

    @policy_function(unit=Unit.CALENDAR_YEAR)
    def doubled(geburtsjahr: int) -> int:
        return geburtsjahr * 2  # bug: scaling a calendar point

    with pytest.raises(UnitConsistencyError, match="combines a calendar point"):
        fail_if_environment_units_are_inconsistent(
            env={
                "geburtsjahr": geburtsjahr,
                "doubled": doubled,
            },
            grouping_levels=GROUPING_LEVELS,
        )


def test_calendar_point_difference_declared_as_a_calendar_year_is_caught():
    """The S1 failure mode: a year difference is a duration, so declaring the
    result ``CALENDAR_YEAR`` (a point) is inconsistent and is flagged."""

    @policy_function(unit=Unit.CALENDAR_YEAR)
    def age(policy_year: int, geburtsjahr: int) -> int:
        return policy_year - geburtsjahr  # a duration, not a calendar year

    with pytest.raises(UnitConsistencyError, match="age"):
        fail_if_environment_units_are_inconsistent(
            env={
                "policy_year": _policy_year(),
                "geburtsjahr": geburtsjahr,
                "age": age,
            },
            grouping_levels=GROUPING_LEVELS,
        )


def test_ordering_a_calendar_point_against_another_unit_is_caught():
    """An ordering runs no forward pint op, so a calendar point gets no
    delegate-to-pint dispensation there: ordered against anything but a
    same-axis point, it is a unit mix (GEP 10)."""

    @policy_function(unit=Unit.DIMENSIONLESS)
    def nonsense(geburtsjahr: int, income_m: float) -> bool:
        return geburtsjahr >= income_m  # bug: a calendar point vs a flow

    with pytest.raises(UnitConsistencyError, match="non-equivalent"):
        fail_if_environment_units_are_inconsistent(
            env={
                "geburtsjahr": geburtsjahr,
                "income_m": income_m,
                "nonsense": nonsense,
            },
            grouping_levels=GROUPING_LEVELS,
        )


def test_ordering_a_calendar_point_against_a_duration_is_caught():
    """A point and a duration share ``[time]`` but not an algebra: equivalence
    decides points by identity, so ordering them is a unit mix (GEP 10)."""

    @policy_function(unit=Unit.DIMENSIONLESS)
    def nonsense(geburtsjahr: int, statutory_age: int) -> bool:
        return geburtsjahr >= statutory_age  # bug: a point vs a duration

    with pytest.raises(UnitConsistencyError, match="non-equivalent"):
        fail_if_environment_units_are_inconsistent(
            env={
                "geburtsjahr": geburtsjahr,
                "statutory_age": statutory_age,
                "nonsense": nonsense,
            },
            grouping_levels=GROUPING_LEVELS,
        )


def test_ordering_two_same_axis_calendar_points_passes():
    """Ordering two points on the same calendar axis is sound
    (``geburtsjahr <= policy_year``): identical units, so the ordering screen
    passes without any calendar dispensation."""

    @policy_function(unit=Unit.DIMENSIONLESS)
    def born_by_policy_year(policy_year: int, geburtsjahr: int) -> bool:
        return geburtsjahr <= policy_year

    fail_if_environment_units_are_inconsistent(
        env={
            "policy_year": _policy_year(),
            "geburtsjahr": geburtsjahr,
            "born_by_policy_year": born_by_policy_year,
        },
        grouping_levels=GROUPING_LEVELS,
    )


def test_subtracting_calendar_points_of_different_axes_is_caught():
    """Two *different* offset units of the same [time] dimension are the trap:
    pint subtracts ``calendar_month - calendar_year`` with a silent /12 while the
    run-time subtraction is raw and unconverted, so a cross-axis point - point is
    rejected rather than delegated to pint (defect #2, GEP 10)."""

    @policy_input(unit=Unit.CALENDAR_MONTH)
    def some_calendar_month() -> int:
        """A month point on the calendar."""

    @policy_function(unit=Unit.MONTHS)
    def nonsense(some_calendar_month: int, geburtsjahr: int) -> int:
        return some_calendar_month - geburtsjahr  # bug: subtract points across axes

    with pytest.raises(UnitConsistencyError, match="non-equivalent"):
        fail_if_environment_units_are_inconsistent(
            env={
                "some_calendar_month": some_calendar_month,
                "geburtsjahr": geburtsjahr,
                "nonsense": nonsense,
            },
            grouping_levels=GROUPING_LEVELS,
        )


def test_adding_a_currency_to_a_calendar_point_is_reported_as_a_calendar_misuse():
    """A calendar point plus a foreign dimension raises pint ``DimensionalityError``;
    it is a genuine calendar bug, so it reports as a calendar misuse rather than
    falling into the blanket ``verify_units=False`` advice (defect #7, GEP 10)."""

    @policy_function(unit=Unit.CALENDAR_YEAR)
    def nonsense(geburtsjahr: int, income_m: float) -> float:
        return geburtsjahr + income_m  # bug: a calendar point plus a currency

    with pytest.raises(UnitConsistencyError, match="combines a calendar point"):
        fail_if_environment_units_are_inconsistent(
            env={
                "geburtsjahr": geburtsjahr,
                "income_m": income_m,
                "nonsense": nonsense,
            },
            grouping_levels=GROUPING_LEVELS,
        )


def test_flow_time_converter_body_passes():
    """``per_m_to_per_y`` rebases a monthly flow to a yearly one; the dry-run models
    the period rebase, so the body checks against its ``_y`` declaration with no
    opt-out (GEP 10, time converters)."""

    @policy_input(unit=Unit.CURRENCY.PER_MONTH)
    def betrag_m() -> float:
        """A monthly flow."""

    @policy_function(unit=Unit.CURRENCY.PER_YEAR)
    def betrag_y(betrag_m: float) -> float:
        return per_m_to_per_y(betrag_m)

    fail_if_environment_units_are_inconsistent(
        env={"betrag_m": betrag_m, "betrag_y": betrag_y},
        grouping_levels=GROUPING_LEVELS,
    )


def test_duration_time_converter_body_passes():
    """``m_to_y`` rebases a ``MONTHS`` duration to ``YEARS``; the classic
    ``m_to_y(months) >= grenze`` shape checks without an opt-out (GEP 10)."""

    @policy_input(unit=Unit.MONTHS)
    def wartezeit() -> int:
        """A waiting time in months (a duration)."""

    @policy_input(unit=Unit.YEARS)
    def wartezeitgrenze() -> int:
        """A threshold in years."""

    @policy_function(unit=Unit.DIMENSIONLESS)
    def wartezeit_erfüllt(wartezeit: int, wartezeitgrenze: int) -> bool:
        return m_to_y(wartezeit) >= wartezeitgrenze

    fail_if_environment_units_are_inconsistent(
        env={
            "wartezeit": wartezeit,
            "wartezeitgrenze": wartezeitgrenze,
            "wartezeit_erfüllt": wartezeit_erfüllt,
        },
        grouping_levels=GROUPING_LEVELS,
    )


def test_wrong_direction_time_converter_is_caught():
    """A converter for the wrong period rebases to a unit that disagrees with the
    declaration, so the misuse is caught rather than silently passed (GEP 10)."""

    @policy_input(unit=Unit.MONTHS)
    def wartezeit() -> int:
        """A duration in months."""

    @policy_function(unit=Unit.YEARS)
    def nonsense(wartezeit: int) -> float:
        return y_to_m(wartezeit)  # wrong: a MONTHS duration fed to a year->month rebase

    with pytest.raises(UnitConsistencyError, match="nonsense"):
        fail_if_environment_units_are_inconsistent(
            env={"wartezeit": wartezeit, "nonsense": nonsense},
            grouping_levels=GROUPING_LEVELS,
        )


def test_month_date_nodes_are_cyclic_ordinals():
    """``policy_month`` carries a month-of-year (1-12): a cyclic ordinal, hence
    ``DIMENSIONLESS`` (GEP 10), so comparing it to another ordinal is plain
    dimensionless arithmetic."""

    @policy_function(unit=Unit.DIMENSIONLESS)
    def had_birthday(policy_month: int, geburtsmonat: int) -> bool:
        return policy_month >= geburtsmonat

    fail_if_environment_units_are_inconsistent(
        env={
            "policy_month": _policy_month(),
            "geburtsmonat": geburtsmonat,
            "had_birthday": had_birthday,
        },
        grouping_levels=GROUPING_LEVELS,
    )


def test_month_date_node_shifted_by_a_duration_is_caught():
    """Shifting the cyclic ``policy_month`` by a months duration wraps at run
    time — the silent fold the ordinal/point split exists to catch. As a
    dimensionless ordinal it does not add to a ``MONTHS`` duration (GEP 10)."""

    @policy_function(unit=Unit.DIMENSIONLESS)
    def nonsense(policy_month: int, months_paid: int) -> int:
        return policy_month + months_paid  # bug: an ordinal shifted like a point

    with pytest.raises(UnitConsistencyError, match="nonsense"):
        fail_if_environment_units_are_inconsistent(
            env={
                "policy_month": _policy_month(),
                "months_paid": months_paid,
                "nonsense": nonsense,
            },
            grouping_levels=GROUPING_LEVELS,
        )


def test_error_names_the_failing_branch():
    """A branch-confined failure names the branch in the body's own terms and
    reports the other combinations clean (GEP 10)."""

    @policy_function(unit=Unit.CURRENCY.PER_MONTH)
    def betrag_m(bonus_y: float, is_exempt: bool) -> float:
        if is_exempt:
            return 0.0
        return bonus_y  # bug: a yearly flow under a monthly declaration

    with pytest.raises(
        UnitConsistencyError,
        match=(
            r"on the branch where `is_exempt` is False\. "
            r"All other branch combinations match the declaration\."
        ),
    ):
        fail_if_environment_units_are_inconsistent(
            env={
                "bonus_y": bonus_y,
                "is_exempt": is_exempt,
                "betrag_m": betrag_m,
            },
            grouping_levels=GROUPING_LEVELS,
        )


def test_error_names_a_comparison_driven_branch():
    """A branch decided by a comparison is named by that comparison's operands
    (GEP 10)."""

    @policy_function(unit=Unit.CURRENCY.PER_MONTH)
    def gated_m(income_m: float, other_income_m: float, bonus_y: float) -> float:
        if income_m < other_income_m:
            return income_m
        return bonus_y  # bug: a yearly flow under a monthly declaration

    with pytest.raises(
        UnitConsistencyError,
        match=r"on the branch where `income_m < other_income_m` is False",
    ):
        fail_if_environment_units_are_inconsistent(
            env={
                "income_m": income_m,
                "other_income_m": other_income_m,
                "bonus_y": bonus_y,
                "gated_m": gated_m,
            },
            grouping_levels=GROUPING_LEVELS,
        )


def test_boolean_body_bad_comparison_is_caught():
    """A boolean-returning body is dry-run like any other (GEP 10).

    Its truth-value output carries no unit, but the comparison inside it does:
    ``wealth`` is a ``CURRENCY`` stock and ``bonus_y`` a ``CURRENCY / year``
    flow, so the ``>=`` mixes non-equivalent units. Before, the boolean output
    made the whole body skip the check and this slipped through silently.
    """

    @policy_function(unit=Unit.DIMENSIONLESS)
    def wealthy(wealth: float, bonus_y: float) -> bool:
        return wealth >= bonus_y  # bug: a stock compared with a yearly flow

    with pytest.raises(UnitConsistencyError, match="wealthy"):
        fail_if_environment_units_are_inconsistent(
            env={
                "wealth": wealth,
                "bonus_y": bonus_y,
                "wealthy": wealthy,
            },
            grouping_levels=GROUPING_LEVELS,
        )


def test_boolean_body_with_logical_ops_passes():
    """A boolean body combining clean truth values with ``&``/``|``/``~`` is not
    a false positive: every operand is a dimensionless truth value."""

    @policy_function(unit=Unit.DIMENSIONLESS)
    def eligible(income_m: float, other_income_m: float, is_exempt: bool) -> bool:
        return ((income_m >= other_income_m) | is_exempt) & (~is_exempt)

    fail_if_environment_units_are_inconsistent(
        env={
            "income_m": income_m,
            "other_income_m": other_income_m,
            "is_exempt": is_exempt,
            "eligible": eligible,
        },
        grouping_levels=GROUPING_LEVELS,
    )


def test_logical_op_on_unit_carrying_operand_is_caught():
    """A logical operator applied to a real quantity (not a truth value) is a
    bug the run-time arrays would silently swallow, so the dry-run rejects it.

    ``age`` is a ``YEARS`` quantity, so ``age & is_exempt`` ANDs a duration into
    a logical combination — caught on either side via the reflected dunders.
    """

    @policy_input(unit=Unit.YEARS)
    def age() -> int:
        """An age in years (a ``YEARS`` quantity)."""

    @policy_function(unit=Unit.DIMENSIONLESS)
    def nonsense(age: int, is_exempt: bool) -> bool:
        return age & is_exempt  # bug: '&' on a YEARS quantity

    with pytest.raises(UnitConsistencyError, match="nonsense"):
        fail_if_environment_units_are_inconsistent(
            env={
                "age": age,
                "is_exempt": is_exempt,
                "nonsense": nonsense,
            },
            grouping_levels=GROUPING_LEVELS,
        )


def test_not_of_a_leveled_boolean_keeps_its_level():
    """``not`` on a leveled boolean keeps its level, exactly as ``~`` does (defect
    #5, GEP 10): ``flag_fam and not other_flag_fam`` stays fam-level and matches
    the fam-level declaration — no spurious level error. Before the fix, ``not``
    dropped the stand-in to a plain bool and the combine mislevelled the result."""

    @policy_input(unit=Unit.DIMENSIONLESS.PER_FAM)
    def flag_fam() -> bool:
        """A fam-level indicator."""

    @policy_input(unit=Unit.DIMENSIONLESS.PER_FAM)
    def other_flag_fam() -> bool:
        """Another fam-level indicator."""

    @policy_function(leaf_name="combined_fam", unit=Unit.DIMENSIONLESS.PER_FAM)
    def combined_fam(flag_fam: bool, other_flag_fam: bool) -> bool:
        return flag_fam and not other_flag_fam

    fail_if_environment_units_are_inconsistent(
        env={
            "flag_fam": flag_fam,
            "other_flag_fam": other_flag_fam,
            "combined_fam": combined_fam,
        },
        grouping_levels=GROUPING_LEVELS,
    )


def test_not_of_a_non_boolean_quantity_is_caught():
    """``not`` on a non-boolean (a currency) is a bug that ``~`` catches; its scalar
    spelling must too — the dry-run models ``not`` as ``logical_not`` (defect #5,
    GEP 10)."""

    @policy_function(unit=Unit.DIMENSIONLESS)
    def flag(income_m: float) -> bool:
        return not income_m  # bug: `not` on a currency

    with pytest.raises(UnitConsistencyError, match="non-boolean"):
        fail_if_environment_units_are_inconsistent(
            env={"income_m": income_m, "flag": flag},
            grouping_levels=GROUPING_LEVELS,
        )


def test_boolean_body_at_correct_group_level_passes():
    """A fam-level predicate comparing fam-level quantities infers ``1 / [fam]``,
    matching its ``_fam`` name (GEP 10)."""

    @policy_input(unit=Unit.CURRENCY.PER_MONTH.PER_FAM)
    def income_m_fam() -> float:
        """Family income."""

    @policy_input(unit=Unit.CURRENCY.PER_MONTH.PER_FAM)
    def threshold_m_fam() -> float:
        """Family subsistence threshold."""

    @policy_function(
        leaf_name="requirement_fulfilled_fam", unit=Unit.DIMENSIONLESS.PER_FAM
    )
    def requirement_fulfilled_fam(income_m_fam: float, threshold_m_fam: float) -> bool:
        return income_m_fam < threshold_m_fam

    fail_if_environment_units_are_inconsistent(
        env={
            "income_m_fam": income_m_fam,
            "threshold_m_fam": threshold_m_fam,
            "requirement_fulfilled_fam": requirement_fulfilled_fam,
        },
        grouping_levels=GROUPING_LEVELS,
    )


def test_boolean_body_at_wrong_group_level_is_caught():
    """A ``DIMENSIONLESS_PER_FAM`` predicate that actually compares *person*-level
    quantities infers ``1 / [person]`` and is caught (GEP 10) — the bug leveled
    booleans fix.

    Before, the boolean result was level-less and bypassed the suffix-level check.
    """

    @policy_function(
        leaf_name="requirement_fulfilled_fam", unit=Unit.DIMENSIONLESS.PER_FAM
    )
    def requirement_fulfilled_fam(income_m: float, other_income_m: float) -> bool:
        return income_m < other_income_m  # person-level operands, but a _fam name

    with pytest.raises(UnitConsistencyError, match="requirement_fulfilled_fam"):
        fail_if_environment_units_are_inconsistent(
            env={
                "income_m": income_m,
                "other_income_m": other_income_m,
                "requirement_fulfilled_fam": requirement_fulfilled_fam,
            },
            grouping_levels=GROUPING_LEVELS,
        )


def test_logical_combine_of_mixed_levels_downcasts_to_person():
    """``|`` of a fam-level and a person-level indicator is a person-level boolean
    (the combine rule), matching an unsuffixed name (GEP 10)."""

    @policy_input(unit=Unit.CURRENCY.PER_MONTH.PER_FAM)
    def income_m_fam() -> float:
        """Family income."""

    @policy_input(unit=Unit.CURRENCY.PER_MONTH.PER_FAM)
    def threshold_m_fam() -> float:
        """Family subsistence threshold."""

    @policy_function(unit=Unit.DIMENSIONLESS)
    def eligible(
        income_m: float,
        other_income_m: float,
        income_m_fam: float,
        threshold_m_fam: float,
    ) -> bool:
        return (income_m_fam < threshold_m_fam) | (income_m < other_income_m)

    fail_if_environment_units_are_inconsistent(
        env={
            "income_m": income_m,
            "other_income_m": other_income_m,
            "income_m_fam": income_m_fam,
            "threshold_m_fam": threshold_m_fam,
            "eligible": eligible,
        },
        grouping_levels=GROUPING_LEVELS,
    )


def test_python_or_of_mixed_levels_is_rewritten_and_downcasts_to_person():
    """Author-written ``or`` combines leveled booleans exactly like ``|`` (GEP 10).

    Python ``or`` short-circuits through ``__bool__`` and on its own would return a
    single, uncombined operand; the dry-run rewrites ``and``/``or`` to ``&``/``|``
    first (mirroring the array vectorizer), so a fam-level ``or`` a person-level
    indicator downcasts to a person-level boolean, matching the unsuffixed name —
    the ``wealth_tax.exempt_from_wealth_tax`` shape.
    """

    @policy_input(unit=Unit.CURRENCY.PER_MONTH.PER_FAM)
    def income_m_fam() -> float:
        """Family income."""

    @policy_input(unit=Unit.CURRENCY.PER_MONTH.PER_FAM)
    def threshold_m_fam() -> float:
        """Family subsistence threshold."""

    @policy_function(unit=Unit.DIMENSIONLESS)
    def eligible(
        income_m: float,
        other_income_m: float,
        income_m_fam: float,
        threshold_m_fam: float,
    ) -> bool:
        return income_m_fam < threshold_m_fam or income_m < other_income_m

    fail_if_environment_units_are_inconsistent(
        env={
            "income_m": income_m,
            "other_income_m": other_income_m,
            "income_m_fam": income_m_fam,
            "threshold_m_fam": threshold_m_fam,
            "eligible": eligible,
        },
        grouping_levels=GROUPING_LEVELS,
    )


def test_python_and_on_unit_carrying_operand_is_still_caught():
    """The ``and``→``&`` rewrite keeps the operand screen: ``and``-ing a real
    quantity into a logical combination is still rejected (GEP 10).

    ``age`` is a ``YEARS`` quantity, so ``age and is_exempt`` is the same bug as
    ``age & is_exempt`` — the rewrite must not let it slip through.
    """

    @policy_input(unit=Unit.YEARS)
    def age() -> int:
        """An age in years (a ``YEARS`` quantity)."""

    @policy_function(unit=Unit.DIMENSIONLESS)
    def nonsense(age: int, is_exempt: bool) -> bool:
        return age and is_exempt  # bug: 'and' on a YEARS quantity

    with pytest.raises(UnitConsistencyError, match="nonsense"):
        fail_if_environment_units_are_inconsistent(
            env={
                "age": age,
                "is_exempt": is_exempt,
                "nonsense": nonsense,
            },
            grouping_levels=GROUPING_LEVELS,
        )


# ----------------------------------------------------------------------------
# Cross-level shares (division across grouping levels)
# ----------------------------------------------------------------------------


def test_terminal_cross_level_division_is_caught():
    """Dividing two amounts at *different* group levels leaves a bare ratio of
    levels (``betrag_m_fam / betrag_m_kin`` -> ``[kin]/[fam]``) once the physical
    content cancels. A grouping level cannot outlive its base, so returning that
    residue as a *result* is caught on the level axis (GEP 10)."""

    @policy_input(unit=Unit.CURRENCY.PER_MONTH.PER_FAM)
    def betrag_m_fam() -> float:
        """A monthly family amount."""

    @policy_input(unit=Unit.CURRENCY.PER_MONTH.PER_KIN)
    def betrag_m_kin() -> float:
        """A monthly kin amount."""

    @policy_function(unit=Unit.DIMENSIONLESS)
    def anteil(betrag_m_fam: float, betrag_m_kin: float) -> float:
        return betrag_m_fam / betrag_m_kin

    with pytest.raises(UnitConsistencyError, match="anteil"):
        fail_if_environment_units_are_inconsistent(
            env={
                "betrag_m_fam": betrag_m_fam,
                "betrag_m_kin": betrag_m_kin,
                "anteil": anteil,
            },
            grouping_levels=GROUPING_LEVELS,
        )


def test_terminal_cross_level_division_passes_with_an_explicit_opt_out():
    """A genuine terminal cross-level ratio is a deliberate policy judgement, so
    it takes a local ``verify_units=False`` rather than a blanket exemption
    (GEP 10)."""

    @policy_input(unit=Unit.CURRENCY.PER_MONTH.PER_FAM)
    def betrag_m_fam() -> float:
        """A monthly family amount."""

    @policy_input(unit=Unit.CURRENCY.PER_MONTH.PER_KIN)
    def betrag_m_kin() -> float:
        """A monthly kin amount."""

    @policy_function(unit=Unit.DIMENSIONLESS, verify_units=False)
    def anteil(betrag_m_fam: float, betrag_m_kin: float) -> float:
        return betrag_m_kin / betrag_m_fam

    fail_if_environment_units_are_inconsistent(
        env={
            "betrag_m_fam": betrag_m_fam,
            "betrag_m_kin": betrag_m_kin,
            "anteil": anteil,
        },
        grouping_levels=GROUPING_LEVELS,
    )


def test_bedarfsanteilsmethode_cross_level_share_consumed_by_multiplication_passes():
    """The GETTSIM idiom: a person's share of a group claim,
    ``(bedarf_m / bedarf_m_fam) * anspruch_m_fam``. The cross-level result
    ``[fam]/[person]`` is consumed by the multiply, landing on a person-level flow
    that matches the declaration — no exemption needed, and unchanged by the
    cross-level rule (GEP 10)."""

    @policy_input(unit=Unit.CURRENCY.PER_MONTH)
    def bedarf_m() -> float:
        """A person's monthly need."""

    @policy_input(unit=Unit.CURRENCY.PER_MONTH.PER_FAM)
    def bedarf_m_fam() -> float:
        """The family's pooled monthly need."""

    @policy_input(unit=Unit.CURRENCY.PER_MONTH.PER_FAM)
    def anspruch_m_fam() -> float:
        """The family's monthly claim."""

    @policy_function(unit=Unit.CURRENCY.PER_MONTH)
    def betrag_m(bedarf_m: float, bedarf_m_fam: float, anspruch_m_fam: float) -> float:
        return (bedarf_m / bedarf_m_fam) * anspruch_m_fam

    fail_if_environment_units_are_inconsistent(
        env={
            "bedarf_m": bedarf_m,
            "bedarf_m_fam": bedarf_m_fam,
            "anspruch_m_fam": anspruch_m_fam,
            "betrag_m": betrag_m,
        },
        grouping_levels=GROUPING_LEVELS,
    )


def test_cross_level_share_declared_with_concrete_content_is_caught():
    """A cross-level division leaves a physically dimensionless result; declaring
    it with concrete content (``CURRENCY_PER_MONTH`` rather than
    ``DIMENSIONLESS``) is caught on the physical axis, before the level axis is
    even reached (GEP 10)."""

    @policy_input(unit=Unit.CURRENCY.PER_MONTH.PER_FAM)
    def betrag_m_fam() -> float:
        """A monthly family amount."""

    @policy_input(unit=Unit.CURRENCY.PER_MONTH.PER_KIN)
    def betrag_m_kin() -> float:
        """A monthly kin amount."""

    @policy_function(leaf_name="anteil_m", unit=Unit.CURRENCY.PER_MONTH)
    def anteil_m(betrag_m_fam: float, betrag_m_kin: float) -> float:
        return betrag_m_fam / betrag_m_kin  # [kin]/[fam] cross-level result, not money

    with pytest.raises(UnitConsistencyError, match="anteil_m"):
        fail_if_environment_units_are_inconsistent(
            env={
                "betrag_m_fam": betrag_m_fam,
                "betrag_m_kin": betrag_m_kin,
                "anteil_m": anteil_m,
            },
            grouping_levels=GROUPING_LEVELS,
        )


def test_head_count_at_wrong_group_level_is_still_caught():
    """A head count carries a ``[person]`` numerator and is a declarable,
    level-checked unit. A ``[person]/[fam]`` count declared at the kin level is
    caught (GEP 10) — it is not mistaken for a cross-level share."""

    @policy_input(unit=Unit.PERSON_COUNT.PER_FAM)
    def anzahl_personen_fam() -> int:
        """A head count per family — ``[person]/[fam]``."""

    @policy_function(leaf_name="anzahl_personen_kin", unit=Unit.PERSON_COUNT.PER_KIN)
    def anzahl_personen_kin(anzahl_personen_fam: int) -> int:
        return anzahl_personen_fam  # a [person]/[fam] count under a _kin name

    with pytest.raises(UnitConsistencyError, match="anzahl_personen_kin"):
        fail_if_environment_units_are_inconsistent(
            env={
                "anzahl_personen_fam": anzahl_personen_fam,
                "anzahl_personen_kin": anzahl_personen_kin,
            },
            grouping_levels=GROUPING_LEVELS,
        )


# ----------------------------------------------------------------------------
# `cast_unit`: the expression-level escape hatch
# ----------------------------------------------------------------------------


def test_cross_level_comparison_without_cast_is_caught():
    """Comparing a group extreme against a level-less threshold mixes levels
    (``month/[fam]`` against ``month``), so the ordering screen rejects it —
    even where the law mandates exactly this test (GEP 10)."""

    @policy_input(unit=Unit.MONTHS.PER_FAM)
    def age_youngest_months_fam() -> float:
        """The family's youngest member's age — a property of the family."""

    @policy_input(unit=Unit.MONTHS)
    def age_limit_months() -> float:
        """An age threshold; a level-less duration."""

    @policy_function(unit=Unit.DIMENSIONLESS)
    def eligible(age_youngest_months_fam: float, age_limit_months: float) -> bool:
        return age_youngest_months_fam <= age_limit_months

    with pytest.raises(UnitConsistencyError, match="eligible"):
        fail_if_environment_units_are_inconsistent(
            env={
                "age_youngest_months_fam": age_youngest_months_fam,
                "age_limit_months": age_limit_months,
                "eligible": eligible,
            },
            grouping_levels=GROUPING_LEVELS,
        )


def test_cross_level_comparison_with_cast_passes():
    """The policy-mandated per-person reading — each person sees their family's
    extreme — is stated at the site with ``cast_unit``; the rest of the body
    stays checked (GEP 10)."""

    @policy_input(unit=Unit.MONTHS.PER_FAM)
    def age_youngest_months_fam() -> float:
        """The family's youngest member's age — a property of the family."""

    @policy_input(unit=Unit.MONTHS)
    def age_limit_months() -> float:
        """An age threshold; a level-less duration."""

    @policy_function(unit=Unit.DIMENSIONLESS)
    def eligible(age_youngest_months_fam: float, age_limit_months: float) -> bool:
        return (
            cast_unit(value=age_youngest_months_fam, unit=Unit.MONTHS)
            <= age_limit_months
        )

    fail_if_environment_units_are_inconsistent(
        env={
            "age_youngest_months_fam": age_youngest_months_fam,
            "age_limit_months": age_limit_months,
            "eligible": eligible,
        },
        grouping_levels=GROUPING_LEVELS,
    )


def test_level_less_inference_under_a_declared_group_level_is_caught():
    """The declared-vs-inferred level match is exact: a body whose arithmetic
    yields no level cannot silently claim the declared group level; the error
    points at ``cast_unit`` (GEP 10)."""

    @policy_input(unit=Unit.MONTHS)
    def age_limit_months() -> float:
        """An age threshold; a level-less duration."""

    @policy_function(unit=Unit.MONTHS.PER_FAM)
    def doubled_limit_months_fam(age_limit_months: float) -> float:
        return age_limit_months * 2.0

    with pytest.raises(UnitConsistencyError, match="cast_unit"):
        fail_if_environment_units_are_inconsistent(
            env={
                "age_limit_months": age_limit_months,
                "doubled_limit_months_fam": doubled_limit_months_fam,
            },
            grouping_levels=GROUPING_LEVELS,
        )


def test_cast_at_the_return_states_the_declared_group_level():
    """An intensive group property computed from level-less material states its
    level with ``cast_unit`` at the return (GEP 10)."""

    @policy_input(unit=Unit.MONTHS)
    def age_limit_months() -> float:
        """An age threshold; a level-less duration."""

    @policy_function(unit=Unit.MONTHS.PER_FAM)
    def doubled_limit_months_fam(age_limit_months: float) -> float:
        return cast_unit(value=age_limit_months * 2.0, unit=Unit.MONTHS.PER_FAM)

    fail_if_environment_units_are_inconsistent(
        env={
            "age_limit_months": age_limit_months,
            "doubled_limit_months_fam": doubled_limit_months_fam,
        },
        grouping_levels=GROUPING_LEVELS,
    )


def test_group_share_times_group_total_squares_the_level_and_is_caught():
    """A group-owned share times a group total squares the level
    (``1/[fam] * CURRENCY/month/[fam]`` → ``…/[fam]**2``); the level signature
    is compared with exponents, so the product cannot silently claim the
    declared single level (GEP 10)."""

    @policy_input(unit=Unit.DIMENSIONLESS.PER_FAM)
    def parents_share_fam() -> float:
        """The parents' share of the family's need — the family's property."""

    @policy_input(unit=Unit.CURRENCY.PER_MONTH.PER_FAM)
    def need_m_fam() -> float:
        """The family's monthly need."""

    @policy_function(unit=Unit.CURRENCY.PER_MONTH.PER_FAM)
    def parents_need_m_fam(parents_share_fam: float, need_m_fam: float) -> float:
        return parents_share_fam * need_m_fam

    with pytest.raises(UnitConsistencyError, match="cast_unit"):
        fail_if_environment_units_are_inconsistent(
            env={
                "parents_share_fam": parents_share_fam,
                "need_m_fam": need_m_fam,
                "parents_need_m_fam": parents_need_m_fam,
            },
            grouping_levels=GROUPING_LEVELS,
        )


def test_group_share_times_group_total_passes_with_cast():
    """Where the law mandates the group-share product, the cast states the
    intended result at the site (GEP 10)."""

    @policy_input(unit=Unit.DIMENSIONLESS.PER_FAM)
    def parents_share_fam() -> float:
        """The parents' share of the family's need — the family's property."""

    @policy_input(unit=Unit.CURRENCY.PER_MONTH.PER_FAM)
    def need_m_fam() -> float:
        """The family's monthly need."""

    @policy_function(unit=Unit.CURRENCY.PER_MONTH.PER_FAM)
    def parents_need_m_fam(parents_share_fam: float, need_m_fam: float) -> float:
        return cast_unit(
            value=parents_share_fam * need_m_fam, unit=Unit.CURRENCY.PER_MONTH.PER_FAM
        )

    fail_if_environment_units_are_inconsistent(
        env={
            "parents_share_fam": parents_share_fam,
            "need_m_fam": need_m_fam,
            "parents_need_m_fam": parents_need_m_fam,
        },
        grouping_levels=GROUPING_LEVELS,
    )


def test_cast_tags_a_dimensioned_literal_in_an_ordering_comparison():
    """A genuine dimensioned bound that must stay inline is tagged in place;
    the tagged literal is still screened, so a wrong-period tag is caught
    (GEP 10)."""

    @policy_function(unit=Unit.DIMENSIONLESS)
    def poor(income_m: float) -> bool:
        return income_m < cast_unit(value=1000.0, unit=Unit.CURRENCY.PER_MONTH)

    fail_if_environment_units_are_inconsistent(
        env={"income_m": income_m, "poor": poor},
        grouping_levels=GROUPING_LEVELS,
    )

    @policy_function(unit=Unit.DIMENSIONLESS)
    def poor_buggy(income_m: float) -> bool:
        return income_m < cast_unit(
            value=1000.0, unit=Unit.CURRENCY.PER_YEAR
        )  # wrong period

    with pytest.raises(UnitConsistencyError, match="poor_buggy"):
        fail_if_environment_units_are_inconsistent(
            env={"income_m": income_m, "poor_buggy": poor_buggy},
            grouping_levels=GROUPING_LEVELS,
        )


def test_cast_in_a_vectorized_body_is_screened_identically():
    """The dry-run's ``xnp`` shim and the cast compose: a literal cap tagged in
    place inside ``xnp.minimum`` passes where the bare literal would be
    rejected (GEP 10)."""

    @policy_function(
        unit=Unit.CURRENCY.PER_MONTH, vectorization_strategy="not_required"
    )
    def capped_income_m(income_m: FloatColumn, xnp: ModuleType) -> FloatColumn:
        return xnp.minimum(
            income_m, cast_unit(value=2000.0, unit=Unit.CURRENCY.PER_MONTH)
        )

    fail_if_environment_units_are_inconsistent(
        env={"income_m": income_m, "capped_income_m": capped_income_m},
        grouping_levels=GROUPING_LEVELS,
    )


def test_cast_to_a_concrete_currency_is_rejected():
    """Bodies are currency-agnostic, so a cast pinning a concrete currency is a
    definition error — reported as such, not as an un-evaluable body (GEP 10)."""

    @policy_function(unit=Unit.CURRENCY.PER_MONTH)
    def pinned_m(income_m: float) -> float:
        return cast_unit(value=income_m, unit="CASTAR_PER_MONTH")

    with pytest.raises(UnitDefinitionError, match="currency-agnostic"):
        fail_if_environment_units_are_inconsistent(
            env={"income_m": income_m, "pinned_m": pinned_m},
            grouping_levels=GROUPING_LEVELS,
        )


def test_dimensionless_inference_cannot_claim_a_group_owned_declaration():
    """A plain dimensionless result slips every level screen, so it cannot
    claim a group-owned declaration: a fam predicate over level-less shares
    states its level with ``cast_unit``; the person grain stays lenient
    (GEP 10)."""

    @policy_input(unit=Unit.DIMENSIONLESS)
    def share_of_need() -> float:
        """A level-less share."""

    @policy_input(unit=Unit.DIMENSIONLESS)
    def threshold_share() -> float:
        """A level-less share."""

    @policy_function(unit=Unit.DIMENSIONLESS.PER_FAM)
    def requirement_fulfilled_fam(share_of_need: float, threshold_share: float) -> bool:
        return share_of_need < threshold_share

    with pytest.raises(UnitConsistencyError, match="cast_unit"):
        fail_if_environment_units_are_inconsistent(
            env={
                "share_of_need": share_of_need,
                "threshold_share": threshold_share,
                "requirement_fulfilled_fam": requirement_fulfilled_fam,
            },
            grouping_levels=GROUPING_LEVELS,
        )

    @policy_function(unit=Unit.DIMENSIONLESS.PER_FAM)
    def requirement_fulfilled_cast_fam(
        share_of_need: float, threshold_share: float
    ) -> bool:
        return cast_unit(
            value=share_of_need < threshold_share, unit=Unit.DIMENSIONLESS.PER_FAM
        )

    fail_if_environment_units_are_inconsistent(
        env={
            "share_of_need": share_of_need,
            "threshold_share": threshold_share,
            "requirement_fulfilled_fam": requirement_fulfilled_cast_fam,
        },
        grouping_levels=GROUPING_LEVELS,
    )


def test_adding_a_nonzero_bare_literal_to_a_quantity_is_caught():
    """``income_m + 100.0`` hides a monthly amount in the literal; ``+``/``-``
    screen literals exactly as the ordering comparisons do — promote to a
    parameter, tag with ``cast_unit``, or use 0 (GEP 10)."""

    @policy_function(unit=Unit.CURRENCY.PER_MONTH)
    def bumped_income_m(income_m: float) -> float:
        return income_m + 100.0

    with pytest.raises(UnitConsistencyError, match="bare literal"):
        fail_if_environment_units_are_inconsistent(
            env={"income_m": income_m, "bumped_income_m": bumped_income_m},
            grouping_levels=GROUPING_LEVELS,
        )

    @policy_function(unit=Unit.CURRENCY.PER_MONTH)
    def bumped_income_cast_m(income_m: float) -> float:
        return income_m + cast_unit(value=100.0, unit=Unit.CURRENCY.PER_MONTH)

    fail_if_environment_units_are_inconsistent(
        env={"income_m": income_m, "bumped_income_m": bumped_income_cast_m},
        grouping_levels=GROUPING_LEVELS,
    )


def test_nonzero_literal_return_under_a_dimensioned_declaration_is_caught():
    """``return 25.0`` under a currency declaration is a hidden dimensioned
    constant: only ``0`` falls through (the eligibility guard); anything else
    is promoted to a parameter or tagged with ``cast_unit`` (GEP 10)."""

    @policy_function(unit=Unit.CURRENCY.PER_MONTH)
    def lump_m(is_exempt: bool, income_m: float) -> float:
        if is_exempt:
            return 25.0
        return income_m

    with pytest.raises(UnitConsistencyError, match="bare literal"):
        fail_if_environment_units_are_inconsistent(
            env={"is_exempt": is_exempt, "income_m": income_m, "lump_m": lump_m},
            grouping_levels=GROUPING_LEVELS,
        )

    @policy_function(unit=Unit.CURRENCY.PER_MONTH)
    def lump_cast_m(is_exempt: bool, income_m: float) -> float:
        if is_exempt:
            return cast_unit(value=25.0, unit=Unit.CURRENCY.PER_MONTH)
        return income_m

    fail_if_environment_units_are_inconsistent(
        env={"is_exempt": is_exempt, "income_m": income_m, "lump_m": lump_cast_m},
        grouping_levels=GROUPING_LEVELS,
    )


def test_path_cap_truncation_demands_opt_out(monkeypatch):
    """Exceeding the path cap demands an opt-out, never passes silently (GEP 10).

    With the cap lowered to 4, a body with three independent boolean gates has
    2**3 = 8 reachable paths; the explorer must stop and report rather than
    return success with most paths unchecked.
    """
    monkeypatch.setattr("ttsim.interface_dag_elements.unit_checks._MAX_PATHS", 4)

    @policy_input(unit=Unit.DIMENSIONLESS)
    def flag_a() -> bool:
        """A boolean gate."""

    @policy_input(unit=Unit.DIMENSIONLESS)
    def flag_b() -> bool:
        """A boolean gate."""

    @policy_input(unit=Unit.DIMENSIONLESS)
    def flag_c() -> bool:
        """A boolean gate."""

    @policy_function(unit=Unit.CURRENCY.PER_YEAR)
    def many_branches_y(
        wealth: float,
        tax_rate_y: float,
        flag_a: bool,
        flag_b: bool,
        flag_c: bool,
    ) -> float:
        out = wealth * tax_rate_y
        if flag_a:
            out = out * 2.0
        if flag_b:
            out = out * 3.0
        if flag_c:
            out = out * 4.0
        return out

    with pytest.raises(UnitConsistencyError, match="branch paths"):
        fail_if_environment_units_are_inconsistent(
            env={
                "wealth": wealth,
                "tax_rate_y": make_flow_rate(),
                "flag_a": flag_a,
                "flag_b": flag_b,
                "flag_c": flag_c,
                "many_branches_y": many_branches_y,
            },
            grouping_levels=GROUPING_LEVELS,
        )


def test_bare_nonzero_literal_in_ordering_is_caught():
    """A bare non-zero literal in an ordering comparison carries the other
    operand's unit, so it is rejected — promote it to a parameter (GEP 10)."""

    @policy_function(unit=Unit.DIMENSIONLESS)
    def rich(wealth: float) -> bool:
        return wealth > 1_000_000.0  # bug: the bound silently becomes CURRENCY

    with pytest.raises(UnitConsistencyError, match="rich"):
        fail_if_environment_units_are_inconsistent(
            env={"wealth": wealth, "rich": rich},
            grouping_levels=GROUPING_LEVELS,
        )


def test_zero_literal_and_dimensionless_self_in_ordering_pass():
    """The two allowed cases: a literal ``0`` (sign test) against any quantity,
    and any bare literal when the quantity itself is dimensionless (GEP 10)."""

    @policy_input(unit=Unit.DIMENSIONLESS)
    def some_rate() -> float:
        """A dimensionless share."""

    @policy_function(unit=Unit.DIMENSIONLESS)
    def has_wealth(wealth: float) -> bool:
        return wealth > 0.0  # 0 is the allowed inline literal

    @policy_function(unit=Unit.DIMENSIONLESS)
    def high_rate(some_rate: float) -> bool:
        return some_rate > 0.5  # self is dimensionless, so a bare literal is fine

    fail_if_environment_units_are_inconsistent(
        env={
            "wealth": wealth,
            "some_rate": some_rate,
            "has_wealth": has_wealth,
            "high_rate": high_rate,
        },
        grouping_levels=GROUPING_LEVELS,
    )


def test_opaque_return_demands_opt_out():
    """A body returning an opaque value (a tuple, a dataclass) is neither a
    checkable quantity nor a plain scalar, so it must opt out (GEP 10)."""

    @policy_function(unit=Unit.CURRENCY.PER_YEAR)
    def packaged_y(wealth: float, tax_rate_y: float) -> tuple[float, float]:
        return (wealth * tax_rate_y, wealth)  # opaque: a tuple, not a scalar

    with pytest.raises(UnitConsistencyError, match="packaged_y"):
        fail_if_environment_units_are_inconsistent(
            env={
                "wealth": wealth,
                "tax_rate_y": make_flow_rate(),
                "packaged_y": packaged_y,
            },
            grouping_levels=GROUPING_LEVELS,
        )


def test_adding_different_period_flows_is_caught():
    """``_m + _y`` is unit-blind at run time, so it must be flagged.

    pint would silently auto-convert ``CURRENCY / month + CURRENCY / year`` to
    the left operand's unit during the dry-run (matching the ``_m`` declaration)
    and hide the bug; the additive unit check rejects the non-equivalent operands
    before pint sees them (GEP 10).
    """

    @policy_function(unit=Unit.CURRENCY.PER_MONTH)
    def total_m(income_m: float, bonus_y: float) -> float:
        return income_m + bonus_y  # bug: adds a monthly and a yearly flow

    with pytest.raises(UnitConsistencyError, match="total_m"):
        fail_if_environment_units_are_inconsistent(
            env={"income_m": income_m, "bonus_y": bonus_y, "total_m": total_m},
            grouping_levels=GROUPING_LEVELS,
        )


def test_adding_stock_and_flow_is_caught():
    """A cross-dimension addition (stock + flow) raises a ``DimensionalityError``
    in pint, which the dry-run otherwise swallows; the additive check flags it."""

    @policy_function(unit=Unit.CURRENCY)
    def stock_plus_flow(wealth: float, income_m: float) -> float:
        return wealth + income_m  # bug: a stock plus a monthly flow

    with pytest.raises(UnitConsistencyError, match="stock_plus_flow"):
        fail_if_environment_units_are_inconsistent(
            env={
                "wealth": wealth,
                "income_m": income_m,
                "stock_plus_flow": stock_plus_flow,
            },
            grouping_levels=GROUPING_LEVELS,
        )


def test_comparing_different_period_flows_is_caught():
    """Ordering comparisons are unit-blind at run time too: comparing a monthly
    flow to a yearly one is flagged even when both return arms are consistent."""

    @policy_function(unit=Unit.CURRENCY.PER_MONTH)
    def gated_income_m(income_m: float, bonus_y: float) -> float:
        if income_m > bonus_y:  # bug: compares a monthly flow to a yearly one
            return income_m * 2.0
        return income_m

    with pytest.raises(UnitConsistencyError, match="gated_income_m"):
        fail_if_environment_units_are_inconsistent(
            env={
                "income_m": income_m,
                "bonus_y": bonus_y,
                "gated_income_m": gated_income_m,
            },
            grouping_levels=GROUPING_LEVELS,
        )


def test_adding_same_period_flows_does_not_false_positive():
    """Two operands in equivalent units add cleanly — no false positive."""

    @policy_function(unit=Unit.CURRENCY.PER_MONTH)
    def total_two_monthly_m(income_m: float, other_income_m: float) -> float:
        return income_m + other_income_m

    fail_if_environment_units_are_inconsistent(
        env={
            "income_m": income_m,
            "other_income_m": other_income_m,
            "total_two_monthly_m": total_two_monthly_m,
        },
        grouping_levels=GROUPING_LEVELS,
    )


def test_adding_bare_literal_does_not_false_positive():
    """Only ``0`` is allowed inline, so the ``x + 0.0`` guard stays lenient."""

    @policy_function(unit=Unit.CURRENCY.PER_MONTH)
    def income_floor_m(income_m: float) -> float:
        return income_m + 0.0

    fail_if_environment_units_are_inconsistent(
        env={"income_m": income_m, "income_floor_m": income_floor_m},
        grouping_levels=GROUPING_LEVELS,
    )


def test_dimensionless_inference_falls_back_to_declaration():
    @policy_function(unit=Unit.CURRENCY)
    def early_return(wealth: float) -> float:  # noqa: ARG001
        return 0.0

    fail_if_environment_units_are_inconsistent(
        env={"wealth": wealth, "early_return": early_return},
        grouping_levels=GROUPING_LEVELS,
    )


def test_undryrunnable_body_without_opt_out_is_rejected():
    # A body the dry-run cannot evaluate is not waved through silently (GEP 10):
    # the author must opt out explicitly with verify_units=False.
    @policy_function(unit=Unit.CURRENCY)
    def not_dry_runnable(wealth: float) -> float:
        return wealth.this_attribute_does_not_exist()  # ty: ignore[unresolved-attribute]

    with pytest.raises(UnitConsistencyError, match="verify_units=False"):
        fail_if_environment_units_are_inconsistent(
            env={"wealth": wealth, "not_dry_runnable": not_dry_runnable},
            grouping_levels=GROUPING_LEVELS,
        )


def test_undryrunnable_body_with_opt_out_passes():
    # The same body, explicitly opted out, is accepted: its declared unit stands.
    @policy_function(unit=Unit.CURRENCY, verify_units=False)
    def not_dry_runnable(wealth: float) -> float:
        return wealth.this_attribute_does_not_exist()  # ty: ignore[unresolved-attribute]

    fail_if_environment_units_are_inconsistent(
        env={"wealth": wealth, "not_dry_runnable": not_dry_runnable},
        grouping_levels=GROUPING_LEVELS,
    )


# ----------------------------------------------------------------------------
# Vectorized bodies: the xnp stand-in and the piecewise_polynomial / look_up /
# join primitives are screened at their edges (GEP 10). Their happy paths run
# end-to-end through the mettsim worked example (housing_benefits: minimum +
# look_up, payroll/property tax: piecewise, child_tax_credit: join), so these
# tests pin only what a silent shim regression would hide from that check —
# a screen that stops screening fails no test anywhere else.
# ----------------------------------------------------------------------------


def _make_lookup_param(**kwargs: Any) -> ConsecutiveIntLookupTableParam:
    return ConsecutiveIntLookupTableParam(
        value=ConsecutiveIntLookupTableParamValue(
            xnp=numpy,
            values_to_look_up=numpy.array([100.0, 200.0]),
            bases_to_subtract=numpy.array([2020]),
        ),
        start_date=_START,
        end_date=_END,
        **kwargs,
    )


def test_where_arms_are_screened_for_equivalence():
    """``xnp.where`` merges its two arms into one column, so they must carry
    equivalent units (as for ``+``): equivalent arms pass, mixed periods are
    flagged (GEP 10). No mettsim policy function uses ``where``, so both sides
    are pinned here."""

    @policy_function(
        unit=Unit.CURRENCY.PER_MONTH, vectorization_strategy="not_required"
    )
    def gated_m(
        is_exempt: BoolColumn,
        income_m: FloatColumn,
        other_income_m: FloatColumn,
        xnp: ModuleType,
    ) -> FloatColumn:
        return xnp.where(is_exempt, income_m, other_income_m)

    fail_if_environment_units_are_inconsistent(
        env={
            "is_exempt": is_exempt,
            "income_m": income_m,
            "other_income_m": other_income_m,
            "gated_m": gated_m,
        },
        grouping_levels=GROUPING_LEVELS,
    )

    @policy_function(
        unit=Unit.CURRENCY.PER_MONTH, vectorization_strategy="not_required"
    )
    def gated_buggy_m(
        is_exempt: BoolColumn,
        income_m: FloatColumn,
        bonus_y: FloatColumn,
        xnp: ModuleType,
    ) -> FloatColumn:
        return xnp.where(is_exempt, income_m, bonus_y)  # bug: arms mix periods

    with pytest.raises(UnitConsistencyError, match="gated_buggy_m"):
        fail_if_environment_units_are_inconsistent(
            env={
                "is_exempt": is_exempt,
                "income_m": income_m,
                "bonus_y": bonus_y,
                "gated_buggy_m": gated_buggy_m,
            },
            grouping_levels=GROUPING_LEVELS,
        )


def test_where_mixing_a_calendar_point_and_a_duration_is_caught():
    """``xnp.where`` runs no forward pint op, so a calendar-point arm gets no
    delegate-to-pint dispensation: an arm mix of a point and a duration is
    flagged (GEP 10)."""

    @policy_function(unit=Unit.CALENDAR_YEAR, vectorization_strategy="not_required")
    def year_or_age(
        is_exempt: BoolColumn,
        geburtsjahr: IntColumn,
        statutory_age: IntColumn,
        xnp: ModuleType,
    ) -> IntColumn:
        return xnp.where(is_exempt, geburtsjahr, statutory_age)  # bug: point/duration

    with pytest.raises(UnitConsistencyError, match="year_or_age"):
        fail_if_environment_units_are_inconsistent(
            env={
                "is_exempt": is_exempt,
                "geburtsjahr": geburtsjahr,
                "statutory_age": statutory_age,
                "year_or_age": year_or_age,
            },
            grouping_levels=GROUPING_LEVELS,
        )


def test_vectorized_minimum_with_a_bare_literal_bound_is_caught():
    """``xnp.minimum``/``maximum`` screen like an ordering comparison (they are
    the vectorized ``min``/``max``): a bare non-zero literal bound silently
    carries the other operand's unit, so promote it to a parameter (GEP 10)."""

    @policy_function(
        unit=Unit.CURRENCY.PER_MONTH, vectorization_strategy="not_required"
    )
    def capped_m(income_m: FloatColumn, xnp: ModuleType) -> FloatColumn:
        return xnp.minimum(income_m, 1_000.0)  # bug: a bare literal cap

    with pytest.raises(UnitConsistencyError, match="capped_m"):
        fail_if_environment_units_are_inconsistent(
            env={"income_m": income_m, "capped_m": capped_m},
            grouping_levels=GROUPING_LEVELS,
        )


def test_clip_with_a_bare_nonzero_literal_bound_is_caught():
    """``xnp.clip`` screens each bound as an ordering operand: a zero bound is
    the allowed sign test, a bare non-zero bound is rejected (GEP 10)."""

    @policy_function(
        unit=Unit.CURRENCY.PER_MONTH, vectorization_strategy="not_required"
    )
    def clipped_m(income_m: FloatColumn, xnp: ModuleType) -> FloatColumn:
        return xnp.clip(income_m, 0.0, 5_000.0)  # bug: a bare literal ceiling

    with pytest.raises(UnitConsistencyError, match="clipped_m"):
        fail_if_environment_units_are_inconsistent(
            env={"income_m": income_m, "clipped_m": clipped_m},
            grouping_levels=GROUPING_LEVELS,
        )


def test_unmodelled_xnp_op_demands_opt_out():
    """An xnp op the dry-run does not model falls through to raw NumPy and is
    reported as needing ``verify_units=False`` — never silently passed (GEP 10)."""

    @policy_function(
        unit=Unit.CURRENCY.PER_MONTH, vectorization_strategy="not_required"
    )
    def cumulative_m(income_m: FloatColumn, xnp: ModuleType) -> FloatColumn:
        return xnp.cumsum(income_m)

    with pytest.raises(UnitConsistencyError, match="verify_units=False"):
        fail_if_environment_units_are_inconsistent(
            env={"income_m": income_m, "cumulative_m": cumulative_m},
            grouping_levels=GROUPING_LEVELS,
        )


def test_schedule_call_with_wrong_domain_unit_is_caught():
    """A ``piecewise_polynomial`` call is screened at its edges (GEP 10): the
    argument must match the schedule's declared ``input_unit``."""
    schedule = _make_schedule_param(input_unit=CASTAR, output_unit=CASTAR_PER_YEAR)

    @policy_function(unit=Unit.CURRENCY.PER_YEAR)
    def levy_y(
        income_m: float, schedule: PiecewisePolynomialParamValue, xnp: ModuleType
    ) -> float:
        # bug: a monthly income into a wealth-domain schedule
        return piecewise_polynomial(x=income_m, parameters=schedule, xnp=xnp)

    with pytest.raises(UnitConsistencyError, match="levy_y"):
        fail_if_environment_units_are_inconsistent(
            env={"income_m": income_m, "schedule": schedule, "levy_y": levy_y},
            grouping_levels=GROUPING_LEVELS,
        )


def test_schedule_output_disagreeing_with_the_declaration_is_caught():
    """The call produces the schedule's ``output_unit``, which the declaration
    check verifies. Only this mismatch proves the output unit is real: were the
    shim to produce a bare stand-in, the dimensionless fallback would mask it
    in every happy-path run."""
    schedule = _make_schedule_param(input_unit=CASTAR, output_unit=CASTAR_PER_YEAR)

    @policy_function(unit=Unit.CURRENCY.PER_MONTH)
    def levy_m(
        wealth: float, schedule: PiecewisePolynomialParamValue, xnp: ModuleType
    ) -> float:
        # the schedule produces a yearly flow, but the node claims a monthly one
        return piecewise_polynomial(x=wealth, parameters=schedule, xnp=xnp)

    with pytest.raises(UnitConsistencyError, match="levy_m"):
        fail_if_environment_units_are_inconsistent(
            env={"wealth": wealth, "schedule": schedule, "levy_m": levy_m},
            grouping_levels=GROUPING_LEVELS,
        )


def test_lookup_call_with_wrong_domain_unit_is_caught():
    by_year = _make_lookup_param(
        input_unit=Unit.CALENDAR_YEAR, output_unit=CASTAR_PER_MONTH
    )

    @policy_function(unit=Unit.CURRENCY.PER_MONTH)
    def max_amount_m(
        statutory_age: int, by_year: ConsecutiveIntLookupTableParamValue
    ) -> float:
        # bug: an age (a duration) as the calendar-year key
        return by_year.look_up(statutory_age)

    with pytest.raises(UnitConsistencyError, match="max_amount_m"):
        fail_if_environment_units_are_inconsistent(
            env={
                "statutory_age": statutory_age,
                "by_year": by_year,
                "max_amount_m": max_amount_m,
            },
            grouping_levels=GROUPING_LEVELS,
        )


def test_join_target_level_disagreeing_with_the_declaration_is_caught():
    """A ``join`` gather hands on the target column's unit, grouping level
    included (GEP 10) — proven by contradiction: were the shim to drop the
    unit, this level mismatch could not be detected. (mettsim's checked join
    body gathers a dimensionless target, so it cannot pin this.)"""

    @policy_input(unit=Unit.CURRENCY.PER_MONTH.PER_FAM)
    def income_m_fam() -> float: ...

    @policy_function(
        unit=Unit.CURRENCY.PER_MONTH, vectorization_strategy="not_required"
    )
    def recipient_family_income_m(
        p_id: IntColumn,
        p_id_recipient: IntColumn,
        income_m_fam: FloatColumn,
        xnp: ModuleType,
    ) -> FloatColumn:
        # bug: the gathered target is the family's [fam] amount, but the node
        # declares a person-level one
        return join(
            foreign_key=p_id_recipient,
            primary_key=p_id,
            target=income_m_fam,
            value_if_foreign_key_is_missing=0.0,
            xnp=xnp,
        )

    with pytest.raises(UnitConsistencyError, match="recipient_family_income_m"):
        fail_if_environment_units_are_inconsistent(
            env={
                "p_id": p_id,
                "p_id_recipient": p_id_recipient,
                "income_m_fam": income_m_fam,
                "recipient_family_income_m": recipient_family_income_m,
            },
            grouping_levels=GROUPING_LEVELS,
        )


def test_body_consuming_identifier_infers_dimensionless_and_passes():
    # An identifier is a dimensionless quantity (GEP 10); a body multiplying it
    # infers dimensionless, which falls back to the declaration (no false
    # positive) rather than being skipped as a previously-exempt input.
    @policy_function(unit=Unit.CURRENCY)
    def depends_on_identifier(p_id: int) -> float:
        return p_id * 2.0

    fail_if_environment_units_are_inconsistent(
        env={"p_id": p_id, "depends_on_identifier": depends_on_identifier},
        grouping_levels=GROUPING_LEVELS,
    )


def test_concrete_mismatch_is_caught():
    @policy_function(unit=Unit.CURRENCY)
    def mislabelled(age_in_years: float) -> float:
        return age_in_years * 2.0

    @policy_input(unit=Unit.YEARS)
    def age_in_years() -> float:
        """An age."""

    with pytest.raises(UnitConsistencyError, match="mislabelled"):
        fail_if_environment_units_are_inconsistent(
            env={"age_in_years": age_in_years, "mislabelled": mislabelled},
            grouping_levels=GROUPING_LEVELS,
        )


def test_dict_param_subscripting_is_verifiable():
    """Per-leaf units make dict-consuming bodies dry-runnable (GEP 10, #121)."""
    schedule = DictParam(
        value={"child_amount_y": 100.0, "max_age": 18},
        unit={"child_amount_y": "CASTAR_PER_YEAR", "max_age": "YEARS"},
        start_date=_START,
        end_date=_END,
    )

    @policy_function(unit=Unit.CURRENCY.PER_YEAR)
    def claim_of_child_y(is_exempt: bool, schedule: dict) -> float:
        if is_exempt:
            return 0.0
        return schedule["child_amount_y"]

    fail_if_environment_units_are_inconsistent(
        env={
            "is_exempt": is_exempt,
            "schedule": schedule,
            "claim_of_child_y": claim_of_child_y,
        },
        grouping_levels=GROUPING_LEVELS,
    )

    @policy_function(unit=Unit.CURRENCY.PER_MONTH)
    def claim_of_child_m(is_exempt: bool, schedule: dict) -> float:
        if is_exempt:
            return 0.0
        return schedule["child_amount_y"]  # yearly leaf in a monthly node

    with pytest.raises(UnitConsistencyError, match="claim_of_child_m"):
        fail_if_environment_units_are_inconsistent(
            env={
                "is_exempt": is_exempt,
                "schedule": schedule,
                "claim_of_child_m": claim_of_child_m,
            },
            grouping_levels=GROUPING_LEVELS,
        )


def test_uniform_dict_param_is_subscriptable_in_dry_run():
    subsistence = DictParam(
        value={"per_spouse": 500.0},
        unit=CASTAR_PER_MONTH,
        start_date=_START,
        end_date=_END,
    )

    @policy_input(unit=Unit.DIMENSIONLESS)
    def number_of_adults() -> int:
        """A head count — dimensionless (GEP 10)."""

    @policy_function(unit=Unit.CURRENCY.PER_MONTH)
    def subsistence_income_m(subsistence: dict, number_of_adults: int) -> float:
        return subsistence["per_spouse"] * number_of_adults

    fail_if_environment_units_are_inconsistent(
        env={
            "subsistence": subsistence,
            "number_of_adults": number_of_adults,
            "subsistence_income_m": subsistence_income_m,
        },
        grouping_levels=GROUPING_LEVELS,
    )


# ----------------------------------------------------------------------------
# Structured param functions (unit=UNSET_UNIT): plucks are cast at the site
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class _AgeBounds:
    min_age: int
    max_age: int


@dataclass(frozen=True)
class _ChildRate:
    amount_m: float
    bounds: _AgeBounds


def make_raw_child_rate() -> RawParam:
    return RawParam(
        value={"amount_m": 100.0, "bounds": {"min_age": 0, "max_age": 18}},
        unit={
            "amount_m": "CASTAR_PER_MONTH",
            "bounds": {"min_age": "YEARS", "max_age": "YEARS"},
        },
        start_date=_START,
        end_date=_END,
    )


@param_function(unit=UNSET_UNIT)
def child_rate(raw_child_rate: RawParamValue) -> _ChildRate:
    """A structured builder: its output is a dataclass, not a quantity."""
    return _ChildRate(
        amount_m=raw_child_rate["amount_m"],
        bounds=_AgeBounds(
            min_age=raw_child_rate["bounds"]["min_age"],
            max_age=raw_child_rate["bounds"]["max_age"],
        ),
    )


@policy_input(unit=Unit.YEARS)
def age() -> int:
    """A duration in years (a person's age)."""


def test_missing_check_accepts_structured_param_function():
    fail_if_environment_units_are_missing(
        env={"raw_child_rate": make_raw_child_rate(), "child_rate": child_rate},
        grouping_levels=GROUPING_LEVELS,
    )


def test_missing_check_reports_uncovered_require_converter_leaf():
    raw = RawParam(
        value={"amount_m": 100.0, "bounds": {"min_age": 0, "max_age": 18}},
        unit={"amount_m": "CASTAR_PER_MONTH", "bounds": {"min_age": "YEARS"}},
        start_date=_START,
        end_date=_END,
    )
    with pytest.raises(
        UnitDefinitionError, match=r"raw_child_rate\[bounds\]\[max_age\]"
    ):
        fail_if_environment_units_are_missing(
            env={"raw_child_rate": raw},
            grouping_levels=GROUPING_LEVELS,
        )


def test_structured_plucks_with_casts_are_verifiable():
    """Casting each pluck keeps the rest of the body checked (GEP 10)."""

    @policy_function(unit=Unit.CURRENCY.PER_MONTH)
    def child_benefit_m(age: int, child_rate: _ChildRate) -> float:
        amount_m = cast_unit(value=child_rate.amount_m, unit=Unit.CURRENCY.PER_MONTH)
        max_age = cast_unit(value=child_rate.bounds.max_age, unit=Unit.YEARS)
        if age <= max_age:
            return amount_m
        return 0.0

    fail_if_environment_units_are_inconsistent(
        env={
            "age": age,
            "raw_child_rate": make_raw_child_rate(),
            "child_rate": child_rate,
            "child_benefit_m": child_benefit_m,
        },
        grouping_levels=GROUPING_LEVELS,
    )


def test_structured_pluck_used_without_cast_is_caught():
    @policy_function(unit=Unit.CURRENCY.PER_MONTH)
    def child_benefit_m(age: int, child_rate: _ChildRate) -> float:
        if age <= child_rate.bounds.max_age:  # bug: pluck used as a quantity
            return cast_unit(value=child_rate.amount_m, unit=Unit.CURRENCY.PER_MONTH)
        return 0.0

    with pytest.raises(UnitConsistencyError, match="cast_unit"):
        fail_if_environment_units_are_inconsistent(
            env={
                "age": age,
                "child_rate": child_rate,
                "child_benefit_m": child_benefit_m,
            },
            grouping_levels=GROUPING_LEVELS,
        )


def test_structured_pluck_returned_without_cast_is_caught():
    @policy_function(unit=Unit.CURRENCY.PER_MONTH)
    def child_benefit_m(child_rate: _ChildRate) -> float:
        return child_rate.amount_m  # bug: returned without stating its unit

    with pytest.raises(UnitConsistencyError, match="at the pluck"):
        fail_if_environment_units_are_inconsistent(
            env={"child_rate": child_rate, "child_benefit_m": child_benefit_m},
            grouping_levels=GROUPING_LEVELS,
        )


def test_structured_cast_too_coarse_fails_on_the_deeper_pluck():
    """A cast on a sub-structure yields a plain quantity, so the next deeper
    pluck fails loudly — a too-coarse cast can never silently mis-tag."""

    @policy_function(unit=Unit.CURRENCY.PER_MONTH)
    def child_benefit_m(age: int, child_rate: _ChildRate) -> float:
        bounds = cast_unit(value=child_rate.bounds, unit=Unit.YEARS)  # too coarse
        if age <= bounds.max_age:
            return cast_unit(value=child_rate.amount_m, unit=Unit.CURRENCY.PER_MONTH)
        return 0.0

    with pytest.raises(UnitConsistencyError, match="verify_units=False"):
        fail_if_environment_units_are_inconsistent(
            env={
                "age": age,
                "child_rate": child_rate,
                "child_benefit_m": child_benefit_m,
            },
            grouping_levels=GROUPING_LEVELS,
        )


# ----------------------------------------------------------------------------
# Annotated parameter dataclasses: fields state their units, plucks resolve
# (GEP 10)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class _AnnotatedAgeBounds:
    min_age: Annotated[int, Unit.YEARS]
    max_age: Annotated[int, Unit.YEARS]


@dataclass(frozen=True)
class _AnnotatedChildRate:
    amount_m: Annotated[float, Unit.CURRENCY.PER_MONTH]
    bounds: _AnnotatedAgeBounds


@param_function(unit=UNSET_UNIT)
def annotated_child_rate(raw_child_rate: RawParamValue) -> _AnnotatedChildRate:
    """A structured builder whose dataclass states each field's unit."""
    return _AnnotatedChildRate(
        amount_m=raw_child_rate["amount_m"],
        bounds=_AnnotatedAgeBounds(
            min_age=raw_child_rate["bounds"]["min_age"],
            max_age=raw_child_rate["bounds"]["max_age"],
        ),
    )


def test_annotated_fields_resolve_plucks_without_casts():
    @policy_function(unit=Unit.CURRENCY.PER_MONTH)
    def child_benefit_m(age: int, annotated_child_rate: _AnnotatedChildRate) -> float:
        if age <= annotated_child_rate.bounds.max_age:
            return annotated_child_rate.amount_m
        return 0.0

    fail_if_environment_units_are_inconsistent(
        env={
            "age": age,
            "raw_child_rate": make_raw_child_rate(),
            "annotated_child_rate": annotated_child_rate,
            "child_benefit_m": child_benefit_m,
        },
        grouping_levels=GROUPING_LEVELS,
    )


def test_annotated_pluck_misuse_is_caught():
    @policy_function(unit=Unit.CURRENCY.PER_MONTH)
    def child_benefit_m(age: int, annotated_child_rate: _AnnotatedChildRate) -> float:
        return annotated_child_rate.amount_m + age  # bug: money plus a duration

    with pytest.raises(UnitConsistencyError, match="non-equivalent units"):
        fail_if_environment_units_are_inconsistent(
            env={
                "age": age,
                "raw_child_rate": make_raw_child_rate(),
                "annotated_child_rate": annotated_child_rate,
                "child_benefit_m": child_benefit_m,
            },
            grouping_levels=GROUPING_LEVELS,
        )


@dataclass(frozen=True)
class _PartiallyAnnotatedRate:
    amount_m: Annotated[float, Unit.CURRENCY.PER_MONTH]
    max_age: int


@param_function(unit=UNSET_UNIT)
def partially_annotated_rate(raw_child_rate: RawParamValue) -> _PartiallyAnnotatedRate:
    return _PartiallyAnnotatedRate(
        amount_m=raw_child_rate["amount_m"],
        max_age=raw_child_rate["bounds"]["max_age"],
    )


def test_unannotated_field_keeps_the_cast_requirement():
    @policy_function(unit=Unit.CURRENCY.PER_MONTH)
    def benefit_m(age: int, partially_annotated_rate: _PartiallyAnnotatedRate) -> float:
        if age <= partially_annotated_rate.max_age:  # bug: opaque pluck, no cast
            return partially_annotated_rate.amount_m
        return 0.0

    with pytest.raises(UnitConsistencyError, match="cast_unit"):
        fail_if_environment_units_are_inconsistent(
            env={
                "age": age,
                "raw_child_rate": make_raw_child_rate(),
                "partially_annotated_rate": partially_annotated_rate,
                "benefit_m": benefit_m,
            },
            grouping_levels=GROUPING_LEVELS,
        )


@dataclass(frozen=True)
class _DriftingChildRate:
    amount_m: Annotated[float, Unit.YEARS]


@param_function(unit=UNSET_UNIT)
def drifting_child_rate(raw_child_rate: RawParamValue) -> _DriftingChildRate:
    return _DriftingChildRate(amount_m=raw_child_rate["amount_m"])


def test_field_annotation_drifting_from_the_unit_mapping_is_rejected():
    # The YAML mapping declares CASTAR_PER_MONTH for the `amount_m` leaf; the
    # field of the same path claims YEARS. The number would convert as money
    # and check as a duration — the drift check makes that loud (GEP 10).
    with pytest.raises(UnitConsistencyError, match="state the same unit"):
        fail_if_environment_units_are_inconsistent(
            env={
                "raw_child_rate": make_raw_child_rate(),
                "drifting_child_rate": drifting_child_rate,
            },
            grouping_levels=GROUPING_LEVELS,
        )


@dataclass(frozen=True)
class _ConcreteCurrencyRate:
    amount_m: Annotated[float, CASTAR_PER_MONTH]


@param_function(unit=UNSET_UNIT)
def concrete_currency_rate(raw_child_rate: RawParamValue) -> _ConcreteCurrencyRate:
    return _ConcreteCurrencyRate(amount_m=raw_child_rate["amount_m"])


def test_concrete_currency_field_annotation_is_rejected():
    with pytest.raises(UnitDefinitionError, match="concrete currency"):
        fail_if_environment_units_are_inconsistent(
            env={
                "raw_child_rate": make_raw_child_rate(),
                "concrete_currency_rate": concrete_currency_rate,
            },
            grouping_levels=GROUPING_LEVELS,
        )


@dataclass(frozen=True)
class _ContainerRate:
    amounts_m: Annotated[dict[str, float], Unit.CURRENCY.PER_MONTH]


@param_function(unit=UNSET_UNIT)
def container_rate(raw_child_rate: RawParamValue) -> _ContainerRate:
    return _ContainerRate(
        amounts_m={str(key): float(value) for key, value in raw_child_rate.items()}
    )


def test_container_field_annotation_is_rejected():
    with pytest.raises(UnitDefinitionError, match="scalar field"):
        fail_if_environment_units_are_inconsistent(
            env={
                "raw_child_rate": make_raw_child_rate(),
                "container_rate": container_rate,
            },
            grouping_levels=GROUPING_LEVELS,
        )


@param_function(unit=UNSET_UNIT)
def built_schedule(
    raw_schedule: RawParamValue, xnp: ModuleType
) -> PiecewisePolynomialParamValue:
    """A converter-built schedule: opaque to the dry-run (GEP 10)."""
    return PiecewisePolynomialParamValue(
        thresholds=xnp.asarray(raw_schedule["thresholds"]),
        intercepts=xnp.asarray(raw_schedule["intercepts"]),
        coefficients=xnp.asarray(raw_schedule["coefficients"]),
    )


def test_piecewise_call_on_converter_built_schedule_is_cast_at_the_call():
    @policy_function(unit=Unit.CURRENCY.PER_YEAR)
    def levy_y(
        bonus_y: float,
        built_schedule: PiecewisePolynomialParamValue,
        xnp: ModuleType,
    ) -> float:
        return cast_unit(
            value=piecewise_polynomial(x=bonus_y, parameters=built_schedule, xnp=xnp),
            unit=Unit.CURRENCY.PER_YEAR,
        )

    fail_if_environment_units_are_inconsistent(
        env={"bonus_y": bonus_y, "built_schedule": built_schedule, "levy_y": levy_y},
        grouping_levels=GROUPING_LEVELS,
    )


def test_piecewise_call_on_converter_built_schedule_without_cast_is_caught():
    @policy_function(unit=Unit.CURRENCY.PER_YEAR)
    def levy_y(
        bonus_y: float,
        built_schedule: PiecewisePolynomialParamValue,
        xnp: ModuleType,
    ) -> float:
        return piecewise_polynomial(x=bonus_y, parameters=built_schedule, xnp=xnp)

    with pytest.raises(UnitConsistencyError, match="at the pluck"):
        fail_if_environment_units_are_inconsistent(
            env={
                "bonus_y": bonus_y,
                "built_schedule": built_schedule,
                "levy_y": levy_y,
            },
            grouping_levels=GROUPING_LEVELS,
        )


# ----------------------------------------------------------------------------
# Converter-built schedules from input/output-unit require_converters screen
# like parameter-declared ones (GEP 10)
# ----------------------------------------------------------------------------


def make_raw_levy_schedule() -> RawParam:
    return RawParam(
        value={"top_rate": 0.2, "ceiling": 1000},
        input_unit="CASTAR",
        output_unit="CASTAR_PER_YEAR",
        start_date=_START,
        end_date=_END,
    )


@param_function(unit=UNSET_UNIT)
def levy_schedule(
    raw_levy_schedule: RawParamValue, xnp: ModuleType
) -> PiecewisePolynomialParamValue:
    """A converter with input/output units declared: consumers screen against them."""
    return PiecewisePolynomialParamValue(
        thresholds=xnp.asarray([0.0, raw_levy_schedule["ceiling"]]),
        intercepts=xnp.asarray([0.0, 0.0]),
        coefficients=xnp.asarray([[0.0], [raw_levy_schedule["top_rate"]]]),
    )


def test_axes_declared_schedule_screens_consumer_without_cast():
    @policy_function(unit=Unit.CURRENCY.PER_YEAR)
    def levy_y(
        wealth: float,
        levy_schedule: PiecewisePolynomialParamValue,
        xnp: ModuleType,
    ) -> float:
        return piecewise_polynomial(x=wealth, parameters=levy_schedule, xnp=xnp)

    fail_if_environment_units_are_inconsistent(
        env={
            "wealth": wealth,
            "raw_levy_schedule": make_raw_levy_schedule(),
            "levy_schedule": levy_schedule,
            "levy_y": levy_y,
        },
        grouping_levels=GROUPING_LEVELS,
    )


def test_axes_declared_schedule_rejects_wrong_domain_argument():
    @policy_function(unit=Unit.CURRENCY.PER_YEAR)
    def levy_y(
        age: int,
        levy_schedule: PiecewisePolynomialParamValue,
        xnp: ModuleType,
    ) -> float:
        return piecewise_polynomial(x=age, parameters=levy_schedule, xnp=xnp)

    with pytest.raises(UnitConsistencyError, match="non-equivalent units"):
        fail_if_environment_units_are_inconsistent(
            env={
                "age": age,
                "raw_levy_schedule": make_raw_levy_schedule(),
                "levy_schedule": levy_schedule,
                "levy_y": levy_y,
            },
            grouping_levels=GROUPING_LEVELS,
        )


def test_axes_declared_schedule_output_reaches_the_consumer_declaration():
    @policy_function(unit=Unit.CURRENCY.PER_MONTH)
    def levy_m(
        wealth: float,
        levy_schedule: PiecewisePolynomialParamValue,
        xnp: ModuleType,
    ) -> float:
        return piecewise_polynomial(x=wealth, parameters=levy_schedule, xnp=xnp)

    with pytest.raises(UnitConsistencyError, match="but its body infers"):
        fail_if_environment_units_are_inconsistent(
            env={
                "wealth": wealth,
                "raw_levy_schedule": make_raw_levy_schedule(),
                "levy_schedule": levy_schedule,
                "levy_m": levy_m,
            },
            grouping_levels=GROUPING_LEVELS,
        )


def test_axes_consumer_without_schedule_return_annotation_is_rejected():
    @param_function(unit=UNSET_UNIT)
    def levy_params(raw_levy_schedule: RawParamValue) -> dict[str, float]:
        return dict(raw_levy_schedule)

    with pytest.raises(UnitConsistencyError, match="annotated as returning"):
        fail_if_environment_units_are_inconsistent(
            env={
                "raw_levy_schedule": make_raw_levy_schedule(),
                "levy_params": levy_params,
            },
            grouping_levels=GROUPING_LEVELS,
        )


def test_axes_consumer_with_quantity_unit_is_rejected():
    @param_function(unit=Unit.CURRENCY.PER_YEAR)
    def levy_ceiling_y(raw_levy_schedule: RawParamValue) -> float:
        return raw_levy_schedule["ceiling"]

    with pytest.raises(UnitConsistencyError, match="UNSET_UNIT"):
        fail_if_environment_units_are_inconsistent(
            env={
                "raw_levy_schedule": make_raw_levy_schedule(),
                "levy_ceiling_y": levy_ceiling_y,
            },
            grouping_levels=GROUPING_LEVELS,
        )


def test_converter_of_two_input_output_unit_params_is_rejected():
    @param_function(unit=UNSET_UNIT)
    def merged_schedule(
        raw_levy_schedule: RawParamValue,
        raw_second_levy_schedule: RawParamValue,  # noqa: ARG001
        xnp: ModuleType,
    ) -> PiecewisePolynomialParamValue:
        return PiecewisePolynomialParamValue(
            thresholds=xnp.asarray([0.0, raw_levy_schedule["ceiling"]]),
            intercepts=xnp.asarray([0.0, 0.0]),
            coefficients=xnp.asarray([[0.0], [0.0]]),
        )

    with pytest.raises(UnitConsistencyError, match="exactly one"):
        fail_if_environment_units_are_inconsistent(
            env={
                "raw_levy_schedule": make_raw_levy_schedule(),
                "raw_second_levy_schedule": make_raw_levy_schedule(),
                "merged_schedule": merged_schedule,
            },
            grouping_levels=GROUPING_LEVELS,
        )


# ----------------------------------------------------------------------------
# Per-function body opt-out (verify_units=False)
# ----------------------------------------------------------------------------


def test_verify_units_false_skips_body_inference():
    # A body whose dry-run would otherwise flag a mismatch (stock * share is a
    # stock, not the declared yearly flow) is not dry-run when it opts out; the
    # declared unit still stands (GEP 10).
    @policy_function(unit=Unit.CURRENCY.PER_YEAR, verify_units=False)
    def amount_optout_y(wealth: float, tax_rate: float, is_exempt: bool) -> float:
        if is_exempt:
            return 0.0
        return wealth * tax_rate

    fail_if_environment_units_are_inconsistent(
        env={
            "wealth": wealth,
            "is_exempt": is_exempt,
            "tax_rate": make_dimensionless_rate(),
            "amount_optout_y": amount_optout_y,
        },
        grouping_levels=GROUPING_LEVELS,
    )


def test_verify_units_false_still_checks_consumers_against_declared_unit():
    # The opt-out is local to the body: the declared unit is still the edge
    # contract, so a consumer that misuses the producer is still caught.
    @policy_function(unit=Unit.CURRENCY.PER_YEAR, verify_units=False)
    def producer_y(wealth: float) -> float:
        return wealth.this_does_not_exist()  # ty: ignore[unresolved-attribute]

    @policy_function(unit=Unit.YEARS)
    def consumer(producer_y: float) -> float:
        return producer_y

    with pytest.raises(UnitConsistencyError, match="consumer"):
        fail_if_environment_units_are_inconsistent(
            env={"wealth": wealth, "producer_y": producer_y, "consumer": consumer},
            grouping_levels=GROUPING_LEVELS,
        )


# ----------------------------------------------------------------------------
# Aggregation decorators
# ----------------------------------------------------------------------------


def test_count_aggregation_auto_assigns_person_per_group():
    @agg_by_group_function(agg_type=AggType.COUNT)
    def number_of_individuals_fam(fam_id: int) -> int:
        """A head count per family — PERSON_COUNT_PER_FAM (GEP 10)."""

    assert number_of_individuals_fam.unit == Unit.PERSON_COUNT.PER_FAM


def test_any_aggregation_auto_assigns_dimensionless_at_target_level():
    # ANY/ALL yield a boolean at the target level (GEP 10): a group aggregation
    # auto-assigns DIMENSIONLESS_PER_<target>, the person leaf being implied.
    @agg_by_group_function(agg_type=AggType.ANY)
    def any_exempt_fam(is_exempt: bool, fam_id: int) -> bool:
        """Any member exempt."""

    assert any_exempt_fam.unit == Unit.DIMENSIONLESS.PER_FAM


def test_sum_aggregation_requires_explicit_unit():
    @agg_by_group_function(agg_type=AggType.SUM, unit=Unit.CURRENCY)
    def wealth_fam(wealth: float, fam_id: int) -> float:
        """Family wealth."""

    assert wealth_fam.unit is Unit.CURRENCY

    @agg_by_group_function(agg_type=AggType.SUM)
    def unannotated_fam(wealth: float, fam_id: int) -> float:
        """Missing its unit."""

    assert unannotated_fam.unit is UNSET_UNIT
    with pytest.raises(UnitDefinitionError, match="unannotated_fam"):
        fail_if_environment_units_are_missing(
            env={"unannotated_fam": unannotated_fam},
            grouping_levels=GROUPING_LEVELS,
        )


def test_sum_aggregation_over_booleans_declares_dimensionless():
    # A sum over a boolean column is a plain head count — declared
    # dimensionless via an explicit `unit=Unit.DIMENSIONLESS` (GEP 10).
    @agg_by_group_function(agg_type=AggType.SUM, unit=Unit.DIMENSIONLESS)
    def number_of_exempt_fam(is_exempt: bool, fam_id: int) -> int:
        """The number of exempt members per family."""

    assert number_of_exempt_fam.unit is Unit.DIMENSIONLESS
    fail_if_environment_units_are_missing(
        env={"number_of_exempt_fam": number_of_exempt_fam},
        grouping_levels=GROUPING_LEVELS,
    )


def test_aggregation_must_spell_the_derived_grouping_level():
    """An aggregation's declared unit must be precise and complete: a ``_fam`` sum
    declaring a bare ``CURRENCY`` (omitting the derived ``[fam]`` level) is rejected
    — there is no implicit matching of group levels, the author spells it (GEP 10)."""

    @agg_by_group_function(agg_type=AggType.SUM, unit=Unit.CURRENCY)
    def wealth_fam(wealth: float, fam_id: int) -> float:
        """A family sum that fails to spell its [fam] level."""

    with pytest.raises(UnitConsistencyError, match="wealth_fam"):
        fail_if_environment_units_are_inconsistent(
            env={"wealth": wealth, "wealth_fam": wealth_fam},
            grouping_levels=GROUPING_LEVELS,
        )


def test_aggregation_with_the_precise_derived_unit_passes():
    """The full, precise declaration — kind, period, *and* level — matches the
    derived unit and passes (GEP 10)."""

    @agg_by_group_function(agg_type=AggType.SUM, unit=Unit.CURRENCY.PER_FAM)
    def wealth_fam(wealth: float, fam_id: int) -> float:
        """Family wealth, level spelled."""

    fail_if_environment_units_are_inconsistent(
        env={"wealth": wealth, "wealth_fam": wealth_fam},
        grouping_levels=GROUPING_LEVELS,
    )


def test_aggregation_with_spelled_wrong_grouping_level_is_caught():
    """A spelled grouping level that contradicts the derivation is rejected: a
    ``_fam`` sum declaring ``CURRENCY_PER_KIN`` derives ``[fam]`` (GEP 10)."""

    @agg_by_group_function(agg_type=AggType.SUM, unit=Unit.CURRENCY.PER_KIN)
    def wealth_fam(wealth: float, fam_id: int) -> float:
        """A family sum mis-declared at the kin level."""

    with pytest.raises(UnitConsistencyError, match="wealth_fam"):
        fail_if_environment_units_are_inconsistent(
            env={"wealth": wealth, "wealth_fam": wealth_fam},
            grouping_levels=GROUPING_LEVELS,
        )


def test_aggregation_decorator_rejects_invalid_unit():
    # Strings are not tokens: the decorator's type contract only admits
    # `Unit` members (or None), enforced by the beartype claw.
    with pytest.raises(AggregationDefinitionError, match="unit"):

        @agg_by_group_function(agg_type=AggType.SUM, unit="kelvin")  # ty: ignore[invalid-argument-type]
        def bad_fam(wealth: float, fam_id: int) -> float:
            """Invalid unit."""


# ----------------------------------------------------------------------------
# Param functions
# ----------------------------------------------------------------------------


def test_param_function_unit_resolves_via_leaf_name_suffix():
    @param_function(unit=Unit.CURRENCY.PER_MONTH.PER_FAM)
    def max_amount_m_fam(policy_year: int) -> float:
        return float(policy_year)

    resolved = resolve_environment_units(
        env={"max_amount_m_fam": max_amount_m_fam},
        grouping_levels=GROUPING_LEVELS,
    )
    # The `_fam` suffix puts this flow at the family level (GEP 10).
    assert units_are_equivalent(
        left=_scalar_unit(resolved=resolved, qname="max_amount_m_fam"),
        right=parse_unit("CURRENCY / month / grouping_level_fam"),
    )


# ----------------------------------------------------------------------------
# Parameters must pin down their concrete currency (GEP 10)
# ----------------------------------------------------------------------------


def test_scalar_param_with_agnostic_currency_token_fails():
    threshold = ScalarParam(
        value=100.0,
        unit=Unit.CURRENCY,
        start_date=_START,
        end_date=_END,
    )
    with pytest.raises(UnitDefinitionError, match="pin down the concrete currency"):
        resolve_environment_units(
            env={"threshold": threshold}, grouping_levels=GROUPING_LEVELS
        )


def test_dict_param_leaf_with_agnostic_currency_token_fails():
    schedule = DictParam(
        value={"child_amount_y": 100.0},
        unit={"child_amount_y": "CURRENCY_PER_YEAR"},
        start_date=_START,
        end_date=_END,
    )
    with pytest.raises(UnitDefinitionError, match="pin down the concrete currency"):
        resolve_environment_units(
            env={"schedule": schedule}, grouping_levels=GROUPING_LEVELS
        )


def test_concrete_currency_token_resolves_like_agnostic_counterpart():
    # Union semantics: the concrete currency drives only the build-time
    # conversion, never the dimensionality.
    threshold = ScalarParam(
        value=100.0,
        unit=CASTAR,
        start_date=_START,
        end_date=_END,
    )
    resolved = resolve_environment_units(
        env={"threshold": threshold}, grouping_levels=GROUPING_LEVELS
    )
    assert units_are_equivalent(
        left=_scalar_unit(resolved=resolved, qname="threshold"),
        right=parse_unit("CURRENCY / grouping_level_person"),
    )


# ----------------------------------------------------------------------------
# Mapping parameters declare per-axis units (GEP 10)
# ----------------------------------------------------------------------------


def _make_schedule_param(**kwargs: Any) -> PiecewisePolynomialParam:
    return PiecewisePolynomialParam(
        value=PiecewisePolynomialParamValue(
            thresholds=numpy.array([0.0, 1.0]),
            intercepts=numpy.array([0.0]),
            coefficients=numpy.array([[0.1]]),
        ),
        start_date=_START,
        end_date=_END,
        **kwargs,
    )


def test_param_mapping_object_rejects_unit_declaration():
    with pytest.raises(UnitDefinitionError, match=r"input_unit.*instead of"):
        _make_schedule_param(unit=CASTAR_PER_YEAR)


def test_param_mapping_object_resolves_output_axis():
    # An income schedule: both axes are currency flows, each spelling its period.
    schedule = _make_schedule_param(
        input_unit=CASTAR_PER_YEAR,
        output_unit=CASTAR_PER_YEAR,
    )
    resolved = resolve_environment_units(
        env={"schedule": schedule}, grouping_levels=GROUPING_LEVELS
    )
    assert units_are_equivalent(
        left=_scalar_unit(resolved=resolved, qname="schedule"),
        right=parse_unit("CURRENCY / year / grouping_level_person"),
    )


def test_param_mapping_object_complete_input_axis_with_flow_output():
    # The property-tax shape: hectares in, a yearly currency flow out.
    schedule = _make_schedule_param(
        input_unit=Unit.HECTARE,
        output_unit=CASTAR_PER_YEAR,
    )
    resolved = resolve_environment_units(
        env={"schedule": schedule}, grouping_levels=GROUPING_LEVELS
    )
    assert units_are_equivalent(
        left=_scalar_unit(resolved=resolved, qname="schedule"),
        right=parse_unit("CURRENCY / year / grouping_level_person"),
    )


def test_param_mapping_object_rejects_agnostic_currency_axis():
    with pytest.raises(UnitDefinitionError, match="pin down the concrete currency"):
        resolve_environment_units(
            env={
                "schedule": _make_schedule_param(
                    input_unit=Unit.CURRENCY.PER_YEAR,
                    output_unit=CASTAR_PER_YEAR,
                )
            },
            grouping_levels=GROUPING_LEVELS,
        )


def test_param_mapping_object_missing_axis_units_are_reported():
    with pytest.raises(UnitDefinitionError) as excinfo:
        fail_if_environment_units_are_missing(
            env={"schedule": _make_schedule_param()},
            grouping_levels=GROUPING_LEVELS,
        )
    assert "schedule (input_unit)" in str(excinfo.value)
    assert "schedule (output_unit)" in str(excinfo.value)


def test_auto_generated_boolean_group_aggregate_passes_the_build():
    """The regression behind defect #1: requesting the group aggregate of a boolean
    auto-generates a SUM node, whose framework-minted token must match what the
    resolver derives (a head count). Before the fix the minter produced
    DIMENSIONLESS_PER_FAM while the resolver derived [person]/[fam], so the build
    rejected its own auto-assignment. It must now pass unchanged (GEP 10)."""

    @policy_function(leaf_name="is_adult", unit=Unit.DIMENSIONLESS)
    def is_adult() -> bool:
        return True

    aggs = create_agg_by_group_functions(
        column_functions={"is_adult": is_adult},
        qname_policy_environment={},
        input_columns=set(),
        tt_targets={"is_adult_fam"},
        grouping_levels=("fam",),
    )
    # No UnitConsistencyError: the minted token and the derived unit agree.
    fail_if_environment_units_are_inconsistent(
        env={"is_adult": is_adult, "is_adult_fam": aggs["is_adult_fam"]},
        grouping_levels=GROUPING_LEVELS,
    )


def test_opted_out_aggregation_declares_its_own_level():
    """``verify_units=False`` on an aggregation skips the declared-vs-derived check
    and resolves the *declared* unit, so a MEAN can be stated ``PER_KIN`` (a kin
    property) even though the algebra derives it as the person's (GEP 10)."""

    @policy_input(unit=Unit.CURRENCY)
    def wealth() -> float: ...

    @agg_by_group_function(
        agg_type=AggType.MEAN, unit=Unit.CURRENCY.PER_KIN, verify_units=False
    )
    def average_wealth_kin(kin_id: int, wealth: float) -> float: ...

    env = {"wealth": wealth, "average_wealth_kin": average_wealth_kin}
    resolved = resolve_environment_units(env=env, grouping_levels=GROUPING_LEVELS)
    assert units_are_equivalent(
        left=_scalar_unit(resolved=resolved, qname="average_wealth_kin"),
        right=parse_unit("CURRENCY / grouping_level_kin"),
    )
    # No declared-vs-derived rejection despite the MEAN deriving the person level.
    fail_if_environment_units_are_inconsistent(env=env, grouping_levels=GROUPING_LEVELS)


def test_aggregation_without_opt_out_still_rejects_a_wrong_level():
    """Without the opt-out, the same ``PER_KIN`` declaration on a MEAN is rejected —
    the opt-out is the only way to override the derivation (GEP 10)."""

    @policy_input(unit=Unit.CURRENCY)
    def wealth() -> float: ...

    @agg_by_group_function(agg_type=AggType.MEAN, unit=Unit.CURRENCY.PER_KIN)
    def average_wealth_kin(kin_id: int, wealth: float) -> float: ...

    with pytest.raises(UnitConsistencyError, match="average_wealth_kin"):
        fail_if_environment_units_are_inconsistent(
            env={"wealth": wealth, "average_wealth_kin": average_wealth_kin},
            grouping_levels=GROUPING_LEVELS,
        )


def test_count_and_sum_of_boolean_both_mint_head_counts():
    """A COUNT and a SUM over a boolean are both head counts (GEP 10): each
    resolves to [person]/[target], not DIMENSIONLESS."""

    @agg_by_group_function(agg_type=AggType.COUNT)
    def number_of_individuals_fam(fam_id: int) -> int: ...

    @policy_input(unit=Unit.DIMENSIONLESS)
    def is_adult() -> bool: ...

    @agg_by_group_function(agg_type=AggType.SUM, unit=Unit.DIMENSIONLESS)
    def number_of_adults_fam(fam_id: int, is_adult: bool) -> int: ...

    resolved = resolve_environment_units(
        env={
            "number_of_individuals_fam": number_of_individuals_fam,
            "is_adult": is_adult,
            "number_of_adults_fam": number_of_adults_fam,
        },
        grouping_levels=GROUPING_LEVELS,
    )
    head_count = parse_unit("grouping_level_person / grouping_level_fam")
    assert units_are_equivalent(
        left=_scalar_unit(resolved=resolved, qname="number_of_individuals_fam"),
        right=head_count,
    )
    assert units_are_equivalent(
        left=_scalar_unit(resolved=resolved, qname="number_of_adults_fam"),
        right=head_count,
    )


def test_per_capita_division_bridges_via_head_count():
    """A group total divided by a head count type-checks to a per-person amount:
    (CURRENCY/[fam]) / ([person]/[fam]) = CURRENCY/[person] (GEP 10)."""

    @agg_by_group_function(agg_type=AggType.COUNT)
    def number_of_individuals_fam(fam_id: int) -> int: ...

    @policy_input(unit=Unit.CURRENCY.PER_MONTH.PER_FAM)
    def rent_m_fam() -> float: ...

    @policy_function(unit=Unit.CURRENCY.PER_MONTH)
    def rent_per_head_m(rent_m_fam: float, number_of_individuals_fam: int) -> float:
        return rent_m_fam / number_of_individuals_fam

    env = {
        "number_of_individuals_fam": number_of_individuals_fam,
        "rent_m_fam": rent_m_fam,
        "rent_per_head_m": rent_per_head_m,
    }
    resolved = resolve_environment_units(env=env, grouping_levels=GROUPING_LEVELS)
    assert units_are_equivalent(
        left=_scalar_unit(resolved=resolved, qname="rent_per_head_m"),
        right=parse_unit("CURRENCY / month / grouping_level_person"),
    )
    # The [fam] cancels against the count's [person]/[fam] — no level mismatch.
    fail_if_environment_units_are_inconsistent(env=env, grouping_levels=GROUPING_LEVELS)


def test_cross_group_level_subtraction_in_a_body_is_caught():
    """Subtracting two different group levels is a level mismatch (GEP 10).

    ``income_m_fam`` is ``CURRENCY/month/[fam]`` and ``income_m_kin``
    ``CURRENCY/month/[kin]``. Broadcast replicates each onto persons but leaves
    the *unit* level untouched, so the subtraction stays ``[fam] - [kin]`` — the
    headline cross-level bug the dry-run must reject.
    """

    @policy_input(unit=Unit.CURRENCY.PER_MONTH.PER_FAM)
    def income_m_fam() -> float: ...

    @policy_input(unit=Unit.CURRENCY.PER_MONTH.PER_KIN)
    def income_m_kin() -> float: ...

    @policy_function(unit=Unit.CURRENCY.PER_MONTH.PER_FAM)
    def difference_m_fam(income_m_fam: float, income_m_kin: float) -> float:
        return income_m_fam - income_m_kin

    with pytest.raises(UnitConsistencyError, match="difference_m_fam"):
        fail_if_environment_units_are_inconsistent(
            env={
                "income_m_fam": income_m_fam,
                "income_m_kin": income_m_kin,
                "difference_m_fam": difference_m_fam,
            },
            grouping_levels=GROUPING_LEVELS,
        )


def test_person_versus_group_level_subtraction_in_a_body_is_caught():
    """A person-level quantity minus a group-level one is a mismatch (GEP 10).

    ``income_m`` carries no group suffix, so it is ``CURRENCY/month/[person]``;
    ``freibetrag_m_fam`` is ``CURRENCY/month/[fam]``. Combining them needs an
    explicit per-capita reconciliation, so the bare subtraction is rejected.
    """

    @policy_input(unit=Unit.CURRENCY.PER_MONTH)
    def income_m() -> float: ...

    @policy_input(unit=Unit.CURRENCY.PER_MONTH.PER_FAM)
    def freibetrag_m_fam() -> float: ...

    @policy_function(unit=Unit.CURRENCY.PER_MONTH)
    def difference_m(income_m: float, freibetrag_m_fam: float) -> float:
        return income_m - freibetrag_m_fam

    with pytest.raises(UnitConsistencyError, match="difference_m"):
        fail_if_environment_units_are_inconsistent(
            env={
                "income_m": income_m,
                "freibetrag_m_fam": freibetrag_m_fam,
                "difference_m": difference_m,
            },
            grouping_levels=GROUPING_LEVELS,
        )


def test_person_versus_group_level_ordering_comparison_is_caught():
    """An ordering comparison across levels is a mismatch (GEP 10).

    The canonical "person income below a group threshold" shape: ``income_m``
    (``[person]``) against ``schwelle_m_fam`` (``[fam]``). Ordering two
    non-equivalent quantities is rejected — a distinct dry-run path from `<`.
    """

    @policy_input(unit=Unit.CURRENCY.PER_MONTH)
    def income_m() -> float: ...

    @policy_input(unit=Unit.CURRENCY.PER_MONTH.PER_FAM)
    def schwelle_m_fam() -> float: ...

    @policy_function(unit=Unit.DIMENSIONLESS)
    def below_threshold(income_m: float, schwelle_m_fam: float) -> bool:
        return income_m < schwelle_m_fam

    with pytest.raises(UnitConsistencyError, match="below_threshold"):
        fail_if_environment_units_are_inconsistent(
            env={
                "income_m": income_m,
                "schwelle_m_fam": schwelle_m_fam,
                "below_threshold": below_threshold,
            },
            grouping_levels=GROUPING_LEVELS,
        )


def test_same_group_level_addition_in_a_body_passes():
    """Control: combining two quantities at the *same* group level is fine.

    Proves the cross-level checks above reject on the level mismatch, not on
    merely seeing a group suffix — ``a_m_fam + b_m_fam`` is ``[fam] + [fam]``.
    """

    @policy_input(unit=Unit.CURRENCY.PER_MONTH.PER_FAM)
    def a_m_fam() -> float: ...

    @policy_input(unit=Unit.CURRENCY.PER_MONTH.PER_FAM)
    def b_m_fam() -> float: ...

    @policy_function(unit=Unit.CURRENCY.PER_MONTH.PER_FAM)
    def total_m_fam(a_m_fam: float, b_m_fam: float) -> float:
        return a_m_fam + b_m_fam

    fail_if_environment_units_are_inconsistent(
        env={"a_m_fam": a_m_fam, "b_m_fam": b_m_fam, "total_m_fam": total_m_fam},
        grouping_levels=GROUPING_LEVELS,
    )


# ----------------------------------------------------------------------------
# Aggregations: declared unit must match the derived unit (GEP 10)
# ----------------------------------------------------------------------------


def test_sum_over_boolean_declared_dimensionless_is_caught():
    """A SUM over a boolean is a head count; declaring it DIMENSIONLESS is wrong.

    It derives `[person]/[fam]` (the persons the indicator is true for), so the
    declaration must be PERSON_COUNT_PER_FAM, not DIMENSIONLESS (GEP 10).
    """

    @policy_input(unit=Unit.DIMENSIONLESS)
    def adult() -> bool: ...

    @agg_by_group_function(agg_type=AggType.SUM, unit=Unit.DIMENSIONLESS)
    def number_of_adults_fam(adult: bool, fam_id: int) -> int: ...

    with pytest.raises(UnitConsistencyError, match="number_of_adults_fam"):
        fail_if_environment_units_are_inconsistent(
            env={"adult": adult, "number_of_adults_fam": number_of_adults_fam},
            grouping_levels=GROUPING_LEVELS,
        )


def test_sum_over_boolean_declared_person_per_group_passes():
    @policy_input(unit=Unit.DIMENSIONLESS)
    def adult() -> bool: ...

    @agg_by_group_function(agg_type=AggType.SUM, unit=Unit.PERSON_COUNT.PER_FAM)
    def number_of_adults_fam(adult: bool, fam_id: int) -> int: ...

    fail_if_environment_units_are_inconsistent(
        env={"adult": adult, "number_of_adults_fam": number_of_adults_fam},
        grouping_levels=GROUPING_LEVELS,
    )


def test_sum_of_currency_declared_with_wrong_kind_is_caught():
    """A SUM of a currency flow derives currency; declaring YEARS is rejected."""

    @policy_input(unit=Unit.CURRENCY.PER_MONTH)
    def income_m() -> float: ...

    @agg_by_group_function(agg_type=AggType.SUM, unit=Unit.YEARS)
    def income_m_fam(income_m: float, fam_id: int) -> float: ...

    with pytest.raises(UnitConsistencyError, match="income_m_fam"):
        fail_if_environment_units_are_inconsistent(
            env={"income_m": income_m, "income_m_fam": income_m_fam},
            grouping_levels=GROUPING_LEVELS,
        )


def test_max_over_level_carrying_source_carries_the_target_group_level():
    """Aggregations follow the *base*, not the agg type: a MAX of a level-carrying
    person income carries the target group level like a SUM. A `_fam` MAX is
    CURRENCY/month/[fam], not the source [person], and is declared `..._PER_FAM`.
    """

    @policy_input(unit=Unit.CURRENCY.PER_MONTH)
    def income_m() -> float: ...

    @agg_by_group_function(agg_type=AggType.MAX, unit=Unit.CURRENCY.PER_MONTH.PER_FAM)
    def income_max_m_fam(income_m: float, fam_id: int) -> float: ...

    env = {"income_m": income_m, "income_max_m_fam": income_max_m_fam}
    resolved = resolve_environment_units(env=env, grouping_levels=GROUPING_LEVELS)
    max_unit = _scalar_unit(resolved=resolved, qname="income_max_m_fam")
    # The MAX carries the target [fam] level, not the source [person] level.
    assert units_are_equivalent(
        left=max_unit,
        right=divide_by_grouping_level(
            unit=parse_unit("CURRENCY / month"), level="fam"
        ),
    )
    assert not units_are_equivalent(
        left=max_unit,
        right=divide_by_grouping_level(
            unit=parse_unit("CURRENCY / month"), level=PERSON_LEVEL
        ),
    )
    # The `_PER_FAM` declaration is consistent with what it derives.
    fail_if_environment_units_are_inconsistent(env=env, grouping_levels=GROUPING_LEVELS)
