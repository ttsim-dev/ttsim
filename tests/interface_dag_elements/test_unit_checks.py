"""Tests for the environment-level conservative unit checks (GEP 10, #121)."""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any

# Importing the mettsim package registers the castar (the base currency), so
# the concrete currency tokens exist and the params-must-pin-down-their-
# currency rule is active (GEP 10) regardless of test-collection order.
import mettsim.middle_earth  # noqa: F401
import numpy
import pint
import pytest

from ttsim.exceptions import (
    AggregationDefinitionError,
    UnitConsistencyError,
    UnitDefinitionError,
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
    Unit,
    agg_by_group_function,
    coerce_unit_token,
    group_creation_function,
    join,
    param_function,
    parse_unit,
    piecewise_polynomial,
    policy_function,
    policy_input,
    units_are_equivalent,
)
from ttsim.tt.param_objects import (
    ConsecutiveIntLookupTableParam,
    ConsecutiveIntLookupTableParamValue,
    DictParam,
    PiecewisePolynomialParam,
    PiecewisePolynomialParamValue,
    ScalarParam,
)
from ttsim.tt.units import (
    PERSON_LEVEL,
    divide_by_grouping_level,
)
from ttsim.typing import BoolColumn, FloatColumn, IntColumn

if TYPE_CHECKING:
    from types import ModuleType

GROUPING_LEVELS = ("fam", "kin")

_START = datetime.date(2020, 1, 1)
_END = datetime.date(2030, 12, 31)

# Parameters must pin down the concrete currency their numbers are written in
# (GEP 10); these are mettsim's concrete (castar) compositional spellings.
CASTAR_PER_YEAR = coerce_unit_token("CASTAR_PER_YEAR", where="test setup")
CASTAR_PER_MONTH = coerce_unit_token("CASTAR_PER_MONTH", where="test setup")
CASTAR = coerce_unit_token("CASTAR", where="test setup")


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


@group_creation_function()
def fam_id(p_id: IntColumn, xnp: object) -> IntColumn:  # noqa: ARG001
    """Group creation; auto-assigned DIMENSIONLESS (GEP 10)."""
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

    with pytest.raises(UnitConsistencyError, match="combines calendar points"):
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

    with pytest.raises(UnitConsistencyError, match="combines calendar points"):
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
    """A bare literal carries no unit, so an ``x + 0.0`` guard stays lenient."""

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
        left=_scalar_unit(resolved, "number_of_individuals_fam"), right=head_count
    )
    assert units_are_equivalent(
        left=_scalar_unit(resolved, "number_of_adults_fam"), right=head_count
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
        left=_scalar_unit(resolved, "rent_per_head_m"),
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


def test_max_resolves_to_the_target_group_level():
    """A MAX aggregation resolves to its *target* group level, like SUM (GEP 10,
    T8): the `_xx` suffix and the unit's grouping level are always in sync, so a
    `_fam` MAX of a person income is CURRENCY/month/[fam] — not the source
    [person] level. The correct declaration spells `..._PER_FAM`.
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
        right=divide_by_grouping_level(parse_unit("CURRENCY / month"), "fam"),
    )
    assert not units_are_equivalent(
        left=max_unit,
        right=divide_by_grouping_level(parse_unit("CURRENCY / month"), PERSON_LEVEL),
    )
    # The `_PER_FAM` declaration is consistent with what it derives.
    fail_if_environment_units_are_inconsistent(env=env, grouping_levels=GROUPING_LEVELS)
