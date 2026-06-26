"""Tests for the environment-level conservative unit checks (GEP 10, #121)."""

from __future__ import annotations

import datetime
from typing import Any

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
)
from ttsim.tt import (
    UNSET_UNIT,
    AggType,
    FKType,
    Unit,
    agg_by_group_function,
    coerce_unit_token,
    group_creation_function,
    param_function,
    parse_unit,
    policy_function,
    policy_input,
    units_are_equivalent,
)
from ttsim.tt.param_objects import (
    DictParam,
    PiecewisePolynomialParam,
    PiecewisePolynomialParamValue,
    ScalarParam,
)
from ttsim.typing import IntColumn

GROUPING_LEVELS = ("fam", "kin")

_START = datetime.date(2020, 1, 1)
_END = datetime.date(2030, 12, 31)

# Parameters must pin down the concrete currency their numbers are written in
# (GEP 10); these are mettsim's concrete variants of the agnostic tokens.
CASTAR_FLOW = coerce_unit_token("CASTAR_FLOW", where="test setup")
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


@policy_input()
def unannotated_income_y() -> float:
    """Missing its mandatory unit."""


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
    # A wealth-tax rate is a share per year: `DIMENSIONLESS_FLOW`, the period
    # supplied by the `tax_rate_y` name suffix (GEP 10).
    return ScalarParam(
        value=0.01,
        unit=Unit.DIMENSIONLESS_FLOW,
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


@policy_function(unit=Unit.CURRENCY_FLOW)
def amount_y(wealth: float, tax_rate_y: float, is_exempt: bool) -> float:
    """The wealth-tax pattern: stock times a per-year rate, guarded by an
    exemption. ``tax_rate_y`` is a share per year, so the product is a flow."""
    if is_exempt:
        return 0.0
    return wealth * tax_rate_y


@policy_function(unit=Unit.CURRENCY_FLOW)
def amount_buggy_y(wealth: float, tax_rate: float, is_exempt: bool) -> float:
    """The bug: ``tax_rate`` is a plain dimensionless share, so ``wealth *
    tax_rate`` is a stock, not the declared yearly flow."""
    if is_exempt:
        return 0.0
    return wealth * tax_rate


@policy_input(unit=Unit.CURRENCY_FLOW)
def income_m() -> float:
    """A monthly flow of currency (CURRENCY / month)."""


@policy_input(unit=Unit.CURRENCY_FLOW)
def other_income_m() -> float:
    """A second monthly flow of currency (CURRENCY / month)."""


@policy_input(unit=Unit.CURRENCY_FLOW)
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
    @policy_input()
    def some_id() -> int:
        """An identifier without `unit=`."""

    @policy_input()
    def some_flag() -> bool:
        """A boolean without `unit=`."""

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
        unit={"child_amount_y": "CASTAR_FLOW", "max_age": "YEARS"},
        start_date=_START,
        end_date=_END,
    )
    resolved = resolve_environment_units(
        env={"schedule": schedule}, grouping_levels=GROUPING_LEVELS
    )
    unit_tree = _unit_tree(resolved=resolved, qname="schedule")
    assert units_are_equivalent(
        left=unit_tree["child_amount_y"], right=parse_unit("CURRENCY / year")
    )
    assert units_are_equivalent(left=unit_tree["max_age"], right=parse_unit("year"))


def test_dict_param_leaf_suffix_must_coincide_with_reference_period():
    # Strict coincidence (GEP 10): wherever two period sources apply to the
    # same leaf they must agree — there is no precedence order.
    schedule = DictParam(
        value={"child_amount_y": 100.0},
        unit={"child_amount_y": "CASTAR_FLOW"},
        reference_period="Month",
        start_date=_START,
        end_date=_END,
    )
    with pytest.raises(UnitDefinitionError, match="coincide"):
        resolve_environment_units(
            env={"schedule": schedule}, grouping_levels=GROUPING_LEVELS
        )


def test_dict_param_leaf_suffix_agreeing_with_reference_period_passes():
    schedule = DictParam(
        value={"child_amount_y": 100.0},
        unit={"child_amount_y": "CASTAR_FLOW"},
        reference_period="Year",
        start_date=_START,
        end_date=_END,
    )
    resolved = resolve_environment_units(
        env={"schedule": schedule}, grouping_levels=GROUPING_LEVELS
    )
    assert units_are_equivalent(
        left=_unit_tree(resolved=resolved, qname="schedule")["child_amount_y"],
        right=parse_unit("CURRENCY / year"),
    )


def test_dict_param_suffixless_flow_leaf_takes_reference_period():
    # Integer keys cannot carry a suffix; the dict-level reference_period
    # supplies their period (GEP 10).
    amount_by_rank = DictParam(
        value={1: 250.0, 2: 250.0},
        unit={1: "CASTAR_FLOW", 2: "CASTAR_FLOW"},
        reference_period="Month",
        start_date=_START,
        end_date=_END,
    )
    resolved = resolve_environment_units(
        env={"amount_by_rank": amount_by_rank}, grouping_levels=GROUPING_LEVELS
    )
    assert units_are_equivalent(
        left=_unit_tree(resolved=resolved, qname="amount_by_rank")[1],
        right=parse_unit("CURRENCY / month"),
    )


def test_dict_param_suffixless_flow_leaf_without_period_source_fails():
    schedule = DictParam(
        value={"base_amount": 100.0},
        unit={"base_amount": "CASTAR_FLOW"},
        start_date=_START,
        end_date=_END,
    )
    with pytest.raises(UnitDefinitionError, match="no period source"):
        resolve_environment_units(
            env={"schedule": schedule}, grouping_levels=GROUPING_LEVELS
        )


def test_dict_param_dangling_reference_period_fails():
    schedule = DictParam(
        value={"max_age": 18},
        unit={"max_age": "YEARS"},
        reference_period="Year",
        start_date=_START,
        end_date=_END,
    )
    with pytest.raises(UnitDefinitionError, match="dangling"):
        resolve_environment_units(
            env={"schedule": schedule}, grouping_levels=GROUPING_LEVELS
        )


def test_dict_param_complete_token_on_suffixed_leaf_key_fails():
    schedule = DictParam(
        value={"amount_y": 100.0},
        unit={"amount_y": "CASTAR"},
        start_date=_START,
        end_date=_END,
    )
    with pytest.raises(UnitDefinitionError, match="denotes a flow"):
        resolve_environment_units(
            env={"schedule": schedule}, grouping_levels=GROUPING_LEVELS
        )


def test_dict_param_mixed_periods_via_suffixes_are_allowed():
    # Each flow leaf carries its own explicit suffix: nothing implicit.
    schedule = DictParam(
        value={"base_amount_m": 100.0, "annual_bonus_y": 50.0},
        unit={"base_amount_m": "CASTAR_FLOW", "annual_bonus_y": "CASTAR_FLOW"},
        start_date=_START,
        end_date=_END,
    )
    resolved = resolve_environment_units(
        env={"schedule": schedule}, grouping_levels=GROUPING_LEVELS
    )
    unit_tree = _unit_tree(resolved=resolved, qname="schedule")
    assert units_are_equivalent(
        left=unit_tree["base_amount_m"], right=parse_unit("CURRENCY / month")
    )
    assert units_are_equivalent(
        left=unit_tree["annual_bonus_y"], right=parse_unit("CURRENCY / year")
    )


def test_dict_param_missing_leaf_unit_is_reported():
    schedule = DictParam(
        value={"child_amount_y": 100.0, "max_age": 18},
        unit={"child_amount_y": "CASTAR_FLOW"},
        start_date=_START,
        end_date=_END,
    )
    with pytest.raises(UnitDefinitionError, match=r"schedule\[max_age\]"):
        fail_if_environment_units_are_missing(
            env={"schedule": schedule},
            grouping_levels=GROUPING_LEVELS,
        )


def test_scalar_param_rejects_reference_period():
    # A scalar parameter takes its period from a time suffix on its name, not
    # from reference_period (GEP 10).
    lump_sum = ScalarParam(
        value=100.0,
        unit=CASTAR_FLOW,
        reference_period="Year",
        start_date=_START,
        end_date=_END,
    )
    with pytest.raises(UnitDefinitionError, match="reference_period"):
        resolve_environment_units(
            env={"lump_sum_deduction_y": lump_sum}, grouping_levels=GROUPING_LEVELS
        )


def test_scalar_flow_param_resolves_via_name_suffix():
    lump_sum = ScalarParam(
        value=100.0,
        unit=CASTAR_FLOW,
        start_date=_START,
        end_date=_END,
    )
    resolved = resolve_environment_units(
        env={"lump_sum_deduction_y": lump_sum}, grouping_levels=GROUPING_LEVELS
    )
    assert units_are_equivalent(
        left=_scalar_unit(resolved=resolved, qname="lump_sum_deduction_y"),
        right=parse_unit("CURRENCY / year"),
    )


def test_scalar_param_with_suffixed_name_requires_flow_token():
    threshold = ScalarParam(
        value=100.0,
        unit=CASTAR,
        start_date=_START,
        end_date=_END,
    )
    with pytest.raises(UnitDefinitionError, match="denotes a flow"):
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

    @policy_input()
    def use_wealth() -> bool:
        """Boolean input selecting the stock branch."""

    @policy_function(unit=Unit.CURRENCY_FLOW)
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

    @policy_input()
    def use_wealth() -> bool:
        """Boolean input selecting between two flow branches."""

    @policy_function(unit=Unit.CURRENCY_FLOW)
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

    @policy_function(unit=Unit.CURRENCY_FLOW)
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

    @policy_function(unit=Unit.CURRENCY_FLOW)
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

    @policy_function(unit=Unit.CURRENCY_FLOW)
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

    @policy_function(unit=Unit.CURRENCY_FLOW)
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

    @policy_function(unit=Unit.CURRENCY_FLOW)
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

    @policy_function(unit=Unit.CURRENCY_FLOW)
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

    @policy_function(unit=Unit.CURRENCY_FLOW)
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

    @policy_function(unit=Unit.CURRENCY_FLOW)
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
        unit={"child_amount_y": "CASTAR_FLOW", "max_age": "YEARS"},
        start_date=_START,
        end_date=_END,
    )

    @policy_function(unit=Unit.CURRENCY_FLOW)
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

    @policy_function(unit=Unit.CURRENCY_FLOW)
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
        unit=CASTAR_FLOW,
        reference_period="Month",
        start_date=_START,
        end_date=_END,
    )

    @policy_input(unit=Unit.DIMENSIONLESS)
    def number_of_adults() -> int:
        """A head count — dimensionless (GEP 10)."""

    @policy_function(unit=Unit.CURRENCY_FLOW)
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
    @policy_function(unit=Unit.CURRENCY_FLOW, verify_units=False)
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
    @policy_function(unit=Unit.CURRENCY_FLOW, verify_units=False)
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


def test_count_aggregation_auto_assigns_headcount():
    @agg_by_group_function(agg_type=AggType.COUNT)
    def number_of_individuals_fam(fam_id: int) -> int:
        """A head count per family — the HEADCOUNT token (GEP 10)."""

    assert number_of_individuals_fam.unit is Unit.HEADCOUNT


def test_any_aggregation_auto_assigns_dimensionless():
    # ANY/ALL yield a boolean, which is a dimensionless quantity (GEP 10): the
    # aggregation auto-assigns Unit.DIMENSIONLESS.
    @agg_by_group_function(agg_type=AggType.ANY)
    def any_exempt_fam(is_exempt: bool, fam_id: int) -> bool:
        """Any member exempt."""

    assert any_exempt_fam.unit is Unit.DIMENSIONLESS


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
    @param_function(unit=Unit.CURRENCY_FLOW)
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
        unit={"child_amount_y": "CURRENCY_FLOW"},
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
        right=parse_unit("CURRENCY"),
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
        _make_schedule_param(unit=CASTAR_FLOW)


def test_param_mapping_object_resolves_output_axis():
    # An income schedule: both axes are currency flows sharing the single
    # reference_period.
    schedule = _make_schedule_param(
        input_unit=CASTAR_FLOW,
        output_unit=CASTAR_FLOW,
        reference_period="Year",
    )
    resolved = resolve_environment_units(
        env={"schedule": schedule}, grouping_levels=GROUPING_LEVELS
    )
    assert units_are_equivalent(
        left=_scalar_unit(resolved=resolved, qname="schedule"),
        right=parse_unit("CURRENCY / year"),
    )


def test_param_mapping_object_complete_input_axis_with_flow_output():
    # The property-tax shape: hectares in, a yearly currency flow out. The
    # reference_period feeds only the flow axis.
    schedule = _make_schedule_param(
        input_unit=Unit.HECTARES,
        output_unit=CASTAR_FLOW,
        reference_period="Year",
    )
    resolved = resolve_environment_units(
        env={"schedule": schedule}, grouping_levels=GROUPING_LEVELS
    )
    assert units_are_equivalent(
        left=_scalar_unit(resolved=resolved, qname="schedule"),
        right=parse_unit("CURRENCY / year"),
    )


def test_param_mapping_object_rejects_agnostic_currency_axis():
    with pytest.raises(UnitDefinitionError, match="pin down the concrete currency"):
        resolve_environment_units(
            env={
                "schedule": _make_schedule_param(
                    input_unit=Unit.CURRENCY_FLOW,
                    output_unit=CASTAR_FLOW,
                    reference_period="Year",
                )
            },
            grouping_levels=GROUPING_LEVELS,
        )


def test_param_mapping_object_rejects_dangling_reference_period():
    with pytest.raises(UnitDefinitionError, match="dangling reference_period"):
        resolve_environment_units(
            env={
                "schedule": _make_schedule_param(
                    input_unit=Unit.HECTARES,
                    output_unit=CASTAR,
                    reference_period="Year",
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

    @policy_input(unit=Unit.CURRENCY_FLOW)
    def rent_m_fam() -> float: ...

    @policy_function(unit=Unit.CURRENCY_FLOW)
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
