"""Tests for abstract unit inference through function bodies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any

import numpy
import pytest

from ttsim.exceptions import (
    UnitConsistencyError,
    UnitDefinitionError,
)
from ttsim.time_converters import m_to_y, per_m_to_per_y, y_to_m
from ttsim.tt import (
    UNSET_UNIT,
    InputOutputUnits,
    TTSIMUnit,
    cast_ttsim_unit,
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
from ttsim.typing import (
    BoolColumn,
    DatetimeColumn,
    FloatColumn,
    IntColumn,
    RawParamValue,
)
from ttsim.unit_resolution import _resolved_return_structure
from ttsim.unit_validation import (
    fail_if_environment_units_are_inconsistent,
    fail_if_environment_units_are_missing,
)

if TYPE_CHECKING:
    from types import ModuleType

from tests.test_unit_fixtures import (
    _END,
    _START,
    CASTAR,
    CASTAR_PER_MONTH,
    CASTAR_PER_YEAR,
    GROUPING_LEVELS,
    UNIT_SYSTEM,
    amount_y,
    bonus_y,
    geburtsjahr,
    income_m,
    is_exempt,
    make_flow_rate,
    p_id,
    p_id_recipient,
    statutory_age,
    wealth,
)


def make_dimensionless_rate() -> ScalarParam:
    # The bug the consistency check must catch: a plain dimensionless share with
    # no period, so `wealth * rate` is a stock where a flow node expects a flow.
    return ScalarParam(
        value=0.01,
        unit=TTSIMUnit.DIMENSIONLESS,
        start_date=_START,
        end_date=_END,
    )


@policy_input(unit=TTSIMUnit.CURRENCY)
def wealth_threshold() -> float:
    """A wealth threshold; a stock of currency. Comparing a quantity against a
    bare inline literal is rejected (GEP 10), so the bound is a named producer
    rather than a magic number."""


@policy_function(unit=TTSIMUnit.CURRENCY.PER_YEAR)
def amount_buggy_y(wealth: float, tax_rate: float, is_exempt: bool) -> float:
    """The bug: ``tax_rate`` is a plain dimensionless share, so ``wealth *
    tax_rate`` is a stock, not the declared yearly flow."""
    if is_exempt:
        return 0.0
    return wealth * tax_rate


@policy_input(unit=TTSIMUnit.CURRENCY.PER_MONTH)
def other_income_m() -> float:
    """A second monthly flow of currency (CURRENCY / month)."""


@policy_input(unit=TTSIMUnit.DIMENSIONLESS)
def geburtsmonat() -> int:
    """A month-of-year (1-12): a cyclic ordinal, not a calendar point (GEP 10)."""


@policy_input(unit=TTSIMUnit.MONTHS)
def months_paid() -> int:
    """A duration in months."""


# Conservative body verification


def _wealth_tax_env() -> dict:
    return {
        "wealth": wealth,
        "is_exempt": is_exempt,
        "tax_rate_y": make_flow_rate(),
        "amount_y": amount_y,
    }


def test_stock_times_rate_with_time_component_passes():
    fail_if_environment_units_are_inconsistent(
        env=_wealth_tax_env(), grouping_levels=GROUPING_LEVELS, unit_system=UNIT_SYSTEM
    )


def test_stock_times_rate_without_time_component_is_caught():
    """``wealth * tax_rate`` must resolve to a flow.

    ``tax_rate`` is a plain dimensionless share, so ``wealth * tax_rate`` is a
    stock while the node declares a yearly flow. The exemption branch returns
    0.0 (the explicit zero-literal exception); the path explorer exercises the
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
            unit_system=UNIT_SYSTEM,
        )


def test_multi_boolean_guard_bug_only_on_mixed_assignment_is_caught():
    """Branch exploration covers multi-boolean guards.

    The bug branch is reached only when ``is_exempt=False`` AND
    ``use_wealth=True`` — a mixed assignment (all-truthy hits the exempt
    return, all-falsy hits the correct flow branch). The path explorer walks
    every reachable combination.
    """

    @policy_input(unit=TTSIMUnit.DIMENSIONLESS)
    def use_wealth() -> bool:
        """Boolean input selecting the stock branch."""

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_YEAR)
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
            unit_system=UNIT_SYSTEM,
        )


def test_multi_boolean_guard_all_paths_consistent_passes():
    """The enumeration adds no false positive when every path is consistent."""

    @policy_input(unit=TTSIMUnit.DIMENSIONLESS)
    def use_wealth() -> bool:
        """Boolean input selecting between two flow branches."""

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_YEAR)
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
        unit_system=UNIT_SYSTEM,
    )


def test_numeric_driven_branch_bug_is_caught():
    """Numeric-driven branches are explored, not fixed.

    The bug lives on the high-wealth arm of a numeric comparison with no
    boolean input at all. A single representative magnitude would fix the
    comparison to one arm; the path explorer forces both arms and reaches the
    stock-returning branch. The threshold is a ``CURRENCY`` parameter, so the
    comparison itself is sound (equivalent units).
    """

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_YEAR)
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
            unit_system=UNIT_SYSTEM,
        )


def test_numeric_driven_branch_with_dimensionless_arm_does_not_false_positive():
    """The literal-zero guard arm stays unit-polymorphic under exploration.

    Forcing the high-wealth arm infers a dimensionless ``0.0`` (the conservative
    fallback), so the ubiquitous ``if ...: return 0.0`` pattern is not flagged.
    """

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_YEAR)
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
        unit_system=UNIT_SYSTEM,
    )


# Unit-check robustness: representative magnitudes are all 1.0, so a body may
# divide by a zero-magnitude difference of same-unit quantities or call the
# builtin ``round`` — both are dimensionally fine and must check cleanly.


def test_division_by_zero_magnitude_difference_infers_the_quotient_unit():
    """A body dividing by ``a - b`` of two equal-unit flows checks cleanly.

    The two monthly flows are each represented at magnitude 1.0, so their
    difference is 0; the unit check cares only about the quotient's *unit*
    (``currency / (currency/month) = months``), not that the magnitude is
    finite.
    """

    @policy_function(unit=TTSIMUnit.MONTHS)
    def months_to_close_the_gap(
        wealth: float, income_m: float, other_income_m: float
    ) -> float:
        return wealth / (income_m - other_income_m)

    fail_if_environment_units_are_inconsistent(
        env={
            "wealth": wealth,
            "income_m": income_m,
            "other_income_m": other_income_m,
            "months_to_close_the_gap": months_to_close_the_gap,
        },
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )


def test_division_by_zero_magnitude_difference_still_catches_a_unit_mismatch():
    """The zero-division fallback keeps the quotient's unit, so a wrong
    declaration is still caught: the ``months`` quotient is declared as a
    currency."""

    @policy_function(unit=TTSIMUnit.CURRENCY)
    def gap_declared_as_currency(
        wealth: float, income_m: float, other_income_m: float
    ) -> float:
        return wealth / (income_m - other_income_m)

    with pytest.raises(UnitConsistencyError, match="gap_declared_as_currency"):
        fail_if_environment_units_are_inconsistent(
            env={
                "wealth": wealth,
                "income_m": income_m,
                "other_income_m": other_income_m,
                "gap_declared_as_currency": gap_declared_as_currency,
            },
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_builtin_round_is_unit_preserving_and_checks_cleanly():
    """A body calling the builtin ``round`` on a monthly flow checks cleanly:
    ``round`` preserves the unit."""

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH)
    def rounded_income_m(income_m: float) -> float:
        return round(income_m)

    fail_if_environment_units_are_inconsistent(
        env={"income_m": income_m, "rounded_income_m": rounded_income_m},
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )


def test_builtin_round_preserves_the_unit_so_a_mismatch_is_caught():
    """``round`` is unit-preserving, so a body rounding a monthly flow but
    declaring a yearly one is still caught."""

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_YEAR)
    def rounded_income_mislabelled_y(income_m: float) -> float:
        return round(income_m)

    with pytest.raises(UnitConsistencyError, match="rounded_income_mislabelled_y"):
        fail_if_environment_units_are_inconsistent(
            env={
                "income_m": income_m,
                "rounded_income_mislabelled_y": rounded_income_mislabelled_y,
            },
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_astype_is_unit_preserving_and_checks_cleanly():
    """``astype`` re-types the magnitude and leaves the unit alone, so a body
    flooring a monthly flow and casting it to int keeps that unit."""

    @policy_function(
        unit=TTSIMUnit.CURRENCY.PER_MONTH, vectorization_strategy="not_required"
    )
    def floored_income_m(income_m: FloatColumn, xnp: ModuleType) -> IntColumn:
        return xnp.floor(income_m).astype(int)

    fail_if_environment_units_are_inconsistent(
        env={"income_m": income_m, "floored_income_m": floored_income_m},
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )


def test_astype_to_bool_drops_the_physical_dimension():
    """A cast to ``bool`` yields an indicator, so a body casting a monthly flow
    but declaring it as currency is caught."""

    @policy_function(
        unit=TTSIMUnit.CURRENCY.PER_MONTH, vectorization_strategy="not_required"
    )
    def has_income_mislabelled_m(
        income_m: FloatColumn,
        xnp: ModuleType,  # noqa: ARG001
    ) -> BoolColumn:
        return income_m.astype(bool)

    with pytest.raises(
        UnitConsistencyError, match=r"has_income_mislabelled_m.*dimensionless"
    ):
        fail_if_environment_units_are_inconsistent(
            env={
                "income_m": income_m,
                "has_income_mislabelled_m": has_income_mislabelled_m,
            },
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_astype_to_bool_is_a_dimensionless_indicator():
    """A cast to ``bool`` declared ``DIMENSIONLESS`` checks cleanly."""

    @policy_function(
        unit=TTSIMUnit.DIMENSIONLESS, vectorization_strategy="not_required"
    )
    def has_income(
        income_m: FloatColumn,
        xnp: ModuleType,  # noqa: ARG001
    ) -> BoolColumn:
        return income_m.astype(bool)

    fail_if_environment_units_are_inconsistent(
        env={"income_m": income_m, "has_income": has_income},
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )


def test_astype_to_an_unsupported_dtype_demands_an_opt_out():
    """A dtype with no unit reading leaves the body un-evaluable."""

    @policy_function(
        unit=TTSIMUnit.CURRENCY.PER_MONTH, vectorization_strategy="not_required"
    )
    def as_dates_m(
        income_m: FloatColumn,
        xnp: ModuleType,  # noqa: ARG001
    ) -> DatetimeColumn:
        return income_m.astype("datetime64[D]")

    with pytest.raises(UnitConsistencyError, match="verify_units=False"):
        fail_if_environment_units_are_inconsistent(
            env={"income_m": income_m, "as_dates_m": as_dates_m},
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_astype_accepts_the_keyword_dtype_form():
    """Both backends accept `.astype(dtype=...)`, so the unit check must too."""

    @policy_function(
        unit=TTSIMUnit.CURRENCY.PER_MONTH, vectorization_strategy="not_required"
    )
    def floored_income_kwarg_m(income_m: FloatColumn, xnp: ModuleType) -> IntColumn:
        return xnp.floor(income_m).astype(dtype=int)

    fail_if_environment_units_are_inconsistent(
        env={"income_m": income_m, "floored_income_kwarg_m": floored_income_kwarg_m},
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )


def test_astype_preserves_the_unit_so_a_mismatch_is_caught():
    """``astype`` is unit-preserving, so a body flooring a monthly flow but
    declaring a yearly one is still caught."""

    @policy_function(
        unit=TTSIMUnit.CURRENCY.PER_YEAR, vectorization_strategy="not_required"
    )
    def floored_income_mislabelled_y(
        income_m: FloatColumn, xnp: ModuleType
    ) -> IntColumn:
        return xnp.floor(income_m).astype(int)

    with pytest.raises(UnitConsistencyError, match="floored_income_mislabelled_y"):
        fail_if_environment_units_are_inconsistent(
            env={
                "income_m": income_m,
                "floored_income_mislabelled_y": floored_income_mislabelled_y,
            },
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


# Calendar points vs durations (GEP 10, S1)


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
    ``YEARS``; the unit check accepts it through pint's offset algebra."""

    @policy_function(unit=TTSIMUnit.YEARS)
    def age(policy_year: int, geburtsjahr: int) -> int:
        return policy_year - geburtsjahr

    fail_if_environment_units_are_inconsistent(
        env={
            "policy_year": _policy_year(),
            "geburtsjahr": geburtsjahr,
            "age": age,
        },
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )


def test_duration_shifts_a_calendar_point_to_a_calendar_point():
    """A ``YEARS`` duration added to a calendar year yields a calendar year
    (``geburtsjahr + statutory_age``), declared ``CALENDAR_YEAR``."""

    @policy_function(unit=TTSIMUnit.CALENDAR_YEAR)
    def retirement_year(geburtsjahr: int, statutory_age: int) -> int:
        return geburtsjahr + statutory_age

    fail_if_environment_units_are_inconsistent(
        env={
            "geburtsjahr": geburtsjahr,
            "statutory_age": statutory_age,
            "retirement_year": retirement_year,
        },
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )


def test_adding_two_calendar_points_is_caught():
    """``point + point`` has no affine meaning; pint refuses it and the unit
    check reports a calendar misuse (GEP 10)."""

    @policy_function(unit=TTSIMUnit.CALENDAR_YEAR)
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
            unit_system=UNIT_SYSTEM,
        )


def test_scaling_a_calendar_point_is_caught():
    """A calendar point cannot be scaled (it is affine, not multiplicative)."""

    @policy_function(unit=TTSIMUnit.CALENDAR_YEAR)
    def doubled(geburtsjahr: int) -> int:
        return geburtsjahr * 2  # bug: scaling a calendar point

    with pytest.raises(UnitConsistencyError, match="combines a calendar point"):
        fail_if_environment_units_are_inconsistent(
            env={
                "geburtsjahr": geburtsjahr,
                "doubled": doubled,
            },
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_calendar_point_difference_declared_as_a_calendar_year_is_caught():
    """The S1 failure mode: a year difference is a duration, so declaring the
    result ``CALENDAR_YEAR`` (a point) is inconsistent and is flagged."""

    @policy_function(unit=TTSIMUnit.CALENDAR_YEAR)
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
            unit_system=UNIT_SYSTEM,
        )


def test_ordering_a_calendar_point_against_another_unit_is_caught():
    """An ordering runs no forward pint op, so a calendar point gets no
    delegate-to-pint dispensation there: ordered against anything but a
    same-axis point, it is a unit mix (GEP 10)."""

    @policy_function(unit=TTSIMUnit.DIMENSIONLESS)
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
            unit_system=UNIT_SYSTEM,
        )


def test_ordering_a_calendar_point_against_a_duration_is_caught():
    """A point and a duration share ``[time]`` but not an algebra: equivalence
    decides points by identity, so ordering them is a unit mix (GEP 10)."""

    @policy_function(unit=TTSIMUnit.DIMENSIONLESS)
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
            unit_system=UNIT_SYSTEM,
        )


def test_ordering_two_same_axis_calendar_points_passes():
    """Ordering two points on the same calendar axis is sound
    (``geburtsjahr <= policy_year``): identical units, so the ordering screen
    passes without any calendar dispensation. The comparison of two bare points
    yields a bare, individual boolean (GEP 10)."""

    @policy_function(unit=TTSIMUnit.DIMENSIONLESS)
    def born_by_policy_year(policy_year: int, geburtsjahr: int) -> bool:
        return geburtsjahr <= policy_year

    fail_if_environment_units_are_inconsistent(
        env={
            "policy_year": _policy_year(),
            "geburtsjahr": geburtsjahr,
            "born_by_policy_year": born_by_policy_year,
        },
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )


def test_subtracting_calendar_points_of_different_axes_is_caught():
    """Two *different* offset units of the same [time] dimension are the trap:
    pint subtracts ``calendar_month - calendar_year`` with a silent /12 while the
    run-time subtraction is raw and unconverted, so a cross-axis point - point is
    rejected rather than delegated to pint (GEP 10)."""

    @policy_input(unit=TTSIMUnit.CALENDAR_MONTH)
    def some_calendar_month() -> int:
        """A month point on the calendar."""

    @policy_function(unit=TTSIMUnit.MONTHS)
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
            unit_system=UNIT_SYSTEM,
        )


def test_adding_a_currency_to_a_calendar_point_is_reported_as_a_calendar_misuse():
    """A calendar point plus a foreign dimension raises pint ``DimensionalityError``;
    it is a genuine calendar bug, so it reports as a calendar misuse rather than
    falling into the blanket ``verify_units=False`` advice (GEP 10)."""

    @policy_function(unit=TTSIMUnit.CALENDAR_YEAR)
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
            unit_system=UNIT_SYSTEM,
        )


def test_flow_time_converter_body_passes():
    """``per_m_to_per_y`` rebases a monthly flow to a yearly one; the unit check
    models the period rebase, so the body checks against its ``_y`` declaration with
    no opt-out (GEP 10, time converters)."""

    @policy_input(unit=TTSIMUnit.CURRENCY.PER_MONTH)
    def betrag_m() -> float:
        """A monthly flow."""

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_YEAR)
    def betrag_y(betrag_m: float) -> float:
        return per_m_to_per_y(betrag_m)

    fail_if_environment_units_are_inconsistent(
        env={"betrag_m": betrag_m, "betrag_y": betrag_y},
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )


def test_duration_time_converter_body_passes():
    """``m_to_y`` rebases a ``MONTHS`` duration to ``YEARS``; the classic
    ``m_to_y(months) >= grenze`` shape checks without an opt-out (GEP 10)."""

    @policy_input(unit=TTSIMUnit.MONTHS)
    def wartezeit() -> int:
        """A waiting time in months (a duration)."""

    @policy_input(unit=TTSIMUnit.YEARS)
    def wartezeitgrenze() -> int:
        """A threshold in years."""

    @policy_function(unit=TTSIMUnit.DIMENSIONLESS)
    def wartezeit_erfüllt(wartezeit: int, wartezeitgrenze: int) -> bool:
        return m_to_y(wartezeit) >= wartezeitgrenze

    fail_if_environment_units_are_inconsistent(
        env={
            "wartezeit": wartezeit,
            "wartezeitgrenze": wartezeitgrenze,
            "wartezeit_erfüllt": wartezeit_erfüllt,
        },
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )


def test_wrong_direction_time_converter_is_caught():
    """A converter for the wrong period rebases to a unit that disagrees with the
    declaration, so the misuse is caught rather than silently passed (GEP 10)."""

    @policy_input(unit=TTSIMUnit.MONTHS)
    def wartezeit() -> int:
        """A duration in months."""

    @policy_function(unit=TTSIMUnit.YEARS)
    def nonsense(wartezeit: int) -> float:
        return y_to_m(wartezeit)  # wrong: a MONTHS duration fed to a year->month rebase

    with pytest.raises(UnitConsistencyError, match="nonsense"):
        fail_if_environment_units_are_inconsistent(
            env={"wartezeit": wartezeit, "nonsense": nonsense},
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_month_date_nodes_are_cyclic_ordinals():
    """``policy_month`` carries a month-of-year (1-12): a cyclic ordinal, hence
    ``DIMENSIONLESS`` (GEP 10), so comparing it to another ordinal is plain
    dimensionless arithmetic."""

    @policy_function(unit=TTSIMUnit.DIMENSIONLESS)
    def had_birthday(policy_month: int, geburtsmonat: int) -> bool:
        return policy_month >= geburtsmonat

    fail_if_environment_units_are_inconsistent(
        env={
            "policy_month": _policy_month(),
            "geburtsmonat": geburtsmonat,
            "had_birthday": had_birthday,
        },
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )


def test_month_date_node_shifted_by_a_duration_is_caught():
    """Shifting the cyclic ``policy_month`` by a months duration wraps at run
    time — the silent fold the ordinal/point split exists to catch. As a
    dimensionless ordinal it does not add to a ``MONTHS`` duration (GEP 10)."""

    @policy_function(unit=TTSIMUnit.DIMENSIONLESS)
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
            unit_system=UNIT_SYSTEM,
        )


def test_error_names_the_failing_branch():
    """A branch-confined failure names the branch in the body's own terms and
    reports the other combinations clean (GEP 10)."""

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH)
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
            unit_system=UNIT_SYSTEM,
        )


def test_error_names_a_comparison_driven_branch():
    """A branch decided by a comparison is named by that comparison's operands
    (GEP 10)."""

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH)
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
            unit_system=UNIT_SYSTEM,
        )


def test_boolean_body_bad_comparison_is_caught():
    """A boolean-returning body is unit-checked like any other (GEP 10).

    Its truth-value output carries no unit, but the comparison inside it does:
    ``wealth`` is a ``CURRENCY`` stock and ``bonus_y`` a ``CURRENCY / year``
    flow, so the ``>=`` mixes non-equivalent units. A boolean output does not
    exempt the body — the comparison inside it is checked all the same.
    """

    @policy_function(unit=TTSIMUnit.DIMENSIONLESS)
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
            unit_system=UNIT_SYSTEM,
        )


def test_boolean_body_with_logical_ops_passes():
    """A boolean body combining clean truth values with ``&``/``|``/``~`` is not
    a false positive: every operand is a dimensionless truth value."""

    @policy_function(unit=TTSIMUnit.DIMENSIONLESS)
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
        unit_system=UNIT_SYSTEM,
    )


def test_logical_op_on_unit_carrying_operand_is_caught():
    """A logical operator applied to a real quantity (not a truth value) is a
    bug the run-time arrays would silently swallow, so the unit check rejects it.

    ``age`` is a ``YEARS`` quantity, so ``age & is_exempt`` ANDs a duration into
    a logical combination — caught on either side via the reflected dunders.
    """

    @policy_input(unit=TTSIMUnit.YEARS)
    def age() -> int:
        """An age in years (a ``YEARS`` quantity)."""

    @policy_function(unit=TTSIMUnit.DIMENSIONLESS)
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
            unit_system=UNIT_SYSTEM,
        )


def test_not_of_a_leveled_boolean_keeps_its_level():
    """``not`` on a leveled boolean keeps its level, exactly as ``~`` does (GEP 10):
    ``flag_fam and not other_flag_fam`` stays fam-level and matches the fam-level
    declaration — no spurious level error."""

    @policy_input(unit=TTSIMUnit.DIMENSIONLESS.PER_FAM)
    def flag_fam() -> bool:
        """A fam-level indicator."""

    @policy_input(unit=TTSIMUnit.DIMENSIONLESS.PER_FAM)
    def other_flag_fam() -> bool:
        """Another fam-level indicator."""

    @policy_function(leaf_name="combined_fam", unit=TTSIMUnit.DIMENSIONLESS.PER_FAM)
    def combined_fam(flag_fam: bool, other_flag_fam: bool) -> bool:
        return flag_fam and not other_flag_fam

    fail_if_environment_units_are_inconsistent(
        env={
            "flag_fam": flag_fam,
            "other_flag_fam": other_flag_fam,
            "combined_fam": combined_fam,
        },
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )


def test_not_of_a_non_boolean_quantity_is_caught():
    """``not`` on a non-boolean (a currency) is a bug that ``~`` catches; its scalar
    spelling must too — the unit check models ``not`` as ``logical_not`` (GEP 10)."""

    @policy_function(unit=TTSIMUnit.DIMENSIONLESS)
    def flag(income_m: float) -> bool:
        return not income_m  # bug: `not` on a currency

    with pytest.raises(UnitConsistencyError, match="non-boolean"):
        fail_if_environment_units_are_inconsistent(
            env={"income_m": income_m, "flag": flag},
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_dimensioned_value_as_a_branch_condition_is_caught():
    """Only a boolean may control a branch: an `if` on a currency stock is a bug
    the unit check reports, exactly as `not` on a currency is (GEP 10)."""

    @policy_function(unit=TTSIMUnit.CURRENCY)
    def remaining_wealth(wealth: float) -> float:
        if wealth:  # bug: a stock is not a truth value
            return wealth
        return 0.0

    with pytest.raises(UnitConsistencyError, match="truth value"):
        fail_if_environment_units_are_inconsistent(
            env={"wealth": wealth, "remaining_wealth": remaining_wealth},
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_dimensioned_value_as_a_conditional_expression_condition_is_caught():
    """A conditional expression's condition is a truth context like any other, so
    a dimensioned selector is caught there too (GEP 10)."""

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH)
    def gated_income_m(income_m: float, wealth: float) -> float:
        return income_m if wealth else 0.0  # bug: a stock is not a truth value

    with pytest.raises(UnitConsistencyError, match="truth value"):
        fail_if_environment_units_are_inconsistent(
            env={
                "income_m": income_m,
                "wealth": wealth,
                "gated_income_m": gated_income_m,
            },
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_leveled_boolean_as_a_branch_condition_passes():
    """A group-level indicator is a truth value, so it may control a branch — the
    truth-context screen rejects physical content, not a grouping level (GEP 10)."""

    @policy_input(unit=TTSIMUnit.DIMENSIONLESS.PER_FAM)
    def is_exempt_fam() -> bool:
        """A fam-level indicator."""

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH)
    def gated_income_m(income_m: float, is_exempt_fam: bool) -> float:
        if is_exempt_fam:
            return 0.0
        return income_m

    fail_if_environment_units_are_inconsistent(
        env={
            "income_m": income_m,
            "is_exempt_fam": is_exempt_fam,
            "gated_income_m": gated_income_m,
        },
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )


def test_boolean_body_at_correct_group_level_passes():
    """A fam-level predicate comparing fam-level quantities infers ``1 / [fam]``,
    matching its ``_fam`` name (GEP 10)."""

    @policy_input(unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_FAM)
    def income_m_fam() -> float:
        """Family income."""

    @policy_input(unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_FAM)
    def threshold_m_fam() -> float:
        """Family subsistence threshold."""

    @policy_function(
        leaf_name="requirement_fulfilled_fam", unit=TTSIMUnit.DIMENSIONLESS.PER_FAM
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
        unit_system=UNIT_SYSTEM,
    )


def test_boolean_body_at_wrong_group_level_is_caught():
    """A ``DIMENSIONLESS_PER_FAM`` predicate that actually compares bare
    (individual) quantities infers a bare boolean and is caught against its
    ``1 / [fam]`` declaration (GEP 10)."""

    @policy_function(
        leaf_name="requirement_fulfilled_fam", unit=TTSIMUnit.DIMENSIONLESS.PER_FAM
    )
    def requirement_fulfilled_fam(income_m: float, other_income_m: float) -> bool:
        return income_m < other_income_m  # bare operands, but a _fam name

    with pytest.raises(UnitConsistencyError, match="requirement_fulfilled_fam"):
        fail_if_environment_units_are_inconsistent(
            env={
                "income_m": income_m,
                "other_income_m": other_income_m,
                "requirement_fulfilled_fam": requirement_fulfilled_fam,
            },
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_logical_combine_of_mixed_levels_downcasts_to_bare():
    """``|`` of a fam-level and a bare indicator is a bare, individual boolean
    (the combine rule), matching a bare ``DIMENSIONLESS`` declaration at the
    unsuffixed name (GEP 10)."""

    @policy_input(unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_FAM)
    def income_m_fam() -> float:
        """Family income."""

    @policy_input(unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_FAM)
    def threshold_m_fam() -> float:
        """Family subsistence threshold."""

    @policy_function(unit=TTSIMUnit.DIMENSIONLESS)
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
        unit_system=UNIT_SYSTEM,
    )


def test_python_or_of_mixed_levels_is_rewritten_and_downcasts_to_bare():
    """Author-written ``or`` combines leveled booleans exactly like ``|`` (GEP 10).

    Python ``or`` short-circuits through ``__bool__`` and on its own would return a
    single, uncombined operand; the unit check rewrites ``and``/``or`` to ``&``/``|``
    first (mirroring the array vectorizer), so a fam-level ``or`` a bare indicator
    downcasts to a bare, individual boolean, matching the bare ``DIMENSIONLESS``
    declaration at the unsuffixed name — the ``wealth_tax.exempt_from_wealth_tax``
    shape.
    """

    @policy_input(unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_FAM)
    def income_m_fam() -> float:
        """Family income."""

    @policy_input(unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_FAM)
    def threshold_m_fam() -> float:
        """Family subsistence threshold."""

    @policy_function(unit=TTSIMUnit.DIMENSIONLESS)
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
        unit_system=UNIT_SYSTEM,
    )


def test_python_and_on_unit_carrying_operand_is_still_caught():
    """The ``and``→``&`` rewrite keeps the operand screen: ``and``-ing a real
    quantity into a logical combination is still rejected (GEP 10).

    ``age`` is a ``YEARS`` quantity, so ``age and is_exempt`` is the same bug as
    ``age & is_exempt`` — the rewrite must not let it slip through.
    """

    @policy_input(unit=TTSIMUnit.YEARS)
    def age() -> int:
        """An age in years (a ``YEARS`` quantity)."""

    @policy_function(unit=TTSIMUnit.DIMENSIONLESS)
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
            unit_system=UNIT_SYSTEM,
        )


# Cross-level shares (division across grouping levels)


def test_terminal_cross_level_division_is_caught():
    """Dividing two amounts at *different* group levels leaves a bare ratio of
    levels (``betrag_m_fam / betrag_m_kin`` -> ``[kin]/[fam]``) once the physical
    content cancels. A grouping level cannot outlive its base, so returning that
    residue as a *result* is caught on the level axis (GEP 10)."""

    @policy_input(unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_FAM)
    def betrag_m_fam() -> float:
        """A monthly family amount."""

    @policy_input(unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_KIN)
    def betrag_m_kin() -> float:
        """A monthly kin amount."""

    @policy_function(unit=TTSIMUnit.DIMENSIONLESS)
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
            unit_system=UNIT_SYSTEM,
        )


def test_terminal_cross_level_division_passes_with_an_explicit_opt_out():
    """A genuine terminal cross-level ratio is a deliberate policy judgement, so
    it takes a local ``verify_units=False`` rather than a blanket exemption
    (GEP 10)."""

    @policy_input(unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_FAM)
    def betrag_m_fam() -> float:
        """A monthly family amount."""

    @policy_input(unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_KIN)
    def betrag_m_kin() -> float:
        """A monthly kin amount."""

    @policy_function(unit=TTSIMUnit.DIMENSIONLESS, verify_units=False)
    def anteil(betrag_m_fam: float, betrag_m_kin: float) -> float:
        return betrag_m_kin / betrag_m_fam

    fail_if_environment_units_are_inconsistent(
        env={
            "betrag_m_fam": betrag_m_fam,
            "betrag_m_kin": betrag_m_kin,
            "anteil": anteil,
        },
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )


def test_bedarfsanteilsmethode_cross_level_share_consumed_by_multiplication_passes():
    """The GETTSIM idiom: a person's share of a group claim,
    ``(bedarf_m / bedarf_m_fam) * anspruch_m_fam``. The cross-level result
    ``[fam]`` is consumed by the multiply, landing on a bare per-person flow that
    matches the declaration — no exemption needed, and unchanged by the
    cross-level rule (GEP 10)."""

    @policy_input(unit=TTSIMUnit.CURRENCY.PER_MONTH)
    def bedarf_m() -> float:
        """A person's monthly need."""

    @policy_input(unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_FAM)
    def bedarf_m_fam() -> float:
        """The family's pooled monthly need."""

    @policy_input(unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_FAM)
    def anspruch_m_fam() -> float:
        """The family's monthly claim."""

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH)
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
        unit_system=UNIT_SYSTEM,
    )


def test_cross_level_share_declared_with_concrete_content_is_caught():
    """A cross-level division leaves a physically dimensionless result; declaring
    it with concrete content (``CURRENCY_PER_MONTH`` rather than
    ``DIMENSIONLESS``) is caught on the physical axis, before the level axis is
    even reached (GEP 10)."""

    @policy_input(unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_FAM)
    def betrag_m_fam() -> float:
        """A monthly family amount."""

    @policy_input(unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_KIN)
    def betrag_m_kin() -> float:
        """A monthly kin amount."""

    @policy_function(leaf_name="anteil_m", unit=TTSIMUnit.CURRENCY.PER_MONTH)
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
            unit_system=UNIT_SYSTEM,
        )


def test_head_count_at_wrong_group_level_is_still_caught():
    """A head count is a dimensionless, declarable, level-checked unit. A
    ``1/[fam]`` count declared at the kin level is caught (GEP 10) — it is not
    mistaken for a cross-level share."""

    @policy_input(unit=TTSIMUnit.DIMENSIONLESS.PER_FAM)
    def anzahl_personen_fam() -> int:
        """A head count per family — ``1/[fam]``."""

    @policy_function(
        leaf_name="anzahl_personen_kin", unit=TTSIMUnit.DIMENSIONLESS.PER_KIN
    )
    def anzahl_personen_kin(anzahl_personen_fam: int) -> int:
        return anzahl_personen_fam  # a 1/[fam] count under a _kin name

    with pytest.raises(UnitConsistencyError, match="anzahl_personen_kin"):
        fail_if_environment_units_are_inconsistent(
            env={
                "anzahl_personen_fam": anzahl_personen_fam,
                "anzahl_personen_kin": anzahl_personen_kin,
            },
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


# `cast_ttsim_unit`: the expression-level escape hatch


def test_cross_level_comparison_without_cast_is_caught():
    """Comparing a group extreme against a level-less threshold mixes levels
    (``month/[fam]`` against ``month``), so the ordering screen rejects it —
    even where the law mandates exactly this test (GEP 10)."""

    @policy_input(unit=TTSIMUnit.MONTHS.PER_FAM)
    def age_youngest_months_fam() -> float:
        """The family's youngest member's age — a property of the family."""

    @policy_input(unit=TTSIMUnit.MONTHS)
    def age_limit_months() -> float:
        """An age threshold; a level-less duration."""

    @policy_function(unit=TTSIMUnit.DIMENSIONLESS)
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
            unit_system=UNIT_SYSTEM,
        )


def test_cross_level_comparison_with_cast_passes():
    """The policy-mandated per-person reading — each person sees their family's
    extreme — is stated at the site with ``cast_ttsim_unit``; the rest of the body
    stays checked (GEP 10)."""

    @policy_input(unit=TTSIMUnit.MONTHS.PER_FAM)
    def age_youngest_months_fam() -> float:
        """The family's youngest member's age — a property of the family."""

    @policy_input(unit=TTSIMUnit.MONTHS)
    def age_limit_months() -> float:
        """An age threshold; a level-less duration."""

    @policy_function(unit=TTSIMUnit.DIMENSIONLESS)
    def eligible(age_youngest_months_fam: float, age_limit_months: float) -> bool:
        return (
            cast_ttsim_unit(value=age_youngest_months_fam, unit=TTSIMUnit.MONTHS)
            <= age_limit_months
        )

    fail_if_environment_units_are_inconsistent(
        env={
            "age_youngest_months_fam": age_youngest_months_fam,
            "age_limit_months": age_limit_months,
            "eligible": eligible,
        },
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )


def test_level_less_inference_under_a_declared_group_level_is_caught():
    """The declared-vs-inferred level match is exact: a body whose arithmetic
    yields no level cannot silently claim the declared group level; the error
    points at ``cast_ttsim_unit`` (GEP 10)."""

    @policy_input(unit=TTSIMUnit.MONTHS)
    def age_limit_months() -> float:
        """An age threshold; a level-less duration."""

    @policy_function(unit=TTSIMUnit.MONTHS.PER_FAM)
    def doubled_limit_months_fam(age_limit_months: float) -> float:
        return age_limit_months * 2.0

    with pytest.raises(UnitConsistencyError, match="cast_ttsim_unit"):
        fail_if_environment_units_are_inconsistent(
            env={
                "age_limit_months": age_limit_months,
                "doubled_limit_months_fam": doubled_limit_months_fam,
            },
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_cast_at_the_return_states_the_declared_group_level():
    """An intensive group property computed from level-less material states its
    level with ``cast_ttsim_unit`` at the return (GEP 10)."""

    @policy_input(unit=TTSIMUnit.MONTHS)
    def age_limit_months() -> float:
        """An age threshold; a level-less duration."""

    @policy_function(unit=TTSIMUnit.MONTHS.PER_FAM)
    def doubled_limit_months_fam(age_limit_months: float) -> float:
        return cast_ttsim_unit(
            value=age_limit_months * 2.0, unit=TTSIMUnit.MONTHS.PER_FAM
        )

    fail_if_environment_units_are_inconsistent(
        env={
            "age_limit_months": age_limit_months,
            "doubled_limit_months_fam": doubled_limit_months_fam,
        },
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )


def test_group_share_times_group_total_squares_the_level_and_is_caught():
    """A group-owned share times a group total squares the level
    (``1/[fam] * CURRENCY/month/[fam]`` → ``…/[fam]**2``); the level signature
    is compared with exponents, so the product cannot silently claim the
    declared single level (GEP 10)."""

    @policy_input(unit=TTSIMUnit.DIMENSIONLESS.PER_FAM)
    def parents_share_fam() -> float:
        """The parents' share of the family's need — the family's property."""

    @policy_input(unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_FAM)
    def need_m_fam() -> float:
        """The family's monthly need."""

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_FAM)
    def parents_need_m_fam(parents_share_fam: float, need_m_fam: float) -> float:
        return parents_share_fam * need_m_fam

    with pytest.raises(UnitConsistencyError, match="cast_ttsim_unit"):
        fail_if_environment_units_are_inconsistent(
            env={
                "parents_share_fam": parents_share_fam,
                "need_m_fam": need_m_fam,
                "parents_need_m_fam": parents_need_m_fam,
            },
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_group_share_times_group_total_passes_with_cast():
    """Where the law mandates the group-share product, the cast states the
    intended result at the site (GEP 10)."""

    @policy_input(unit=TTSIMUnit.DIMENSIONLESS.PER_FAM)
    def parents_share_fam() -> float:
        """The parents' share of the family's need — the family's property."""

    @policy_input(unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_FAM)
    def need_m_fam() -> float:
        """The family's monthly need."""

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_FAM)
    def parents_need_m_fam(parents_share_fam: float, need_m_fam: float) -> float:
        return cast_ttsim_unit(
            value=parents_share_fam * need_m_fam,
            unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_FAM,
        )

    fail_if_environment_units_are_inconsistent(
        env={
            "parents_share_fam": parents_share_fam,
            "need_m_fam": need_m_fam,
            "parents_need_m_fam": parents_need_m_fam,
        },
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )


def test_cast_tags_a_dimensioned_literal_in_an_ordering_comparison():
    """A genuine dimensioned bound that must stay inline is tagged in place;
    the tagged literal is still screened, so a wrong-period tag is caught
    (GEP 10)."""

    @policy_function(unit=TTSIMUnit.DIMENSIONLESS)
    def poor(income_m: float) -> bool:
        return income_m < cast_ttsim_unit(
            value=1000.0, unit=TTSIMUnit.CURRENCY.PER_MONTH
        )

    fail_if_environment_units_are_inconsistent(
        env={"income_m": income_m, "poor": poor},
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )

    @policy_function(unit=TTSIMUnit.DIMENSIONLESS)
    def poor_buggy(income_m: float) -> bool:
        return income_m < cast_ttsim_unit(
            value=1000.0, unit=TTSIMUnit.CURRENCY.PER_YEAR
        )  # wrong period

    with pytest.raises(UnitConsistencyError, match="poor_buggy"):
        fail_if_environment_units_are_inconsistent(
            env={"income_m": income_m, "poor_buggy": poor_buggy},
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_cast_in_a_vectorized_body_is_screened_identically():
    """The unit check's ``xnp`` stand-in and the cast compose: a literal cap tagged in
    place inside ``xnp.minimum`` passes where the bare literal would be
    rejected (GEP 10)."""

    @policy_function(
        unit=TTSIMUnit.CURRENCY.PER_MONTH, vectorization_strategy="not_required"
    )
    def capped_income_m(income_m: FloatColumn, xnp: ModuleType) -> FloatColumn:
        return xnp.minimum(
            income_m, cast_ttsim_unit(value=2000.0, unit=TTSIMUnit.CURRENCY.PER_MONTH)
        )

    fail_if_environment_units_are_inconsistent(
        env={"income_m": income_m, "capped_income_m": capped_income_m},
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )


def test_cast_to_a_concrete_currency_is_rejected():
    """Only parameters and rounding specs pin down concrete currencies, so a
    cast pinning one is a definition error — reported as such, not as an
    un-evaluable body (GEP 10)."""

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH)
    def pinned_m(income_m: float) -> float:
        return cast_ttsim_unit(value=income_m, unit="CASTAR_PER_MONTH")

    with pytest.raises(UnitDefinitionError, match="agnostic CURRENCY"):
        fail_if_environment_units_are_inconsistent(
            env={"income_m": income_m, "pinned_m": pinned_m},
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_dimensionless_inference_cannot_claim_a_group_owned_declaration():
    """A family predicate over level-less shares must state its family level."""

    @policy_input(unit=TTSIMUnit.DIMENSIONLESS)
    def share_of_need() -> float:
        """A level-less share."""

    @policy_input(unit=TTSIMUnit.DIMENSIONLESS)
    def threshold_share() -> float:
        """A level-less share."""

    @policy_function(unit=TTSIMUnit.DIMENSIONLESS.PER_FAM)
    def requirement_fulfilled_fam(share_of_need: float, threshold_share: float) -> bool:
        return share_of_need < threshold_share

    with pytest.raises(UnitConsistencyError, match="cast_ttsim_unit"):
        fail_if_environment_units_are_inconsistent(
            env={
                "share_of_need": share_of_need,
                "threshold_share": threshold_share,
                "requirement_fulfilled_fam": requirement_fulfilled_fam,
            },
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )

    @policy_function(unit=TTSIMUnit.DIMENSIONLESS.PER_FAM)
    def requirement_fulfilled_cast_fam(
        share_of_need: float, threshold_share: float
    ) -> bool:
        return cast_ttsim_unit(
            value=share_of_need < threshold_share, unit=TTSIMUnit.DIMENSIONLESS.PER_FAM
        )

    fail_if_environment_units_are_inconsistent(
        env={
            "share_of_need": share_of_need,
            "threshold_share": threshold_share,
            "requirement_fulfilled_fam": requirement_fulfilled_cast_fam,
        },
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )


def test_adding_a_nonzero_bare_literal_to_a_quantity_is_caught():
    """``income_m + 100.0`` hides a monthly amount in the literal; ``+``/``-``
    screen literals exactly as the ordering comparisons do — promote to a
    parameter, tag with ``cast_ttsim_unit``, or use 0 (GEP 10)."""

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH)
    def bumped_income_m(income_m: float) -> float:
        return income_m + 100.0

    with pytest.raises(UnitConsistencyError, match="bare literal"):
        fail_if_environment_units_are_inconsistent(
            env={"income_m": income_m, "bumped_income_m": bumped_income_m},
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH)
    def bumped_income_cast_m(income_m: float) -> float:
        return income_m + cast_ttsim_unit(
            value=100.0, unit=TTSIMUnit.CURRENCY.PER_MONTH
        )

    fail_if_environment_units_are_inconsistent(
        env={"income_m": income_m, "bumped_income_m": bumped_income_cast_m},
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )


def test_nonzero_literal_return_under_a_dimensioned_declaration_is_caught():
    """``return 25.0`` under a currency declaration is a hidden dimensioned
    constant: only ``0`` falls through (the eligibility guard); anything else
    is promoted to a parameter or tagged with ``cast_ttsim_unit`` (GEP 10)."""

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH)
    def lump_m(is_exempt: bool, income_m: float) -> float:
        if is_exempt:
            return 25.0
        return income_m

    with pytest.raises(UnitConsistencyError, match="bare literal"):
        fail_if_environment_units_are_inconsistent(
            env={"is_exempt": is_exempt, "income_m": income_m, "lump_m": lump_m},
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH)
    def lump_cast_m(is_exempt: bool, income_m: float) -> float:
        if is_exempt:
            return cast_ttsim_unit(value=25.0, unit=TTSIMUnit.CURRENCY.PER_MONTH)
        return income_m

    fail_if_environment_units_are_inconsistent(
        env={"is_exempt": is_exempt, "income_m": income_m, "lump_m": lump_cast_m},
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )


def test_path_cap_truncation_demands_opt_out(monkeypatch):
    """Exceeding the path cap demands an opt-out, never passes silently (GEP 10).

    With the cap lowered to 4, a body with three independent boolean gates has
    2**3 = 8 reachable paths; the explorer must stop and report rather than
    return success with most paths unchecked.
    """
    monkeypatch.setattr("ttsim._unit_inference._MAX_PATHS", 4)

    @policy_input(unit=TTSIMUnit.DIMENSIONLESS)
    def flag_a() -> bool:
        """A boolean gate."""

    @policy_input(unit=TTSIMUnit.DIMENSIONLESS)
    def flag_b() -> bool:
        """A boolean gate."""

    @policy_input(unit=TTSIMUnit.DIMENSIONLESS)
    def flag_c() -> bool:
        """A boolean gate."""

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_YEAR)
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
            unit_system=UNIT_SYSTEM,
        )


def test_bare_nonzero_literal_in_ordering_is_caught():
    """A bare non-zero literal in an ordering comparison carries the other
    operand's unit, so it is rejected — promote it to a parameter (GEP 10)."""

    @policy_function(unit=TTSIMUnit.DIMENSIONLESS)
    def rich(wealth: float) -> bool:
        return wealth > 1_000_000.0  # bug: the bound silently becomes CURRENCY

    with pytest.raises(UnitConsistencyError, match="rich"):
        fail_if_environment_units_are_inconsistent(
            env={"wealth": wealth, "rich": rich},
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_zero_literal_and_dimensionless_self_in_ordering_pass():
    """The two allowed cases: a literal ``0`` (sign test) against any quantity,
    and any bare literal when the quantity itself is dimensionless (GEP 10)."""

    @policy_input(unit=TTSIMUnit.DIMENSIONLESS)
    def some_rate() -> float:
        """A dimensionless share."""

    @policy_function(unit=TTSIMUnit.DIMENSIONLESS)
    def has_wealth(wealth: float) -> bool:
        return wealth > 0.0  # 0 is the allowed inline literal

    @policy_function(unit=TTSIMUnit.DIMENSIONLESS)
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
        unit_system=UNIT_SYSTEM,
    )


def test_opaque_return_demands_opt_out():
    """A body returning an opaque value (a tuple, a dataclass) is neither a
    checkable quantity nor a plain scalar, so it must opt out (GEP 10)."""

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_YEAR)
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
            unit_system=UNIT_SYSTEM,
        )


def test_adding_different_period_flows_is_caught():
    """``_m + _y`` is unit-blind at run time, so it must be flagged.

    pint would silently auto-convert ``CURRENCY / month + CURRENCY / year`` to
    the left operand's unit during the unit check (matching the ``_m`` declaration)
    and hide the bug; the additive unit check rejects the non-equivalent operands
    before pint sees them (GEP 10).
    """

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH)
    def total_m(income_m: float, bonus_y: float) -> float:
        return income_m + bonus_y  # bug: adds a monthly and a yearly flow

    with pytest.raises(UnitConsistencyError, match="total_m"):
        fail_if_environment_units_are_inconsistent(
            env={"income_m": income_m, "bonus_y": bonus_y, "total_m": total_m},
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_adding_stock_and_flow_is_caught():
    """A cross-dimension addition (stock + flow) raises a ``DimensionalityError``
    in pint, which the unit check otherwise swallows; the additive check flags it."""

    @policy_function(unit=TTSIMUnit.CURRENCY)
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
            unit_system=UNIT_SYSTEM,
        )


def test_comparing_different_period_flows_is_caught():
    """Ordering comparisons are unit-blind at run time too: comparing a monthly
    flow to a yearly one is flagged even when both return arms are consistent."""

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH)
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
            unit_system=UNIT_SYSTEM,
        )


def test_adding_same_period_flows_does_not_false_positive():
    """Two operands in equivalent units add cleanly — no false positive."""

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH)
    def total_two_monthly_m(income_m: float, other_income_m: float) -> float:
        return income_m + other_income_m

    fail_if_environment_units_are_inconsistent(
        env={
            "income_m": income_m,
            "other_income_m": other_income_m,
            "total_two_monthly_m": total_two_monthly_m,
        },
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )


def test_adding_bare_literal_does_not_false_positive():
    """Only ``0`` is allowed inline, so the ``x + 0.0`` guard stays lenient."""

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH)
    def income_floor_m(income_m: float) -> float:
        return income_m + 0.0

    fail_if_environment_units_are_inconsistent(
        env={"income_m": income_m, "income_floor_m": income_floor_m},
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )


def test_zero_literal_falls_back_to_declaration():
    @policy_function(unit=TTSIMUnit.CURRENCY)
    def early_return(wealth: float) -> float:  # noqa: ARG001
        return 0.0

    fail_if_environment_units_are_inconsistent(
        env={"wealth": wealth, "early_return": early_return},
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )


def test_unevaluable_body_without_opt_out_is_rejected():
    # A body the unit check cannot evaluate is not waved through silently (GEP 10):
    # the author must opt out explicitly with verify_units=False.
    @policy_function(unit=TTSIMUnit.CURRENCY)
    def not_evaluable(wealth: float) -> float:
        return wealth.this_attribute_does_not_exist()  # ty: ignore[unresolved-attribute]

    with pytest.raises(UnitConsistencyError, match="verify_units=False"):
        fail_if_environment_units_are_inconsistent(
            env={"wealth": wealth, "not_evaluable": not_evaluable},
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_unevaluable_body_with_opt_out_passes():
    # The same body, explicitly opted out, is accepted: its declared unit stands.
    @policy_function(unit=TTSIMUnit.CURRENCY, verify_units=False)
    def not_evaluable(wealth: float) -> float:
        return wealth.this_attribute_does_not_exist()  # ty: ignore[unresolved-attribute]

    fail_if_environment_units_are_inconsistent(
        env={"wealth": wealth, "not_evaluable": not_evaluable},
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )


# Vectorized bodies: the xnp stand-in and the piecewise_polynomial / look_up /
# join primitives are screened at their edges (GEP 10). Their happy paths run
# end-to-end through the mettsim worked example (housing_benefits: minimum +
# look_up, payroll/property tax: piecewise, child_tax_credit: join), so these
# tests pin only what a silent stand-in regression would hide from that check —
# a screen that stops screening fails no test anywhere else.


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
        unit=TTSIMUnit.CURRENCY.PER_MONTH, vectorization_strategy="not_required"
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
        unit_system=UNIT_SYSTEM,
    )

    @policy_function(
        unit=TTSIMUnit.CURRENCY.PER_MONTH, vectorization_strategy="not_required"
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
            unit_system=UNIT_SYSTEM,
        )


def test_where_condition_must_be_a_boolean():
    """`xnp.where` screens its condition before unifying its arms, so the
    vectorized spelling rejects a dimensioned selector exactly as the scalar `if`
    does (GEP 10)."""

    @policy_function(
        unit=TTSIMUnit.CURRENCY.PER_MONTH, vectorization_strategy="not_required"
    )
    def gated_m(
        wealth: FloatColumn,
        income_m: FloatColumn,
        other_income_m: FloatColumn,
        xnp: ModuleType,
    ) -> FloatColumn:
        # bug: a stock selects, the arms are fine
        return xnp.where(wealth, income_m, other_income_m)

    with pytest.raises(UnitConsistencyError, match="truth value"):
        fail_if_environment_units_are_inconsistent(
            env={
                "wealth": wealth,
                "income_m": income_m,
                "other_income_m": other_income_m,
                "gated_m": gated_m,
            },
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_where_mixing_a_calendar_point_and_a_duration_is_caught():
    """``xnp.where`` runs no forward pint op, so a calendar-point arm gets no
    delegate-to-pint dispensation: an arm mix of a point and a duration is
    flagged (GEP 10)."""

    @policy_function(
        unit=TTSIMUnit.CALENDAR_YEAR, vectorization_strategy="not_required"
    )
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
            unit_system=UNIT_SYSTEM,
        )


def test_vectorized_minimum_with_a_bare_literal_bound_is_caught():
    """``xnp.minimum``/``maximum`` screen like an ordering comparison (they are
    the vectorized ``min``/``max``): a bare non-zero literal bound silently
    carries the other operand's unit, so promote it to a parameter (GEP 10)."""

    @policy_function(
        unit=TTSIMUnit.CURRENCY.PER_MONTH, vectorization_strategy="not_required"
    )
    def capped_m(income_m: FloatColumn, xnp: ModuleType) -> FloatColumn:
        return xnp.minimum(income_m, 1_000.0)  # bug: a bare literal cap

    with pytest.raises(UnitConsistencyError, match="capped_m"):
        fail_if_environment_units_are_inconsistent(
            env={"income_m": income_m, "capped_m": capped_m},
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_clip_with_a_bare_nonzero_literal_bound_is_caught():
    """``xnp.clip`` screens each bound as an ordering operand: a zero bound is
    the allowed sign test, a bare non-zero bound is rejected (GEP 10)."""

    @policy_function(
        unit=TTSIMUnit.CURRENCY.PER_MONTH, vectorization_strategy="not_required"
    )
    def clipped_m(income_m: FloatColumn, xnp: ModuleType) -> FloatColumn:
        return xnp.clip(income_m, 0.0, 5_000.0)  # bug: a bare literal ceiling

    with pytest.raises(UnitConsistencyError, match="clipped_m"):
        fail_if_environment_units_are_inconsistent(
            env={"income_m": income_m, "clipped_m": clipped_m},
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_scalar_max_zero_floor_preserves_unit_when_result_is_continued():
    """Scalar ``max(x, 0.0)`` carries ``x``'s unit on every branch, exactly as
    the vectorized ``xnp.maximum`` does, so the clamped value stays usable: the
    net/gross ratio ``max(income_m, 0.0) / income_m`` is dimensionless without a
    ``cast_ttsim_unit`` on the zero floor (GEP 10). The literal spelling
    matters: the check rebinds the *names* ``max``/``min``, so each is
    exercised by its own body."""

    @policy_function(unit=TTSIMUnit.DIMENSIONLESS)
    def ratio(income_m: float) -> float:
        return max(income_m, 0.0) / income_m

    fail_if_environment_units_are_inconsistent(
        env={"income_m": income_m, "ratio": ratio},
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )


def test_scalar_min_zero_floor_preserves_unit_when_result_is_continued():
    """Scalar ``min(x, 0.0)`` likewise keeps ``x``'s unit on both branches."""

    @policy_function(unit=TTSIMUnit.DIMENSIONLESS)
    def ratio(income_m: float) -> float:
        return min(income_m, 0.0) / income_m

    fail_if_environment_units_are_inconsistent(
        env={"income_m": income_m, "ratio": ratio},
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )


def test_scalar_max_with_a_bare_nonzero_literal_bound_is_caught():
    """Scalar ``max``/``min`` screen their bounds like the vectorized ops: a bare
    non-zero literal silently carries the operand's unit and is rejected — only
    the zero floor falls through (GEP 10)."""

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH)
    def capped_m(income_m: float) -> float:
        return max(income_m, 1_000.0)  # bug: a bare literal floor

    with pytest.raises(UnitConsistencyError, match="capped_m"):
        fail_if_environment_units_are_inconsistent(
            env={"income_m": income_m, "capped_m": capped_m},
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_unmodelled_xnp_op_demands_opt_out():
    """An xnp op the unit check does not model falls through to raw NumPy and is
    reported as needing ``verify_units=False`` — never silently passed (GEP 10)."""

    @policy_function(
        unit=TTSIMUnit.CURRENCY.PER_MONTH, vectorization_strategy="not_required"
    )
    def cumulative_m(income_m: FloatColumn, xnp: ModuleType) -> FloatColumn:
        return xnp.cumsum(income_m)

    with pytest.raises(UnitConsistencyError, match="verify_units=False"):
        fail_if_environment_units_are_inconsistent(
            env={"income_m": income_m, "cumulative_m": cumulative_m},
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


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


def test_schedule_call_with_wrong_domain_unit_is_caught():
    """A ``piecewise_polynomial`` call is screened at its edges (GEP 10): the
    argument must match the schedule's declared ``input_unit``."""
    schedule = _make_schedule_param(input_unit=CASTAR, output_unit=CASTAR_PER_YEAR)

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_YEAR)
    def levy_y(
        income_m: float, schedule: PiecewisePolynomialParamValue, xnp: ModuleType
    ) -> float:
        # bug: a monthly income into a wealth-domain schedule
        return piecewise_polynomial(x=income_m, parameters=schedule, xnp=xnp)

    with pytest.raises(UnitConsistencyError, match="levy_y"):
        fail_if_environment_units_are_inconsistent(
            env={"income_m": income_m, "schedule": schedule, "levy_y": levy_y},
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_schedule_output_disagreeing_with_the_declaration_is_caught():
    """The call produces the schedule's ``output_unit``, which the declaration
    check verifies. This mismatch confirms that the validator propagates the
    schedule's output unit rather than treating the result as dimensionless."""
    schedule = _make_schedule_param(input_unit=CASTAR, output_unit=CASTAR_PER_YEAR)

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH)
    def levy_m(
        wealth: float, schedule: PiecewisePolynomialParamValue, xnp: ModuleType
    ) -> float:
        # the schedule produces a yearly flow, but the node claims a monthly one
        return piecewise_polynomial(x=wealth, parameters=schedule, xnp=xnp)

    with pytest.raises(UnitConsistencyError, match="levy_m"):
        fail_if_environment_units_are_inconsistent(
            env={"wealth": wealth, "schedule": schedule, "levy_m": levy_m},
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_lookup_call_with_wrong_domain_unit_is_caught():
    by_year = _make_lookup_param(
        input_unit=TTSIMUnit.CALENDAR_YEAR, output_unit=CASTAR_PER_MONTH
    )

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH)
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
            unit_system=UNIT_SYSTEM,
        )


def test_lookup_indexed_by_a_computed_literal_counter_is_evaluable():
    """A lookup keyed by a dimensionless counter built from literals resolves to
    the schedule's output unit without an opt-out: the output unit is fixed by the
    schedule regardless of the index (GEP 10). The body anchors on its own branch
    path, so a bare-literal index needs no unit-carrying domain argument."""
    by_stufe = _make_lookup_param(
        input_unit=TTSIMUnit.DIMENSIONLESS, output_unit=CASTAR_PER_MONTH
    )

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH)
    def abzug_m(
        income_m: float,
        other_income_m: float,
        by_stufe: ConsecutiveIntLookupTableParamValue,
    ) -> float:
        stufe = 0
        if income_m > 0:
            stufe = stufe + 1
        if other_income_m > 0:
            stufe = stufe + 1
        return by_stufe.look_up(stufe)

    fail_if_environment_units_are_inconsistent(
        env={
            "income_m": income_m,
            "other_income_m": other_income_m,
            "by_stufe": by_stufe,
            "abzug_m": abzug_m,
        },
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )


def test_lookup_indexed_by_a_computed_literal_counter_still_checks_the_output_unit():
    """The counter-indexed lookup still produces the schedule's ``output_unit``,
    which the declaration check verifies: a node declaring a yearly flow over a
    monthly-flow schedule is caught (GEP 10)."""
    by_stufe = _make_lookup_param(
        input_unit=TTSIMUnit.DIMENSIONLESS, output_unit=CASTAR_PER_MONTH
    )

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_YEAR)
    def abzug_y(
        income_m: float, by_stufe: ConsecutiveIntLookupTableParamValue
    ) -> float:
        # the schedule produces a monthly flow, but the node claims a yearly one
        stufe = 1 if income_m > 0 else 0
        return by_stufe.look_up(stufe)

    with pytest.raises(UnitConsistencyError, match="abzug_y"):
        fail_if_environment_units_are_inconsistent(
            env={"income_m": income_m, "by_stufe": by_stufe, "abzug_y": abzug_y},
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_schedule_with_a_dimensionful_input_axis_rejects_a_bare_literal_index():
    """A schedule declaring a dimensionful input axis is never keyed by a bare
    literal: such an index bypasses the input-axis contract, so the body still
    demands an explicit opt-out rather than passing silently (GEP 10)."""
    schedule = _make_schedule_param(
        input_unit=TTSIMUnit.HECTARE, output_unit=CASTAR_PER_YEAR
    )

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_YEAR)
    def levy_y(schedule: PiecewisePolynomialParamValue, xnp: ModuleType) -> float:
        # bug: a bare literal where the schedule expects an area
        return piecewise_polynomial(x=1.0, parameters=schedule, xnp=xnp)

    with pytest.raises(UnitConsistencyError, match="verify_units=False"):
        fail_if_environment_units_are_inconsistent(
            env={"schedule": schedule, "levy_y": levy_y},
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_join_with_a_dimensioned_key_is_caught():
    """A gather's keys are identifiers, so a dimensioned column used as a key is a
    bug — a currency never identifies a row (GEP 10)."""

    @policy_function(
        unit=TTSIMUnit.CURRENCY.PER_MONTH, vectorization_strategy="not_required"
    )
    def recipient_income_m(
        p_id: IntColumn,
        wealth: FloatColumn,
        income_m: FloatColumn,
        xnp: ModuleType,
    ) -> FloatColumn:
        return join(
            foreign_key=wealth,  # bug: a stock is not an identifier
            primary_key=p_id,
            target=income_m,
            value_if_foreign_key_is_missing=0.0,
            xnp=xnp,
        )

    with pytest.raises(UnitConsistencyError, match="identifier"):
        fail_if_environment_units_are_inconsistent(
            env={
                "p_id": p_id,
                "wealth": wealth,
                "income_m": income_m,
                "recipient_income_m": recipient_income_m,
            },
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_join_fallback_not_in_the_targets_unit_is_caught():
    """The missing-key fallback becomes part of the gathered column, so it must
    carry the target's unit; a yearly fallback under a monthly target is a bug
    that only unmatched keys would ever expose (GEP 10)."""

    @policy_function(
        unit=TTSIMUnit.CURRENCY.PER_MONTH, vectorization_strategy="not_required"
    )
    def recipient_income_m(
        p_id: IntColumn,
        p_id_recipient: IntColumn,
        income_m: FloatColumn,
        xnp: ModuleType,
    ) -> FloatColumn:
        return join(
            foreign_key=p_id_recipient,
            primary_key=p_id,
            target=income_m,
            # bug: a yearly amount fills the unmatched rows of a monthly column
            value_if_foreign_key_is_missing=cast_ttsim_unit(
                100.0, TTSIMUnit.CURRENCY.PER_YEAR
            ),
            xnp=xnp,
        )

    with pytest.raises(UnitConsistencyError, match="missing-key fallback"):
        fail_if_environment_units_are_inconsistent(
            env={
                "p_id": p_id,
                "p_id_recipient": p_id_recipient,
                "income_m": income_m,
                "recipient_income_m": recipient_income_m,
            },
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_in_body_reduction_requires_an_opt_out():
    """A reduction changes which rows a value belongs to, and the unit check has no
    array-axis metadata to derive that from, so it demands an explicit opt-out
    rather than passing the operand's unit through (GEP 10)."""

    @policy_function(unit=TTSIMUnit.CURRENCY, vectorization_strategy="not_required")
    def total_wealth(wealth: FloatColumn, xnp: ModuleType) -> FloatColumn:
        return xnp.sum(wealth)

    with pytest.raises(UnitConsistencyError, match="verify_units=False"):
        fail_if_environment_units_are_inconsistent(
            env={"wealth": wealth, "total_wealth": total_wealth},
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_join_target_level_disagreeing_with_the_declaration_is_caught():
    """A ``join`` gather hands on the target column's unit, grouping level
    included (GEP 10) — proven by contradiction: were the stand-in to drop the
    unit, this level mismatch could not be detected. (mettsim's checked join
    body gathers a dimensionless target, so it cannot pin this.)"""

    @policy_input(unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_FAM)
    def income_m_fam() -> float: ...

    @policy_function(
        unit=TTSIMUnit.CURRENCY.PER_MONTH, vectorization_strategy="not_required"
    )
    def recipient_family_income_m(
        p_id: IntColumn,
        p_id_recipient: IntColumn,
        income_m_fam: FloatColumn,
        xnp: ModuleType,
    ) -> FloatColumn:
        # bug: the gathered target is the family's [fam] amount, but the node
        # declares a bare (individual) one
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
            unit_system=UNIT_SYSTEM,
        )


def test_concrete_mismatch_is_caught():
    @policy_function(unit=TTSIMUnit.CURRENCY)
    def mislabelled(age_in_years: float) -> float:
        return age_in_years * 2.0

    @policy_input(unit=TTSIMUnit.YEARS)
    def age_in_years() -> float:
        """An age."""

    with pytest.raises(UnitConsistencyError, match="mislabelled"):
        fail_if_environment_units_are_inconsistent(
            env={"age_in_years": age_in_years, "mislabelled": mislabelled},
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_dict_param_subscripting_is_verifiable():
    """Per-leaf units make dict-consuming bodies unit-checkable (GEP 10)."""
    schedule = DictParam(
        value={"child_amount_y": 100.0, "max_age": 18},
        unit={"child_amount_y": "CASTAR_PER_YEAR", "max_age": "YEARS"},
        start_date=_START,
        end_date=_END,
    )

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_YEAR)
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
        unit_system=UNIT_SYSTEM,
    )

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH)
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
            unit_system=UNIT_SYSTEM,
        )


def test_uniform_dict_param_is_subscriptable_in_unit_check():
    subsistence = DictParam(
        value={"per_spouse": 500.0},
        unit=CASTAR_PER_MONTH,
        start_date=_START,
        end_date=_END,
    )

    @policy_input(unit=TTSIMUnit.DIMENSIONLESS)
    def number_of_adults() -> int:
        """A head count — dimensionless (GEP 10)."""

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH)
    def subsistence_income_m(subsistence: dict, number_of_adults: int) -> float:
        return subsistence["per_spouse"] * number_of_adults

    fail_if_environment_units_are_inconsistent(
        env={
            "subsistence": subsistence,
            "number_of_adults": number_of_adults,
            "subsistence_income_m": subsistence_income_m,
        },
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )


# Structured param functions (unit=UNSET_UNIT): plucks are cast at the site


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


@policy_input(unit=TTSIMUnit.YEARS)
def age() -> int:
    """A duration in years (a person's age)."""


def test_structured_plucks_with_casts_are_verifiable():
    """Casting each pluck keeps the rest of the body checked (GEP 10)."""

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH)
    def child_benefit_m(age: int, child_rate: _ChildRate) -> float:
        amount_m = cast_ttsim_unit(
            value=child_rate.amount_m, unit=TTSIMUnit.CURRENCY.PER_MONTH
        )
        max_age = cast_ttsim_unit(value=child_rate.bounds.max_age, unit=TTSIMUnit.YEARS)
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
        unit_system=UNIT_SYSTEM,
    )


def test_structured_pluck_used_without_cast_is_caught():
    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH)
    def child_benefit_m(age: int, child_rate: _ChildRate) -> float:
        if age <= child_rate.bounds.max_age:  # bug: pluck used as a quantity
            return cast_ttsim_unit(
                value=child_rate.amount_m, unit=TTSIMUnit.CURRENCY.PER_MONTH
            )
        return 0.0

    with pytest.raises(UnitConsistencyError, match="cast_ttsim_unit"):
        fail_if_environment_units_are_inconsistent(
            env={
                "age": age,
                "child_rate": child_rate,
                "child_benefit_m": child_benefit_m,
            },
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_structured_pluck_returned_without_cast_is_caught():
    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH)
    def child_benefit_m(child_rate: _ChildRate) -> float:
        return child_rate.amount_m  # bug: returned without stating its unit

    with pytest.raises(UnitConsistencyError, match="at the pluck"):
        fail_if_environment_units_are_inconsistent(
            env={"child_rate": child_rate, "child_benefit_m": child_benefit_m},
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_structured_cast_too_coarse_fails_on_the_deeper_pluck():
    """A cast on a sub-structure yields a plain quantity, so the next deeper
    pluck fails loudly — a too-coarse cast can never silently mis-tag."""

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH)
    def child_benefit_m(age: int, child_rate: _ChildRate) -> float:
        bounds = cast_ttsim_unit(
            value=child_rate.bounds, unit=TTSIMUnit.YEARS
        )  # too coarse
        if age <= bounds.max_age:
            return cast_ttsim_unit(
                value=child_rate.amount_m, unit=TTSIMUnit.CURRENCY.PER_MONTH
            )
        return 0.0

    with pytest.raises(UnitConsistencyError, match="verify_units=False"):
        fail_if_environment_units_are_inconsistent(
            env={
                "age": age,
                "child_rate": child_rate,
                "child_benefit_m": child_benefit_m,
            },
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


# Annotated parameter dataclasses: fields state their units, plucks resolve
# (GEP 10)


@dataclass(frozen=True)
class _AnnotatedAgeBounds:
    min_age: Annotated[int, TTSIMUnit.YEARS]
    max_age: Annotated[int, TTSIMUnit.YEARS]


@dataclass(frozen=True)
class _AnnotatedChildRate:
    amount_m: Annotated[float, TTSIMUnit.CURRENCY.PER_MONTH]
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
    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH)
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
        unit_system=UNIT_SYSTEM,
    )


def test_annotated_pluck_misuse_is_caught():
    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH)
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
            unit_system=UNIT_SYSTEM,
        )


@dataclass(frozen=True)
class _PartiallyAnnotatedRate:
    amount_m: Annotated[float, TTSIMUnit.CURRENCY.PER_MONTH]
    max_age: int


@param_function(unit=UNSET_UNIT)
def partially_annotated_rate(raw_child_rate: RawParamValue) -> _PartiallyAnnotatedRate:
    return _PartiallyAnnotatedRate(
        amount_m=raw_child_rate["amount_m"],
        max_age=raw_child_rate["bounds"]["max_age"],
    )


def test_unannotated_field_keeps_the_cast_requirement():
    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH)
    def benefit_m(age: int, partially_annotated_rate: _PartiallyAnnotatedRate) -> float:
        if age <= partially_annotated_rate.max_age:  # bug: opaque pluck, no cast
            return partially_annotated_rate.amount_m
        return 0.0

    with pytest.raises(UnitConsistencyError, match="cast_ttsim_unit"):
        fail_if_environment_units_are_inconsistent(
            env={
                "age": age,
                "raw_child_rate": make_raw_child_rate(),
                "partially_annotated_rate": partially_annotated_rate,
                "benefit_m": benefit_m,
            },
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_lookup_indexed_by_an_opaque_structured_pluck_keeps_the_cast_requirement():
    """An opaque (unannotated) structured pluck used as a lookup index is not a
    bare literal: it still owes a `cast_ttsim_unit` or an annotation, so the body
    demands an opt-out rather than passing on the literal-index fallback (GEP 10)."""
    by_stufe = _make_lookup_param(
        input_unit=TTSIMUnit.DIMENSIONLESS, output_unit=CASTAR_PER_MONTH
    )

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH)
    def benefit_m(
        by_stufe: ConsecutiveIntLookupTableParamValue,
        partially_annotated_rate: _PartiallyAnnotatedRate,
    ) -> float:
        # bug: an opaque pluck as the index, neither annotated nor cast
        return by_stufe.look_up(partially_annotated_rate.max_age)

    with pytest.raises(UnitConsistencyError, match="verify_units=False"):
        fail_if_environment_units_are_inconsistent(
            env={
                "by_stufe": by_stufe,
                "raw_child_rate": make_raw_child_rate(),
                "partially_annotated_rate": partially_annotated_rate,
                "benefit_m": benefit_m,
            },
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


@param_function(unit=UNSET_UNIT)
def annotated_child_rate_by_group(
    raw_child_rate: RawParamValue,
) -> dict[str, _AnnotatedChildRate]:
    """A mapping of category to an annotated rate dataclass."""
    return {
        "kleinkind": _AnnotatedChildRate(
            amount_m=raw_child_rate["amount_m"],
            bounds=_AnnotatedAgeBounds(
                min_age=raw_child_rate["bounds"]["min_age"],
                max_age=raw_child_rate["bounds"]["max_age"],
            ),
        ),
    }


def test_annotated_fields_resolve_plucks_through_mapping_of_dataclass():
    """`param[key].field` on a `dict[str, <annotated dataclass>]` carries the
    field's declared unit, so a consistent body needs no cast."""

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH)
    def child_benefit_m(
        age: int,
        annotated_child_rate_by_group: dict[str, _AnnotatedChildRate],
    ) -> float:
        if age <= annotated_child_rate_by_group["kleinkind"].bounds.max_age:
            return annotated_child_rate_by_group["kleinkind"].amount_m
        return 0.0

    fail_if_environment_units_are_inconsistent(
        env={
            "age": age,
            "raw_child_rate": make_raw_child_rate(),
            "annotated_child_rate_by_group": annotated_child_rate_by_group,
            "child_benefit_m": child_benefit_m,
        },
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )


def test_mapping_of_dataclass_pluck_misuse_is_caught():
    """A field pluck through the mapping keeps its unit, so a dimensionally wrong
    use (money plus a duration) is still caught."""

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH)
    def child_benefit_m(
        age: int,
        annotated_child_rate_by_group: dict[str, _AnnotatedChildRate],
    ) -> float:
        return annotated_child_rate_by_group["kleinkind"].amount_m + age

    with pytest.raises(UnitConsistencyError, match="non-equivalent units"):
        fail_if_environment_units_are_inconsistent(
            env={
                "age": age,
                "raw_child_rate": make_raw_child_rate(),
                "annotated_child_rate_by_group": annotated_child_rate_by_group,
                "child_benefit_m": child_benefit_m,
            },
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_resolved_return_structure_handles_a_runtime_mapping_annotation():
    """The non-string annotation path resolves the value dataclass of a mapping,
    so a param function whose module lacks `from __future__ import annotations`
    still gets field-unit tracing."""

    def producer(raw_child_rate: object) -> object: ...

    producer.__annotations__ = {"return": dict[str, _AnnotatedChildRate]}
    cls, item_cls = _resolved_return_structure(producer)
    assert cls is None
    assert item_cls is _AnnotatedChildRate


# Annotated schedule fields: a lookup nested in a parameter dataclass declares its
# two axes at the field with `InputOutputUnits`, so `params.field.look_up(...)`
# screens its index against the input axis and produces the output axis, no cast
# (GEP 10)


@dataclass(frozen=True)
class _AnnotatedRentBounds:
    max_age: Annotated[int, TTSIMUnit.YEARS]
    rate_m: Annotated[
        ConsecutiveIntLookupTableParamValue,
        InputOutputUnits(
            input_unit=TTSIMUnit.YEARS, output_unit=TTSIMUnit.CURRENCY.PER_MONTH
        ),
    ]


@dataclass(frozen=True)
class _AnnotatedLookupRate:
    factor: Annotated[
        ConsecutiveIntLookupTableParamValue,
        InputOutputUnits(
            input_unit=TTSIMUnit.YEARS, output_unit=TTSIMUnit.DIMENSIONLESS
        ),
    ]
    stufe_rate_m: Annotated[
        ConsecutiveIntLookupTableParamValue,
        InputOutputUnits(
            input_unit=TTSIMUnit.DIMENSIONLESS, output_unit=TTSIMUnit.CURRENCY.PER_MONTH
        ),
    ]
    bounds: _AnnotatedRentBounds


@param_function(unit=UNSET_UNIT)
def annotated_lookup_rate(xnp: ModuleType) -> _AnnotatedLookupRate:
    """A structured builder whose dataclass fields include lookup schedules."""
    return _AnnotatedLookupRate(
        factor=ConsecutiveIntLookupTableParamValue(
            xnp=xnp,
            values_to_look_up=xnp.array([1.0, 2.0]),
            bases_to_subtract=xnp.array([1]),
        ),
        stufe_rate_m=ConsecutiveIntLookupTableParamValue(
            xnp=xnp,
            values_to_look_up=xnp.array([30.0, 40.0]),
            bases_to_subtract=xnp.array([1]),
        ),
        bounds=_AnnotatedRentBounds(
            max_age=18,
            rate_m=ConsecutiveIntLookupTableParamValue(
                xnp=xnp,
                values_to_look_up=xnp.array([45.0, 60.0]),
                bases_to_subtract=xnp.array([1]),
            ),
        ),
    )


def test_annotated_schedule_field_look_up_resolves_without_cast():
    """A field annotated with a lookup type carries its output unit: the pluck
    yields a schedule, so `look_up` produces that unit with no `cast_ttsim_unit`."""

    @policy_function(unit=TTSIMUnit.DIMENSIONLESS)
    def scaling(
        statutory_age: int, annotated_lookup_rate: _AnnotatedLookupRate
    ) -> float:
        return annotated_lookup_rate.factor.look_up(statutory_age)

    fail_if_environment_units_are_inconsistent(
        env={
            "statutory_age": statutory_age,
            "annotated_lookup_rate": annotated_lookup_rate,
            "scaling": scaling,
        },
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )


def test_annotated_schedule_field_look_up_output_unit_is_applied():
    """The field's output unit is really applied, not silenced: a money-per-month
    look-up added to a duration is a non-equivalent-unit sum."""

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH)
    def rent_m(
        statutory_age: int, annotated_lookup_rate: _AnnotatedLookupRate
    ) -> float:
        return (
            annotated_lookup_rate.bounds.rate_m.look_up(statutory_age) + statutory_age
        )

    with pytest.raises(UnitConsistencyError, match="non-equivalent units"):
        fail_if_environment_units_are_inconsistent(
            env={
                "statutory_age": statutory_age,
                "annotated_lookup_rate": annotated_lookup_rate,
                "rent_m": rent_m,
            },
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_annotated_schedule_field_resolves_through_nested_dataclass():
    """A schedule field on a nested dataclass resolves through the nested pluck,
    so `params.nested.the_lookup.look_up(...)` screens against the field unit."""

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH)
    def rent_m(
        statutory_age: int, annotated_lookup_rate: _AnnotatedLookupRate
    ) -> float:
        return annotated_lookup_rate.bounds.rate_m.look_up(statutory_age)

    fail_if_environment_units_are_inconsistent(
        env={
            "statutory_age": statutory_age,
            "annotated_lookup_rate": annotated_lookup_rate,
            "rent_m": rent_m,
        },
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )


def test_annotated_schedule_field_look_up_by_a_literal_index_is_evaluable():
    """A schedule field with a dimensionless input axis keyed by a bare literal
    resolves to the field's output unit without an opt-out, just as a top-level
    schedule parameter does: a dimensionless index need not be a quantity for the
    output to be known (GEP 10)."""

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH)
    def rent_m(annotated_lookup_rate: _AnnotatedLookupRate) -> float:
        return annotated_lookup_rate.stufe_rate_m.look_up(1)

    fail_if_environment_units_are_inconsistent(
        env={
            "annotated_lookup_rate": annotated_lookup_rate,
            "rent_m": rent_m,
        },
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )


def test_resolved_return_structure_handles_a_runtime_dataclass_annotation():
    """The non-string annotation path resolves a plain dataclass return."""

    def producer(raw_child_rate: object) -> object: ...

    producer.__annotations__ = {"return": _AnnotatedChildRate}
    cls, item_cls = _resolved_return_structure(producer)
    assert cls is _AnnotatedChildRate
    assert item_cls is None


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
            unit_system=UNIT_SYSTEM,
        )


@dataclass(frozen=True)
class _ContainerRate:
    amounts_m: Annotated[dict[str, float], TTSIMUnit.CURRENCY.PER_MONTH]


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
            unit_system=UNIT_SYSTEM,
        )


@dataclass(frozen=True)
class _NestedContainerRate:
    bounds: _ContainerRate


@param_function(unit=UNSET_UNIT)
def never_plucked_bad_rate(raw_child_rate: RawParamValue) -> _NestedContainerRate:
    """A structured builder with a malformed annotation nobody plucks."""
    return _NestedContainerRate(
        bounds=_ContainerRate(
            amounts_m={str(key): float(value) for key, value in raw_child_rate.items()}
        )
    )


def test_never_plucked_malformed_field_annotation_is_rejected():
    """A malformed field annotation on a structured param function is caught at
    build time even when no body ever plucks it — the eager pass walks the nested
    dataclass tree (GEP 10)."""
    with pytest.raises(UnitDefinitionError, match="scalar field"):
        fail_if_environment_units_are_inconsistent(
            env={
                "raw_child_rate": make_raw_child_rate(),
                "never_plucked_bad_rate": never_plucked_bad_rate,
            },
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


# A schedule builder declares its two axes with `unit=InputOutputUnits(...)` and
# opts out of body verification; consumers screen against those axes (GEP 10)


def make_raw_levy_schedule() -> RawParam:
    return RawParam(
        value={"top_rate": 0.2, "ceiling": 1000},
        input_unit="CASTAR",
        output_unit="CASTAR_PER_YEAR",
        start_date=_START,
        end_date=_END,
    )


@param_function(
    unit=InputOutputUnits(
        input_unit=TTSIMUnit.CURRENCY, output_unit=TTSIMUnit.CURRENCY.PER_YEAR
    ),
    verify_units=False,
)
def levy_schedule(
    raw_levy_schedule: RawParamValue, xnp: ModuleType
) -> PiecewisePolynomialParamValue:
    """A schedule builder declaring its two axes: consumers screen against them."""
    return PiecewisePolynomialParamValue(
        thresholds=xnp.asarray([0.0, raw_levy_schedule["ceiling"]]),
        intercepts=xnp.asarray([0.0, 0.0]),
        coefficients=xnp.asarray([[0.0], [raw_levy_schedule["top_rate"]]]),
    )


def test_axes_declared_schedule_screens_consumer_without_cast():
    @policy_function(unit=TTSIMUnit.CURRENCY.PER_YEAR)
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
        unit_system=UNIT_SYSTEM,
    )


def test_axes_declared_schedule_rejects_wrong_domain_argument():
    @policy_function(unit=TTSIMUnit.CURRENCY.PER_YEAR)
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
            unit_system=UNIT_SYSTEM,
        )


def test_axes_declared_schedule_output_reaches_the_consumer_declaration():
    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH)
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
            unit_system=UNIT_SYSTEM,
        )


def test_input_output_unit_on_non_schedule_return_is_rejected():
    """A quantity or dict return declaring `InputOutputUnits(...)` is a contract
    error: only a schedule has two axes (GEP 10)."""

    @param_function(
        unit=InputOutputUnits(
            input_unit=TTSIMUnit.CURRENCY, output_unit=TTSIMUnit.CURRENCY.PER_YEAR
        ),
        verify_units=False,
    )
    def levy_params(raw_levy_schedule: RawParamValue) -> dict[str, float]:
        return dict(raw_levy_schedule)

    with pytest.raises(UnitConsistencyError, match="not annotated as returning"):
        fail_if_environment_units_are_inconsistent(
            env={
                "raw_levy_schedule": make_raw_levy_schedule(),
                "levy_params": levy_params,
            },
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_schedule_return_with_quantity_unit_is_rejected():
    """A schedule-returning param function must declare `InputOutputUnits(...)`, not
    a single quantity unit (GEP 10)."""

    @param_function(unit=TTSIMUnit.CURRENCY.PER_YEAR, verify_units=False)
    def levy_schedule_bad(
        raw_levy_schedule: RawParamValue, xnp: ModuleType
    ) -> PiecewisePolynomialParamValue:
        return PiecewisePolynomialParamValue(
            thresholds=xnp.asarray([0.0, raw_levy_schedule["ceiling"]]),
            intercepts=xnp.asarray([0.0, 0.0]),
            coefficients=xnp.asarray([[0.0], [raw_levy_schedule["top_rate"]]]),
        )

    with pytest.raises(UnitConsistencyError, match="InputOutputUnits"):
        fail_if_environment_units_are_inconsistent(
            env={
                "raw_levy_schedule": make_raw_levy_schedule(),
                "levy_schedule_bad": levy_schedule_bad,
            },
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_schedule_return_with_unset_unit_is_rejected():
    """A schedule-returning param function may not declare `unit=UNSET_UNIT`: that
    marks a structured value, not a schedule (GEP 10)."""

    @param_function(unit=UNSET_UNIT)
    def levy_schedule_unset(
        raw_levy_schedule: RawParamValue, xnp: ModuleType
    ) -> PiecewisePolynomialParamValue:
        return PiecewisePolynomialParamValue(
            thresholds=xnp.asarray([0.0, raw_levy_schedule["ceiling"]]),
            intercepts=xnp.asarray([0.0, 0.0]),
            coefficients=xnp.asarray([[0.0], [raw_levy_schedule["top_rate"]]]),
        )

    with pytest.raises(UnitConsistencyError, match="InputOutputUnits"):
        fail_if_environment_units_are_inconsistent(
            env={
                "raw_levy_schedule": make_raw_levy_schedule(),
                "levy_schedule_unset": levy_schedule_unset,
            },
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_schedule_builder_without_verify_units_false_is_rejected():
    """A schedule builder's body cannot be unit-verified, so it must state
    `verify_units=False` explicitly (GEP 10)."""

    @param_function(
        unit=InputOutputUnits(
            input_unit=TTSIMUnit.CURRENCY, output_unit=TTSIMUnit.CURRENCY.PER_YEAR
        ),
    )
    def levy_schedule_verified(
        raw_levy_schedule: RawParamValue, xnp: ModuleType
    ) -> PiecewisePolynomialParamValue:
        return PiecewisePolynomialParamValue(
            thresholds=xnp.asarray([0.0, raw_levy_schedule["ceiling"]]),
            intercepts=xnp.asarray([0.0, 0.0]),
            coefficients=xnp.asarray([[0.0], [raw_levy_schedule["top_rate"]]]),
        )

    with pytest.raises(UnitConsistencyError, match="verify_units=False"):
        fail_if_environment_units_are_inconsistent(
            env={
                "raw_levy_schedule": make_raw_levy_schedule(),
                "levy_schedule_verified": levy_schedule_verified,
            },
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_lookup_table_builder_with_currency_input_unit_is_rejected():
    """A lookup table is keyed by consecutive integers, so its `input_unit` may
    not be a currency (GEP 10)."""

    @param_function(
        unit=InputOutputUnits(
            input_unit=TTSIMUnit.CURRENCY, output_unit=TTSIMUnit.CURRENCY.PER_YEAR
        ),
        verify_units=False,
    )
    def levy_lookup_bad(xnp: ModuleType) -> ConsecutiveIntLookupTableParamValue:
        return ConsecutiveIntLookupTableParamValue(
            xnp=xnp,
            values_to_look_up=xnp.array([45.0, 60.0]),
            bases_to_subtract=xnp.array([1]),
        )

    with pytest.raises(UnitConsistencyError, match="keyed by consecutive integers"):
        fail_if_environment_units_are_inconsistent(
            env={"levy_lookup_bad": levy_lookup_bad},
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_schedule_builder_with_concrete_currency_axis_is_rejected():
    """A builder's `InputOutputUnits` axes are currency-agnostic, exactly like a
    column/function declaration: a concrete-currency axis is a definition error
    that names the offending builder (GEP 10)."""

    @param_function(
        unit=InputOutputUnits(
            input_unit=TTSIMUnit.CURRENCY, output_unit=CASTAR_PER_YEAR
        ),
        verify_units=False,
    )
    def levy_schedule_concrete(
        raw_levy_schedule: RawParamValue, xnp: ModuleType
    ) -> PiecewisePolynomialParamValue:
        return PiecewisePolynomialParamValue(
            thresholds=xnp.asarray([0.0, raw_levy_schedule["ceiling"]]),
            intercepts=xnp.asarray([0.0, 0.0]),
            coefficients=xnp.asarray([[0.0], [raw_levy_schedule["top_rate"]]]),
        )

    with pytest.raises(
        UnitDefinitionError, match=r"levy_schedule_concrete.*concrete currency"
    ):
        fail_if_environment_units_are_inconsistent(
            env={
                "raw_levy_schedule": make_raw_levy_schedule(),
                "levy_schedule_concrete": levy_schedule_concrete,
            },
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_schedule_builder_declared_axes_flow_into_look_up_without_cast():
    """A lookup-table builder declares its two axes with `InputOutputUnits`;
    ``look_up`` screens its index against the input axis and yields the output at
    the consumer with no ``cast_ttsim_unit`` (GEP 10)."""

    @param_function(
        unit=InputOutputUnits(
            input_unit=TTSIMUnit.YEARS, output_unit=TTSIMUnit.SQUARE_METER.PER_FAM
        ),
        verify_units=False,
    )
    def eligible_area(xnp: ModuleType) -> ConsecutiveIntLookupTableParamValue:
        return ConsecutiveIntLookupTableParamValue(
            xnp=xnp,
            values_to_look_up=xnp.array([45.0, 60.0]),
            bases_to_subtract=xnp.array([1]),
        )

    @policy_function(unit=TTSIMUnit.SQUARE_METER.PER_FAM)
    def area_fam(
        statutory_age: int, eligible_area: ConsecutiveIntLookupTableParamValue
    ) -> float:
        return eligible_area.look_up(statutory_age)

    fail_if_environment_units_are_inconsistent(
        env={
            "statutory_age": statutory_age,
            "eligible_area": eligible_area,
            "area_fam": area_fam,
        },
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )


def test_schedule_builder_declared_axes_screen_a_mismatched_consumer():
    """The declared output axis is a real contract: a ``look_up`` result fed into
    a disagreeing consumer declaration is caught (GEP 10)."""

    @param_function(
        unit=InputOutputUnits(
            input_unit=TTSIMUnit.YEARS, output_unit=TTSIMUnit.SQUARE_METER.PER_FAM
        ),
        verify_units=False,
    )
    def eligible_area(xnp: ModuleType) -> ConsecutiveIntLookupTableParamValue:
        return ConsecutiveIntLookupTableParamValue(
            xnp=xnp,
            values_to_look_up=xnp.array([45.0, 60.0]),
            bases_to_subtract=xnp.array([1]),
        )

    @policy_function(unit=TTSIMUnit.SQUARE_METER)
    def area_per_person(
        statutory_age: int, eligible_area: ConsecutiveIntLookupTableParamValue
    ) -> float:
        return eligible_area.look_up(statutory_age)

    with pytest.raises(UnitConsistencyError, match="area_per_person"):
        fail_if_environment_units_are_inconsistent(
            env={
                "statutory_age": statutory_age,
                "eligible_area": eligible_area,
                "area_per_person": area_per_person,
            },
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


# Field-marker mismatches: a schedule field needs `InputOutputUnits`, a scalar field
# a single `CompositeUnit` (GEP 10)


@dataclass(frozen=True)
class _ScheduleFieldWithBareUnit:
    rate_m: Annotated[ConsecutiveIntLookupTableParamValue, TTSIMUnit.CURRENCY.PER_MONTH]


@param_function(unit=UNSET_UNIT)
def bare_unit_schedule_field(xnp: ModuleType) -> _ScheduleFieldWithBareUnit:
    return _ScheduleFieldWithBareUnit(
        rate_m=ConsecutiveIntLookupTableParamValue(
            xnp=xnp,
            values_to_look_up=xnp.array([1.0, 2.0]),
            bases_to_subtract=xnp.array([1]),
        )
    )


def test_bare_unit_on_schedule_field_is_rejected():
    with pytest.raises(UnitDefinitionError, match="InputOutputUnits"):
        fail_if_environment_units_are_inconsistent(
            env={"bare_unit_schedule_field": bare_unit_schedule_field},
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


@dataclass(frozen=True)
class _ScalarFieldWithIOUnit:
    amount_m: Annotated[
        float,
        InputOutputUnits(
            input_unit=TTSIMUnit.YEARS, output_unit=TTSIMUnit.CURRENCY.PER_MONTH
        ),
    ]


@param_function(unit=UNSET_UNIT)
def io_unit_scalar_field(raw_child_rate: RawParamValue) -> _ScalarFieldWithIOUnit:
    return _ScalarFieldWithIOUnit(amount_m=raw_child_rate["amount_m"])


def test_input_output_unit_on_scalar_field_is_rejected():
    with pytest.raises(UnitDefinitionError, match="not a lookup/piecewise value"):
        fail_if_environment_units_are_inconsistent(
            env={
                "raw_child_rate": make_raw_child_rate(),
                "io_unit_scalar_field": io_unit_scalar_field,
            },
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


# Multi-dimensional lookups: a tuple `input_unit` screens each `look_up` argument
# against its own axis positionally (GEP 10)


def _two_axis_lookup(xnp: ModuleType) -> ConsecutiveIntLookupTableParamValue:
    return ConsecutiveIntLookupTableParamValue(
        xnp=xnp,
        values_to_look_up=xnp.array([1.0, 2.0]),
        bases_to_subtract=xnp.array([1]),
    )


@param_function(
    unit=InputOutputUnits(
        input_unit=(TTSIMUnit.DIMENSIONLESS, TTSIMUnit.YEARS),
        output_unit=TTSIMUnit.CURRENCY.PER_MONTH,
    ),
    verify_units=False,
)
def two_axis_table(xnp: ModuleType) -> ConsecutiveIntLookupTableParamValue:
    return _two_axis_lookup(xnp)


def test_tuple_input_axes_screen_look_up_arguments_positionally():
    """A two-axis lookup screens argument 0 against axis 0 and argument 1 against
    axis 1; a call whose units match each axis passes (GEP 10)."""

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH)
    def benefit_m(
        geburtsmonat: int,
        statutory_age: int,
        two_axis_table: ConsecutiveIntLookupTableParamValue,
    ) -> float:
        return two_axis_table.look_up(geburtsmonat, statutory_age)

    fail_if_environment_units_are_inconsistent(
        env={
            "geburtsmonat": geburtsmonat,
            "statutory_age": statutory_age,
            "two_axis_table": two_axis_table,
            "benefit_m": benefit_m,
        },
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )


def test_tuple_input_axes_positional_mismatch_is_caught():
    """Swapping the arguments screens the year-valued age against the dimensionless
    axis 0, a non-equivalent-unit look-up (GEP 10)."""

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH)
    def benefit_m(
        geburtsmonat: int,
        statutory_age: int,
        two_axis_table: ConsecutiveIntLookupTableParamValue,
    ) -> float:
        return two_axis_table.look_up(statutory_age, geburtsmonat)

    with pytest.raises(UnitConsistencyError, match="non-equivalent units"):
        fail_if_environment_units_are_inconsistent(
            env={
                "geburtsmonat": geburtsmonat,
                "statutory_age": statutory_age,
                "two_axis_table": two_axis_table,
                "benefit_m": benefit_m,
            },
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_tuple_input_axes_wrong_argument_count_is_caught():
    """A call supplying a different number of arguments than declared axes is a
    unit-check error naming the counts (GEP 10)."""

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH)
    def benefit_m(
        statutory_age: int,
        two_axis_table: ConsecutiveIntLookupTableParamValue,
    ) -> float:
        return two_axis_table.look_up(statutory_age)

    with pytest.raises(UnitConsistencyError, match="2 input axes with 1 argument"):
        fail_if_environment_units_are_inconsistent(
            env={
                "statutory_age": statutory_age,
                "two_axis_table": two_axis_table,
                "benefit_m": benefit_m,
            },
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_tuple_input_axis_currency_element_on_lookup_table_is_rejected():
    """A lookup table is keyed by consecutive integers, so no axis of a tuple
    `input_unit` may be a currency (GEP 10)."""

    @param_function(
        unit=InputOutputUnits(
            input_unit=(TTSIMUnit.DIMENSIONLESS, TTSIMUnit.CURRENCY),
            output_unit=TTSIMUnit.CURRENCY.PER_MONTH,
        ),
        verify_units=False,
    )
    def bad_table(xnp: ModuleType) -> ConsecutiveIntLookupTableParamValue:
        return _two_axis_lookup(xnp)

    with pytest.raises(UnitConsistencyError, match="keyed by consecutive integers"):
        fail_if_environment_units_are_inconsistent(
            env={"bad_table": bad_table},
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_tuple_input_axis_on_piecewise_builder_is_rejected():
    """`piecewise_polynomial` takes one domain argument, so a tuple `input_unit`
    on a piecewise builder is a contract error (GEP 10)."""

    @param_function(
        unit=InputOutputUnits(
            input_unit=(TTSIMUnit.CURRENCY, TTSIMUnit.DIMENSIONLESS),
            output_unit=TTSIMUnit.CURRENCY.PER_YEAR,
        ),
        verify_units=False,
    )
    def bad_piecewise(
        raw_levy_schedule: RawParamValue, xnp: ModuleType
    ) -> PiecewisePolynomialParamValue:
        return PiecewisePolynomialParamValue(
            thresholds=xnp.asarray([0.0, raw_levy_schedule["ceiling"]]),
            intercepts=xnp.asarray([0.0, 0.0]),
            coefficients=xnp.asarray([[0.0], [raw_levy_schedule["top_rate"]]]),
        )

    with pytest.raises(UnitConsistencyError, match="single domain argument"):
        fail_if_environment_units_are_inconsistent(
            env={
                "raw_levy_schedule": make_raw_levy_schedule(),
                "bad_piecewise": bad_piecewise,
            },
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_tuple_input_axis_concrete_currency_element_is_rejected():
    """A concrete-currency element in a tuple `input_unit` is rejected — builder
    axes are agnostic, and the message names the builder, not "a field
    annotation" (GEP 10)."""

    @param_function(
        unit=InputOutputUnits(
            input_unit=(TTSIMUnit.DIMENSIONLESS, CASTAR),
            output_unit=TTSIMUnit.CURRENCY.PER_MONTH,
        ),
        verify_units=False,
    )
    def concrete_axis_table(xnp: ModuleType) -> ConsecutiveIntLookupTableParamValue:
        return _two_axis_lookup(xnp)

    with pytest.raises(
        UnitDefinitionError, match=r"concrete_axis_table.*concrete currency"
    ):
        fail_if_environment_units_are_inconsistent(
            env={"concrete_axis_table": concrete_axis_table},
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


@dataclass(frozen=True)
class _TwoAxisFieldRate:
    table: Annotated[
        ConsecutiveIntLookupTableParamValue,
        InputOutputUnits(
            input_unit=(TTSIMUnit.DIMENSIONLESS, TTSIMUnit.YEARS),
            output_unit=TTSIMUnit.CURRENCY.PER_MONTH,
        ),
    ]


@param_function(unit=UNSET_UNIT)
def two_axis_field_rate(xnp: ModuleType) -> _TwoAxisFieldRate:
    return _TwoAxisFieldRate(table=_two_axis_lookup(xnp))


def test_schedule_field_with_tuple_input_axes_screens_positionally():
    """A schedule-typed field may declare a tuple `input_unit`; the plucked lookup
    screens each argument against its own axis (GEP 10)."""

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH)
    def benefit_m(
        geburtsmonat: int,
        statutory_age: int,
        two_axis_field_rate: _TwoAxisFieldRate,
    ) -> float:
        return two_axis_field_rate.table.look_up(geburtsmonat, statutory_age)

    fail_if_environment_units_are_inconsistent(
        env={
            "geburtsmonat": geburtsmonat,
            "statutory_age": statutory_age,
            "two_axis_field_rate": two_axis_field_rate,
            "benefit_m": benefit_m,
        },
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )


def test_schedule_field_with_tuple_input_axes_positional_mismatch_is_caught():
    """The field's tuple axes are really applied: swapping the arguments screens
    the year-valued age against the dimensionless axis 0 (GEP 10)."""

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH)
    def benefit_m(
        geburtsmonat: int,
        statutory_age: int,
        two_axis_field_rate: _TwoAxisFieldRate,
    ) -> float:
        return two_axis_field_rate.table.look_up(statutory_age, geburtsmonat)

    with pytest.raises(UnitConsistencyError, match="non-equivalent units"):
        fail_if_environment_units_are_inconsistent(
            env={
                "geburtsmonat": geburtsmonat,
                "statutory_age": statutory_age,
                "two_axis_field_rate": two_axis_field_rate,
                "benefit_m": benefit_m,
            },
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


# The decorator-site rules are mirrored at schedule fields, and fire eagerly —
# whether or not any body plucks the field (GEP 10)


@dataclass(frozen=True)
class _PiecewiseFieldWithTupleAxes:
    schedule: Annotated[
        PiecewisePolynomialParamValue,
        InputOutputUnits(
            input_unit=(TTSIMUnit.CURRENCY, TTSIMUnit.DIMENSIONLESS),
            output_unit=TTSIMUnit.CURRENCY.PER_YEAR,
        ),
    ]


@param_function(unit=UNSET_UNIT)
def piecewise_field_with_tuple_axes(
    xnp: ModuleType,
) -> _PiecewiseFieldWithTupleAxes:
    """A structured builder with a never-plucked piecewise field carrying a tuple."""
    return _PiecewiseFieldWithTupleAxes(
        schedule=PiecewisePolynomialParamValue(
            thresholds=xnp.asarray([0.0, 1000.0]),
            intercepts=xnp.asarray([0.0, 0.0]),
            coefficients=xnp.asarray([[0.0], [0.2]]),
        )
    )


def test_tuple_input_axis_on_piecewise_field_is_rejected():
    """`piecewise_polynomial` takes one domain argument, so a tuple `input_unit`
    on a piecewise-typed field is a definition error, caught eagerly even though
    no body plucks the field (GEP 10)."""
    with pytest.raises(UnitDefinitionError, match="single domain argument"):
        fail_if_environment_units_are_inconsistent(
            env={"piecewise_field_with_tuple_axes": piecewise_field_with_tuple_axes},
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


@dataclass(frozen=True)
class _LookupFieldWithCurrencyAxis:
    table: Annotated[
        ConsecutiveIntLookupTableParamValue,
        InputOutputUnits(
            input_unit=(TTSIMUnit.DIMENSIONLESS, TTSIMUnit.CURRENCY),
            output_unit=TTSIMUnit.CURRENCY.PER_MONTH,
        ),
    ]


@param_function(unit=UNSET_UNIT)
def lookup_field_with_currency_axis(
    xnp: ModuleType,
) -> _LookupFieldWithCurrencyAxis:
    """A structured builder with a never-plucked lookup field keyed by a currency."""
    return _LookupFieldWithCurrencyAxis(table=_two_axis_lookup(xnp))


def test_currency_input_axis_on_lookup_field_is_rejected():
    """A lookup table is keyed by consecutive integers, so no axis of a lookup-typed
    field may be a currency; caught eagerly even without a pluck (GEP 10)."""
    with pytest.raises(UnitDefinitionError, match="keyed by consecutive integers"):
        fail_if_environment_units_are_inconsistent(
            env={"lookup_field_with_currency_axis": lookup_field_with_currency_axis},
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


# Per-function body opt-out (verify_units=False)


def test_verify_units_false_skips_body_inference():
    # A body the unit check would otherwise flag a mismatch in (stock * share is
    # a stock, not the declared yearly flow) is not checked when it opts out; the
    # declared unit still stands (GEP 10).
    @policy_function(unit=TTSIMUnit.CURRENCY.PER_YEAR, verify_units=False)
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
        unit_system=UNIT_SYSTEM,
    )


def test_verify_units_false_still_checks_consumers_against_declared_unit():
    # The opt-out is local to the body: the declared unit is still the edge
    # contract, so a consumer that misuses the producer is still caught.
    @policy_function(unit=TTSIMUnit.CURRENCY.PER_YEAR, verify_units=False)
    def producer_y(wealth: float) -> float:
        return wealth.this_does_not_exist()  # ty: ignore[unresolved-attribute]

    @policy_function(unit=TTSIMUnit.YEARS)
    def consumer(producer_y: float) -> float:
        return producer_y

    with pytest.raises(UnitConsistencyError, match="consumer"):
        fail_if_environment_units_are_inconsistent(
            env={"wealth": wealth, "producer_y": producer_y, "consumer": consumer},
            grouping_levels=GROUPING_LEVELS,
            unit_system=UNIT_SYSTEM,
        )


def test_cast_unit_literal_clamp_floor_inside_max_is_checkable():
    """A ``cast_ttsim_unit(0, …)`` clamp floor inside ``max()`` is checkable: the
    tagged literal participates in the ordering screen like any quantity, so the
    body infers its declared unit rather than reporting as un-evaluable
    (GEP 10)."""

    @policy_input(unit=TTSIMUnit.CURRENCY.PER_MONTH)
    def wage_m() -> float: ...

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH)
    def clamped_wage_m(wage_m: float) -> float:
        return max(wage_m, cast_ttsim_unit(0, TTSIMUnit.CURRENCY.PER_MONTH))

    fail_if_environment_units_are_inconsistent(
        env={"wage_m": wage_m, "clamped_wage_m": clamped_wage_m},
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )


def test_cast_unit_literal_participates_in_every_screened_op():
    """A cast literal is a first-class unit-check operand across the screened ops, not
    only ``max``/``min``: arithmetic (``*``/``+``), an ordering comparison, and a
    ``where`` (ternary) all check with a cast literal on one side (GEP 10)."""

    @policy_input(unit=TTSIMUnit.CURRENCY.PER_MONTH)
    def wage_m() -> float: ...

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH)
    def floored_m(wage_m: float) -> float:
        floor = cast_ttsim_unit(0, TTSIMUnit.CURRENCY.PER_MONTH)
        bonus = cast_ttsim_unit(100, TTSIMUnit.CURRENCY.PER_MONTH) * cast_ttsim_unit(
            1, TTSIMUnit.DIMENSIONLESS
        )
        return (wage_m + bonus) if wage_m > floor else floor

    fail_if_environment_units_are_inconsistent(
        env={"wage_m": wage_m, "floored_m": floored_m},
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )


def test_cross_group_level_subtraction_in_a_body_is_caught():
    """Subtracting two different group levels is a level mismatch (GEP 10).

    ``income_m_fam`` is ``CURRENCY/month/[fam]`` and ``income_m_kin``
    ``CURRENCY/month/[kin]``. Broadcast replicates each onto persons but leaves
    the *unit* level untouched, so the subtraction stays ``[fam] - [kin]`` — the
    headline cross-level bug the unit check must reject.
    """

    @policy_input(unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_FAM)
    def income_m_fam() -> float: ...

    @policy_input(unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_KIN)
    def income_m_kin() -> float: ...

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_FAM)
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
            unit_system=UNIT_SYSTEM,
        )


def test_person_versus_group_level_subtraction_in_a_body_is_caught():
    """A bare (individual) quantity minus a group-level one is a mismatch (GEP 10).

    ``income_m`` carries no group suffix, so it is bare ``CURRENCY/month``;
    ``freibetrag_m_fam`` is ``CURRENCY/month/[fam]``. Combining them needs an
    explicit per-capita reconciliation, so the bare subtraction is rejected.
    """

    @policy_input(unit=TTSIMUnit.CURRENCY.PER_MONTH)
    def income_m() -> float: ...

    @policy_input(unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_FAM)
    def freibetrag_m_fam() -> float: ...

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH)
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
            unit_system=UNIT_SYSTEM,
        )


def test_same_group_level_addition_in_a_body_passes():
    """Control: combining two quantities at the *same* group level is fine.

    Proves the cross-level checks above reject on the level mismatch, not on
    merely seeing a group suffix — ``a_m_fam + b_m_fam`` is ``[fam] + [fam]``.
    """

    @policy_input(unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_FAM)
    def a_m_fam() -> float: ...

    @policy_input(unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_FAM)
    def b_m_fam() -> float: ...

    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_FAM)
    def total_m_fam(a_m_fam: float, b_m_fam: float) -> float:
        return a_m_fam + b_m_fam

    fail_if_environment_units_are_inconsistent(
        env={"a_m_fam": a_m_fam, "b_m_fam": b_m_fam, "total_m_fam": total_m_fam},
        grouping_levels=GROUPING_LEVELS,
        unit_system=UNIT_SYSTEM,
    )


def test_missing_check_accepts_structured_param_function():
    fail_if_environment_units_are_missing(
        {"raw_child_rate": make_raw_child_rate(), "child_rate": child_rate}
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
        fail_if_environment_units_are_missing({"raw_child_rate": raw})
