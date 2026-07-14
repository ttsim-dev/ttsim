"""Tests for the grouping-level dimensions and the [person] count (GEP 10).

These exercise the core level mechanics directly on the unit primitives:
levels as non-convertible base dimensions, the level-as-denominator
resolution, the [person] count bridge, cross-level rejection, and the
level-aware aggregation (SUM/MIN/MAX take the target level, MEAN the
individual, COUNT mints ``[person]/[target]``, ANY/ALL a boolean at the
target).
"""

from __future__ import annotations

import pytest

from ttsim.exceptions import UnitDefinitionError
from ttsim.tt import (
    AggType,
    Unit,
)
from ttsim.tt.currencies import (
    isolated_currency_registration,
    register_currency,
)
from ttsim.tt.grouping_levels import register_grouping_levels
from ttsim.tt.units import (
    CURRENCY_TOKEN,
    PERSON_LEVEL,
    UNIT_REGISTRY,
    base_is_level_carrying,
    divide_by_grouping_level,
    grouping_level_count_unit,
    parse_unit,
    resolve_compositional_column_unit,
    resolve_compositional_param_unit,
    resolve_compositional_unit,
    resolved_unit_for_aggregation,
    unit_for_aggregation,
    units_are_equivalent,
)

# ----------------------------------------------------------------------------
# #118: grouping levels as base dimensions + the [person] count dimension
# ----------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _register_middle_earth_levels():
    """Register a representative level set for the level-aware tests.

    Levels are discovered per build; the orchestration (issue #119) registers
    them. ``person`` is always present. Registration is idempotent, so the
    autouse fixture is safe across tests.
    """
    register_grouping_levels(["hh", "bg", "sn"])


def test_register_grouping_levels_always_registers_person():
    register_grouping_levels([])
    # `person` resolves to its own base dimension (the [person] count dimension).
    assert (
        divide_by_grouping_level(unit=parse_unit("CURRENCY"), level=PERSON_LEVEL)
        is not None
    )


def test_register_grouping_levels_is_idempotent():
    # Re-registering an already-known level is a tolerated no-op.
    register_grouping_levels(["hh"])
    register_grouping_levels(["hh", "bg"])
    first = divide_by_grouping_level(unit=parse_unit("CURRENCY"), level="hh")
    second = divide_by_grouping_level(unit=parse_unit("CURRENCY"), level="hh")
    assert units_are_equivalent(left=first, right=second)


def test_each_level_is_its_own_base_dimension():
    # No conversion between levels: hh and bg denominators are distinct dimensions.
    at_hh = divide_by_grouping_level(unit=parse_unit("CURRENCY / month"), level="hh")
    at_bg = divide_by_grouping_level(unit=parse_unit("CURRENCY / month"), level="bg")
    assert at_hh.dimensionality != at_bg.dimensionality
    assert not units_are_equivalent(left=at_hh, right=at_bg)


def test_unregistered_grouping_level_is_rejected():
    with pytest.raises(UnitDefinitionError, match="Unknown grouping level"):
        divide_by_grouping_level(unit=parse_unit("CURRENCY"), level="eg_not_registered")


# ----------------------------------------------------------------------------
# Which tokens carry a level (level-carrying/level-less default)
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("base", "level_carrying"),
    [
        ("CURRENCY", True),
        ("PERSON_COUNT", True),
        ("SQUARE_METER", True),
        ("HECTARE", True),
        ("YEARS", False),
        ("MONTHS", False),
        ("DAYS", False),
        ("CALENDAR_YEAR", False),
        ("CALENDAR_MONTH", False),
        ("CALENDAR_DAY", False),
        ("DIMENSIONLESS", False),
        ("HOURS", True),
    ],
)
def test_base_is_level_carrying_defaults(base, level_carrying):
    assert base_is_level_carrying(base) is level_carrying


def test_concrete_currency_base_is_level_carrying():
    # Defined relative to CASTAR, the base currency the mettsim import registers;
    # the registration is isolated so it does not leak to other tests.
    with isolated_currency_registration():
        register_currency(name="LEVEL_TEST_COIN", definition="CASTAR / 2")
        assert base_is_level_carrying("LEVEL_TEST_COIN")


# ----------------------------------------------------------------------------
# A level is a denominator (divide_by_grouping_level)
# ----------------------------------------------------------------------------


def test_currency_flow_resolves_with_hh_denominator_at_level_hh():
    base = resolve_compositional_unit(Unit.CURRENCY.PER_MONTH)
    at_hh = divide_by_grouping_level(unit=base, level="hh")
    assert units_are_equivalent(
        left=at_hh,
        right=divide_by_grouping_level(unit=parse_unit("CURRENCY / month"), level="hh"),
    )


def test_currency_flow_resolves_with_person_denominator_at_individual_level():
    base = resolve_compositional_unit(Unit.CURRENCY.PER_MONTH)
    at_person = divide_by_grouping_level(unit=base, level=PERSON_LEVEL)
    assert units_are_equivalent(
        left=at_person,
        right=divide_by_grouping_level(
            unit=parse_unit("CURRENCY / month"), level=PERSON_LEVEL
        ),
    )
    # Person and hh denominators are different dimensions.
    assert not units_are_equivalent(
        left=at_person, right=divide_by_grouping_level(unit=base, level="hh")
    )


@pytest.mark.parametrize(
    "token",
    [
        Unit.YEARS,
        Unit.HOURS.PER_WEEK,
        Unit.DIMENSIONLESS,
    ],
)
def test_level_less_tokens_resolve_without_a_level(token):
    # A level-less unit carries no grouping denominator: the resolved unit is
    # the plain physical unit, unchanged by any level.
    resolved = resolve_compositional_unit(token)
    person_division = divide_by_grouping_level(
        unit=parse_unit("CURRENCY"), level=PERSON_LEVEL
    )
    # It has no [person] denominator: dividing CURRENCY by person is a different
    # dimension, and the level-less unit shares no level dimension with it.
    assert resolved.dimensionality != person_division.dimensionality


# ----------------------------------------------------------------------------
# The [person] count dimension: the bridge cancels
# ----------------------------------------------------------------------------


def test_count_bridges_hh_to_person_via_division():
    # (CURRENCY/month/[hh]) / ([person]/[hh]) == CURRENCY/month/[person].
    rent_at_hh = divide_by_grouping_level(
        unit=parse_unit("CURRENCY / month"), level="hh"
    )
    count_to_hh = grouping_level_count_unit(target_level="hh")  # [person]/[hh]
    bridged = (
        UNIT_REGISTRY.Quantity(1.0, rent_at_hh)
        / UNIT_REGISTRY.Quantity(1.0, count_to_hh)
    ).units
    expected = divide_by_grouping_level(
        unit=parse_unit("CURRENCY / month"), level=PERSON_LEVEL
    )
    assert units_are_equivalent(left=bridged, right=expected)


def test_count_bridges_person_to_sn_via_multiplication():
    # ([person]/[sn]) * (CURRENCY/year/[person]) == CURRENCY/year/[sn]: a
    # per-person allowance times a head count is a per-group amount.
    count_to_sn = grouping_level_count_unit(target_level="sn")  # [person]/[sn]
    per_person = divide_by_grouping_level(
        unit=parse_unit("CURRENCY / year"), level=PERSON_LEVEL
    )
    product = (
        UNIT_REGISTRY.Quantity(1.0, count_to_sn)
        * UNIT_REGISTRY.Quantity(1.0, per_person)
    ).units
    expected = divide_by_grouping_level(unit=parse_unit("CURRENCY / year"), level="sn")
    assert units_are_equivalent(left=product, right=expected)


def test_cross_level_addition_is_not_equivalent():
    # A unit at [hh] and one at [bg] are different dimensions: adding them across
    # levels is a mismatch the equivalence check catches.
    at_hh = divide_by_grouping_level(unit=parse_unit("CURRENCY / month"), level="hh")
    at_bg = divide_by_grouping_level(unit=parse_unit("CURRENCY / month"), level="bg")
    assert not units_are_equivalent(left=at_hh, right=at_bg)


# ----------------------------------------------------------------------------
# The person leaf is implied, group levels are spelled, on parameters
# ----------------------------------------------------------------------------


def test_spelled_person_level_is_rejected():
    # The person leaf is implied, never spelled (GEP 10): _PER_PERSON is rejected
    # so there is exactly one canonical spelling for a per-person quantity.
    with pytest.raises(UnitDefinitionError, match="implied, never spelled"):
        _ = Unit.CURRENCY.PER_YEAR.PER_PERSON


def test_absent_level_yields_person_leaf():
    # sparerfreibetrag: a per-person yearly amount — the person leaf is implied,
    # so the bare CURRENCY_PER_YEAR resolves to CURRENCY / year / [person].
    resolved = resolve_compositional_param_unit(
        unit=Unit.CURRENCY.PER_YEAR, where="test"
    )
    per_person = divide_by_grouping_level(
        unit=parse_unit("CURRENCY / year"), level=PERSON_LEVEL
    )
    assert units_are_equivalent(left=resolved, right=per_person)


def test_spelled_level_on_stock_param():
    # A non-flow per-group amount: CURRENCY at level hh.
    resolved = resolve_compositional_param_unit(unit=Unit.CURRENCY.PER_HH, where="test")
    expected = divide_by_grouping_level(unit=parse_unit("CURRENCY"), level="hh")
    assert units_are_equivalent(left=resolved, right=expected)


def test_unknown_level_is_rejected():
    with pytest.raises(UnitDefinitionError, match="Unknown grouping level"):
        resolve_compositional_param_unit(
            unit=Unit.CURRENCY.PER_YEAR.PER_LEVEL("not_a_level"), where="test"
        )


def test_person_implied_on_scalar_param_with_name_suffix():
    # A per-person scalar param: the person leaf is implied (not spelled) and the
    # spelled period agrees with the name's time suffix.
    resolved = resolve_compositional_param_unit(
        unit=Unit.CURRENCY.PER_YEAR, time_unit_id="y", where="test"
    )
    expected = divide_by_grouping_level(
        unit=parse_unit("CURRENCY / year"), level=PERSON_LEVEL
    )
    assert units_are_equivalent(left=resolved, right=expected)


# ----------------------------------------------------------------------------
# The level is declared, not read off the suffix, on columns
# ----------------------------------------------------------------------------


def test_column_omitting_the_group_level_is_a_person_property():
    # A per-person amount constant within the group (GEP 10's
    # ``regelbedarf_pro_person_m_bg``): omitting the level at a group suffix
    # leaves the implied person leaf, no ``[bg]``.
    resolved = resolve_compositional_column_unit(
        unit=Unit.CURRENCY.PER_MONTH,
        time_unit_id="m",
        grouping_level="bg",
        where="test",
    )
    expected = divide_by_grouping_level(
        unit=parse_unit("CURRENCY / month"), level=PERSON_LEVEL
    )
    assert units_are_equivalent(left=resolved, right=expected)


def test_intensive_column_omitting_the_level_stays_bare_at_a_group_suffix():
    resolved = resolve_compositional_column_unit(
        unit=Unit.MONTHS, time_unit_id=None, grouping_level="bg", where="test"
    )
    assert units_are_equivalent(left=resolved, right=parse_unit("delta_calendar_month"))


def test_intensive_column_with_a_spelled_group_level_resolves():
    # GEP 10's ``alter_monate_jüngstes_mitglied_fg``: the family's property, so
    # the duration carries the group level — declared, not read off the base.
    resolved = resolve_compositional_column_unit(
        unit=Unit.MONTHS.PER_LEVEL("hh"),
        time_unit_id=None,
        grouping_level="hh",
        where="test",
    )
    expected = divide_by_grouping_level(
        unit=parse_unit("delta_calendar_month"), level="hh"
    )
    assert units_are_equivalent(left=resolved, right=expected)


def test_spelled_group_level_contradicting_the_suffix_is_rejected():
    with pytest.raises(UnitDefinitionError, match="must not contradict"):
        resolve_compositional_column_unit(
            unit=Unit.CURRENCY.PER_MONTH.PER_LEVEL("bg"),
            time_unit_id="m",
            grouping_level="hh",
            where="test",
        )


def test_boolean_omitting_the_level_at_a_group_suffix_is_a_person_indicator():
    resolved = resolve_compositional_column_unit(
        unit=Unit.DIMENSIONLESS,
        time_unit_id=None,
        grouping_level="hh",
        where="test",
        is_boolean=True,
    )
    expected = divide_by_grouping_level(
        unit=UNIT_REGISTRY.dimensionless, level=PERSON_LEVEL
    )
    assert units_are_equivalent(left=resolved, right=expected)


def test_calendar_point_carries_a_level():
    # GEP 10's ``baujahr_immobilie_hh``: the dwelling's construction year is the
    # household's property — a leveled calendar point. Attaching and comparing
    # the level must stay clear of pint's offset-arithmetic rules.
    resolved = resolve_compositional_column_unit(
        unit=Unit.CALENDAR_YEAR.PER_LEVEL("hh"),
        time_unit_id=None,
        grouping_level="hh",
        where="test",
    )
    expected = divide_by_grouping_level(unit=parse_unit("calendar_year"), level="hh")
    assert units_are_equivalent(left=resolved, right=expected)
    assert not units_are_equivalent(left=resolved, right=parse_unit("calendar_year"))


# ----------------------------------------------------------------------------
# Level-aware aggregation (resolved_unit_for_aggregation)
# ----------------------------------------------------------------------------


def test_resolved_aggregation_sum_swaps_person_to_hh():
    # SUM person -> hh swaps the denominator [person] -> [hh].
    source = divide_by_grouping_level(
        unit=parse_unit("CURRENCY / month"), level=PERSON_LEVEL
    )
    result = resolved_unit_for_aggregation(
        source_unit=source,
        agg_type=AggType.SUM,
        target_level="hh",
        source_level=PERSON_LEVEL,
    )
    expected = divide_by_grouping_level(unit=parse_unit("CURRENCY / month"), level="hh")
    assert units_are_equivalent(left=result, right=expected)


def test_resolved_aggregation_count_mints_person_over_target():
    # COUNT to hh yields [person]/[hh], independent of the source.
    source = divide_by_grouping_level(
        unit=parse_unit("CURRENCY / month"), level=PERSON_LEVEL
    )
    result = resolved_unit_for_aggregation(
        source_unit=source,
        agg_type=AggType.COUNT,
        target_level="hh",
        source_level=PERSON_LEVEL,
    )
    assert units_are_equivalent(
        left=result, right=grouping_level_count_unit(target_level="hh")
    )


def test_resolved_aggregation_min_over_level_less_source_acquires_target_level():
    # An extreme is a property of the target group whatever the source's base
    # (GEP 10): an ``_hh`` min of a bare month-duration age carries ``[hh]``.
    source = parse_unit("delta_calendar_month")
    result = resolved_unit_for_aggregation(
        source_unit=source,
        agg_type=AggType.MIN,
        target_level="hh",
        source_level=None,
    )
    expected = divide_by_grouping_level(unit=source, level="hh")
    assert units_are_equivalent(left=result, right=expected)


def test_resolved_aggregation_min_over_level_carrying_source_carries_target_level():
    source = divide_by_grouping_level(unit=parse_unit("CURRENCY"), level=PERSON_LEVEL)
    result = resolved_unit_for_aggregation(
        source_unit=source,
        agg_type=AggType.MIN,
        target_level="hh",
        source_level=PERSON_LEVEL,
    )
    expected = divide_by_grouping_level(unit=parse_unit("CURRENCY"), level="hh")
    assert units_are_equivalent(left=result, right=expected)


@pytest.mark.parametrize("agg_type", [AggType.ANY, AggType.ALL])
def test_resolved_aggregation_any_all_are_boolean_at_target_level(agg_type):
    # A boolean aggregation mints a boolean at its *target* level (GEP 10):
    # `1 / [hh]`, not a level-less dimensionless.
    source = divide_by_grouping_level(unit=parse_unit("CURRENCY"), level=PERSON_LEVEL)
    result = resolved_unit_for_aggregation(
        source_unit=source,
        agg_type=agg_type,
        target_level="hh",
        source_level=PERSON_LEVEL,
    )
    assert units_are_equivalent(
        left=result,
        right=divide_by_grouping_level(unit=UNIT_REGISTRY.dimensionless, level="hh"),
    )


def test_resolved_aggregation_sum_over_level_less_source_acquires_target_level():
    source = parse_unit("working_hour / week")
    result = resolved_unit_for_aggregation(
        source_unit=source,
        agg_type=AggType.SUM,
        target_level="hh",
        source_level=None,
    )
    expected = divide_by_grouping_level(unit=source, level="hh")
    assert units_are_equivalent(left=result, right=expected)


def test_resolved_aggregation_mean_resolves_to_the_individual_level():
    # A per-head average belongs to the person, whatever the target (GEP 10):
    # leveling it to the target would break ``mean · count = sum``.
    source = divide_by_grouping_level(unit=parse_unit(CURRENCY_TOKEN), level="hh")
    result = resolved_unit_for_aggregation(
        source_unit=source,
        agg_type=AggType.MEAN,
        target_level="sn",
        source_level="hh",
    )
    expected = divide_by_grouping_level(
        unit=parse_unit(CURRENCY_TOKEN), level=PERSON_LEVEL
    )
    assert units_are_equivalent(left=result, right=expected)


def test_resolved_aggregation_mean_over_level_less_source_stays_bare():
    # The person-level reading of an intensive base is bare, so an age's mean
    # stays comparable to level-less thresholds.
    source = parse_unit("delta_calendar_month")
    result = resolved_unit_for_aggregation(
        source_unit=source,
        agg_type=AggType.MEAN,
        target_level="hh",
        source_level=None,
    )
    assert units_are_equivalent(left=result, right=source)


def test_resolved_aggregation_mean_over_boolean_source_is_a_bare_share():
    # The mean of an indicator is a share: stripping the boolean's level leaves
    # no base to put at the person leaf.
    source = divide_by_grouping_level(unit=UNIT_REGISTRY.dimensionless, level="hh")
    result = resolved_unit_for_aggregation(
        source_unit=source,
        agg_type=AggType.MEAN,
        target_level="hh",
        source_level="hh",
    )
    assert units_are_equivalent(left=result, right=UNIT_REGISTRY.dimensionless)


def test_resolved_aggregation_min_over_leveled_calendar_point_swaps_level():
    # Re-leveling a calendar point must not trip pint's offset-arithmetic rules:
    # levels attach and strip via *unit* arithmetic (GEP 10).
    source = divide_by_grouping_level(unit=parse_unit("calendar_year"), level="hh")
    result = resolved_unit_for_aggregation(
        source_unit=source,
        agg_type=AggType.MIN,
        target_level="sn",
        source_level="hh",
    )
    expected = divide_by_grouping_level(unit=parse_unit("calendar_year"), level="sn")
    assert units_are_equivalent(left=result, right=expected)


# ----------------------------------------------------------------------------
# PERSON_COUNT: a declarable head count ([person] / [level])
# ----------------------------------------------------------------------------


def test_person_column_at_group_level_matches_a_count():
    # A PERSON_COUNT_PER_HH column resolves to [person]/[hh] — the same unit a COUNT
    # aggregation to hh mints, so a declaration and an aggregation compose and
    # compare cleanly (GEP 10).
    at_hh = resolve_compositional_param_unit(
        unit=Unit.PERSON_COUNT.PER_HH, where="test"
    )
    assert units_are_equivalent(
        left=at_hh, right=grouping_level_count_unit(target_level="hh")
    )


def test_person_at_person_level_is_dimensionless():
    # A head count per individual is [person]/[person] = a plain number. The
    # person leaf is implied, so the bare Unit.PERSON_COUNT count resolves there.
    at_person = resolve_compositional_param_unit(unit=Unit.PERSON_COUNT, where="test")
    assert units_are_equivalent(left=at_person, right=UNIT_REGISTRY.dimensionless)


def test_declared_person_per_group_bridges_like_a_count():
    # A *declared* PERSON_COUNT_PER_HH divides a per-[hh] amount down to a per-person
    # one, exactly as an aggregated COUNT would: the two are interchangeable.
    headcount_at_hh = resolve_compositional_param_unit(
        unit=Unit.PERSON_COUNT.PER_HH, where="test"
    )
    per_hh = divide_by_grouping_level(unit=parse_unit("CURRENCY / month"), level="hh")
    bridged = (
        UNIT_REGISTRY.Quantity(1.0, per_hh)
        / UNIT_REGISTRY.Quantity(1.0, headcount_at_hh)
    ).units
    expected = divide_by_grouping_level(
        unit=parse_unit("CURRENCY / month"), level=PERSON_LEVEL
    )
    assert units_are_equivalent(left=bridged, right=expected)


def test_count_aggregation_token_is_person():
    # COUNT mints the PERSON_COUNT placeholder unit (group and PID alike); the
    # level-aware resolved unit is recomputed downstream.
    assert (
        unit_for_aggregation(source_unit=Unit.DIMENSIONLESS, agg_type=AggType.COUNT)
        == Unit.PERSON_COUNT
    )


@pytest.mark.parametrize("agg_type", [AggType.ANY, AggType.ALL])
def test_any_all_aggregation_token_is_dimensionless(agg_type):
    assert (
        unit_for_aggregation(source_unit=Unit.DIMENSIONLESS, agg_type=agg_type)
        == Unit.DIMENSIONLESS
    )


def test_sum_aggregation_token_takes_the_target_level():
    assert (
        unit_for_aggregation(
            source_unit=Unit.CURRENCY.PER_MONTH,
            agg_type=AggType.SUM,
            target_level="hh",
        )
        == Unit.CURRENCY.PER_MONTH.PER_HH
    )


def test_min_aggregation_token_over_a_level_less_base_takes_the_target_level():
    assert (
        unit_for_aggregation(
            source_unit=Unit.MONTHS, agg_type=AggType.MIN, target_level="hh"
        )
        == Unit.MONTHS.PER_HH
    )


def test_mean_aggregation_token_strips_to_the_individual_spelling():
    # The individual reading is the level-less spelling: the person leaf is
    # implied for a level-carrying base, bare for an intensive one.
    assert (
        unit_for_aggregation(
            source_unit=Unit.CURRENCY.PER_MONTH.PER_HH,
            agg_type=AggType.MEAN,
            target_level="hh",
        )
        == Unit.CURRENCY.PER_MONTH
    )
