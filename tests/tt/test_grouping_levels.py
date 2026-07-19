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
    TTSIMUnit,
)
from ttsim.tt.currencies import UnitSystem
from ttsim.tt.grouping_levels import register_grouping_levels
from ttsim.tt.units import (
    CURRENCY_TOKEN,
    PERSON_LEVEL,
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

# A representative policy system for the level-aware tests: its registry holds
# the grouping levels the level mechanics resolve against.
SYSTEM = UnitSystem(
    base_currency="CASTAR",
    other_currencies={"SILVER_PENNY": "CASTAR / 4"},
    statutory_currencies={"0001-01-01": "CASTAR"},
    grouping_levels=["hh", "bg", "sn"],
)
REGISTRY = SYSTEM.registry


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
    register_grouping_levels(names=["hh", "bg", "sn"], registry=REGISTRY)


def test_register_grouping_levels_always_registers_person():
    register_grouping_levels([], registry=REGISTRY)
    # `person` resolves to its own base dimension (the [person] count dimension).
    assert (
        divide_by_grouping_level(
            unit=parse_unit("CURRENCY", registry=REGISTRY),
            level=PERSON_LEVEL,
            registry=REGISTRY,
        )
        is not None
    )


def test_register_grouping_levels_is_idempotent():
    # Re-registering an already-known level is a tolerated no-op.
    register_grouping_levels(["hh"], registry=REGISTRY)
    register_grouping_levels(["hh", "bg"], registry=REGISTRY)
    first = divide_by_grouping_level(
        unit=parse_unit("CURRENCY", registry=REGISTRY), level="hh", registry=REGISTRY
    )
    second = divide_by_grouping_level(
        unit=parse_unit("CURRENCY", registry=REGISTRY), level="hh", registry=REGISTRY
    )
    assert units_are_equivalent(left=first, right=second, registry=REGISTRY)


def test_each_level_is_its_own_base_dimension():
    # No conversion between levels: hh and bg denominators are distinct dimensions.
    at_hh = divide_by_grouping_level(
        unit=parse_unit("CURRENCY / month", registry=REGISTRY),
        level="hh",
        registry=REGISTRY,
    )
    at_bg = divide_by_grouping_level(
        unit=parse_unit("CURRENCY / month", registry=REGISTRY),
        level="bg",
        registry=REGISTRY,
    )
    assert at_hh.dimensionality != at_bg.dimensionality
    assert not units_are_equivalent(left=at_hh, right=at_bg, registry=REGISTRY)


def test_unregistered_grouping_level_is_rejected():
    with pytest.raises(UnitDefinitionError, match="Unknown grouping level"):
        divide_by_grouping_level(
            unit=parse_unit("CURRENCY", registry=REGISTRY),
            level="eg_not_registered",
            registry=REGISTRY,
        )


# ----------------------------------------------------------------------------
# Which tokens carry a level (level-carrying/level-less default)
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# A level is a denominator (divide_by_grouping_level)
# ----------------------------------------------------------------------------


def test_currency_flow_resolves_with_hh_denominator_at_level_hh():
    base = resolve_compositional_unit(TTSIMUnit.CURRENCY.PER_MONTH, registry=REGISTRY)
    at_hh = divide_by_grouping_level(unit=base, level="hh", registry=REGISTRY)
    assert units_are_equivalent(
        left=at_hh,
        right=divide_by_grouping_level(
            unit=parse_unit("CURRENCY / month", registry=REGISTRY),
            level="hh",
            registry=REGISTRY,
        ),
        registry=REGISTRY,
    )


def test_currency_flow_resolves_with_person_denominator_at_individual_level():
    base = resolve_compositional_unit(TTSIMUnit.CURRENCY.PER_MONTH, registry=REGISTRY)
    at_person = divide_by_grouping_level(
        unit=base, level=PERSON_LEVEL, registry=REGISTRY
    )
    assert units_are_equivalent(
        left=at_person,
        right=divide_by_grouping_level(
            unit=parse_unit("CURRENCY / month", registry=REGISTRY),
            level=PERSON_LEVEL,
            registry=REGISTRY,
        ),
        registry=REGISTRY,
    )
    # Person and hh denominators are different dimensions.
    assert not units_are_equivalent(
        left=at_person,
        right=divide_by_grouping_level(unit=base, level="hh", registry=REGISTRY),
        registry=REGISTRY,
    )


@pytest.mark.parametrize(
    "token",
    [
        TTSIMUnit.YEARS,
        TTSIMUnit.HOURS.PER_WEEK,
        TTSIMUnit.DIMENSIONLESS,
    ],
)
def test_level_less_tokens_resolve_without_a_level(token):
    # A level-less unit carries no grouping denominator: the resolved unit is
    # the plain physical unit, unchanged by any level.
    resolved = resolve_compositional_unit(token, registry=REGISTRY)
    person_division = divide_by_grouping_level(
        unit=parse_unit("CURRENCY", registry=REGISTRY),
        level=PERSON_LEVEL,
        registry=REGISTRY,
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
        unit=parse_unit("CURRENCY / month", registry=REGISTRY),
        level="hh",
        registry=REGISTRY,
    )
    count_to_hh = grouping_level_count_unit(
        target_level="hh", registry=REGISTRY
    )  # [person]/[hh]
    bridged = (
        REGISTRY.Quantity(1.0, rent_at_hh) / REGISTRY.Quantity(1.0, count_to_hh)
    ).units
    expected = divide_by_grouping_level(
        unit=parse_unit("CURRENCY / month", registry=REGISTRY),
        level=PERSON_LEVEL,
        registry=REGISTRY,
    )
    assert units_are_equivalent(left=bridged, right=expected, registry=REGISTRY)


def test_count_bridges_person_to_sn_via_multiplication():
    # ([person]/[sn]) * (CURRENCY/year/[person]) == CURRENCY/year/[sn]: a
    # per-person allowance times a head count is a per-group amount.
    count_to_sn = grouping_level_count_unit(
        target_level="sn", registry=REGISTRY
    )  # [person]/[sn]
    per_person = divide_by_grouping_level(
        unit=parse_unit("CURRENCY / year", registry=REGISTRY),
        level=PERSON_LEVEL,
        registry=REGISTRY,
    )
    product = (
        REGISTRY.Quantity(1.0, count_to_sn) * REGISTRY.Quantity(1.0, per_person)
    ).units
    expected = divide_by_grouping_level(
        unit=parse_unit("CURRENCY / year", registry=REGISTRY),
        level="sn",
        registry=REGISTRY,
    )
    assert units_are_equivalent(left=product, right=expected, registry=REGISTRY)


def test_cross_level_addition_is_not_equivalent():
    # A unit at [hh] and one at [bg] are different dimensions: adding them across
    # levels is a mismatch the equivalence check catches.
    at_hh = divide_by_grouping_level(
        unit=parse_unit("CURRENCY / month", registry=REGISTRY),
        level="hh",
        registry=REGISTRY,
    )
    at_bg = divide_by_grouping_level(
        unit=parse_unit("CURRENCY / month", registry=REGISTRY),
        level="bg",
        registry=REGISTRY,
    )
    assert not units_are_equivalent(left=at_hh, right=at_bg, registry=REGISTRY)


# ----------------------------------------------------------------------------
# The person leaf is implied, group levels are spelled, on parameters
# ----------------------------------------------------------------------------


def test_person_level_can_be_spelled():
    # The person level is spelled like a group level (GEP 10): PER_PERSON
    # resolves to CURRENCY / year / [person].
    resolved = resolve_compositional_param_unit(
        unit=TTSIMUnit.CURRENCY.PER_YEAR.PER_PERSON, where="test", registry=REGISTRY
    )
    per_person = divide_by_grouping_level(
        unit=parse_unit("CURRENCY / year", registry=REGISTRY),
        level=PERSON_LEVEL,
        registry=REGISTRY,
    )
    assert units_are_equivalent(left=resolved, right=per_person, registry=REGISTRY)


def test_absent_level_is_level_neutral():
    # A bare CURRENCY_PER_YEAR carries no grouping level — level-neutral (GEP 10),
    # so it multiplies a leveled quantity without polluting its level. A per-person
    # amount spells its level (PER_PERSON); see test_person_level_can_be_spelled.
    resolved = resolve_compositional_param_unit(
        unit=TTSIMUnit.CURRENCY.PER_YEAR, where="test", registry=REGISTRY
    )
    assert units_are_equivalent(
        left=resolved,
        right=parse_unit("CURRENCY / year", registry=REGISTRY),
        registry=REGISTRY,
    )


def test_spelled_level_on_stock_param():
    # A non-flow per-group amount: CURRENCY at level hh.
    resolved = resolve_compositional_param_unit(
        unit=TTSIMUnit.CURRENCY.PER_HH, where="test", registry=REGISTRY
    )
    expected = divide_by_grouping_level(
        unit=parse_unit("CURRENCY", registry=REGISTRY), level="hh", registry=REGISTRY
    )
    assert units_are_equivalent(left=resolved, right=expected, registry=REGISTRY)


def test_unknown_level_is_rejected():
    with pytest.raises(UnitDefinitionError, match="Unknown grouping level"):
        resolve_compositional_param_unit(
            unit=TTSIMUnit.CURRENCY.PER_YEAR.PER_LEVEL("not_a_level"),
            where="test",
            registry=REGISTRY,
        )


def test_person_level_spelled_on_scalar_param_with_name_suffix():
    # A per-person scalar param spells PER_PERSON, and its spelled period agrees
    # with the name's time suffix.
    resolved = resolve_compositional_param_unit(
        unit=TTSIMUnit.CURRENCY.PER_YEAR.PER_PERSON,
        time_unit_id="y",
        where="test",
        registry=REGISTRY,
    )
    expected = divide_by_grouping_level(
        unit=parse_unit("CURRENCY / year", registry=REGISTRY),
        level=PERSON_LEVEL,
        registry=REGISTRY,
    )
    assert units_are_equivalent(left=resolved, right=expected, registry=REGISTRY)


# ----------------------------------------------------------------------------
# The level is declared, not read off the suffix, on columns
# ----------------------------------------------------------------------------


def test_column_omitting_the_level_is_level_neutral():
    # Omitting the level at a group suffix makes the column level-neutral — no
    # ``[bg]`` and no person leaf (GEP 10). A per-person amount constant within
    # the group spells PER_PERSON instead.
    resolved = resolve_compositional_column_unit(
        unit=TTSIMUnit.CURRENCY.PER_MONTH,
        time_unit_id="m",
        grouping_level="bg",
        where="test",
        registry=REGISTRY,
    )
    assert units_are_equivalent(
        left=resolved,
        right=parse_unit("CURRENCY / month", registry=REGISTRY),
        registry=REGISTRY,
    )


def test_intensive_column_omitting_the_level_stays_bare_at_a_group_suffix():
    resolved = resolve_compositional_column_unit(
        unit=TTSIMUnit.MONTHS,
        time_unit_id=None,
        grouping_level="bg",
        where="test",
        registry=REGISTRY,
    )
    assert units_are_equivalent(
        left=resolved,
        right=parse_unit("delta_calendar_month", registry=REGISTRY),
        registry=REGISTRY,
    )


def test_intensive_column_with_a_spelled_group_level_resolves():
    # GEP 10's ``alter_monate_jüngstes_mitglied_fg``: the family's property, so
    # the duration carries the group level — declared, not read off the base.
    resolved = resolve_compositional_column_unit(
        unit=TTSIMUnit.MONTHS.PER_LEVEL("hh"),
        time_unit_id=None,
        grouping_level="hh",
        where="test",
        registry=REGISTRY,
    )
    expected = divide_by_grouping_level(
        unit=parse_unit("delta_calendar_month", registry=REGISTRY),
        level="hh",
        registry=REGISTRY,
    )
    assert units_are_equivalent(left=resolved, right=expected, registry=REGISTRY)


def test_spelled_group_level_contradicting_the_suffix_is_rejected():
    with pytest.raises(UnitDefinitionError, match="must not contradict"):
        resolve_compositional_column_unit(
            unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_LEVEL("bg"),
            time_unit_id="m",
            grouping_level="hh",
            where="test",
            registry=REGISTRY,
        )


def test_person_level_is_allowed_on_a_group_suffixed_name():
    # A per-person amount constant within a group may be stored under a group
    # suffix (GEP 10): ``betrag_m_hh`` declaring CURRENCY_PER_MONTH_PER_PERSON is
    # the household member's amount, and resolves to CURRENCY / month / [person] —
    # PER_PERSON never contradicts the suffix, only a mismatched group level does.
    resolved = resolve_compositional_column_unit(
        unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_PERSON,
        time_unit_id="m",
        grouping_level="hh",
        where="test",
        registry=REGISTRY,
    )
    expected = divide_by_grouping_level(
        unit=parse_unit("CURRENCY / month", registry=REGISTRY),
        level=PERSON_LEVEL,
        registry=REGISTRY,
    )
    assert units_are_equivalent(left=resolved, right=expected, registry=REGISTRY)


def test_boolean_omitting_the_level_is_level_neutral():
    # A bare DIMENSIONLESS boolean is a level-neutral flag (GEP 10): omitting the
    # level yields plain dimensionless, no person leaf. A person-level indicator
    # spells DIMENSIONLESS_PER_PERSON, a group one its group.
    resolved = resolve_compositional_column_unit(
        unit=TTSIMUnit.DIMENSIONLESS,
        time_unit_id=None,
        grouping_level="hh",
        where="test",
        registry=REGISTRY,
    )
    assert units_are_equivalent(
        left=resolved, right=REGISTRY.dimensionless, registry=REGISTRY
    )


def test_person_level_indicator_carries_the_person_leaf():
    # A person-level boolean spells its level (GEP 10): DIMENSIONLESS_PER_PERSON
    # resolves to 1 / [person], so summing it over a group mints a head count.
    resolved = resolve_compositional_column_unit(
        unit=TTSIMUnit.DIMENSIONLESS.PER_PERSON,
        time_unit_id=None,
        grouping_level="person",
        where="test",
        registry=REGISTRY,
    )
    expected = divide_by_grouping_level(
        unit=REGISTRY.dimensionless, level=PERSON_LEVEL, registry=REGISTRY
    )
    assert units_are_equivalent(left=resolved, right=expected, registry=REGISTRY)


def test_calendar_point_carries_a_level():
    # GEP 10's ``baujahr_immobilie_hh``: the dwelling's construction year is the
    # household's property — a leveled calendar point. Attaching and comparing
    # the level must stay clear of pint's offset-arithmetic rules.
    resolved = resolve_compositional_column_unit(
        unit=TTSIMUnit.CALENDAR_YEAR.PER_LEVEL("hh"),
        time_unit_id=None,
        grouping_level="hh",
        where="test",
        registry=REGISTRY,
    )
    expected = divide_by_grouping_level(
        unit=parse_unit("calendar_year", registry=REGISTRY),
        level="hh",
        registry=REGISTRY,
    )
    assert units_are_equivalent(left=resolved, right=expected, registry=REGISTRY)
    assert not units_are_equivalent(
        left=resolved,
        right=parse_unit("calendar_year", registry=REGISTRY),
        registry=REGISTRY,
    )


# ----------------------------------------------------------------------------
# Level-aware aggregation (resolved_unit_for_aggregation)
# ----------------------------------------------------------------------------


def test_resolved_aggregation_sum_swaps_person_to_hh():
    # SUM person -> hh swaps the denominator [person] -> [hh].
    source = divide_by_grouping_level(
        unit=parse_unit("CURRENCY / month", registry=REGISTRY),
        level=PERSON_LEVEL,
        registry=REGISTRY,
    )
    result = resolved_unit_for_aggregation(
        source_unit=source,
        agg_type=AggType.SUM,
        target_level="hh",
        source_level=PERSON_LEVEL,
        registry=REGISTRY,
    )
    expected = divide_by_grouping_level(
        unit=parse_unit("CURRENCY / month", registry=REGISTRY),
        level="hh",
        registry=REGISTRY,
    )
    assert units_are_equivalent(left=result, right=expected, registry=REGISTRY)


def test_resolved_aggregation_count_mints_person_over_target():
    # COUNT to hh yields [person]/[hh], independent of the source.
    source = divide_by_grouping_level(
        unit=parse_unit("CURRENCY / month", registry=REGISTRY),
        level=PERSON_LEVEL,
        registry=REGISTRY,
    )
    result = resolved_unit_for_aggregation(
        source_unit=source,
        agg_type=AggType.COUNT,
        target_level="hh",
        source_level=PERSON_LEVEL,
        registry=REGISTRY,
    )
    assert units_are_equivalent(
        left=result,
        right=grouping_level_count_unit(target_level="hh", registry=REGISTRY),
        registry=REGISTRY,
    )


def test_resolved_aggregation_min_over_level_less_source_acquires_target_level():
    # An extreme is a property of the target group whatever the source's base
    # (GEP 10): an ``_hh`` min of a bare month-duration age carries ``[hh]``.
    source = parse_unit("delta_calendar_month", registry=REGISTRY)
    result = resolved_unit_for_aggregation(
        source_unit=source,
        agg_type=AggType.MIN,
        target_level="hh",
        source_level=None,
        registry=REGISTRY,
    )
    expected = divide_by_grouping_level(unit=source, level="hh", registry=REGISTRY)
    assert units_are_equivalent(left=result, right=expected, registry=REGISTRY)


def test_resolved_aggregation_min_over_level_carrying_source_carries_target_level():
    source = divide_by_grouping_level(
        unit=parse_unit("CURRENCY", registry=REGISTRY),
        level=PERSON_LEVEL,
        registry=REGISTRY,
    )
    result = resolved_unit_for_aggregation(
        source_unit=source,
        agg_type=AggType.MIN,
        target_level="hh",
        source_level=PERSON_LEVEL,
        registry=REGISTRY,
    )
    expected = divide_by_grouping_level(
        unit=parse_unit("CURRENCY", registry=REGISTRY), level="hh", registry=REGISTRY
    )
    assert units_are_equivalent(left=result, right=expected, registry=REGISTRY)


@pytest.mark.parametrize("agg_type", [AggType.ANY, AggType.ALL])
def test_resolved_aggregation_any_all_are_boolean_at_target_level(agg_type):
    # A boolean aggregation mints a boolean at its *target* level (GEP 10):
    # `1 / [hh]`, not a level-less dimensionless.
    source = divide_by_grouping_level(
        unit=parse_unit("CURRENCY", registry=REGISTRY),
        level=PERSON_LEVEL,
        registry=REGISTRY,
    )
    result = resolved_unit_for_aggregation(
        source_unit=source,
        agg_type=agg_type,
        target_level="hh",
        source_level=PERSON_LEVEL,
        registry=REGISTRY,
    )
    assert units_are_equivalent(
        left=result,
        right=divide_by_grouping_level(
            unit=REGISTRY.dimensionless, level="hh", registry=REGISTRY
        ),
        registry=REGISTRY,
    )


def test_resolved_aggregation_sum_over_level_less_source_acquires_target_level():
    source = parse_unit("working_hour / week", registry=REGISTRY)
    result = resolved_unit_for_aggregation(
        source_unit=source,
        agg_type=AggType.SUM,
        target_level="hh",
        source_level=None,
        registry=REGISTRY,
    )
    expected = divide_by_grouping_level(unit=source, level="hh", registry=REGISTRY)
    assert units_are_equivalent(left=result, right=expected, registry=REGISTRY)


def test_resolved_aggregation_mean_resolves_to_the_individual_level():
    # A per-head average belongs to the person, whatever the target (GEP 10):
    # leveling it to the target would break ``mean · count = sum``.
    source = divide_by_grouping_level(
        unit=parse_unit(CURRENCY_TOKEN, registry=REGISTRY),
        level="hh",
        registry=REGISTRY,
    )
    result = resolved_unit_for_aggregation(
        source_unit=source,
        agg_type=AggType.MEAN,
        target_level="sn",
        source_level="hh",
        registry=REGISTRY,
    )
    expected = divide_by_grouping_level(
        unit=parse_unit(CURRENCY_TOKEN, registry=REGISTRY),
        level=PERSON_LEVEL,
        registry=REGISTRY,
    )
    assert units_are_equivalent(left=result, right=expected, registry=REGISTRY)


def test_resolved_aggregation_mean_over_level_less_source_stays_bare():
    # The person-level reading of an intensive base is bare, so an age's mean
    # stays comparable to level-less thresholds.
    source = parse_unit("delta_calendar_month", registry=REGISTRY)
    result = resolved_unit_for_aggregation(
        source_unit=source,
        agg_type=AggType.MEAN,
        target_level="hh",
        source_level=None,
        registry=REGISTRY,
    )
    assert units_are_equivalent(left=result, right=source, registry=REGISTRY)


def test_resolved_aggregation_mean_over_boolean_source_is_a_bare_share():
    # The mean of an indicator is a share: stripping the boolean's level leaves
    # no base to put at the person leaf.
    source = divide_by_grouping_level(
        unit=REGISTRY.dimensionless, level="hh", registry=REGISTRY
    )
    result = resolved_unit_for_aggregation(
        source_unit=source,
        agg_type=AggType.MEAN,
        target_level="hh",
        source_level="hh",
        registry=REGISTRY,
    )
    assert units_are_equivalent(
        left=result, right=REGISTRY.dimensionless, registry=REGISTRY
    )


def test_resolved_aggregation_min_over_leveled_calendar_point_swaps_level():
    # Re-leveling a calendar point must not trip pint's offset-arithmetic rules:
    # levels attach and strip via *unit* arithmetic (GEP 10).
    source = divide_by_grouping_level(
        unit=parse_unit("calendar_year", registry=REGISTRY),
        level="hh",
        registry=REGISTRY,
    )
    result = resolved_unit_for_aggregation(
        source_unit=source,
        agg_type=AggType.MIN,
        target_level="sn",
        source_level="hh",
        registry=REGISTRY,
    )
    expected = divide_by_grouping_level(
        unit=parse_unit("calendar_year", registry=REGISTRY),
        level="sn",
        registry=REGISTRY,
    )
    assert units_are_equivalent(left=result, right=expected, registry=REGISTRY)


# ----------------------------------------------------------------------------
# PERSON_COUNT: a declarable head count ([person] / [level])
# ----------------------------------------------------------------------------


def test_person_column_at_group_level_matches_a_count():
    # A PERSON_COUNT_PER_HH column resolves to [person]/[hh] — the same unit a COUNT
    # aggregation to hh mints, so a declaration and an aggregation compose and
    # compare cleanly (GEP 10).
    at_hh = resolve_compositional_param_unit(
        unit=TTSIMUnit.PERSON_COUNT.PER_HH, where="test", registry=REGISTRY
    )
    assert units_are_equivalent(
        left=at_hh,
        right=grouping_level_count_unit(target_level="hh", registry=REGISTRY),
        registry=REGISTRY,
    )


def test_bare_person_count_is_a_headcount_numerator():
    # A bare PERSON_COUNT is the [person] head-count numerator, level-neutral —
    # no implied person leaf (GEP 10).
    resolved = resolve_compositional_param_unit(
        unit=TTSIMUnit.PERSON_COUNT, where="test", registry=REGISTRY
    )
    assert units_are_equivalent(
        left=resolved,
        right=parse_unit("grouping_level_person", registry=REGISTRY),
        registry=REGISTRY,
    )


def test_person_count_per_person_is_dimensionless():
    # A head count per individual is [person] / [person] = a plain number: the
    # person level is spelled, PERSON_COUNT_PER_PERSON.
    at_person = resolve_compositional_param_unit(
        unit=TTSIMUnit.PERSON_COUNT.PER_PERSON, where="test", registry=REGISTRY
    )
    assert units_are_equivalent(
        left=at_person, right=REGISTRY.dimensionless, registry=REGISTRY
    )


def test_declared_person_per_group_bridges_like_a_count():
    # A *declared* PERSON_COUNT_PER_HH divides a per-[hh] amount down to a per-person
    # one, exactly as an aggregated COUNT would: the two are interchangeable.
    headcount_at_hh = resolve_compositional_param_unit(
        unit=TTSIMUnit.PERSON_COUNT.PER_HH, where="test", registry=REGISTRY
    )
    per_hh = divide_by_grouping_level(
        unit=parse_unit("CURRENCY / month", registry=REGISTRY),
        level="hh",
        registry=REGISTRY,
    )
    bridged = (
        REGISTRY.Quantity(1.0, per_hh) / REGISTRY.Quantity(1.0, headcount_at_hh)
    ).units
    expected = divide_by_grouping_level(
        unit=parse_unit("CURRENCY / month", registry=REGISTRY),
        level=PERSON_LEVEL,
        registry=REGISTRY,
    )
    assert units_are_equivalent(left=bridged, right=expected, registry=REGISTRY)


def test_count_aggregation_token_is_person_count_at_target():
    # COUNT mints PERSON_COUNT at its target level; at the person target (an
    # agg_by_p_id COUNT) that is PERSON_COUNT_PER_PERSON = [person] / [person].
    assert (
        unit_for_aggregation(
            source_unit=TTSIMUnit.DIMENSIONLESS, agg_type=AggType.COUNT
        )
        == TTSIMUnit.PERSON_COUNT.PER_PERSON
    )


@pytest.mark.parametrize("agg_type", [AggType.ANY, AggType.ALL])
def test_any_all_aggregation_token_is_dimensionless_at_target(agg_type):
    # ANY / ALL yield a boolean at the target level; at the person target that is
    # DIMENSIONLESS_PER_PERSON = 1 / [person].
    assert (
        unit_for_aggregation(source_unit=TTSIMUnit.DIMENSIONLESS, agg_type=agg_type)
        == TTSIMUnit.DIMENSIONLESS.PER_PERSON
    )


def test_sum_aggregation_token_takes_the_target_level():
    assert (
        unit_for_aggregation(
            source_unit=TTSIMUnit.CURRENCY.PER_MONTH,
            agg_type=AggType.SUM,
            target_level="hh",
        )
        == TTSIMUnit.CURRENCY.PER_MONTH.PER_HH
    )


def test_min_aggregation_token_over_a_level_less_base_takes_the_target_level():
    assert (
        unit_for_aggregation(
            source_unit=TTSIMUnit.MONTHS, agg_type=AggType.MIN, target_level="hh"
        )
        == TTSIMUnit.MONTHS.PER_HH
    )


def test_mean_aggregation_token_swaps_a_leveled_source_to_the_person_leaf():
    # A per-head average belongs to the person: a leveled non-dimensionless source
    # swaps its group level for the spelled person leaf.
    assert (
        unit_for_aggregation(
            source_unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_HH,
            agg_type=AggType.MEAN,
            target_level="hh",
        )
        == TTSIMUnit.CURRENCY.PER_MONTH.PER_PERSON
    )


def test_mean_aggregation_token_of_a_boolean_stays_level_neutral():
    # A share — the mean of a person-level boolean — is level-neutral.
    assert (
        unit_for_aggregation(
            source_unit=TTSIMUnit.DIMENSIONLESS.PER_PERSON,
            agg_type=AggType.MEAN,
            target_level="hh",
        )
        == TTSIMUnit.DIMENSIONLESS
    )


def test_mean_aggregation_token_of_a_dimensionless_flow_keeps_the_person_leaf():
    # A dimensionless *flow* (a per-month rate) is not a bare share: its level-
    # stripped token still carries the period, so the mean keeps the person leaf,
    # matching what resolved_unit_for_aggregation derives.
    assert (
        unit_for_aggregation(
            source_unit=TTSIMUnit.DIMENSIONLESS.PER_MONTH.PER_PERSON,
            agg_type=AggType.MEAN,
            target_level="hh",
        )
        == TTSIMUnit.DIMENSIONLESS.PER_MONTH.PER_PERSON
    )


def test_mean_aggregation_token_of_a_physically_cancelling_ratio_is_neutral():
    # An area-over-area ratio at the person level resolves to dimensionless (the
    # physical parts cancel), so its mean is a level-neutral share — matching what
    # resolved_unit_for_aggregation derives after stripping the level.
    assert (
        unit_for_aggregation(
            source_unit=TTSIMUnit.SQUARE_METER.PER_SQUARE_METER.PER_PERSON,
            agg_type=AggType.MEAN,
            target_level="hh",
        )
        == TTSIMUnit.SQUARE_METER.PER_SQUARE_METER
    )
