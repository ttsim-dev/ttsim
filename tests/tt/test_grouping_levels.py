"""Tests for the grouping-level dimensions and the [person] count (GEP 10).

These exercise the core level mechanics directly on the unit primitives:
levels as non-convertible base dimensions, the level-as-denominator
resolution, the [person] count bridge, cross-level rejection, the
``reference_level`` parameter facet, and the level-aware aggregation
(SUM swaps, COUNT mints ``[person]/[target]``, MIN/MAX/MEAN preserve).
"""

from __future__ import annotations

import pytest

from ttsim.exceptions import UnitDefinitionError
from ttsim.tt import (
    UNIT_REGISTRY,
    AggType,
    Unit,
    coerce_unit_token,
    parse_unit,
    units_are_equivalent,
)
from ttsim.tt.units import (
    CURRENCY_TOKEN,
    PERSON_LEVEL,
    divide_by_grouping_level,
    grouping_level_count_unit,
    register_currency,
    register_grouping_levels,
    resolve_column_unit,
    resolve_param_unit,
    resolve_scalar_param_unit,
    resolved_unit_for_aggregation,
    unit_for_aggregation,
    unit_token_carries_level,
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
    assert divide_by_grouping_level(parse_unit("CURRENCY"), PERSON_LEVEL) is not None


def test_register_grouping_levels_is_idempotent():
    # Re-registering an already-known level is a tolerated no-op.
    register_grouping_levels(["hh"])
    register_grouping_levels(["hh", "bg"])
    first = divide_by_grouping_level(parse_unit("CURRENCY"), "hh")
    second = divide_by_grouping_level(parse_unit("CURRENCY"), "hh")
    assert units_are_equivalent(left=first, right=second)


def test_each_level_is_its_own_base_dimension():
    # No conversion between levels: hh and bg denominators are distinct dimensions.
    at_hh = divide_by_grouping_level(parse_unit("CURRENCY / month"), "hh")
    at_bg = divide_by_grouping_level(parse_unit("CURRENCY / month"), "bg")
    assert at_hh.dimensionality != at_bg.dimensionality
    assert not units_are_equivalent(left=at_hh, right=at_bg)


def test_unregistered_grouping_level_is_rejected():
    with pytest.raises(UnitDefinitionError, match="Unknown grouping level"):
        divide_by_grouping_level(parse_unit("CURRENCY"), "eg_not_registered")


# ----------------------------------------------------------------------------
# Which tokens carry a level (extensive/intensive default)
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("token", "carries"),
    [
        (Unit.CURRENCY, True),
        (Unit.CURRENCY_FLOW, True),
        (Unit.SQUARE_METERS, True),
        (Unit.HECTARES, True),
        (Unit.YEARS, False),
        (Unit.MONTHS, False),
        (Unit.DAYS, False),
        (Unit.CALENDAR_YEAR, False),
        (Unit.CALENDAR_MONTH, False),
        (Unit.CALENDAR_DAY, False),
        (Unit.DIMENSIONLESS, False),
        (Unit.DIMENSIONLESS_FLOW, False),
        (Unit.HOURS_FLOW, False),
        (Unit.CURRENCY_PER_SQUARE_METER_FLOW, False),
    ],
)
def test_unit_token_carries_level_defaults(token, carries):
    assert unit_token_carries_level(token) is carries


def test_concrete_currency_token_inherits_level_default():
    # A registered currency inherits its agnostic counterpart's level default.
    # Defined relative to the always-present CURRENCY reference unit so the test
    # is independent of which base currency the suite has registered.
    register_currency("LEVEL_TEST_COIN", definition=f"{CURRENCY_TOKEN} / 2")
    flow = coerce_unit_token("LEVEL_TEST_COIN_FLOW", where="test")
    assert unit_token_carries_level(flow)


# ----------------------------------------------------------------------------
# A level is a denominator (divide_by_grouping_level)
# ----------------------------------------------------------------------------


def test_currency_flow_resolves_with_hh_denominator_at_level_hh():
    base = resolve_column_unit(token=Unit.CURRENCY_FLOW, time_unit_id="m")
    at_hh = divide_by_grouping_level(base, "hh")
    assert units_are_equivalent(
        left=at_hh,
        right=divide_by_grouping_level(parse_unit("CURRENCY / month"), "hh"),
    )


def test_currency_flow_resolves_with_person_denominator_at_individual_level():
    base = resolve_column_unit(token=Unit.CURRENCY_FLOW, time_unit_id="m")
    at_person = divide_by_grouping_level(base, PERSON_LEVEL)
    assert units_are_equivalent(
        left=at_person,
        right=divide_by_grouping_level(parse_unit("CURRENCY / month"), PERSON_LEVEL),
    )
    # Person and hh denominators are different dimensions.
    assert not units_are_equivalent(
        left=at_person, right=divide_by_grouping_level(base, "hh")
    )


@pytest.mark.parametrize(
    ("token", "time_unit_id"),
    [
        (Unit.YEARS, None),
        (Unit.HOURS_FLOW, "w"),
        (Unit.DIMENSIONLESS, None),
    ],
)
def test_level_less_tokens_resolve_without_a_level(token, time_unit_id):
    # A level-less token carries no grouping denominator: the resolved unit is
    # the plain physical unit, unchanged by any level.
    resolved = resolve_column_unit(token=token, time_unit_id=time_unit_id)
    person_division = divide_by_grouping_level(parse_unit("CURRENCY"), PERSON_LEVEL)
    # It has no [person] denominator: dividing CURRENCY by person is a different
    # dimension, and the level-less unit shares no level dimension with it.
    assert resolved.dimensionality != person_division.dimensionality


# ----------------------------------------------------------------------------
# The [person] count dimension: the bridge cancels
# ----------------------------------------------------------------------------


def test_count_bridges_hh_to_person_via_division():
    # (CURRENCY/month/[hh]) / ([person]/[hh]) == CURRENCY/month/[person].
    rent_at_hh = divide_by_grouping_level(parse_unit("CURRENCY / month"), "hh")
    count_to_hh = grouping_level_count_unit(target_level="hh")  # [person]/[hh]
    bridged = (
        UNIT_REGISTRY.Quantity(1.0, rent_at_hh)
        / UNIT_REGISTRY.Quantity(1.0, count_to_hh)
    ).units
    expected = divide_by_grouping_level(parse_unit("CURRENCY / month"), PERSON_LEVEL)
    assert units_are_equivalent(left=bridged, right=expected)


def test_count_bridges_person_to_sn_via_multiplication():
    # ([person]/[sn]) * (CURRENCY/year/[person]) == CURRENCY/year/[sn]: a
    # per-person allowance times a head count is a per-group amount.
    count_to_sn = grouping_level_count_unit(target_level="sn")  # [person]/[sn]
    per_person = divide_by_grouping_level(parse_unit("CURRENCY / year"), PERSON_LEVEL)
    product = (
        UNIT_REGISTRY.Quantity(1.0, count_to_sn)
        * UNIT_REGISTRY.Quantity(1.0, per_person)
    ).units
    expected = divide_by_grouping_level(parse_unit("CURRENCY / year"), "sn")
    assert units_are_equivalent(left=product, right=expected)


def test_cross_level_addition_is_not_equivalent():
    # A unit at [hh] and one at [bg] are different dimensions: adding them across
    # levels is a mismatch the equivalence check catches.
    at_hh = divide_by_grouping_level(parse_unit("CURRENCY / month"), "hh")
    at_bg = divide_by_grouping_level(parse_unit("CURRENCY / month"), "bg")
    assert not units_are_equivalent(left=at_hh, right=at_bg)


# ----------------------------------------------------------------------------
# reference_level on parameters
# ----------------------------------------------------------------------------


def test_reference_level_person_yields_per_person_unit():
    # sparerfreibetrag: EUR_FLOW, reference_period Year, reference_level Person.
    resolved = resolve_param_unit(
        token=Unit.CURRENCY_FLOW,
        reference_period="Year",
        reference_level=PERSON_LEVEL,
    )
    expected = divide_by_grouping_level(parse_unit("CURRENCY / year"), PERSON_LEVEL)
    assert units_are_equivalent(left=resolved, right=expected)


def test_absent_reference_level_yields_no_level():
    resolved = resolve_param_unit(
        token=Unit.CURRENCY_FLOW, reference_period="Year", reference_level=None
    )
    per_person = divide_by_grouping_level(parse_unit("CURRENCY / year"), PERSON_LEVEL)
    assert units_are_equivalent(left=resolved, right=parse_unit("CURRENCY / year"))
    assert not units_are_equivalent(left=resolved, right=per_person)


def test_reference_level_on_stock_param():
    # A non-flow per-group amount: CURRENCY at level hh.
    resolved = resolve_param_unit(
        token=Unit.CURRENCY, reference_period=None, reference_level="hh"
    )
    expected = divide_by_grouping_level(parse_unit("CURRENCY"), "hh")
    assert units_are_equivalent(left=resolved, right=expected)


def test_reference_level_unknown_level_is_rejected():
    with pytest.raises(UnitDefinitionError, match="Unknown grouping level"):
        resolve_param_unit(
            token=Unit.CURRENCY_FLOW,
            reference_period="Year",
            reference_level="not_a_level",
        )


def test_reference_level_allowed_on_scalar_param():
    # Scalar params forbid reference_period but DO allow reference_level (they
    # have no aggregation suffix to read a level from).
    resolved = resolve_scalar_param_unit(
        token=Unit.CURRENCY_FLOW, time_unit_id="y", reference_level=PERSON_LEVEL
    )
    expected = divide_by_grouping_level(parse_unit("CURRENCY / year"), PERSON_LEVEL)
    assert units_are_equivalent(left=resolved, right=expected)


# ----------------------------------------------------------------------------
# Level-aware aggregation (resolved_unit_for_aggregation)
# ----------------------------------------------------------------------------


def test_resolved_aggregation_sum_swaps_person_to_hh():
    # SUM person -> hh swaps the denominator [person] -> [hh].
    source = divide_by_grouping_level(parse_unit("CURRENCY / month"), PERSON_LEVEL)
    result = resolved_unit_for_aggregation(
        source_unit=source,
        agg_type=AggType.SUM,
        target_level="hh",
        source_level=PERSON_LEVEL,
    )
    expected = divide_by_grouping_level(parse_unit("CURRENCY / month"), "hh")
    assert units_are_equivalent(left=result, right=expected)


def test_resolved_aggregation_count_mints_person_over_target():
    # COUNT to hh yields [person]/[hh], independent of the source.
    source = divide_by_grouping_level(parse_unit("CURRENCY / month"), PERSON_LEVEL)
    result = resolved_unit_for_aggregation(
        source_unit=source,
        agg_type=AggType.COUNT,
        target_level="hh",
        source_level=PERSON_LEVEL,
    )
    assert units_are_equivalent(
        left=result, right=grouping_level_count_unit(target_level="hh")
    )


def test_resolved_aggregation_min_preserves_level_less_source():
    # MIN of a level-less MONTHS source stays level-less MONTHS.
    source = parse_unit("month")
    result = resolved_unit_for_aggregation(
        source_unit=source,
        agg_type=AggType.MIN,
        target_level="fg",
        source_level=None,
    )
    assert units_are_equivalent(left=result, right=parse_unit("month"))


def test_resolved_aggregation_min_preserves_person_level_source():
    # MIN of a CURRENCY/[person] source stays CURRENCY/[person] (does not swap).
    source = divide_by_grouping_level(parse_unit("CURRENCY"), PERSON_LEVEL)
    result = resolved_unit_for_aggregation(
        source_unit=source,
        agg_type=AggType.MIN,
        target_level="hh",
        source_level=PERSON_LEVEL,
    )
    assert units_are_equivalent(left=result, right=source)


@pytest.mark.parametrize("agg_type", [AggType.ANY, AggType.ALL])
def test_resolved_aggregation_any_all_are_dimensionless(agg_type):
    source = divide_by_grouping_level(parse_unit("CURRENCY"), PERSON_LEVEL)
    result = resolved_unit_for_aggregation(
        source_unit=source,
        agg_type=agg_type,
        target_level="hh",
        source_level=PERSON_LEVEL,
    )
    assert units_are_equivalent(left=result, right=UNIT_REGISTRY.dimensionless)


def test_resolved_aggregation_sum_over_level_less_source_is_unchanged():
    # A level-less source summed has no level to swap.
    source = parse_unit("hour / week")
    result = resolved_unit_for_aggregation(
        source_unit=source,
        agg_type=AggType.SUM,
        target_level="hh",
        source_level=None,
    )
    assert units_are_equivalent(left=result, right=parse_unit("hour / week"))


# ----------------------------------------------------------------------------
# HEADCOUNT: a declarable head count ([person] / [level])
# ----------------------------------------------------------------------------


def test_headcount_carries_level():
    # The reference level enters as the denominator, like currency and area.
    assert unit_token_carries_level(Unit.HEADCOUNT)


def test_headcount_forbids_a_time_suffix():
    # HEADCOUNT is complete as written (a stock, not a flow); a time suffix on the
    # name denotes a flow and is rejected.
    with pytest.raises(UnitDefinitionError):
        resolve_column_unit(token=Unit.HEADCOUNT, time_unit_id="m")


def test_headcount_column_at_group_level_matches_a_count():
    # A HEADCOUNT column at a group level resolves to [person]/[hh] — the same unit
    # a COUNT aggregation to hh mints, so a declaration and an aggregation compose
    # and compare cleanly (GEP 10).
    base = resolve_column_unit(token=Unit.HEADCOUNT, time_unit_id=None)
    at_hh = divide_by_grouping_level(base, "hh")
    assert units_are_equivalent(
        left=at_hh, right=grouping_level_count_unit(target_level="hh")
    )


def test_headcount_at_person_level_is_dimensionless():
    # A head count per individual is [person]/[person] = a plain number.
    base = resolve_column_unit(token=Unit.HEADCOUNT, time_unit_id=None)
    at_person = divide_by_grouping_level(base, PERSON_LEVEL)
    assert units_are_equivalent(left=at_person, right=UNIT_REGISTRY.dimensionless)


def test_declared_headcount_bridges_like_a_count():
    # A *declared* HEADCOUNT/[hh] divides a per-[hh] amount down to a per-person
    # one, exactly as an aggregated COUNT would: the two are interchangeable.
    headcount_at_hh = divide_by_grouping_level(
        resolve_column_unit(token=Unit.HEADCOUNT, time_unit_id=None), "hh"
    )
    per_hh = divide_by_grouping_level(parse_unit("CURRENCY / month"), "hh")
    bridged = (
        UNIT_REGISTRY.Quantity(1.0, per_hh)
        / UNIT_REGISTRY.Quantity(1.0, headcount_at_hh)
    ).units
    expected = divide_by_grouping_level(parse_unit("CURRENCY / month"), PERSON_LEVEL)
    assert units_are_equivalent(left=bridged, right=expected)


def test_headcount_param_per_group():
    resolved = resolve_param_unit(
        token=Unit.HEADCOUNT, reference_period=None, reference_level="hh"
    )
    assert units_are_equivalent(
        left=resolved, right=grouping_level_count_unit(target_level="hh")
    )


def test_headcount_param_per_person_is_dimensionless():
    resolved = resolve_param_unit(
        token=Unit.HEADCOUNT, reference_period=None, reference_level=PERSON_LEVEL
    )
    assert units_are_equivalent(left=resolved, right=UNIT_REGISTRY.dimensionless)


def test_headcount_scalar_param_per_group():
    resolved = resolve_scalar_param_unit(
        token=Unit.HEADCOUNT, time_unit_id=None, reference_level="bg"
    )
    assert units_are_equivalent(
        left=resolved, right=grouping_level_count_unit(target_level="bg")
    )


def test_headcount_param_without_reference_level_is_rejected():
    # A head count is always persons per something; a bare HEADCOUNT param would be
    # an absolute [person] count per nothing.
    with pytest.raises(UnitDefinitionError, match="must set `reference_level`"):
        resolve_param_unit(
            token=Unit.HEADCOUNT, reference_period=None, reference_level=None
        )


def test_headcount_scalar_param_without_reference_level_is_rejected():
    with pytest.raises(UnitDefinitionError, match="must set `reference_level`"):
        resolve_scalar_param_unit(
            token=Unit.HEADCOUNT, time_unit_id=None, reference_level=None
        )


def test_count_aggregation_token_is_headcount():
    # COUNT mints the HEADCOUNT placeholder token (group and PID alike); the
    # level-aware resolved unit is recomputed downstream.
    assert (
        unit_for_aggregation(source_unit=Unit.DIMENSIONLESS, agg_type=AggType.COUNT)
        == Unit.HEADCOUNT
    )


@pytest.mark.parametrize("agg_type", [AggType.ANY, AggType.ALL])
def test_any_all_aggregation_token_is_dimensionless(agg_type):
    assert (
        unit_for_aggregation(source_unit=Unit.DIMENSIONLESS, agg_type=agg_type)
        == Unit.DIMENSIONLESS
    )
