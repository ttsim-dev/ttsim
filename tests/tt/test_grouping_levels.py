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
    parse_unit,
    units_are_equivalent,
)
from ttsim.tt.units import (
    CURRENCY_TOKEN,
    PERSON_LEVEL,
    composite_base_is_extensive,
    divide_by_grouping_level,
    grouping_level_count_unit,
    register_currency,
    register_grouping_levels,
    resolve_compositional_param_unit,
    resolve_compositional_unit,
    resolved_unit_for_aggregation,
    unit_for_aggregation,
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
    ("base", "extensive"),
    [
        ("CURRENCY", True),
        ("PERSON", True),
        ("SQUARE_METER", True),
        ("HECTARE", True),
        ("YEARS", False),
        ("MONTHS", False),
        ("DAYS", False),
        ("CALENDAR_YEAR", False),
        ("CALENDAR_MONTH", False),
        ("CALENDAR_DAY", False),
        ("DIMENSIONLESS", False),
        ("HOURS", False),
    ],
)
def test_composite_base_is_extensive_defaults(base, extensive):
    # The extensive/intensive default: an extensive base gets the implied person
    # leaf level (and must spell a group level); an intensive one carries none.
    assert composite_base_is_extensive(base) is extensive


def test_spelled_level_carries_level():
    # `carries_level` now reports whether a level denominator is *spelled*.
    assert Unit.CURRENCY.PER_HH.carries_level
    assert not Unit.CURRENCY.carries_level


def test_concrete_currency_base_is_extensive():
    # A registered currency is an extensive base, like the agnostic CURRENCY.
    # Defined relative to the always-present CURRENCY reference unit so the test
    # is independent of which base currency the suite has registered.
    register_currency("LEVEL_TEST_COIN", definition=f"{CURRENCY_TOKEN} / 2")
    assert composite_base_is_extensive("LEVEL_TEST_COIN")


# ----------------------------------------------------------------------------
# A level is a denominator (divide_by_grouping_level)
# ----------------------------------------------------------------------------


def test_currency_flow_resolves_with_hh_denominator_at_level_hh():
    base = resolve_compositional_unit(Unit.CURRENCY.PER_MONTH)
    at_hh = divide_by_grouping_level(base, "hh")
    assert units_are_equivalent(
        left=at_hh,
        right=divide_by_grouping_level(parse_unit("CURRENCY / month"), "hh"),
    )


def test_currency_flow_resolves_with_person_denominator_at_individual_level():
    base = resolve_compositional_unit(Unit.CURRENCY.PER_MONTH)
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


def test_spelled_person_level_yields_per_person_unit():
    # sparerfreibetrag: a per-person yearly amount, fully spelled.
    resolved = resolve_compositional_param_unit(
        Unit.CURRENCY.PER_YEAR.PER_PERSON, where="test"
    )
    expected = divide_by_grouping_level(parse_unit("CURRENCY / year"), PERSON_LEVEL)
    assert units_are_equivalent(left=resolved, right=expected)


def test_absent_level_yields_no_level():
    resolved = resolve_compositional_param_unit(Unit.CURRENCY.PER_YEAR, where="test")
    per_person = divide_by_grouping_level(parse_unit("CURRENCY / year"), PERSON_LEVEL)
    assert units_are_equivalent(left=resolved, right=parse_unit("CURRENCY / year"))
    assert not units_are_equivalent(left=resolved, right=per_person)


def test_spelled_level_on_stock_param():
    # A non-flow per-group amount: CURRENCY at level hh.
    resolved = resolve_compositional_param_unit(Unit.CURRENCY.PER_HH, where="test")
    expected = divide_by_grouping_level(parse_unit("CURRENCY"), "hh")
    assert units_are_equivalent(left=resolved, right=expected)


def test_unknown_level_is_rejected():
    with pytest.raises(UnitDefinitionError, match="Unknown grouping level"):
        resolve_compositional_param_unit(
            Unit.CURRENCY.PER_YEAR.PER_LEVEL("not_a_level"), where="test"
        )


def test_spelled_level_on_scalar_param_with_name_suffix():
    # A scalar param spells its level and agrees with its name time suffix.
    resolved = resolve_compositional_param_unit(
        Unit.CURRENCY.PER_YEAR.PER_PERSON, time_unit_id="y", where="test"
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


def test_resolved_aggregation_min_levels_a_level_less_source():
    # MIN of a level-less MONTHS source acquires the target level (GEP 10, T8):
    # the `_xx` suffix and the unit level are always in sync, so even an intensive
    # age aggregated to a group carries that group's level — MONTHS / [hh].
    source = parse_unit("delta_calendar_month")
    result = resolved_unit_for_aggregation(
        source_unit=source,
        agg_type=AggType.MIN,
        target_level="hh",
        source_level=None,
    )
    expected = divide_by_grouping_level(parse_unit("delta_calendar_month"), "hh")
    assert units_are_equivalent(left=result, right=expected)


def test_resolved_aggregation_min_swaps_person_to_target():
    # MIN of a CURRENCY/[person] source swaps to the target level (GEP 10, T8):
    # like SUM, it carries the level its `_xx` suffix claims — CURRENCY / [hh],
    # not the source [person].
    source = divide_by_grouping_level(parse_unit("CURRENCY"), PERSON_LEVEL)
    result = resolved_unit_for_aggregation(
        source_unit=source,
        agg_type=AggType.MIN,
        target_level="hh",
        source_level=PERSON_LEVEL,
    )
    expected = divide_by_grouping_level(parse_unit("CURRENCY"), "hh")
    assert units_are_equivalent(left=result, right=expected)


@pytest.mark.parametrize("agg_type", [AggType.ANY, AggType.ALL])
def test_resolved_aggregation_any_all_are_boolean_at_target_level(agg_type):
    # A boolean aggregation mints a boolean at its *target* level (GEP 10):
    # `1 / [hh]`, not a level-less dimensionless.
    source = divide_by_grouping_level(parse_unit("CURRENCY"), PERSON_LEVEL)
    result = resolved_unit_for_aggregation(
        source_unit=source,
        agg_type=agg_type,
        target_level="hh",
        source_level=PERSON_LEVEL,
    )
    assert units_are_equivalent(
        left=result, right=divide_by_grouping_level(UNIT_REGISTRY.dimensionless, "hh")
    )


def test_resolved_aggregation_sum_over_level_less_source_acquires_target_level():
    # A level-less source summed to a group acquires the target level (GEP 10,
    # T8): the total working hours in a household is working_hour / week / [hh].
    source = parse_unit("working_hour / week")
    result = resolved_unit_for_aggregation(
        source_unit=source,
        agg_type=AggType.SUM,
        target_level="hh",
        source_level=None,
    )
    expected = divide_by_grouping_level(parse_unit("working_hour / week"), "hh")
    assert units_are_equivalent(left=result, right=expected)


# ----------------------------------------------------------------------------
# PERSON: a declarable head count ([person] / [level])
# ----------------------------------------------------------------------------


def test_person_per_group_carries_level():
    # The reference level enters as the denominator, like currency and area.
    assert Unit.PERSON.PER_HH.carries_level


def test_person_column_at_group_level_matches_a_count():
    # A PERSON_PER_HH column resolves to [person]/[hh] — the same unit a COUNT
    # aggregation to hh mints, so a declaration and an aggregation compose and
    # compare cleanly (GEP 10).
    at_hh = resolve_compositional_param_unit(Unit.PERSON.PER_HH, where="test")
    assert units_are_equivalent(
        left=at_hh, right=grouping_level_count_unit(target_level="hh")
    )


def test_person_at_person_level_is_dimensionless():
    # A head count per individual is [person]/[person] = a plain number.
    at_person = resolve_compositional_param_unit(
        Unit.PERSON.PER_LEVEL(PERSON_LEVEL), where="test"
    )
    assert units_are_equivalent(left=at_person, right=UNIT_REGISTRY.dimensionless)


def test_declared_person_per_group_bridges_like_a_count():
    # A *declared* PERSON_PER_HH divides a per-[hh] amount down to a per-person
    # one, exactly as an aggregated COUNT would: the two are interchangeable.
    headcount_at_hh = resolve_compositional_param_unit(Unit.PERSON.PER_HH, where="test")
    per_hh = divide_by_grouping_level(parse_unit("CURRENCY / month"), "hh")
    bridged = (
        UNIT_REGISTRY.Quantity(1.0, per_hh)
        / UNIT_REGISTRY.Quantity(1.0, headcount_at_hh)
    ).units
    expected = divide_by_grouping_level(parse_unit("CURRENCY / month"), PERSON_LEVEL)
    assert units_are_equivalent(left=bridged, right=expected)


def test_count_aggregation_token_is_person():
    # COUNT mints the PERSON placeholder unit (group and PID alike); the
    # level-aware resolved unit is recomputed downstream.
    assert (
        unit_for_aggregation(source_unit=Unit.DIMENSIONLESS, agg_type=AggType.COUNT)
        == Unit.PERSON
    )


@pytest.mark.parametrize("agg_type", [AggType.ANY, AggType.ALL])
def test_any_all_aggregation_token_is_dimensionless(agg_type):
    assert (
        unit_for_aggregation(source_unit=Unit.DIMENSIONLESS, agg_type=agg_type)
        == Unit.DIMENSIONLESS
    )
