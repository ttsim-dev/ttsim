"""Tests for the grouping-level dimensions and the head count (GEP 10).

These exercise the core level mechanics directly on the unit primitives:
levels as non-convertible base dimensions, the level-as-denominator
resolution, the dimensionless head-count bridge, cross-level rejection, and the
level-aware aggregation (SUM/MIN/MAX take the target level, MEAN goes bare,
COUNT mints ``1/[target]``, ANY/ALL a boolean at the target).

There is no ``person`` grouping level (GEP 10): an individual quantity is bare,
carrying no grouping level, and a head count is a dimensionless ``1/[group]``.
"""

from __future__ import annotations

import pint
import pytest

from ttsim.exceptions import UnitDefinitionError
from ttsim.tt import (
    AggType,
    TTSIMUnit,
)
from ttsim.tt.currencies import UnitSystem
from ttsim.tt.grouping_levels import register_grouping_levels
from ttsim.tt.units import (
    _ALLOWED_UNIT_TOKENS,
    CURRENCY_TOKEN,
    _unit_builder_levels,
    divide_by_grouping_level,
    parse_unit,
    resolve_compositional_column_unit,
    resolve_compositional_param_unit,
    resolve_compositional_unit,
    resolved_unit_for_aggregation,
    unit_for_aggregation,
    units_are_equivalent,
)

# A representative policy system for the level-aware tests. Registering the
# levels on its registry mirrors what the build does with the levels derived
# from the policy environment's `*_id` columns.
SYSTEM = UnitSystem(
    base_currency="CASTAR",
    other_currencies={"SILVER_PENNY": "CASTAR / 4"},
    statutory_currencies={"0001-01-01": "CASTAR"},
)
REGISTRY = SYSTEM.registry
register_grouping_levels(names=["hh", "bg", "sn"], registry=REGISTRY)


def test_register_grouping_levels_does_not_register_person():
    # There is no `person` grouping level (GEP 10): it is never registered as a
    # dimension, so dividing by it is rejected as an unknown level.
    register_grouping_levels(names=[], registry=REGISTRY)
    with pytest.raises(UnitDefinitionError, match="Unknown grouping level"):
        divide_by_grouping_level(
            unit=parse_unit(unit_str="CURRENCY", registry=REGISTRY),
            level="person",
            registry=REGISTRY,
        )


def test_register_grouping_levels_is_idempotent():
    # Re-registering an already-known level is a tolerated no-op.
    register_grouping_levels(names=["hh"], registry=REGISTRY)
    register_grouping_levels(names=["hh", "bg"], registry=REGISTRY)
    first = divide_by_grouping_level(
        unit=parse_unit(unit_str="CURRENCY", registry=REGISTRY),
        level="hh",
        registry=REGISTRY,
    )
    second = divide_by_grouping_level(
        unit=parse_unit(unit_str="CURRENCY", registry=REGISTRY),
        level="hh",
        registry=REGISTRY,
    )
    assert units_are_equivalent(left=first, right=second, registry=REGISTRY)


def test_person_is_rejected_as_a_grouping_level():
    """There is no individual grouping level (GEP 10), so a `person_id` column's
    `person` level is refused rather than registered as a dimension."""
    with pytest.raises(UnitDefinitionError, match="not a grouping level"):
        register_grouping_levels(names=["person"], registry=REGISTRY)


def test_non_lowercase_grouping_level_is_rejected():
    """A grouping level is registered verbatim but resolved lower-cased, so a
    non-lower-case name is refused rather than registered unresolvable."""
    with pytest.raises(UnitDefinitionError, match="must be lower-case"):
        register_grouping_levels(names=["HH"], registry=REGISTRY)


def test_grouping_level_colliding_with_a_period_step_is_rejected():
    """A grouping level may not claim a builder step a denominator already owns —
    a `month_id` column's `month` level would turn `PER_MONTH` into a level for
    the whole process."""
    with pytest.raises(UnitDefinitionError, match="already a unit denominator"):
        register_grouping_levels(names=["month"], registry=REGISTRY)


def test_malformed_grouping_level_widens_no_global_vocabulary():
    """A level name pint cannot define fails registration without widening the
    unit-token vocabulary or the fluent builder's level steps."""
    before = set(_ALLOWED_UNIT_TOKENS), set(_unit_builder_levels)
    with pytest.raises(pint.errors.DefinitionSyntaxError):
        register_grouping_levels(names=["["], registry=REGISTRY)
    assert (set(_ALLOWED_UNIT_TOKENS), set(_unit_builder_levels)) == before


def test_each_level_is_its_own_base_dimension():
    # No conversion between levels: hh and bg denominators are distinct dimensions.
    at_hh = divide_by_grouping_level(
        unit=parse_unit(unit_str="CURRENCY / month", registry=REGISTRY),
        level="hh",
        registry=REGISTRY,
    )
    at_bg = divide_by_grouping_level(
        unit=parse_unit(unit_str="CURRENCY / month", registry=REGISTRY),
        level="bg",
        registry=REGISTRY,
    )
    assert at_hh.dimensionality != at_bg.dimensionality
    assert not units_are_equivalent(left=at_hh, right=at_bg, registry=REGISTRY)


def test_unregistered_grouping_level_is_rejected():
    with pytest.raises(UnitDefinitionError, match="Unknown grouping level"):
        divide_by_grouping_level(
            unit=parse_unit(unit_str="CURRENCY", registry=REGISTRY),
            level="eg_not_registered",
            registry=REGISTRY,
        )


def test_currency_flow_resolves_with_hh_denominator_at_level_hh():
    base = resolve_compositional_unit(
        unit=TTSIMUnit.CURRENCY.PER_MONTH, registry=REGISTRY
    )
    at_hh = divide_by_grouping_level(unit=base, level="hh", registry=REGISTRY)
    assert units_are_equivalent(
        left=at_hh,
        right=divide_by_grouping_level(
            unit=parse_unit(unit_str="CURRENCY / month", registry=REGISTRY),
            level="hh",
            registry=REGISTRY,
        ),
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
def test_bare_tokens_resolve_without_a_level(token):
    # A bare unit carries no grouping denominator: the resolved unit is the plain
    # physical unit, unchanged by any level.
    resolved = resolve_compositional_unit(unit=token, registry=REGISTRY)
    hh_division = divide_by_grouping_level(
        unit=parse_unit(unit_str="CURRENCY", registry=REGISTRY),
        level="hh",
        registry=REGISTRY,
    )
    # It has no [hh] denominator: dividing CURRENCY by hh is a different dimension,
    # and the bare unit shares no level dimension with it.
    assert resolved.dimensionality != hh_division.dimensionality


def test_count_bridges_hh_to_bare_via_division():
    # (CURRENCY/month/[hh]) / (1/[hh]) == CURRENCY/month (bare per-person amount).
    rent_at_hh = divide_by_grouping_level(
        unit=parse_unit(unit_str="CURRENCY / month", registry=REGISTRY),
        level="hh",
        registry=REGISTRY,
    )
    count_to_hh = resolved_unit_for_aggregation(
        agg_type=AggType.COUNT, target_level="hh", registry=REGISTRY
    )  # 1/[hh]
    bridged = (
        REGISTRY.Quantity(1.0, rent_at_hh) / REGISTRY.Quantity(1.0, count_to_hh)
    ).units
    expected = parse_unit(unit_str="CURRENCY / month", registry=REGISTRY)
    assert units_are_equivalent(left=bridged, right=expected, registry=REGISTRY)


def test_count_bridges_bare_to_sn_via_multiplication():
    # (1/[sn]) * (CURRENCY/year) == CURRENCY/year/[sn]: a bare per-person allowance
    # times a head count is a per-group amount.
    count_to_sn = resolved_unit_for_aggregation(
        agg_type=AggType.COUNT, target_level="sn", registry=REGISTRY
    )  # 1/[sn]
    per_person = parse_unit(unit_str="CURRENCY / year", registry=REGISTRY)
    product = (
        REGISTRY.Quantity(1.0, count_to_sn) * REGISTRY.Quantity(1.0, per_person)
    ).units
    expected = divide_by_grouping_level(
        unit=parse_unit(unit_str="CURRENCY / year", registry=REGISTRY),
        level="sn",
        registry=REGISTRY,
    )
    assert units_are_equivalent(left=product, right=expected, registry=REGISTRY)


def test_absent_level_is_bare():
    # A bare CURRENCY_PER_YEAR carries no grouping level — it is a per-person /
    # level-neutral amount (GEP 10), so it multiplies a leveled quantity without
    # polluting its level.
    resolved = resolve_compositional_param_unit(
        unit=TTSIMUnit.CURRENCY.PER_YEAR, where="test", registry=REGISTRY
    )
    assert units_are_equivalent(
        left=resolved,
        right=parse_unit(unit_str="CURRENCY / year", registry=REGISTRY),
        registry=REGISTRY,
    )


def test_spelled_level_on_stock_param():
    # A non-flow per-group amount: CURRENCY at level hh.
    resolved = resolve_compositional_param_unit(
        unit=TTSIMUnit.CURRENCY.PER_HH, where="test", registry=REGISTRY
    )
    expected = divide_by_grouping_level(
        unit=parse_unit(unit_str="CURRENCY", registry=REGISTRY),
        level="hh",
        registry=REGISTRY,
    )
    assert units_are_equivalent(left=resolved, right=expected, registry=REGISTRY)


def test_unknown_level_is_rejected():
    with pytest.raises(UnitDefinitionError, match="Unknown grouping level"):
        resolve_compositional_param_unit(
            unit=TTSIMUnit.CURRENCY.PER_YEAR.PER_LEVEL("not_a_level"),
            where="test",
            registry=REGISTRY,
        )


def test_column_omitting_the_level_is_bare():
    # Omitting the level at a group suffix makes the column bare — no ``[bg]``
    # (GEP 10). A per-person amount constant within the group is bare too.
    resolved = resolve_compositional_column_unit(
        unit=TTSIMUnit.CURRENCY.PER_MONTH,
        time_unit_id="m",
        grouping_level="bg",
        where="test",
        registry=REGISTRY,
    )
    assert units_are_equivalent(
        left=resolved,
        right=parse_unit(unit_str="CURRENCY / month", registry=REGISTRY),
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
        right=parse_unit(unit_str="delta_calendar_month", registry=REGISTRY),
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
        unit=parse_unit(unit_str="delta_calendar_month", registry=REGISTRY),
        level="hh",
        registry=REGISTRY,
    )
    assert units_are_equivalent(left=resolved, right=expected, registry=REGISTRY)


def test_spelled_group_level_contradicting_the_suffix_is_rejected():
    with pytest.raises(UnitDefinitionError, match="must match the suffix"):
        resolve_compositional_column_unit(
            unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_LEVEL("bg"),
            time_unit_id="m",
            grouping_level="hh",
            where="test",
            registry=REGISTRY,
        )


def test_spelled_group_level_on_an_unsuffixed_name_is_rejected():
    # An unsuffixed name is bare; spelling a group level on it contradicts the
    # (absent) suffix (GEP 10).
    with pytest.raises(UnitDefinitionError, match="no level"):
        resolve_compositional_column_unit(
            unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_HH,
            time_unit_id="m",
            grouping_level=None,
            where="test",
            registry=REGISTRY,
        )


def test_boolean_omitting_the_level_is_bare():
    # A bare DIMENSIONLESS boolean is a level-neutral / individual flag (GEP 10):
    # omitting the level yields plain dimensionless. A group indicator spells its
    # group.
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


def test_group_level_indicator_carries_its_group_level():
    # A group-level boolean spells its level (GEP 10): DIMENSIONLESS_PER_HH
    # resolves to 1 / [hh].
    resolved = resolve_compositional_column_unit(
        unit=TTSIMUnit.DIMENSIONLESS.PER_HH,
        time_unit_id=None,
        grouping_level="hh",
        where="test",
        registry=REGISTRY,
    )
    expected = divide_by_grouping_level(
        unit=REGISTRY.dimensionless, level="hh", registry=REGISTRY
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
        unit=parse_unit(unit_str="calendar_year", registry=REGISTRY),
        level="hh",
        registry=REGISTRY,
    )
    assert units_are_equivalent(left=resolved, right=expected, registry=REGISTRY)
    assert not units_are_equivalent(
        left=resolved,
        right=parse_unit(unit_str="calendar_year", registry=REGISTRY),
        registry=REGISTRY,
    )


@pytest.mark.parametrize("source_spelling", ["CURRENCY / month", "working_hour / week"])
def test_resolved_aggregation_sum_over_bare_source_acquires_target_level(
    source_spelling,
):
    # SUM of a bare per-person source to hh acquires the [hh] denominator.
    source = parse_unit(unit_str=source_spelling, registry=REGISTRY)
    result = resolved_unit_for_aggregation(
        source_unit=source,
        agg_type=AggType.SUM,
        target_level="hh",
        source_level=None,
        registry=REGISTRY,
    )
    expected = divide_by_grouping_level(unit=source, level="hh", registry=REGISTRY)
    assert units_are_equivalent(left=result, right=expected, registry=REGISTRY)


def test_resolved_aggregation_count_mints_dimensionless_over_target():
    # COUNT to hh yields 1/[hh], independent of the source.
    result = resolved_unit_for_aggregation(
        agg_type=AggType.COUNT,
        target_level="hh",
        registry=REGISTRY,
    )
    assert units_are_equivalent(
        left=result,
        right=divide_by_grouping_level(
            unit=REGISTRY.dimensionless, level="hh", registry=REGISTRY
        ),
        registry=REGISTRY,
    )


def test_resolved_aggregation_count_to_individual_target_is_bare_dimensionless():
    # An agg_by_p_id COUNT (individual target) is bare dimensionless.
    result = resolved_unit_for_aggregation(
        agg_type=AggType.COUNT,
        target_level=None,
        registry=REGISTRY,
    )
    assert units_are_equivalent(
        left=result, right=REGISTRY.dimensionless, registry=REGISTRY
    )


def test_resolved_aggregation_min_over_bare_source_acquires_target_level():
    # An extreme is a property of the target group whatever the source's base
    # (GEP 10): an ``_hh`` min of a bare month-duration age carries ``[hh]``.
    source = parse_unit(unit_str="delta_calendar_month", registry=REGISTRY)
    result = resolved_unit_for_aggregation(
        source_unit=source,
        agg_type=AggType.MIN,
        target_level="hh",
        source_level=None,
        registry=REGISTRY,
    )
    expected = divide_by_grouping_level(unit=source, level="hh", registry=REGISTRY)
    assert units_are_equivalent(left=result, right=expected, registry=REGISTRY)


@pytest.mark.parametrize("agg_type", [AggType.ANY, AggType.ALL])
def test_resolved_aggregation_any_all_are_boolean_at_target_level(agg_type):
    # A boolean aggregation mints a boolean at its *target* level (GEP 10):
    # `1 / [hh]`, not a bare dimensionless.
    result = resolved_unit_for_aggregation(
        agg_type=agg_type,
        target_level="hh",
        registry=REGISTRY,
    )
    assert units_are_equivalent(
        left=result,
        right=divide_by_grouping_level(
            unit=REGISTRY.dimensionless, level="hh", registry=REGISTRY
        ),
        registry=REGISTRY,
    )


def test_resolved_aggregation_mean_resolves_to_bare():
    # A per-head average belongs to the individual, whatever the target (GEP 10):
    # leveling it to the target would break ``mean · count = sum``. The source
    # level is dropped, leaving a bare per-person amount.
    source = divide_by_grouping_level(
        unit=parse_unit(unit_str=CURRENCY_TOKEN, registry=REGISTRY),
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
    expected = parse_unit(unit_str=CURRENCY_TOKEN, registry=REGISTRY)
    assert units_are_equivalent(left=result, right=expected, registry=REGISTRY)


def test_resolved_aggregation_mean_over_bare_source_stays_bare():
    # The individual reading of an intensive base is bare, so an age's mean stays
    # comparable to bare thresholds.
    source = parse_unit(unit_str="delta_calendar_month", registry=REGISTRY)
    result = resolved_unit_for_aggregation(
        source_unit=source,
        agg_type=AggType.MEAN,
        target_level="hh",
        source_level=None,
        registry=REGISTRY,
    )
    assert units_are_equivalent(left=result, right=source, registry=REGISTRY)


def test_resolved_aggregation_mean_over_boolean_source_is_a_bare_share():
    # The mean of an indicator is a share: stripping the boolean's level leaves a
    # bare dimensionless number.
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


def test_resolved_aggregation_sum_to_individual_target_drops_source_level():
    # An agg_by_p_id SUM (individual target) over a leveled source lands on a
    # person as a bare amount.
    source = divide_by_grouping_level(
        unit=parse_unit(unit_str="CURRENCY", registry=REGISTRY),
        level="hh",
        registry=REGISTRY,
    )
    result = resolved_unit_for_aggregation(
        source_unit=source,
        agg_type=AggType.SUM,
        target_level=None,
        source_level="hh",
        registry=REGISTRY,
    )
    assert units_are_equivalent(
        left=result,
        right=parse_unit(unit_str="CURRENCY", registry=REGISTRY),
        registry=REGISTRY,
    )


def test_resolved_aggregation_min_over_leveled_calendar_point_swaps_level():
    # Re-leveling a calendar point must not trip pint's offset-arithmetic rules:
    # levels attach and strip via *unit* arithmetic (GEP 10).
    source = divide_by_grouping_level(
        unit=parse_unit(unit_str="calendar_year", registry=REGISTRY),
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
        unit=parse_unit(unit_str="calendar_year", registry=REGISTRY),
        level="sn",
        registry=REGISTRY,
    )
    assert units_are_equivalent(left=result, right=expected, registry=REGISTRY)


def test_declared_head_count_at_group_level_matches_a_count():
    # A DIMENSIONLESS_PER_HH column resolves to 1/[hh] — the same unit a COUNT
    # aggregation to hh mints, so a declaration and an aggregation compose and
    # compare cleanly (GEP 10).
    at_hh = resolve_compositional_param_unit(
        unit=TTSIMUnit.DIMENSIONLESS.PER_HH, where="test", registry=REGISTRY
    )
    assert units_are_equivalent(
        left=at_hh,
        right=resolved_unit_for_aggregation(
            agg_type=AggType.COUNT, target_level="hh", registry=REGISTRY
        ),
        registry=REGISTRY,
    )


def test_count_aggregation_token_is_bare_dimensionless_at_individual_target():
    # COUNT mints a head count at its target level; at the individual target (an
    # agg_by_p_id COUNT) that is the bare dimensionless unit.
    assert (
        unit_for_aggregation(
            source_unit=TTSIMUnit.DIMENSIONLESS, agg_type=AggType.COUNT
        )
        == TTSIMUnit.DIMENSIONLESS
    )


@pytest.mark.parametrize("agg_type", [AggType.ANY, AggType.ALL])
def test_any_all_aggregation_token_is_bare_dimensionless_at_individual_target(agg_type):
    # ANY / ALL yield a boolean at the target level; at the individual target that
    # is bare DIMENSIONLESS.
    assert (
        unit_for_aggregation(source_unit=TTSIMUnit.DIMENSIONLESS, agg_type=agg_type)
        == TTSIMUnit.DIMENSIONLESS
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


def test_min_aggregation_token_over_a_bare_base_takes_the_target_level():
    assert (
        unit_for_aggregation(
            source_unit=TTSIMUnit.MONTHS, agg_type=AggType.MIN, target_level="hh"
        )
        == TTSIMUnit.MONTHS.PER_HH
    )


def test_mean_aggregation_token_drops_a_leveled_source_to_bare():
    # A per-head average belongs to the individual: a leveled source drops its
    # group level, leaving a bare per-person amount.
    assert (
        unit_for_aggregation(
            source_unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_HH,
            agg_type=AggType.MEAN,
            target_level="hh",
        )
        == TTSIMUnit.CURRENCY.PER_MONTH
    )


def test_mean_aggregation_token_of_a_boolean_is_bare():
    # A share — the mean of a boolean — is bare.
    assert (
        unit_for_aggregation(
            source_unit=TTSIMUnit.DIMENSIONLESS.PER_HH,
            agg_type=AggType.MEAN,
            target_level="hh",
        )
        == TTSIMUnit.DIMENSIONLESS
    )
