"""Tests for the pint-based unit framework (GEP 10, issue #117)."""

from __future__ import annotations

# Importing the mettsim package registers its base currency (``castar``) and
# makes the tracer-bullet policy function importable.
import mettsim.middle_earth  # noqa: F401
import numpy as np
import pytest
from beartype.roar import BeartypeCallHintViolation

from ttsim import unit_converters
from ttsim.exceptions import (
    PolicyFunctionDefinitionError,
    PolicyInputDefinitionError,
    UnitConsistencyError,
    UnitDefinitionError,
)
from ttsim.interface_dag_elements.automatically_added_functions import (
    create_agg_by_group_functions,
    create_time_conversion_functions,
)
from ttsim.tt import (
    UNSET_UNIT,
    AggType,
    CompositeUnit,
    TTSIMUnit,
    UnitSystem,
    cast_ttsim_unit,
    policy_function,
    policy_input,
    register_unit_builder_levels,
)
from ttsim.tt.grouping_levels import register_grouping_levels
from ttsim.tt.units import (
    CURRENCY_TOKEN,
    coerce_to_composite_unit,
    composite_from_resolved_unit,
    fail_if_units_are_missing,
    grouping_level_count_unit,
    is_calendar_point_unit,
    output_unit_in_data_currency,
    parse_compositional_unit,
    parse_unit,
    resolve_compositional_cast_unit,
    resolve_compositional_column_unit,
    resolve_compositional_param_unit,
    resolve_compositional_unit,
    strip_input_quantity_at_boundary,
    token_is_agnostic_currency,
    token_source_currency,
    unit_for_aggregation,
    units_are_equivalent,
)

# A representative policy system for the compositional tests: its registry holds
# Middle Earth's currencies (CASTAR base, SILVER_PENNY) and the GETTSIM-style
# grouping levels the resolution tests build on.
SYSTEM = UnitSystem(
    base_currency="CASTAR",
    other_currencies={"SILVER_PENNY": "CASTAR / 4"},
    statutory_currencies={"0001-01-01": "SILVER_PENNY", "2020-01-01": "CASTAR"},
    grouping_levels=["bg", "hh"],
)
REGISTRY = SYSTEM.registry


@pytest.fixture(autouse=True)
def _registered_levels():
    """The GETTSIM-style grouping levels the compositional tests build on.

    Autouse (and idempotent), so every compositional test sees ``bg`` / ``hh``
    without threading the fixture through its signature.
    """
    register_grouping_levels(names=["bg", "hh"], registry=REGISTRY)
    register_unit_builder_levels(["bg", "hh"])


def _return_one_float() -> float:
    """Helper body for synthesising column functions in tests."""
    return 1.0


def _return_true() -> bool:
    """Helper body for synthesising boolean column functions in tests."""
    return True


# ----------------------------------------------------------------------------
# The unit vocabulary
# ----------------------------------------------------------------------------


def test_currency_token_anchors_currency_dimension():
    assert REGISTRY.Quantity(1.0, CURRENCY_TOKEN).dimensionality == {"[currency]": 1}


def test_quarter_year_is_a_quarter_of_a_year():
    ratio = (
        REGISTRY.Quantity(1.0, "year") / REGISTRY.Quantity(1.0, "quarter_year")
    ).to("dimensionless")
    assert ratio.magnitude == pytest.approx(4.0)


def test_hectare_is_an_area():
    assert REGISTRY.Quantity(1.0, "hectare").dimensionality == {"[length]": 2}


# ----------------------------------------------------------------------------
# The Unit token enumeration (the declaration surface)
# ----------------------------------------------------------------------------


_BASE_SPELLINGS = [
    "CURRENCY",
    "DIMENSIONLESS",
    "PERSON_COUNT",
    "HOURS",
    "SQUARE_METER",
    "HECTARE",
    "YEARS",
    "MONTHS",
    "DAYS",
    "CALENDAR_YEAR",
    "CALENDAR_MONTH",
    "CALENDAR_DAY",
]


@pytest.mark.parametrize("spelling", _BASE_SPELLINGS)
def test_coerce_to_composite_unit_round_trips_base_spellings(spelling):
    token = coerce_to_composite_unit(value=spelling, where="test")
    assert isinstance(token, CompositeUnit)
    assert str(token) == spelling


@pytest.mark.parametrize(
    "spelling",
    [
        "CURRENCY_PER_MONTH",
        "CURRENCY_PER_MONTH_PER_BG",
        "PERSON_COUNT_PER_BG",
        "DIMENSIONLESS_PER_YEAR",
        "HOURS_PER_WEEK",
    ],
)
def test_coerce_to_composite_unit_round_trips_compositional_spellings(spelling):
    token = coerce_to_composite_unit(value=spelling, where="test")
    assert isinstance(token, CompositeUnit)
    assert str(token) == spelling


def test_coerce_to_composite_unit_rejects_none():
    # `None` is no longer a dimensionless declaration (GEP 10): it reaches
    # `coerce_to_composite_unit` only through an internal bug, so the package claw
    # rejects it before the body runs.
    with pytest.raises(BeartypeCallHintViolation):
        coerce_to_composite_unit(value=None, where="test")  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    "value",
    [
        # The old pint-string surface is gone: one token = one meaning. (Bare
        # "CURRENCY" is now the agnostic stock token TTSIMUnit.CURRENCY, so it is
        # valid — only composite/pint spellings remain rejected.)
        "CURRENCY / year",
        "year",
        "hectare",
        "dimensionless",
        "currency_flow",  # case matters: YAML spells the member exactly
        "kelvin",
    ],
)
def test_coerce_to_composite_unit_rejects_non_members(value):
    with pytest.raises(UnitDefinitionError, match="invalid unit declaration"):
        coerce_to_composite_unit(value=value, where="test")


def test_compositional_flow_is_marked_by_a_period():
    # A unit is a flow iff it spells a period denominator (GEP 10).
    assert TTSIMUnit.CURRENCY.PER_MONTH.is_flow
    assert not TTSIMUnit.CURRENCY.is_flow
    assert not TTSIMUnit.YEARS.is_flow


# ----------------------------------------------------------------------------
# parse_unit and the closed pint-token vocabulary (internal surfaces)
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "unit_str",
    [
        "CURRENCY",
        "CURRENCY / hectare",
        "CURRENCY / meter ** 2",
        "hectare",
        "year",
        "quarter_year",
        "month",
        "CASTAR",
    ],
)
def test_parse_unit_accepts_known_units(unit_str):
    parse_unit(unit_str, registry=REGISTRY)


@pytest.mark.parametrize(
    "unit_str",
    [
        "gram",
        "kelvin",
        "ampere",
        "CURRENCY / kelvin",
        # pint built-ins in admissible dimensions are rejected too: TTSIM
        # rejects any unit token it does not know about.
        "count",
        "CURRENCY / count",
        "percent",
        "kilometer",
    ],
)
def test_parse_unit_rejects_unknown_unit_tokens(unit_str):
    with pytest.raises(UnitDefinitionError, match="does not know about"):
        parse_unit(unit_str, registry=REGISTRY)


@pytest.mark.parametrize("unit_str", ["dimensionless", ""])
def test_parse_unit_rejects_dimensionless_spellings(unit_str):
    # There is exactly one way to declare a dimensionless quantity:
    # `DIMENSIONLESS` — a pint-string spelling is rejected.
    with pytest.raises(UnitDefinitionError, match="DIMENSIONLESS"):
        parse_unit(unit_str, registry=REGISTRY)


def test_parse_unit_rejects_unparseable_string():
    with pytest.raises(UnitDefinitionError, match="parse"):
        parse_unit("this is not a unit", registry=REGISTRY)


# ----------------------------------------------------------------------------
# UnitSystem currencies
# ----------------------------------------------------------------------------


def test_base_currency_is_a_currency_dimension():
    assert SYSTEM.registry.Quantity(1.0, "CASTAR").dimensionality == {"[currency]": 1}


def test_relative_currency_bakes_correct_factor():
    # SILVER_PENNY = CASTAR / 4, so a silver penny is a quarter of a castar.
    factor = (
        SYSTEM.registry.Quantity(1.0, "SILVER_PENNY")
        / SYSTEM.registry.Quantity(1.0, "CASTAR")
    ).to("dimensionless")
    assert factor.magnitude == pytest.approx(0.25)


# ----------------------------------------------------------------------------
# Currency declaration tokens (a system's currencies extend the vocabulary)
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spelling", "currency", "is_flow"),
    [
        ("CASTAR", "CASTAR", False),
        ("CASTAR_PER_MONTH", "CASTAR", True),
        ("SILVER_PENNY", "SILVER_PENNY", False),
        ("SILVER_PENNY_PER_YEAR", "SILVER_PENNY", True),
    ],
)
def test_registered_currency_is_a_compositional_base(spelling, currency, is_flow):
    token = coerce_to_composite_unit(value=spelling, where="test")
    assert isinstance(token, CompositeUnit)
    assert str(token) == spelling
    assert token_source_currency(token) == currency
    assert token.is_flow == is_flow


def test_coerce_currency_token_is_idempotent():
    token = coerce_to_composite_unit(value="CASTAR", where="test")
    assert coerce_to_composite_unit(value=token, where="test") is token
    assert coerce_to_composite_unit(value="CASTAR", where="test") == token


def test_token_source_currency():
    assert token_source_currency(
        coerce_to_composite_unit(value="CASTAR_PER_MONTH", where="t")
    ) == ("CASTAR")
    assert token_source_currency(TTSIMUnit.CURRENCY.PER_MONTH) is None
    assert token_source_currency(TTSIMUnit.HECTARE) is None
    assert token_source_currency(None) is None


def test_unregistered_currency_spelling_is_rejected():
    with pytest.raises(UnitDefinitionError, match="invalid unit declaration"):
        coerce_to_composite_unit(value="MITHRIL", where="test")


def test_currency_agnostic_base_rejected_on_column_at_resolution():
    # A function runs in the statutory currency of the policy date, so a
    # concrete currency base is rejected when a column's compositional unit
    # is resolved (GEP 10).
    token = coerce_to_composite_unit(value="CASTAR_PER_MONTH", where="test")
    with pytest.raises(UnitDefinitionError, match="agnostic CURRENCY"):
        resolve_compositional_column_unit(
            unit=token,
            time_unit_id="m",
            grouping_level="person",
            where="A column",
            registry=REGISTRY,
        )


# ----------------------------------------------------------------------------
# units_are_equivalent
# ----------------------------------------------------------------------------


def test_same_unit_is_equivalent():
    assert units_are_equivalent(
        left=parse_unit("hectare", registry=REGISTRY),
        right=parse_unit("hectare", registry=REGISTRY),
        registry=REGISTRY,
    )


def test_base_currency_equivalent_to_currency_token():
    assert units_are_equivalent(
        left=parse_unit("CASTAR", registry=REGISTRY),
        right=parse_unit("CURRENCY", registry=REGISTRY),
        registry=REGISTRY,
    )


def test_month_and_year_flows_are_not_equivalent():
    # Same dimensionality ([currency] / [time]) but different magnitude.
    assert not units_are_equivalent(
        left=parse_unit("CURRENCY / month", registry=REGISTRY),
        right=parse_unit("CURRENCY / year", registry=REGISTRY),
        registry=REGISTRY,
    )


def test_different_dimensions_are_not_equivalent():
    assert not units_are_equivalent(
        left=parse_unit("CURRENCY", registry=REGISTRY),
        right=parse_unit("hectare", registry=REGISTRY),
        registry=REGISTRY,
    )


def test_calendar_point_is_equivalent_to_itself():
    # A calendar point (affine offset unit) cannot be divided; equivalence is
    # decided by identity (GEP 10).
    assert units_are_equivalent(
        left=parse_unit("calendar_year", registry=REGISTRY),
        right=parse_unit("calendar_year", registry=REGISTRY),
        registry=REGISTRY,
    )


def test_calendar_point_is_not_equivalent_to_a_duration():
    # The S1 distinction: a year on the calendar is not a duration in years.
    assert not units_are_equivalent(
        left=parse_unit("calendar_year", registry=REGISTRY),
        right=parse_unit("delta_calendar_year", registry=REGISTRY),
        registry=REGISTRY,
    )
    assert not units_are_equivalent(
        left=parse_unit("calendar_year", registry=REGISTRY),
        right=parse_unit("year", registry=REGISTRY),
        registry=REGISTRY,
    )


def test_calendar_points_on_different_axes_are_not_equivalent():
    assert not units_are_equivalent(
        left=parse_unit("calendar_year", registry=REGISTRY),
        right=parse_unit("calendar_month", registry=REGISTRY),
        registry=REGISTRY,
    )


def test_is_calendar_point_unit():
    assert is_calendar_point_unit(
        parse_unit("calendar_year", registry=REGISTRY), registry=REGISTRY
    )
    assert is_calendar_point_unit(
        parse_unit("calendar_month", registry=REGISTRY), registry=REGISTRY
    )
    assert is_calendar_point_unit(
        parse_unit("calendar_day", registry=REGISTRY), registry=REGISTRY
    )
    # Durations and ordinary units are not points.
    assert not is_calendar_point_unit(
        parse_unit("delta_calendar_year", registry=REGISTRY), registry=REGISTRY
    )
    assert not is_calendar_point_unit(
        parse_unit("year", registry=REGISTRY), registry=REGISTRY
    )
    assert not is_calendar_point_unit(
        parse_unit("CURRENCY / month", registry=REGISTRY), registry=REGISTRY
    )


def test_duration_token_is_equivalent_to_the_plain_time_unit():
    # YEARS / MONTHS / DAYS resolve to the `delta_calendar_*` durations, which
    # are ratio 1 against year / month / day, so existing duration declarations
    # are unchanged.
    assert units_are_equivalent(
        left=resolve_compositional_unit(TTSIMUnit.YEARS, registry=REGISTRY),
        right=parse_unit("year", registry=REGISTRY),
        registry=REGISTRY,
    )
    assert units_are_equivalent(
        left=resolve_compositional_unit(TTSIMUnit.MONTHS, registry=REGISTRY),
        right=parse_unit("month", registry=REGISTRY),
        registry=REGISTRY,
    )


# ----------------------------------------------------------------------------
# Decorator integration
# ----------------------------------------------------------------------------


def test_policy_function_stores_unit():
    @policy_function(unit=TTSIMUnit.CURRENCY.PER_MONTH)
    def betrag_m(satz: float, anzahl: int) -> float:
        return satz * anzahl

    assert betrag_m.unit == TTSIMUnit.CURRENCY.PER_MONTH


def test_policy_function_rejects_invalid_unit_at_decoration():
    # The decorator's type contract only admits `TTSIMUnit` members (or None);
    # the beartype claw rejects anything else at decoration time.
    with pytest.raises(PolicyFunctionDefinitionError, match="unit"):

        @policy_function(unit="kelvin")  # ty: ignore[invalid-argument-type]
        def temperature(x: float) -> float:
            return x


def test_policy_function_rejects_member_spelled_as_string_at_decoration():
    # In Python code there is exactly one way to write a token: the member.
    with pytest.raises(PolicyFunctionDefinitionError, match="CURRENCY_FLOW"):

        @policy_function(unit="CURRENCY_FLOW")  # ty: ignore[invalid-argument-type]
        def betrag_m(x: float) -> float:
            return x


def test_policy_function_explicit_dimensionless():
    @policy_function(unit=TTSIMUnit.DIMENSIONLESS)
    def some_share(x: float) -> float:
        return x

    assert some_share.unit is TTSIMUnit.DIMENSIONLESS


# ----------------------------------------------------------------------------
# #118: time as a dimension — suffix/reference_period resolution
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        (TTSIMUnit.CURRENCY.PER_MONTH, "CURRENCY / month"),
        (TTSIMUnit.CURRENCY.PER_YEAR, "CURRENCY / year"),
        (TTSIMUnit.CURRENCY.PER_QUARTER, "CURRENCY / quarter_year"),
        (TTSIMUnit.CURRENCY.PER_WEEK, "CURRENCY / week"),
        (TTSIMUnit.CURRENCY.PER_DAY, "CURRENCY / day"),
        (TTSIMUnit.DIMENSIONLESS.PER_YEAR, "1 / year"),  # e.g. a wealth-tax rate
        (TTSIMUnit.HOURS.PER_WEEK, "working_hour / week"),  # e.g. working hours
        (
            TTSIMUnit.CURRENCY.PER_SQUARE_METER.PER_MONTH,
            "CURRENCY / meter ** 2 / month",
        ),
        (TTSIMUnit.CURRENCY, "CURRENCY"),  # stock
        (TTSIMUnit.YEARS, "year"),  # a duration, e.g. an age
        (TTSIMUnit.MONTHS, "month"),  # a duration in months
        (TTSIMUnit.DAYS, "day"),  # a duration in days
        (TTSIMUnit.CALENDAR_YEAR, "calendar_year"),  # a point, e.g. a birth year
        (TTSIMUnit.CALENDAR_MONTH, "calendar_month"),
        (TTSIMUnit.CALENDAR_DAY, "calendar_day"),
        (TTSIMUnit.SQUARE_METER, "meter ** 2"),
        (TTSIMUnit.HECTARE, "hectare"),
    ],
)
def test_resolve_compositional_unit_period_mapping(token, expected):
    resolved = resolve_compositional_unit(token, registry=REGISTRY)
    assert units_are_equivalent(
        left=resolved, right=parse_unit(expected, registry=REGISTRY), registry=REGISTRY
    )


def test_resolve_compositional_unit_dimensionless():
    # A share, a rate, a head count: declared `TTSIMUnit.DIMENSIONLESS`.
    assert units_are_equivalent(
        left=resolve_compositional_unit(TTSIMUnit.DIMENSIONLESS, registry=REGISTRY),
        right=REGISTRY.dimensionless,
        registry=REGISTRY,
    )


def test_flow_period_resolution_distinguishes_month_and_year():
    """A monthly flow and its yearly variant resolve to non-equivalent units."""
    betrag_m = resolve_compositional_unit(
        TTSIMUnit.CURRENCY.PER_MONTH, registry=REGISTRY
    )
    betrag_y = resolve_compositional_unit(
        TTSIMUnit.CURRENCY.PER_YEAR, registry=REGISTRY
    )
    assert not units_are_equivalent(left=betrag_m, right=betrag_y, registry=REGISTRY)


# ----------------------------------------------------------------------------
# #118: pint-sourced conversion factors and duration conversion
# ----------------------------------------------------------------------------


def test_unit_converter_factors_sourced_from_pint():
    # The stock converters multiply by the period-per-year factor; check that
    # factor against pint via the public converter functions.
    def per_year(name):
        return (
            (REGISTRY.Quantity(1.0, "year") / REGISTRY.Quantity(1.0, name))
            .to("dimensionless")
            .magnitude
        )

    assert unit_converters.y_to_q(1.0) == pytest.approx(per_year("quarter_year"))
    assert unit_converters.y_to_m(1.0) == pytest.approx(per_year("month"))
    assert unit_converters.y_to_w(1.0) == pytest.approx(per_year("week"))
    assert unit_converters.y_to_d(1.0) == pytest.approx(per_year("day"))


def test_integral_factors_keep_integer_type():
    # Stock conversions must keep ints as ints (e.g. y_to_m of an int stock).
    assert unit_converters.y_to_q(2) == 8
    assert isinstance(unit_converters.y_to_q(2), int)
    assert unit_converters.y_to_m(2) == 24
    assert isinstance(unit_converters.y_to_m(2), int)


def test_duration_conversion_year_to_month():
    """A duration of 2 years is 24 months — sourced from pint."""
    months = REGISTRY.Quantity(2.0, "year").to("month")
    assert months.magnitude == pytest.approx(24.0)


# ----------------------------------------------------------------------------
# #119: mandatory units
# ----------------------------------------------------------------------------


def test_fail_if_units_are_missing_reports_unannotated_nodes():
    with pytest.raises(UnitDefinitionError, match="kindergeld__betrag_m"):
        fail_if_units_are_missing(
            {"kindergeld__betrag_m": UNSET_UNIT, "alter": TTSIMUnit.YEARS},
        )


def test_fail_if_units_are_missing_passes_when_all_annotated():
    fail_if_units_are_missing({"a": TTSIMUnit.CURRENCY, "b": TTSIMUnit.YEARS})


def test_fail_if_units_are_missing_accepts_dimensionless():
    # `DIMENSIONLESS` *is* a declaration: a dimensionless quantity (GEP 10).
    fail_if_units_are_missing(
        {"some_share": TTSIMUnit.DIMENSIONLESS, "alter": TTSIMUnit.YEARS}
    )


def test_policy_input_rejects_invalid_unit_at_decoration():
    with pytest.raises(PolicyInputDefinitionError, match="unit"):

        @policy_input(unit="kelvin")  # ty: ignore[invalid-argument-type]
        def temperature() -> float:
            """Some input."""


# ----------------------------------------------------------------------------
# #119: auto-assigned units for aggregation nodes
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("agg_type", "source_unit", "expected"),
    [
        (AggType.SUM, TTSIMUnit.CURRENCY.PER_MONTH, TTSIMUnit.CURRENCY.PER_MONTH),
        (AggType.MEAN, TTSIMUnit.CURRENCY.PER_MONTH, TTSIMUnit.CURRENCY.PER_MONTH),
        (AggType.MIN, TTSIMUnit.CURRENCY.PER_MONTH, TTSIMUnit.CURRENCY.PER_MONTH),
        (AggType.MAX, TTSIMUnit.CURRENCY.PER_MONTH, TTSIMUnit.CURRENCY.PER_MONTH),
        # A COUNT is a head count: the PERSON_COUNT base, independent of source
        # (GEP 10). The level-aware resolved unit is [person]/[target].
        (AggType.COUNT, TTSIMUnit.CURRENCY.PER_MONTH, TTSIMUnit.PERSON_COUNT),
        # ANY / ALL yield booleans, which are dimensionless quantities (GEP 10),
        # independent of the source.
        (AggType.ANY, TTSIMUnit.CURRENCY.PER_MONTH, TTSIMUnit.DIMENSIONLESS),
        (AggType.ALL, TTSIMUnit.CURRENCY.PER_MONTH, TTSIMUnit.DIMENSIONLESS),
    ],
)
def test_unit_for_aggregation(agg_type, source_unit, expected):
    assert unit_for_aggregation(source_unit=source_unit, agg_type=agg_type) == expected


def test_unit_for_aggregation_preserves_unannotated_source():
    assert (
        unit_for_aggregation(source_unit=UNSET_UNIT, agg_type=AggType.SUM) is UNSET_UNIT
    )
    # COUNT mints a head count and ANY/ALL a boolean, both independent of source.
    assert (
        unit_for_aggregation(source_unit=UNSET_UNIT, agg_type=AggType.COUNT)
        is TTSIMUnit.PERSON_COUNT
    )


def test_unit_for_aggregation_sum_over_boolean_is_a_head_count():
    # A SUM over a boolean counts the persons its flag is true for, so the minted
    # token is a head count's — the same one a COUNT mints (GEP 10) — not a leveled
    # DIMENSIONLESS. This keeps the minter in step with the resolver, which derives
    # the SUM-over-boolean as a head count too, so the build does not reject the
    # framework's own auto-assigned token.
    assert (
        unit_for_aggregation(
            source_unit=TTSIMUnit.DIMENSIONLESS,
            agg_type=AggType.SUM,
            source_is_boolean=True,
        )
        is TTSIMUnit.PERSON_COUNT
    )
    assert (
        unit_for_aggregation(
            source_unit=TTSIMUnit.DIMENSIONLESS,
            agg_type=AggType.SUM,
            target_level="fam",
            source_is_boolean=True,
        )
        == TTSIMUnit.PERSON_COUNT.PER_FAM
    )


def test_time_conversion_variants_rebased_period():
    variants = create_time_conversion_functions(
        qname_policy_environment={
            "betrag_m": policy_function(
                leaf_name="betrag_m", unit=TTSIMUnit.CURRENCY.PER_MONTH
            )(_return_one_float),
        },
        input_columns=set(),
        grouping_levels=("sn", "kin"),
    )
    # Generated betrag_y re-bases the flow's period to its own _y suffix.
    betrag_y_unit = variants["betrag_y"].unit  # ty: ignore[unresolved-attribute]
    assert betrag_y_unit == TTSIMUnit.CURRENCY.PER_YEAR
    assert units_are_equivalent(
        left=resolve_compositional_unit(betrag_y_unit, registry=REGISTRY),
        right=parse_unit("CURRENCY / year", registry=REGISTRY),
        registry=REGISTRY,
    )


def test_auto_aggregation_carries_the_target_level():
    # An auto-aggregation preserves the source's physical base and period and
    # carries the *target* group level, so its declared token is precise (GEP 10).
    aggs = create_agg_by_group_functions(
        column_functions={
            "betrag_m": policy_function(
                leaf_name="betrag_m", unit=TTSIMUnit.CURRENCY.PER_MONTH
            )(_return_one_float),
        },
        qname_policy_environment={},
        input_columns=set(),
        tt_targets={"betrag_m_kin"},
        grouping_levels=("kin",),
    )
    assert (
        aggs["betrag_m_kin"].unit  # ty: ignore[unresolved-attribute]
        == TTSIMUnit.CURRENCY.PER_MONTH.PER_KIN
    )


def test_auto_aggregation_over_a_boolean_source_mints_a_head_count():
    # Requesting the group aggregate of a boolean (e.g. `is_adult_fam`) auto-generates
    # a SUM node. Over a boolean that SUM is a head count, so the framework mints
    # `PERSON_COUNT_PER_FAM`, matching what the resolver derives — the regression
    # that used to fail the build on any boolean group aggregate (GEP 10).
    aggs = create_agg_by_group_functions(
        column_functions={
            "is_adult": policy_function(
                leaf_name="is_adult", unit=TTSIMUnit.DIMENSIONLESS
            )(_return_true),
        },
        qname_policy_environment={},
        input_columns=set(),
        tt_targets={"is_adult_fam"},
        grouping_levels=("fam",),
    )
    assert (
        aggs["is_adult_fam"].unit  # ty: ignore[unresolved-attribute]
        == TTSIMUnit.PERSON_COUNT.PER_FAM
    )


def test_concrete_currency_per_square_meter_base():
    # A concrete currency divided by an area is a valid compositional unit.
    token = coerce_to_composite_unit(
        value="CASTAR_PER_SQUARE_METER_PER_MONTH", where="test"
    )
    assert isinstance(token, CompositeUnit)
    assert token_source_currency(token) == "CASTAR"
    assert token.base == "CASTAR"
    assert token.area == "SQUARE_METER"
    assert token.is_flow


def test_token_is_agnostic_currency():
    assert token_is_agnostic_currency(TTSIMUnit.CURRENCY)
    assert token_is_agnostic_currency(TTSIMUnit.CURRENCY.PER_MONTH)
    assert token_is_agnostic_currency(TTSIMUnit.CURRENCY.PER_SQUARE_METER.PER_MONTH)
    assert not token_is_agnostic_currency(TTSIMUnit.HECTARE)
    assert not token_is_agnostic_currency(TTSIMUnit.DIMENSIONLESS.PER_YEAR)
    assert not token_is_agnostic_currency(None)
    assert not token_is_agnostic_currency(
        coerce_to_composite_unit(value="CASTAR", where="test")
    )


# ----------------------------------------------------------------------------
# #120: currency conversion factor
# ----------------------------------------------------------------------------


def test_currency_conversion_factor():
    # silver_penny = castar / 4, defined by the system's currencies.
    assert SYSTEM.currency_conversion_factor(
        source_currency="CASTAR", target_currency="SILVER_PENNY"
    ) == pytest.approx(4.0)
    assert SYSTEM.currency_conversion_factor(
        source_currency="SILVER_PENNY", target_currency="CASTAR"
    ) == pytest.approx(0.25)
    assert SYSTEM.currency_conversion_factor(
        source_currency="CASTAR", target_currency="CASTAR"
    ) == pytest.approx(1.0)


def test_currency_conversion_factor_rejects_unknown_currency():
    with pytest.raises(UnitDefinitionError, match="not a registered currency"):
        SYSTEM.currency_conversion_factor(
            source_currency="CASTAR", target_currency="dragon_hoard"
        )


# ----------------------------------------------------------------------------
# #120: Layer-2 boundary — validate and strip pint-tagged inputs
# ----------------------------------------------------------------------------


def test_strip_at_boundary_converts_to_data_currency():
    # silver_penny tag, castar data currency -> divide by four.
    tagged = REGISTRY.Quantity(np.array([4.0]), "SILVER_PENNY")
    bare = strip_input_quantity_at_boundary(
        quantity=tagged,
        data_currency="CASTAR",
        column_label="wealth",
        registry=REGISTRY,
    )
    assert not isinstance(bare, REGISTRY.Quantity)
    assert bare == pytest.approx([1.0])


def test_strip_at_boundary_converts_flow_currency_preserving_period():
    # silver_penny / month -> castar / month: only the currency is rescaled. The
    # tag's /month matches the column's `_m` suffix.
    tagged = REGISTRY.Quantity(np.array([4.0]), "SILVER_PENNY / month")
    bare = strip_input_quantity_at_boundary(
        quantity=tagged,
        data_currency="CASTAR",
        column_label="income_m",
        registry=REGISTRY,
    )
    assert bare == pytest.approx([1.0])


def test_strip_at_boundary_fails_on_missing_period():
    # A flow column (`_m`) tagged without a period: strict mismatch.
    tagged = REGISTRY.Quantity(np.array([4.0]), "SILVER_PENNY")
    with pytest.raises(UnitConsistencyError, match="must match the column's suffix"):
        strip_input_quantity_at_boundary(
            quantity=tagged,
            data_currency="CASTAR",
            column_label="income_m",
            registry=REGISTRY,
        )


def test_strip_at_boundary_fails_on_wrong_period():
    # silver_penny / year against a `_m` column: 12-fold footgun, caught.
    tagged = REGISTRY.Quantity(np.array([4.0]), "SILVER_PENNY / year")
    with pytest.raises(UnitConsistencyError, match="month"):
        strip_input_quantity_at_boundary(
            quantity=tagged,
            data_currency="CASTAR",
            column_label="income_m",
            registry=REGISTRY,
        )


def test_strip_at_boundary_fails_on_period_for_unsuffixed_column():
    # A stock column (no suffix) tagged with a period.
    tagged = REGISTRY.Quantity(np.array([4.0]), "SILVER_PENNY / month")
    with pytest.raises(UnitConsistencyError, match="no time suffix"):
        strip_input_quantity_at_boundary(
            quantity=tagged,
            data_currency="CASTAR",
            column_label="wealth",
            registry=REGISTRY,
        )


def test_strip_at_boundary_does_not_flag_numerator_time_unit():
    # An age tagged in years: `year` is a numerator, not a flow period, so an
    # unsuffixed column is fine. Nothing to convert (no currency).
    tagged = REGISTRY.Quantity(np.array([30.0]), "year")
    bare = strip_input_quantity_at_boundary(
        quantity=tagged, data_currency="CASTAR", column_label="age", registry=REGISTRY
    )
    assert bare == pytest.approx([30.0])


def test_strip_at_boundary_keys_period_off_denominator_for_hours_flow():
    # `arbeitsstunden_w` is HOURS_FLOW (`working_hour / week`): the flow period
    # is the denominator (week), not the `[hours]` numerator. The tag's `/week`
    # matches the `_w` suffix; there is no currency, so the value passes through.
    tagged = REGISTRY.Quantity(np.array([40.0]), "working_hour / week")
    bare = strip_input_quantity_at_boundary(
        quantity=tagged,
        data_currency="CASTAR",
        column_label="arbeitsstunden_w",
        registry=REGISTRY,
    )
    assert bare == pytest.approx([40.0])

    # A wrong period is still caught (month != week).
    with pytest.raises(UnitConsistencyError, match="week"):
        strip_input_quantity_at_boundary(
            quantity=REGISTRY.Quantity(np.array([40.0]), "working_hour / month"),
            data_currency="CASTAR",
            column_label="arbeitsstunden_w",
            registry=REGISTRY,
        )


def test_strip_at_boundary_strips_matching_currency():
    tagged = REGISTRY.Quantity(np.array([3.0]), "CASTAR")
    bare = strip_input_quantity_at_boundary(
        quantity=tagged, data_currency="CASTAR", registry=REGISTRY
    )
    assert not isinstance(bare, REGISTRY.Quantity)
    assert list(bare) == [3.0]


def test_strip_at_boundary_passes_non_currency_tag_through():
    # No currency component -> nothing to convert.
    tagged = REGISTRY.Quantity(np.array([5.0]), "working_hour")
    bare = strip_input_quantity_at_boundary(
        quantity=tagged, data_currency="CASTAR", registry=REGISTRY
    )
    assert bare == pytest.approx([5.0])


# ----------------------------------------------------------------------------
# Compositional units (GEP 10 compositional units) — builder, parser, formatter
# ----------------------------------------------------------------------------


def test_builder_round_trips_with_flat_spelling():
    # The fluent `.py` builder and the flat YAML string are the same unit.
    built = TTSIMUnit.CURRENCY.PER_SQUARE_METER.PER_MONTH
    assert str(built) == "CURRENCY_PER_SQUARE_METER_PER_MONTH"
    assert parse_compositional_unit(str(built)) == built


def test_builder_round_trips_with_level():
    built = TTSIMUnit.CURRENCY.PER_MONTH.PER_BG
    assert str(built) == "CURRENCY_PER_MONTH_PER_BG"
    assert parse_compositional_unit("CURRENCY_PER_MONTH_PER_BG") == built


def test_builder_generic_per_level_matches_attribute():
    assert (
        TTSIMUnit.PERSON_COUNT.PER_LEVEL("bg")
        == TTSIMUnit.PERSON_COUNT.PER_BG
        == parse_compositional_unit("PERSON_COUNT_PER_BG")
    )


@pytest.mark.parametrize(
    ("spelling", "base", "area", "period", "level"),
    [
        ("CURRENCY", "CURRENCY", None, None, None),
        ("CURRENCY_PER_MONTH", "CURRENCY", None, "MONTH", None),
        ("CURRENCY_PER_SQUARE_METER", "CURRENCY", "SQUARE_METER", None, None),
        ("PERSON_COUNT_PER_BG", "PERSON_COUNT", None, None, "BG"),
        ("DIMENSIONLESS_PER_YEAR", "DIMENSIONLESS", None, "YEAR", None),
        ("DIMENSIONLESS_PER_BG", "DIMENSIONLESS", None, None, "BG"),
        ("HOURS_PER_WEEK", "HOURS", None, "WEEK", None),
        ("CURRENCY_PER_HOURS", "CURRENCY", "HOURS", None, None),
        (
            "CURRENCY_PER_SQUARE_METER_PER_MONTH_PER_BG",
            "CURRENCY",
            "SQUARE_METER",
            "MONTH",
            "BG",
        ),
    ],
)
def test_parse_compositional_unit_classifies_denominators(
    spelling, base, area, period, level
):
    parsed = parse_compositional_unit(spelling)
    assert parsed == CompositeUnit(base=base, area=area, period=period, level=level)
    assert str(parsed) == spelling


def test_parse_compositional_unit_accepts_concrete_currency_base():
    # Concrete currencies (param YAML only) are valid bases; CASTAR is the
    # mettsim base currency, registered on import.
    parsed = parse_compositional_unit("CASTAR_PER_MONTH")
    assert parsed.base == "CASTAR"
    assert parsed.period == "MONTH"


@pytest.mark.parametrize(
    "spelling",
    [
        "",  # empty
        "FOO_PER_MONTH",  # unknown base
        "CURRENCY_PER_BG_PER_MONTH",  # non-canonical order (level before period)
        "CURRENCY_PER_MONTH_PER_YEAR",  # two periods
        "CURRENCY_PER_SQUARE_METER_PER_SQUARE_METER",  # two areas
        "CURRENCY_PER_SQUARE_METER_PER_HOURS",  # two physical denominators
        "CURRENCY_PER_MONTH_PER_HOURS",  # non-canonical (hours after period)
    ],
)
def test_parse_compositional_unit_rejects_bad_spellings(spelling):
    with pytest.raises(UnitDefinitionError):
        parse_compositional_unit(spelling)


def test_builder_rejects_non_canonical_order():
    # Level before period, an area after a period, a second period — all caught
    # by the staged builder, mirroring the parser.
    with pytest.raises(UnitDefinitionError, match="precede"):
        _ = TTSIMUnit.CURRENCY.PER_BG.PER_MONTH
    with pytest.raises(UnitDefinitionError, match="precede"):
        _ = TTSIMUnit.CURRENCY.PER_MONTH.PER_YEAR


def test_is_flow_property():
    assert TTSIMUnit.CURRENCY.PER_MONTH.is_flow
    assert not parse_compositional_unit("CURRENCY").is_flow


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [
        ("CURRENCY_PER_MONTH", "CURRENCY / month"),
        ("CURRENCY_PER_YEAR", "CURRENCY / year"),
        ("DIMENSIONLESS_PER_YEAR", "1 / year"),
        ("HOURS_PER_WEEK", "working_hour / week"),
        ("CURRENCY_PER_HOURS", "CURRENCY / working_hour"),
        ("CURRENCY_PER_SQUARE_METER_PER_MONTH", "CURRENCY / meter ** 2 / month"),
        ("SQUARE_METER", "meter ** 2"),
        ("HECTARE", "hectare"),
        ("YEARS", "year"),
        ("CALENDAR_YEAR", "calendar_year"),
    ],
)
def test_compositional_unit_resolves_to_expected_pint_unit(spelling, expected):
    # The leftover-token migration map: every compositional spelling resolves to
    # the identical pint unit the legacy token used to produce.
    compositional = resolve_compositional_unit(
        parse_compositional_unit(spelling), registry=REGISTRY
    )
    assert units_are_equivalent(
        left=compositional,
        right=parse_unit(expected, registry=REGISTRY),
        registry=REGISTRY,
    )


def test_person_per_level_resolves_to_head_count():
    # PERSON_COUNT_PER_BG is the old HEADCOUNT at bg: [person] / [bg], the unit a COUNT
    # aggregation to bg mints.
    compositional = resolve_compositional_unit(
        parse_compositional_unit("PERSON_COUNT_PER_BG"), registry=REGISTRY
    )
    assert units_are_equivalent(
        left=compositional,
        right=grouping_level_count_unit(target_level="bg", registry=REGISTRY),
        registry=REGISTRY,
    )


def test_concrete_currency_base_resolves_like_agnostic():
    # For dimensionality a concrete currency means exactly what CURRENCY means.
    concrete = resolve_compositional_unit(
        parse_compositional_unit("CASTAR_PER_MONTH"), registry=REGISTRY
    )
    agnostic = resolve_compositional_unit(
        parse_compositional_unit("CURRENCY_PER_MONTH"), registry=REGISTRY
    )
    assert units_are_equivalent(left=concrete, right=agnostic, registry=REGISTRY)


def test_resolve_compositional_unit_rejects_unregistered_level():
    with pytest.raises(UnitDefinitionError, match="grouping level"):
        resolve_compositional_unit(
            parse_compositional_unit("CURRENCY_PER_NEVERLAND"), registry=REGISTRY
        )


# ----------------------------------------------------------------------------
# Model change 1: the [hours] dimension (GEP 10 compositional units)
# ----------------------------------------------------------------------------


def test_working_hour_is_its_own_dimension():
    # Working hours are `[hours]`, isolated from pint's `[time]`, so they cannot
    # convert to a period and a `working_hour / week` flow does not collapse to a
    # bare dimensionless number the way a `[time] / [time]` hour-flow would.
    assert REGISTRY.Quantity(1.0, "working_hour").dimensionality == {"[hours]": 1}
    assert not REGISTRY.Quantity(1.0, "working_hour").is_compatible_with("day")
    hours_per_week = resolve_compositional_unit(
        TTSIMUnit.HOURS.PER_WEEK, registry=REGISTRY
    )
    assert REGISTRY.Quantity(1.0, hours_per_week).dimensionality == {
        "[hours]": 1,
        "[time]": -1,
    }


def test_bare_time_hour_is_no_longer_an_admissible_token():
    # There is exactly one spelling for working hours; pint's `[time]` `hour` is
    # not admissible (GEP 10).
    with pytest.raises(UnitDefinitionError, match="does not know about"):
        parse_unit("working_hour / hour", registry=REGISTRY)


def test_hours_per_week_rebases_period_only():
    # The one conversion working hours admit: re-basing the [time] period
    # (week -> month) leaves the [hours] numerator untouched.
    per_week = resolve_compositional_unit(
        parse_compositional_unit("HOURS_PER_WEEK"), registry=REGISTRY
    )
    per_month = resolve_compositional_unit(
        parse_compositional_unit("HOURS_PER_MONTH"), registry=REGISTRY
    )
    assert (
        per_week.dimensionality
        == per_month.dimensionality
        == {
            "[hours]": 1,
            "[time]": -1,
        }
    )
    # Different periods are not equivalent (a 52/12 factor apart).
    assert not units_are_equivalent(left=per_week, right=per_month, registry=REGISTRY)


# ----------------------------------------------------------------------------
# `cast_ttsim_unit` and the implied person leaf (GEP 10)
# ----------------------------------------------------------------------------


def test_cast_unit_is_the_identity_at_run_time():
    # Like `typing.cast`: no runtime effect, scalar or column, so the numeric
    # path and JAX tracing are untouched.
    column = np.array([1.0, 2.0])
    assert cast_ttsim_unit(column, TTSIMUnit.CURRENCY.PER_MONTH) is column
    assert cast_ttsim_unit(3.5, TTSIMUnit.MONTHS) == 3.5


def test_cast_target_resolves_like_a_column_declaration():
    # The cast states a unit in the declaration vocabulary: the person leaf is
    # implied for a level-carrying base, an intensive base stays bare.
    assert units_are_equivalent(
        left=resolve_compositional_cast_unit(
            unit=TTSIMUnit.CURRENCY.PER_MONTH, where="test", registry=REGISTRY
        ),
        right=resolve_compositional_column_unit(
            unit=TTSIMUnit.CURRENCY.PER_MONTH,
            time_unit_id="m",
            grouping_level="person",
            where="test",
            registry=REGISTRY,
        ),
        registry=REGISTRY,
    )
    assert units_are_equivalent(
        left=resolve_compositional_cast_unit(
            unit=TTSIMUnit.MONTHS, where="test", registry=REGISTRY
        ),
        right=resolve_compositional_unit(TTSIMUnit.MONTHS, registry=REGISTRY),
        registry=REGISTRY,
    )


def test_cast_target_must_be_currency_agnostic():
    token = coerce_to_composite_unit(value="CASTAR_PER_MONTH", where="test")
    with pytest.raises(UnitDefinitionError, match="agnostic CURRENCY"):
        resolve_compositional_cast_unit(unit=token, where="test", registry=REGISTRY)


def test_hours_denominator_suppresses_the_implied_person_leaf():
    # A wage floor is a price, owned by nobody: the working-hours denominator
    # keeps the person leaf off, so `floor * hours` cancels to a person-level
    # amount (GEP 10) — the same rule as areas.
    expected = parse_unit("CURRENCY / working_hour", registry=REGISTRY)
    assert units_are_equivalent(
        left=resolve_compositional_column_unit(
            unit=TTSIMUnit.CURRENCY.PER_HOURS,
            time_unit_id=None,
            grouping_level="person",
            where="test",
            registry=REGISTRY,
        ),
        right=expected,
        registry=REGISTRY,
    )
    assert units_are_equivalent(
        left=resolve_compositional_param_unit(
            unit=coerce_to_composite_unit(value="CASTAR_PER_HOURS", where="test"),
            where="test",
            registry=REGISTRY,
        ),
        right=expected,
        registry=REGISTRY,
    )


def test_composite_from_resolved_unit_reconstructs_the_hours_denominator():
    # The output-side label round trip for a wage floor: the working-hour
    # denominator maps back to the physical-denominator slot.
    resolved = resolve_compositional_unit(
        unit=TTSIMUnit.CURRENCY.PER_HOURS, registry=REGISTRY
    )
    in_data_currency = output_unit_in_data_currency(
        units=resolved, data_currency="CASTAR", registry=REGISTRY
    )
    assert composite_from_resolved_unit(
        units=in_data_currency, registry=REGISTRY
    ) == coerce_to_composite_unit(value="CASTAR_PER_HOURS", where="test")


def test_area_denominator_suppresses_the_implied_person_leaf():
    # A rent cap is a price, owned by nobody: the area denominator keeps the
    # person leaf off, so `cap * area` cancels to a person-level amount
    # (GEP 10). Both the column and the parameter resolver apply the rule.
    expected = parse_unit("CURRENCY / meter ** 2 / month", registry=REGISTRY)
    assert units_are_equivalent(
        left=resolve_compositional_column_unit(
            unit=TTSIMUnit.CURRENCY.PER_SQUARE_METER.PER_MONTH,
            time_unit_id="m",
            grouping_level="person",
            where="test",
            registry=REGISTRY,
        ),
        right=expected,
        registry=REGISTRY,
    )
    assert units_are_equivalent(
        left=resolve_compositional_param_unit(
            unit=coerce_to_composite_unit(
                value="CASTAR_PER_SQUARE_METER_PER_MONTH", where="test"
            ),
            where="test",
            registry=REGISTRY,
        ),
        right=expected,
        registry=REGISTRY,
    )


def test_bare_area_base_still_carries_the_person_leaf():
    # A dwelling area is owned: without an area *denominator*, the extensive
    # SQUARE_METER base takes the implied person leaf as any owned amount does.
    assert units_are_equivalent(
        left=resolve_compositional_column_unit(
            unit=TTSIMUnit.SQUARE_METER,
            time_unit_id=None,
            grouping_level="person",
            where="test",
            registry=REGISTRY,
        ),
        right=parse_unit("meter ** 2 / grouping_level_person", registry=REGISTRY),
        registry=REGISTRY,
    )
