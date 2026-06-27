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
    UnitInferenceError,
)
from ttsim.interface_dag_elements.automatically_added_functions import (
    create_agg_by_group_functions,
    create_time_conversion_functions,
)
from ttsim.tt import (
    CURRENCY_TOKEN,
    UNIT_REGISTRY,
    UNSET_UNIT,
    AggType,
    CompositeUnit,
    Unit,
    coerce_unit_token,
    currency_conversion_factor,
    fail_if_units_are_missing,
    parse_compositional_unit,
    parse_unit,
    policy_function,
    policy_input,
    register_currency,
    register_unit_builder_levels,
    resolve_compositional_column_unit,
    resolve_compositional_unit,
    strip_input_quantity_at_boundary,
    token_is_agnostic_currency,
    token_source_currency,
    unit_for_aggregation,
    units_are_equivalent,
)
from ttsim.tt.units import (
    grouping_level_count_unit,
    infer_function_unit,
    is_calendar_point_unit,
    register_grouping_levels,
)


@pytest.fixture(autouse=True)
def _registered_levels():
    """The GETTSIM-style grouping levels the compositional tests build on.

    Autouse (and idempotent), so every compositional test sees ``bg`` / ``hh``
    without threading the fixture through its signature.
    """
    register_grouping_levels(["bg", "hh"])
    register_unit_builder_levels(["bg", "hh"])


def _return_one_float() -> float:
    """Helper body for synthesising column functions in tests."""
    return 1.0


# ----------------------------------------------------------------------------
# The unit vocabulary
# ----------------------------------------------------------------------------


def test_currency_token_anchors_currency_dimension():
    assert UNIT_REGISTRY.Quantity(1.0, CURRENCY_TOKEN).dimensionality == {
        "[currency]": 1
    }


def test_quarter_year_is_a_quarter_of_a_year():
    ratio = (
        UNIT_REGISTRY.Quantity(1.0, "year")
        / UNIT_REGISTRY.Quantity(1.0, "quarter_year")
    ).to("dimensionless")
    assert ratio.magnitude == pytest.approx(4.0)


def test_hectare_is_an_area():
    assert UNIT_REGISTRY.Quantity(1.0, "hectare").dimensionality == {"[length]": 2}


# ----------------------------------------------------------------------------
# The Unit token enumeration (the declaration surface)
# ----------------------------------------------------------------------------


_BASE_SPELLINGS = [
    "CURRENCY",
    "DIMENSIONLESS",
    "PERSON",
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
def test_coerce_unit_token_round_trips_base_spellings(spelling):
    token = coerce_unit_token(spelling, where="test")
    assert isinstance(token, CompositeUnit)
    assert str(token) == spelling


@pytest.mark.parametrize(
    "spelling",
    [
        "CURRENCY_PER_MONTH",
        "CURRENCY_PER_MONTH_PER_BG",
        "PERSON_PER_BG",
        "DIMENSIONLESS_PER_YEAR",
        "HOURS_PER_WEEK",
    ],
)
def test_coerce_unit_token_round_trips_compositional_spellings(spelling):
    token = coerce_unit_token(spelling, where="test")
    assert isinstance(token, CompositeUnit)
    assert str(token) == spelling


def test_coerce_unit_token_rejects_none():
    # `None` is no longer a dimensionless declaration (GEP 10): it reaches
    # `coerce_unit_token` only through an internal bug, so the package claw
    # rejects it before the body runs.
    with pytest.raises(BeartypeCallHintViolation):
        coerce_unit_token(None, where="test")  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    "value",
    [
        # The old pint-string surface is gone: one token = one meaning. (Bare
        # "CURRENCY" is now the agnostic stock token Unit.CURRENCY, so it is
        # valid — only composite/pint spellings remain rejected.)
        "CURRENCY / year",
        "year",
        "hectare",
        "dimensionless",
        "currency_flow",  # case matters: YAML spells the member exactly
        "kelvin",
    ],
)
def test_coerce_unit_token_rejects_non_members(value):
    with pytest.raises(UnitDefinitionError, match="invalid unit declaration"):
        coerce_unit_token(value, where="test")


def test_compositional_flow_is_marked_by_a_period():
    # A unit is a flow iff it spells a period denominator (GEP 10).
    assert Unit.CURRENCY.PER_MONTH.is_flow
    assert not Unit.CURRENCY.is_flow
    assert not Unit.YEARS.is_flow


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
    parse_unit(unit_str)


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
        parse_unit(unit_str)


@pytest.mark.parametrize("unit_str", ["dimensionless", ""])
def test_parse_unit_rejects_dimensionless_spellings(unit_str):
    # There is exactly one way to declare a dimensionless quantity:
    # `DIMENSIONLESS` — a pint-string spelling is rejected.
    with pytest.raises(UnitDefinitionError, match="DIMENSIONLESS"):
        parse_unit(unit_str)


def test_parse_unit_rejects_unparseable_string():
    with pytest.raises(UnitDefinitionError, match="parse"):
        parse_unit("this is not a unit")


# ----------------------------------------------------------------------------
# register_currency
# ----------------------------------------------------------------------------


def test_mettsim_base_currency_registered():
    assert UNIT_REGISTRY.Quantity(1.0, "CASTAR").dimensionality == {"[currency]": 1}


def test_register_relative_currency_bakes_correct_factor():
    register_currency("SILVER_PENNY", definition="CASTAR / 4")
    factor = (
        UNIT_REGISTRY.Quantity(1.0, "SILVER_PENNY")
        / UNIT_REGISTRY.Quantity(1.0, "CASTAR")
    ).to("dimensionless")
    assert factor.magnitude == pytest.approx(0.25)


def test_register_currency_idempotent():
    # Re-registering with a consistent definition is a no-op, not an error.
    register_currency("SILVER_PENNY", definition="CASTAR / 4")


def test_register_currency_with_inconsistent_factor_fails():
    # Re-registering an existing currency with a *different* factor must fail
    # loudly rather than silently keep the original definition (GEP 10).
    register_currency("SILVER_PENNY", definition="CASTAR / 4")
    with pytest.raises(UnitDefinitionError, match="must be consistent"):
        register_currency("SILVER_PENNY", definition="CASTAR / 5")


def test_register_second_base_currency_fails():
    with pytest.raises(UnitDefinitionError, match="base currency"):
        register_currency("mithril_coin", base=True)


def test_register_currency_requires_exactly_one_of_base_or_definition():
    with pytest.raises(UnitDefinitionError, match="exactly one"):
        register_currency("bad_coin", base=True, definition="CASTAR")
    with pytest.raises(UnitDefinitionError, match="exactly one"):
        register_currency("bad_coin")


# ----------------------------------------------------------------------------
# Currency declaration tokens (registered currencies extend the vocabulary)
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
    register_currency("SILVER_PENNY", definition="CASTAR / 4")
    token = coerce_unit_token(spelling, where="test")
    assert isinstance(token, CompositeUnit)
    assert str(token) == spelling
    assert token_source_currency(token) == currency
    assert token.is_flow == is_flow


def test_coerce_currency_token_is_idempotent():
    token = coerce_unit_token("CASTAR", where="test")
    assert coerce_unit_token(token, where="test") is token
    assert coerce_unit_token("CASTAR", where="test") == token


def test_token_source_currency():
    assert token_source_currency(coerce_unit_token("CASTAR_PER_MONTH", where="t")) == (
        "CASTAR"
    )
    assert token_source_currency(Unit.CURRENCY.PER_MONTH) is None
    assert token_source_currency(Unit.HECTARE) is None
    assert token_source_currency(None) is None


def test_unregistered_currency_spelling_is_rejected():
    with pytest.raises(UnitDefinitionError, match="invalid unit declaration"):
        coerce_unit_token("MITHRIL", where="test")


def test_currency_agnostic_base_rejected_on_column_at_resolution():
    # Functions are currency-agnostic by design: a concrete currency base is
    # rejected when a column's compositional unit is resolved (GEP 10).
    token = coerce_unit_token("CASTAR_PER_MONTH", where="test")
    with pytest.raises(UnitDefinitionError, match="currency-agnostic"):
        resolve_compositional_column_unit(
            token, time_unit_id="m", grouping_level="person", where="A column"
        )


# ----------------------------------------------------------------------------
# units_are_equivalent
# ----------------------------------------------------------------------------


def test_same_unit_is_equivalent():
    assert units_are_equivalent(left=parse_unit("hectare"), right=parse_unit("hectare"))


def test_base_currency_equivalent_to_currency_token():
    assert units_are_equivalent(left=parse_unit("CASTAR"), right=parse_unit("CURRENCY"))


def test_month_and_year_flows_are_not_equivalent():
    # Same dimensionality ([currency] / [time]) but different magnitude.
    assert not units_are_equivalent(
        left=parse_unit("CURRENCY / month"), right=parse_unit("CURRENCY / year")
    )


def test_different_dimensions_are_not_equivalent():
    assert not units_are_equivalent(
        left=parse_unit("CURRENCY"), right=parse_unit("hectare")
    )


def test_calendar_point_is_equivalent_to_itself():
    # A calendar point (affine offset unit) cannot be divided; equivalence is
    # decided by identity (GEP 10).
    assert units_are_equivalent(
        left=parse_unit("calendar_year"), right=parse_unit("calendar_year")
    )


def test_calendar_point_is_not_equivalent_to_a_duration():
    # The S1 distinction: a year on the calendar is not a duration in years.
    assert not units_are_equivalent(
        left=parse_unit("calendar_year"), right=parse_unit("delta_calendar_year")
    )
    assert not units_are_equivalent(
        left=parse_unit("calendar_year"), right=parse_unit("year")
    )


def test_calendar_points_on_different_axes_are_not_equivalent():
    assert not units_are_equivalent(
        left=parse_unit("calendar_year"), right=parse_unit("calendar_month")
    )


def test_is_calendar_point_unit():
    assert is_calendar_point_unit(parse_unit("calendar_year"))
    assert is_calendar_point_unit(parse_unit("calendar_month"))
    assert is_calendar_point_unit(parse_unit("calendar_day"))
    # Durations and ordinary units are not points.
    assert not is_calendar_point_unit(parse_unit("delta_calendar_year"))
    assert not is_calendar_point_unit(parse_unit("year"))
    assert not is_calendar_point_unit(parse_unit("CURRENCY / month"))


def test_duration_token_is_equivalent_to_the_plain_time_unit():
    # YEARS / MONTHS / DAYS resolve to the `delta_calendar_*` durations, which
    # are ratio 1 against year / month / day, so existing duration declarations
    # are unchanged.
    assert units_are_equivalent(
        left=resolve_compositional_unit(Unit.YEARS),
        right=parse_unit("year"),
    )
    assert units_are_equivalent(
        left=resolve_compositional_unit(Unit.MONTHS),
        right=parse_unit("month"),
    )


# ----------------------------------------------------------------------------
# infer_function_unit (the dry-run)
# ----------------------------------------------------------------------------


def test_infer_multiplication_combines_units():
    def revenue(price_per_area: float, area: float) -> float:
        return price_per_area * area

    inferred = infer_function_unit(
        function=revenue,
        input_units={"price_per_area": "CURRENCY / hectare", "area": "hectare"},
    )
    assert units_are_equivalent(left=inferred, right=parse_unit("CURRENCY"))


def test_infer_raises_on_dimensional_clash():
    def bad_sum(rent: float, price_per_area: float) -> float:
        return rent + price_per_area

    with pytest.raises(UnitInferenceError):
        infer_function_unit(
            function=bad_sum,
            input_units={"rent": "CURRENCY", "price_per_area": "CURRENCY / hectare"},
        )


def test_infer_calendar_point_difference_is_a_duration():
    # The motivating S1 pattern: subtracting two calendar years yields a
    # duration in years, not a calendar year.
    def age(policy_year: int, geburtsjahr: int) -> int:
        return policy_year - geburtsjahr

    inferred = infer_function_unit(
        function=age,
        input_units={"policy_year": "calendar_year", "geburtsjahr": "calendar_year"},
    )
    assert units_are_equivalent(
        left=inferred, right=resolve_compositional_unit(Unit.YEARS)
    )


def test_infer_duration_added_to_calendar_point_is_a_calendar_point():
    def retirement_year(geburtsjahr: int, statutory_age: int) -> int:
        return geburtsjahr + statutory_age

    inferred = infer_function_unit(
        function=retirement_year,
        input_units={
            "geburtsjahr": "calendar_year",
            "statutory_age": "delta_calendar_year",
        },
    )
    assert units_are_equivalent(left=inferred, right=parse_unit("calendar_year"))


def test_infer_raises_when_two_calendar_points_are_added():
    # `point + point` is affine-invalid; pint refuses it and the dry-run reports.
    def bad(policy_year: int, geburtsjahr: int) -> int:
        return policy_year + geburtsjahr

    with pytest.raises(UnitInferenceError):
        infer_function_unit(
            function=bad,
            input_units={
                "policy_year": "calendar_year",
                "geburtsjahr": "calendar_year",
            },
        )


# ----------------------------------------------------------------------------
# Decorator integration
# ----------------------------------------------------------------------------


def test_policy_function_stores_unit():
    @policy_function(unit=Unit.CURRENCY.PER_MONTH)
    def betrag_m(satz: float, anzahl: int) -> float:
        return satz * anzahl

    assert betrag_m.unit == Unit.CURRENCY.PER_MONTH


def test_policy_function_rejects_invalid_unit_at_decoration():
    # The decorator's type contract only admits `Unit` members (or None);
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
    @policy_function(unit=Unit.DIMENSIONLESS)
    def some_share(x: float) -> float:
        return x

    assert some_share.unit is Unit.DIMENSIONLESS


# ----------------------------------------------------------------------------
# #118: time as a dimension — suffix/reference_period resolution
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        (Unit.CURRENCY.PER_MONTH, "CURRENCY / month"),
        (Unit.CURRENCY.PER_YEAR, "CURRENCY / year"),
        (Unit.CURRENCY.PER_QUARTER, "CURRENCY / quarter_year"),
        (Unit.CURRENCY.PER_WEEK, "CURRENCY / week"),
        (Unit.CURRENCY.PER_DAY, "CURRENCY / day"),
        (Unit.DIMENSIONLESS.PER_YEAR, "1 / year"),  # e.g. a wealth-tax rate
        (Unit.HOURS.PER_WEEK, "working_hour / week"),  # e.g. working hours
        (Unit.CURRENCY.PER_SQUARE_METER.PER_MONTH, "CURRENCY / meter ** 2 / month"),
        (Unit.CURRENCY, "CURRENCY"),  # stock
        (Unit.YEARS, "year"),  # a duration, e.g. an age
        (Unit.MONTHS, "month"),  # a duration in months
        (Unit.DAYS, "day"),  # a duration in days
        (Unit.CALENDAR_YEAR, "calendar_year"),  # a point, e.g. a birth year
        (Unit.CALENDAR_MONTH, "calendar_month"),
        (Unit.CALENDAR_DAY, "calendar_day"),
        (Unit.SQUARE_METER, "meter ** 2"),
        (Unit.HECTARE, "hectare"),
    ],
)
def test_resolve_compositional_unit_period_mapping(token, expected):
    resolved = resolve_compositional_unit(token)
    assert units_are_equivalent(left=resolved, right=parse_unit(expected))


def test_resolve_compositional_unit_dimensionless():
    # A share, a rate, a head count: declared `Unit.DIMENSIONLESS`.
    assert units_are_equivalent(
        left=resolve_compositional_unit(Unit.DIMENSIONLESS),
        right=UNIT_REGISTRY.dimensionless,
    )


def test_flow_period_resolution_distinguishes_month_and_year():
    """A monthly flow and its yearly variant resolve to non-equivalent units."""
    betrag_m = resolve_compositional_unit(Unit.CURRENCY.PER_MONTH)
    betrag_y = resolve_compositional_unit(Unit.CURRENCY.PER_YEAR)
    assert not units_are_equivalent(left=betrag_m, right=betrag_y)


# ----------------------------------------------------------------------------
# #118: pint-sourced conversion factors and duration conversion
# ----------------------------------------------------------------------------


def test_unit_converter_factors_sourced_from_pint():
    # The stock converters multiply by the period-per-year factor; check that
    # factor against pint via the public converter functions.
    def per_year(name):
        return (
            (UNIT_REGISTRY.Quantity(1.0, "year") / UNIT_REGISTRY.Quantity(1.0, name))
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
    months = UNIT_REGISTRY.Quantity(2.0, "year").to("month")
    assert months.magnitude == pytest.approx(24.0)


# ----------------------------------------------------------------------------
# #119: mandatory units
# ----------------------------------------------------------------------------


def test_fail_if_units_are_missing_reports_unannotated_nodes():
    with pytest.raises(UnitDefinitionError, match="kindergeld__betrag_m"):
        fail_if_units_are_missing(
            {"kindergeld__betrag_m": UNSET_UNIT, "alter": Unit.YEARS},
        )


def test_fail_if_units_are_missing_passes_when_all_annotated():
    fail_if_units_are_missing({"a": Unit.CURRENCY, "b": Unit.YEARS})


def test_fail_if_units_are_missing_accepts_dimensionless():
    # `DIMENSIONLESS` *is* a declaration: a dimensionless quantity (GEP 10).
    fail_if_units_are_missing({"some_share": Unit.DIMENSIONLESS, "alter": Unit.YEARS})


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
        (AggType.SUM, Unit.CURRENCY.PER_MONTH, Unit.CURRENCY.PER_MONTH),
        (AggType.MEAN, Unit.CURRENCY.PER_MONTH, Unit.CURRENCY.PER_MONTH),
        (AggType.MIN, Unit.CURRENCY.PER_MONTH, Unit.CURRENCY.PER_MONTH),
        (AggType.MAX, Unit.CURRENCY.PER_MONTH, Unit.CURRENCY.PER_MONTH),
        # A COUNT is a head count: the PERSON count base, independent of source
        # (GEP 10). The level-aware resolved unit is [person]/[target].
        (AggType.COUNT, Unit.CURRENCY.PER_MONTH, Unit.PERSON),
        # ANY / ALL yield booleans, which are dimensionless quantities (GEP 10),
        # independent of the source.
        (AggType.ANY, Unit.CURRENCY.PER_MONTH, Unit.DIMENSIONLESS),
        (AggType.ALL, Unit.CURRENCY.PER_MONTH, Unit.DIMENSIONLESS),
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
        is Unit.PERSON
    )


def test_time_conversion_variants_rebased_period():
    variants = create_time_conversion_functions(
        qname_policy_environment={
            "betrag_m": policy_function(
                leaf_name="betrag_m", unit=Unit.CURRENCY.PER_MONTH
            )(_return_one_float),
        },
        input_columns=set(),
        grouping_levels=("sn", "kin"),
    )
    # Generated betrag_y re-bases the flow's period to its own _y suffix.
    betrag_y_unit = variants["betrag_y"].unit  # ty: ignore[unresolved-attribute]
    assert betrag_y_unit == Unit.CURRENCY.PER_YEAR
    assert units_are_equivalent(
        left=resolve_compositional_unit(betrag_y_unit),
        right=parse_unit("CURRENCY / year"),
    )


def test_auto_aggregation_preserves_source_unit():
    aggs = create_agg_by_group_functions(
        column_functions={
            "betrag_m": policy_function(
                leaf_name="betrag_m", unit=Unit.CURRENCY.PER_MONTH
            )(_return_one_float),
        },
        qname_policy_environment={},
        input_columns=set(),
        tt_targets={"betrag_m_kin"},
        grouping_levels=("kin",),
    )
    assert (
        aggs["betrag_m_kin"].unit  # ty: ignore[unresolved-attribute]
        == Unit.CURRENCY.PER_MONTH
    )


# ----------------------------------------------------------------------------
# #119: edge-consistency / per-function body check (data-independent)
# ----------------------------------------------------------------------------


def test_bare_literal_in_mixed_unit_arithmetic_fails():
    """A bare literal added to a unit-carrying value must be tagged (GEP 10)."""

    def regelaltersgrenze(base: float) -> float:
        return base + 1  # untagged literal added to a [time] quantity

    with pytest.raises(UnitInferenceError):
        infer_function_unit(function=regelaltersgrenze, input_units={"base": "year"})


def test_multiplicative_literal_needs_no_tag():
    """A purely multiplicative literal preserves the unit and needs no tag."""

    def halved(betrag: float) -> float:
        return betrag * 0.5

    inferred = infer_function_unit(function=halved, input_units={"betrag": "CURRENCY"})
    assert units_are_equivalent(left=inferred, right=parse_unit("CURRENCY"))


def test_concrete_currency_per_square_meter_base():
    # A concrete currency divided by an area is a valid compositional unit.
    token = coerce_unit_token("CASTAR_PER_SQUARE_METER_PER_MONTH", where="test")
    assert isinstance(token, CompositeUnit)
    assert token_source_currency(token) == "CASTAR"
    assert token.base == "CASTAR"
    assert token.area == "SQUARE_METER"
    assert token.is_flow


def test_token_is_agnostic_currency():
    assert token_is_agnostic_currency(Unit.CURRENCY)
    assert token_is_agnostic_currency(Unit.CURRENCY.PER_MONTH)
    assert token_is_agnostic_currency(Unit.CURRENCY.PER_SQUARE_METER.PER_MONTH)
    assert not token_is_agnostic_currency(Unit.HECTARE)
    assert not token_is_agnostic_currency(Unit.DIMENSIONLESS.PER_YEAR)
    assert not token_is_agnostic_currency(None)
    assert not token_is_agnostic_currency(coerce_unit_token("CASTAR", where="test"))


# ----------------------------------------------------------------------------
# #120: currency conversion factor
# ----------------------------------------------------------------------------


def test_currency_conversion_factor():
    # silver_penny = castar / 4, registered by mettsim on import.
    assert currency_conversion_factor(
        source_currency="CASTAR", run_currency="SILVER_PENNY"
    ) == pytest.approx(4.0)
    assert currency_conversion_factor(
        source_currency="SILVER_PENNY", run_currency="CASTAR"
    ) == pytest.approx(0.25)
    assert currency_conversion_factor(
        source_currency="CASTAR", run_currency="CASTAR"
    ) == pytest.approx(1.0)


def test_currency_conversion_factor_rejects_unknown_currency():
    with pytest.raises(UnitDefinitionError, match="not a registered currency"):
        currency_conversion_factor(
            source_currency="CASTAR", run_currency="dragon_hoard"
        )


# ----------------------------------------------------------------------------
# #120: Layer-2 boundary — validate and strip pint-tagged inputs
# ----------------------------------------------------------------------------


def test_strip_at_boundary_converts_to_run_currency():
    # silver_penny tag, castar run -> divide by four.
    tagged = UNIT_REGISTRY.Quantity(np.array([4.0]), "SILVER_PENNY")
    bare = strip_input_quantity_at_boundary(
        tagged, run_currency="CASTAR", column_label="wealth"
    )
    assert not isinstance(bare, UNIT_REGISTRY.Quantity)
    assert bare == pytest.approx([1.0])


def test_strip_at_boundary_converts_flow_currency_preserving_period():
    # silver_penny / month -> castar / month: only the currency is rescaled. The
    # tag's /month matches the column's `_m` suffix.
    tagged = UNIT_REGISTRY.Quantity(np.array([4.0]), "SILVER_PENNY / month")
    bare = strip_input_quantity_at_boundary(
        tagged, run_currency="CASTAR", column_label="income_m"
    )
    assert bare == pytest.approx([1.0])


def test_strip_at_boundary_fails_on_missing_period():
    # A flow column (`_m`) tagged without a period: strict mismatch.
    tagged = UNIT_REGISTRY.Quantity(np.array([4.0]), "SILVER_PENNY")
    with pytest.raises(UnitConsistencyError, match="must match the column's suffix"):
        strip_input_quantity_at_boundary(
            tagged, run_currency="CASTAR", column_label="income_m"
        )


def test_strip_at_boundary_fails_on_wrong_period():
    # silver_penny / year against a `_m` column: 12-fold footgun, caught.
    tagged = UNIT_REGISTRY.Quantity(np.array([4.0]), "SILVER_PENNY / year")
    with pytest.raises(UnitConsistencyError, match="month"):
        strip_input_quantity_at_boundary(
            tagged, run_currency="CASTAR", column_label="income_m"
        )


def test_strip_at_boundary_fails_on_period_for_unsuffixed_column():
    # A stock column (no suffix) tagged with a period.
    tagged = UNIT_REGISTRY.Quantity(np.array([4.0]), "SILVER_PENNY / month")
    with pytest.raises(UnitConsistencyError, match="no time suffix"):
        strip_input_quantity_at_boundary(
            tagged, run_currency="CASTAR", column_label="wealth"
        )


def test_strip_at_boundary_does_not_flag_numerator_time_unit():
    # An age tagged in years: `year` is a numerator, not a flow period, so an
    # unsuffixed column is fine. Nothing to convert (no currency).
    tagged = UNIT_REGISTRY.Quantity(np.array([30.0]), "year")
    bare = strip_input_quantity_at_boundary(
        tagged, run_currency="CASTAR", column_label="age"
    )
    assert bare == pytest.approx([30.0])


def test_strip_at_boundary_keys_period_off_denominator_for_hours_flow():
    # `arbeitsstunden_w` is HOURS_FLOW (`working_hour / week`): the flow period
    # is the denominator (week), not the `[hours]` numerator. The tag's `/week`
    # matches the `_w` suffix; there is no currency, so the value passes through.
    tagged = UNIT_REGISTRY.Quantity(np.array([40.0]), "working_hour / week")
    bare = strip_input_quantity_at_boundary(
        tagged, run_currency="CASTAR", column_label="arbeitsstunden_w"
    )
    assert bare == pytest.approx([40.0])

    # A wrong period is still caught (month != week).
    with pytest.raises(UnitConsistencyError, match="week"):
        strip_input_quantity_at_boundary(
            UNIT_REGISTRY.Quantity(np.array([40.0]), "working_hour / month"),
            run_currency="CASTAR",
            column_label="arbeitsstunden_w",
        )


def test_strip_at_boundary_strips_matching_currency():
    tagged = UNIT_REGISTRY.Quantity(np.array([3.0]), "CASTAR")
    bare = strip_input_quantity_at_boundary(tagged, run_currency="CASTAR")
    assert not isinstance(bare, UNIT_REGISTRY.Quantity)
    assert list(bare) == [3.0]


def test_strip_at_boundary_passes_non_currency_tag_through():
    # No currency component -> nothing to convert.
    tagged = UNIT_REGISTRY.Quantity(np.array([5.0]), "working_hour")
    bare = strip_input_quantity_at_boundary(tagged, run_currency="CASTAR")
    assert bare == pytest.approx([5.0])


def test_strip_at_boundary_without_run_currency_just_strips():
    tagged = UNIT_REGISTRY.Quantity(np.array([3.0]), "SILVER_PENNY")
    bare = strip_input_quantity_at_boundary(tagged, run_currency=None)
    assert list(bare) == [3.0]


# ----------------------------------------------------------------------------
# Compositional units (GEP 10 compositional units) — builder, parser, formatter
# ----------------------------------------------------------------------------


def test_builder_round_trips_with_flat_spelling():
    # The fluent `.py` builder and the flat YAML string are the same unit.
    built = Unit.CURRENCY.PER_SQUARE_METER.PER_MONTH
    assert str(built) == "CURRENCY_PER_SQUARE_METER_PER_MONTH"
    assert parse_compositional_unit(str(built)) == built


def test_builder_round_trips_with_level():
    built = Unit.CURRENCY.PER_MONTH.PER_BG
    assert str(built) == "CURRENCY_PER_MONTH_PER_BG"
    assert parse_compositional_unit("CURRENCY_PER_MONTH_PER_BG") == built


def test_builder_generic_per_level_matches_attribute():
    assert (
        Unit.PERSON.PER_LEVEL("bg")
        == Unit.PERSON.PER_BG
        == parse_compositional_unit("PERSON_PER_BG")
    )


@pytest.mark.parametrize(
    ("spelling", "base", "area", "period", "level"),
    [
        ("CURRENCY", "CURRENCY", None, None, None),
        ("CURRENCY_PER_MONTH", "CURRENCY", None, "MONTH", None),
        ("CURRENCY_PER_SQUARE_METER", "CURRENCY", "SQUARE_METER", None, None),
        ("PERSON_PER_BG", "PERSON", None, None, "BG"),
        ("DIMENSIONLESS_PER_YEAR", "DIMENSIONLESS", None, "YEAR", None),
        ("DIMENSIONLESS_PER_BG", "DIMENSIONLESS", None, None, "BG"),
        ("HOURS_PER_WEEK", "HOURS", None, "WEEK", None),
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
    ],
)
def test_parse_compositional_unit_rejects_bad_spellings(spelling):
    with pytest.raises(UnitDefinitionError):
        parse_compositional_unit(spelling)


def test_builder_rejects_non_canonical_order():
    # Level before period, an area after a period, a second period — all caught
    # by the staged builder, mirroring the parser.
    with pytest.raises(UnitDefinitionError, match="precede"):
        _ = Unit.CURRENCY.PER_BG.PER_MONTH
    with pytest.raises(UnitDefinitionError, match="precede"):
        _ = Unit.CURRENCY.PER_MONTH.PER_YEAR


def test_is_flow_and_carries_level_properties():
    assert Unit.CURRENCY.PER_MONTH.is_flow
    assert not parse_compositional_unit("CURRENCY").is_flow
    assert Unit.CURRENCY.PER_BG.carries_level
    assert not Unit.CURRENCY.PER_MONTH.carries_level


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [
        ("CURRENCY_PER_MONTH", "CURRENCY / month"),
        ("CURRENCY_PER_YEAR", "CURRENCY / year"),
        ("DIMENSIONLESS_PER_YEAR", "1 / year"),
        ("HOURS_PER_WEEK", "working_hour / week"),
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
    compositional = resolve_compositional_unit(parse_compositional_unit(spelling))
    assert units_are_equivalent(left=compositional, right=parse_unit(expected))


def test_person_per_level_resolves_to_head_count():
    # PERSON_PER_BG is the old HEADCOUNT at bg: [person] / [bg], the unit a COUNT
    # aggregation to bg mints.
    compositional = resolve_compositional_unit(
        parse_compositional_unit("PERSON_PER_BG")
    )
    assert units_are_equivalent(
        left=compositional, right=grouping_level_count_unit(target_level="bg")
    )


def test_concrete_currency_base_resolves_like_agnostic():
    # For dimensionality a concrete currency means exactly what CURRENCY means.
    concrete = resolve_compositional_unit(parse_compositional_unit("CASTAR_PER_MONTH"))
    agnostic = resolve_compositional_unit(
        parse_compositional_unit("CURRENCY_PER_MONTH")
    )
    assert units_are_equivalent(left=concrete, right=agnostic)


def test_resolve_compositional_unit_rejects_unregistered_level():
    with pytest.raises(UnitDefinitionError, match="grouping level"):
        resolve_compositional_unit(parse_compositional_unit("CURRENCY_PER_NEVERLAND"))


# ----------------------------------------------------------------------------
# Model change 1: the [hours] dimension (GEP 10 compositional units)
# ----------------------------------------------------------------------------


def test_working_hour_is_its_own_dimension():
    # Working hours are `[hours]`, isolated from pint's `[time]`, so they cannot
    # convert to a period and a `working_hour / week` flow does not collapse to a
    # bare dimensionless number the way a `[time] / [time]` hour-flow would.
    assert UNIT_REGISTRY.Quantity(1.0, "working_hour").dimensionality == {"[hours]": 1}
    assert not UNIT_REGISTRY.Quantity(1.0, "working_hour").is_compatible_with("day")
    hours_per_week = resolve_compositional_unit(Unit.HOURS.PER_WEEK)
    assert UNIT_REGISTRY.Quantity(1.0, hours_per_week).dimensionality == {
        "[hours]": 1,
        "[time]": -1,
    }


def test_bare_time_hour_is_no_longer_an_admissible_token():
    # There is exactly one spelling for working hours; pint's `[time]` `hour` is
    # not admissible (GEP 10).
    with pytest.raises(UnitDefinitionError, match="does not know about"):
        parse_unit("working_hour / hour")


def test_adding_working_hours_to_a_share_is_caught():
    def total(hours: float, share: float) -> float:
        return hours + share

    with pytest.raises(UnitInferenceError):
        infer_function_unit(
            function=total,
            input_units={
                "hours": resolve_compositional_unit(Unit.HOURS.PER_WEEK),
                "share": UNIT_REGISTRY.dimensionless,
            },
        )


def test_wage_per_working_hour_times_hours_is_a_currency_flow():
    # A wage per working hour is `CURRENCY / working_hour`; multiplying it by
    # working hours per month cancels the `[hours]` and yields `CURRENCY / month`.
    def income(wage: float, hours_m: float) -> float:
        return wage * hours_m

    inferred = infer_function_unit(
        function=income,
        input_units={
            # A per-working-hour wage: `CURRENCY / working_hour`. The `[hours]`
            # dimension has no compositional period denominator, so this internal
            # pint surface is built directly (GEP 10).
            "wage": "CURRENCY / working_hour",
            "hours_m": resolve_compositional_unit(Unit.HOURS.PER_MONTH),
        },
    )
    assert units_are_equivalent(left=inferred, right=parse_unit("CURRENCY / month"))


def test_hours_per_week_rebases_period_only():
    # The one conversion working hours admit: re-basing the [time] period
    # (week -> month) leaves the [hours] numerator untouched.
    per_week = resolve_compositional_unit(parse_compositional_unit("HOURS_PER_WEEK"))
    per_month = resolve_compositional_unit(parse_compositional_unit("HOURS_PER_MONTH"))
    assert (
        per_week.dimensionality
        == per_month.dimensionality
        == {
            "[hours]": 1,
            "[time]": -1,
        }
    )
    # Different periods are not equivalent (a 52/12 factor apart).
    assert not units_are_equivalent(left=per_week, right=per_month)
