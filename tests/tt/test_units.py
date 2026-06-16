"""Tests for the pint-based unit framework (GEP 10, issue #117)."""

from __future__ import annotations

# Importing the mettsim package registers its base currency (``castar``) and
# makes the tracer-bullet policy function importable.
import mettsim.middle_earth  # noqa: F401
import pytest
from beartype.roar import BeartypeCallHintViolation
from mettsim.middle_earth.property_tax.amount import (
    acre_size_after_cap,
)

from ttsim.exceptions import (
    PolicyFunctionDefinitionError,
    UnitConsistencyError,
    UnitDefinitionError,
    UnitInferenceError,
)
from ttsim.tt import (
    CURRENCY_TOKEN,
    UREG,
    CurrencyUnitToken,
    Unit,
    coerce_unit_token,
    fail_if_function_unit_is_inconsistent,
    infer_function_unit,
    parse_unit,
    policy_function,
    register_currency,
    token_source_currency,
    units_are_equivalent,
)
from ttsim.tt.units import unit_token_is_flow

# ----------------------------------------------------------------------------
# The unit vocabulary
# ----------------------------------------------------------------------------


def test_currency_token_anchors_currency_dimension():
    assert UREG.Quantity(1.0, CURRENCY_TOKEN).dimensionality == {"[currency]": 1}


def test_quarter_year_is_a_quarter_of_a_year():
    ratio = (UREG.Quantity(1.0, "year") / UREG.Quantity(1.0, "quarter_year")).to(
        "dimensionless"
    )
    assert ratio.magnitude == pytest.approx(4.0)


def test_hectare_is_an_area():
    assert UREG.Quantity(1.0, "hectare").dimensionality == {"[length]": 2}


# ----------------------------------------------------------------------------
# The Unit token enumeration (the declaration surface)
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("token", list(Unit))
def test_coerce_unit_token_accepts_every_member_spelling(token):
    assert coerce_unit_token(str(token), where="test") is token


def test_coerce_unit_token_rejects_none():
    # `None` is no longer a dimensionless declaration (GEP 10): it reaches
    # `coerce_unit_token` only through an internal bug, so the package claw
    # rejects it before the body runs.
    with pytest.raises(BeartypeCallHintViolation):
        coerce_unit_token(None, where="test")  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    "value",
    [
        # Pint syntax is not a declaration: one token = one meaning. (Bare
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
    with pytest.raises(UnitDefinitionError, match="invalid unit token"):
        coerce_unit_token(value, where="test")


def test_flow_tokens_are_marked_in_the_name():
    # The naming principle: a token needs a period iff it says `_FLOW`.
    for token in Unit:
        assert unit_token_is_flow(token) == token.name.endswith("_FLOW")


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
    assert UREG.Quantity(1.0, "CASTAR").dimensionality == {"[currency]": 1}


def test_register_relative_currency_bakes_correct_factor():
    register_currency("SILVER_PENNY", definition="CASTAR / 4")
    factor = (UREG.Quantity(1.0, "SILVER_PENNY") / UREG.Quantity(1.0, "CASTAR")).to(
        "dimensionless"
    )
    assert factor.magnitude == pytest.approx(0.25)


def test_register_currency_idempotent():
    # Re-registering with a consistent definition is a no-op, not an error.
    register_currency("SILVER_PENNY", definition="CASTAR / 4")


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
        ("CASTAR_FLOW", "CASTAR", True),
        ("SILVER_PENNY", "SILVER_PENNY", False),
        ("SILVER_PENNY_FLOW", "SILVER_PENNY", True),
    ],
)
def test_registration_derives_declaration_tokens(spelling, currency, is_flow):
    register_currency("SILVER_PENNY", definition="CASTAR / 4")
    token = coerce_unit_token(spelling, where="test")
    assert isinstance(token, CurrencyUnitToken)
    assert token == spelling
    assert token.currency == currency
    assert unit_token_is_flow(token) == is_flow


def test_coerce_currency_token_is_idempotent_and_singleton():
    token = coerce_unit_token("CASTAR", where="test")
    assert coerce_unit_token(token, where="test") is token
    assert coerce_unit_token("CASTAR", where="test") is token


def test_token_source_currency():
    assert token_source_currency(coerce_unit_token("CASTAR_FLOW", where="t")) == (
        "CASTAR"
    )
    assert token_source_currency(Unit.CURRENCY_FLOW) is None
    assert token_source_currency(Unit.HECTARES) is None
    assert token_source_currency(None) is None


def test_unregistered_currency_spelling_is_rejected():
    with pytest.raises(UnitDefinitionError, match="invalid unit token"):
        coerce_unit_token("MITHRIL", where="test")


def test_register_currency_rejects_token_collision_with_core_vocabulary():
    # "currency".upper() collides with the core Unit.CURRENCY token.
    with pytest.raises(UnitDefinitionError, match="collides"):
        register_currency("currency", definition="CASTAR / 2")


def test_policy_function_rejects_currency_token_at_decoration():
    # Functions are currency-agnostic by design: the decorator's type
    # contract only admits `Unit` members, so a concrete currency token is
    # rejected by the beartype claw at decoration time.
    token = coerce_unit_token("CASTAR_FLOW", where="test")
    with pytest.raises(PolicyFunctionDefinitionError, match="unit"):

        @policy_function(unit=token)  # ty: ignore[invalid-argument-type]
        def betrag_m(x: float) -> float:
            return x


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


# ----------------------------------------------------------------------------
# Tracer bullet: one real mettsim function passes; a hand-made clash fails
# ----------------------------------------------------------------------------


def test_tracer_bullet_mettsim_function_passes():
    """`acre_size_after_cap` returns an area; the body is sound."""
    fail_if_function_unit_is_inconsistent(
        function=acre_size_after_cap.function,
        declared_unit="hectare",
        input_units={
            "acre_size": "hectare",
            "cap": "hectare",
            "year_from_which_cap_is_applied": "year",
            "evaluation_year": "year",
        },
    )


def test_tracer_bullet_currency_per_area_clash_fails():
    """A hand-made CURRENCY + CURRENCY/area clash fails with a clear error."""

    def property_tax(base_amount: float, rate_per_area: float) -> float:
        # Dimensionally invalid: a currency plus a currency-per-area.
        return base_amount + rate_per_area

    with pytest.raises(UnitInferenceError, match="Dimensionally invalid"):
        fail_if_function_unit_is_inconsistent(
            function=property_tax,
            declared_unit="CURRENCY",
            input_units={
                "base_amount": "CURRENCY",
                "rate_per_area": "CURRENCY / hectare",
            },
        )


def test_inconsistent_declared_unit_fails():
    def amount(area: float) -> float:
        return area

    with pytest.raises(UnitConsistencyError, match="declares unit"):
        fail_if_function_unit_is_inconsistent(
            function=amount,
            declared_unit="CURRENCY",
            input_units={"area": "hectare"},
        )


# ----------------------------------------------------------------------------
# Decorator integration
# ----------------------------------------------------------------------------


def test_policy_function_stores_unit():
    @policy_function(unit=Unit.CURRENCY_FLOW)
    def betrag_m(satz: float, anzahl: int) -> float:
        return satz * anzahl

    assert betrag_m.unit is Unit.CURRENCY_FLOW


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


def test_policy_function_unit_defaults_to_none():
    @policy_function()
    def something(x: float) -> float:
        return x

    assert something.unit is None
