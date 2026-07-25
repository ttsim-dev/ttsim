"""The per-policy-system unit registry (GEP 10)."""

from __future__ import annotations

import datetime

import pytest

from ttsim.exceptions import UnitDefinitionError
from ttsim.tt.units import UnitSystem


@pytest.fixture
def germany() -> UnitSystem:
    return UnitSystem(
        base_currency="EUR",
        other_currencies={"DM": "EUR / 1.95583"},
        statutory_currencies={"0001-01-01": "DM", "2002-01-01": "EUR"},
    )


@pytest.fixture
def middle_earth() -> UnitSystem:
    return UnitSystem(
        base_currency="CASTAR",
        other_currencies={"SILVER_PENNY": "CASTAR / 4"},
        statutory_currencies={"0001-01-01": "SILVER_PENNY", "2020-01-01": "CASTAR"},
    )


def test_systems_with_different_base_currencies_coexist(germany, middle_earth):
    """Two policy systems, each with its own base currency, live side by side."""
    assert (germany.base_currency, middle_earth.base_currency) == ("EUR", "CASTAR")


def test_each_system_converts_within_its_own_currencies(germany, middle_earth):
    """A conversion factor is read off the system that defines both currencies."""
    assert (
        germany.currency_conversion_factor(source_currency="DM", target_currency="EUR"),
        middle_earth.currency_conversion_factor(
            source_currency="SILVER_PENNY", target_currency="CASTAR"
        ),
    ) == (pytest.approx(1 / 1.95583), pytest.approx(0.25))


def test_conversion_across_systems_is_rejected(germany):
    """A currency of another system is not convertible — no silent factor of 1."""
    with pytest.raises(UnitDefinitionError, match="'CASTAR' is not a registered"):
        germany.currency_conversion_factor(
            source_currency="EUR", target_currency="CASTAR"
        )


def test_statutory_currency_follows_each_system_mapping(germany, middle_earth):
    """Each system reads the statutory currency off its own dated mapping."""
    date = datetime.date(2019, 12, 31)
    assert (
        germany.statutory_currency_for_date(date),
        middle_earth.statutory_currency_for_date(date),
    ) == ("EUR", "SILVER_PENNY")
