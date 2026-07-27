"""Shared policy-environment fixtures for the GEP 10 unit test suites.

The declaration-side suite (`test_unit_validation`) and the body-inference suite
(`test_unit_inference`) build their environments from the same handful of
producers — a currency stock, a monthly and a yearly flow, a calendar point, a
duration, an identifier, a group id. They live here so the two suites cannot
drift apart on what "a monthly flow" means.
"""

from __future__ import annotations

import datetime
from typing import Any

import pint

from tests.test_unit_system import TEST_UNIT_SYSTEM
from ttsim.tt import (
    UNSET_UNIT,
    FKType,
    TTSIMUnit,
    group_creation_function,
    policy_function,
    policy_input,
)
from ttsim.tt.param_objects import ScalarParam
from ttsim.tt.units import ttsim_unit_from_yaml_value
from ttsim.typing import IntColumn

UNIT_SYSTEM = TEST_UNIT_SYSTEM
REGISTRY = UNIT_SYSTEM.registry

GROUPING_LEVELS = ("fam", "kin")

_START = datetime.date(2020, 1, 1)
_END = datetime.date(2030, 12, 31)

# Parameters must pin down the concrete currency their numbers are written in
# (GEP 10); these are mettsim's concrete (castar) compositional spellings.
CASTAR_PER_YEAR = ttsim_unit_from_yaml_value(
    value="CASTAR_PER_YEAR", where="test setup"
)
CASTAR_PER_MONTH = ttsim_unit_from_yaml_value(
    value="CASTAR_PER_MONTH", where="test setup"
)
CASTAR = ttsim_unit_from_yaml_value(value="CASTAR", where="test setup")


# Fixture objects


@policy_input(unit=TTSIMUnit.DIMENSIONLESS)
def p_id() -> int:
    """Identifier; a dimensionless quantity (GEP 10)."""


@policy_input(foreign_key_type=FKType.MAY_POINT_TO_SELF, unit=TTSIMUnit.DIMENSIONLESS)
def p_id_recipient() -> int:
    """Person pointer; a dimensionless quantity (GEP 10)."""


@policy_input(unit=TTSIMUnit.CURRENCY)
def wealth() -> float:
    """A stock of currency."""


@policy_input(unit=TTSIMUnit.DIMENSIONLESS)
def is_exempt() -> bool:
    """Boolean input; a dimensionless quantity (GEP 10)."""


@policy_input(unit=UNSET_UNIT)
def unannotated_income_y() -> float:
    """Carries the UNSET sentinel; the missing-units check must report it."""


@group_creation_function(unit=TTSIMUnit.DIMENSIONLESS)
def fam_id(p_id: IntColumn, xnp: object) -> IntColumn:  # noqa: ARG001
    """Group creation (a dimensionless id)."""
    return p_id


def _scalar_unit(resolved: dict[str, Any], qname: str) -> pint.Unit:
    unit = resolved[qname]
    assert isinstance(unit, pint.Unit)
    return unit


def _unit_tree(resolved: dict[str, Any], qname: str) -> dict[str | int, Any]:
    unit = resolved[qname]
    assert isinstance(unit, dict)
    return unit


def make_flow_rate() -> ScalarParam:
    # A wealth-tax rate is a share per year: `DIMENSIONLESS_PER_YEAR`, used at the
    # `tax_rate_y` name (the spelled period agrees with the suffix, GEP 10).
    return ScalarParam(
        value=0.01,
        unit=TTSIMUnit.DIMENSIONLESS.PER_YEAR,
        start_date=_START,
        end_date=_END,
    )


@policy_function(unit=TTSIMUnit.CURRENCY.PER_YEAR)
def amount_y(wealth: float, tax_rate_y: float, is_exempt: bool) -> float:
    """The wealth-tax pattern: stock times a per-year rate, guarded by an
    exemption. ``tax_rate_y`` is a share per year, so the product is a flow."""
    if is_exempt:
        return 0.0
    return wealth * tax_rate_y


@policy_input(unit=TTSIMUnit.CURRENCY.PER_MONTH)
def income_m() -> float:
    """A monthly flow of currency (CURRENCY / month)."""


@policy_input(unit=TTSIMUnit.CURRENCY.PER_YEAR)
def bonus_y() -> float:
    """A yearly flow of currency (CURRENCY / year)."""


@policy_input(unit=TTSIMUnit.CALENDAR_YEAR)
def geburtsjahr() -> int:
    """A birth year: a point on the calendar, not a duration (GEP 10)."""


@policy_input(unit=TTSIMUnit.YEARS)
def statutory_age() -> int:
    """A duration in years (an age threshold)."""
