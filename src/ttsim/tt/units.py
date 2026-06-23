"""The pint-based unit framework (GEP 10).

This module establishes the closed unit vocabulary used to check the
dimensional soundness of the taxes-and-transfers DAG, plus the build-time
machinery that performs the check.

The foundational constraint of GEP 10 is that **pint never wraps a live
array**. A :class:`pint.Quantity` is not a JAX pytree and does not trace under
``jit``. pint is used here in exactly two build-time roles:

- to run the *dry-run* dimensionality check on representative ``Quantity``\\ s
  (:func:`infer_function_unit`); and
- (from issue #118 onwards) to source time- and currency-conversion *factors*,
  which are baked into the numeric workers.

The numeric runtime path stays pure arrays, single currency, JAX-safe.

A declaration is one token of the vocabulary — ``unit=Unit.CURRENCY_FLOW``
in code, ``unit: CURRENCY_FLOW`` in YAML. The vocabulary has two parts: the
core :class:`Unit` enumeration, which grows only by GEP amendment plus TTSIM
PR, and per-package :class:`CurrencyUnitToken`\\ s, which
:func:`register_currency` derives from each registered concrete currency
(``DM``, ``DM_FLOW``, …). The agnostic currency tokens
(:attr:`Unit.CURRENCY` / :attr:`Unit.CURRENCY_FLOW`) denote the
*union* of the registered currencies: for dimensionality checks a concrete
currency token means exactly what its agnostic counterpart means, and in
addition names the currency a parameter's numbers are denominated in, which
the build-time conversion to the run currency reads off the declaration.
Each package's JSON schema for the parameter YAMLs stays enumerable by
listing the core tokens plus that package's own currency tokens.

Tokens come in two kinds: *flow* tokens (named ``…_FLOW``) denote a
per-period quantity and are completed by a period supplied by the name
suffix (columns/functions) or ``reference_period`` (parameters); all other
tokens are *complete* as written and admit no period source. Declarations
never contain pint syntax; internally each token resolves to a pint unit.
A dimensionless quantity (a share, a rate, a head count) declares
:attr:`Unit.DIMENSIONLESS` (``unit: DIMENSIONLESS``) and never combines with a
period source; the per-period dimensionless quantity is its own flow token
(:attr:`Unit.DIMENSIONLESS_FLOW`).

The :data:`CURRENCY_TOKEN` (the literal string ``"CURRENCY"``) is a real
unit anchoring the ``[currency]`` dimension, so the currency tokens resolve
regardless of whether a concrete currency has been registered yet — checks
compare at the dimensionality level and the concrete currency is resolved
separately (issue #120). Columns and functions must declare the agnostic
tokens (they are currency-agnostic by design); parameters must pin down the
concrete currency their numbers are written in.

Literals (GEP 10)
-----------------

The dry-run executes the scalar body on representative ``Quantity``\\ s, so a
*bare numeric literal* combined additively with a unit-carrying value raises a
:class:`~ttsim.exceptions.UnitInferenceError` (pint refuses to add a
dimensionless number to a currency). A literal that is purely a multiplicative
factor (``betrag * 0.5``) needs no tag, because multiplying by a dimensionless
number preserves the unit. Most additive cases dissolve once quantities are
declared correctly — an ordinal such as ``geburtsmonat`` is ``DIMENSIONLESS``, so
``geburtsmonat - 1`` is dimensionless arithmetic. A genuine constant of a real
dimension is promoted to a parameter; a genuine code-level constant lets its
function body opt out of inference with ``@policy_function(verify_units=False)``.
"""

from __future__ import annotations

import enum
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Self

import pint
from pint.util import to_units_container

from ttsim.exceptions import (
    UnitConsistencyError,
    UnitDefinitionError,
    UnitInferenceError,
)
from ttsim.tt.aggregation import AggType

#: Maps a GEP-1 time-unit suffix id (``_y``/``_q``/``_m``/``_w``/``_d``) to the
#: pint unit naming its period. Time is a first-class pint dimension (GEP 10):
#: the suffix supplies the per-period denominator of a flow.
TIME_UNIT_ID_TO_PINT_NAME = {
    "y": "year",
    "q": "quarter_year",
    "m": "month",
    "w": "week",
    "d": "day",
}

#: Matches a trailing GEP-1 time-unit suffix (``…_m``) on a column's qualified
#: name, naming the column's flow period. Used at the Layer-2 input boundary to
#: check a pint tag's period against the column's suffix.
_QNAME_TIME_SUFFIX_PATTERN = re.compile(
    rf"_(?P<time_unit>[{''.join(TIME_UNIT_ID_TO_PINT_NAME)}])$"
)

#: Maps a ``reference_period`` label (the functional flow period of a parameter)
#: to the pint unit naming its period.
REFERENCE_PERIOD_TO_PINT_NAME = {
    "Year": "year",
    "Quarter": "quarter_year",
    "Month": "month",
    "Week": "week",
    "Day": "day",
    "Hour": "hour",
}


#: The pint unit anchoring the ``[currency]`` dimension, used internally to
#: resolve the currency tokens (``CURRENCY_FLOW``, ``CURRENCY``, …)
#: before any concrete currency is registered. Checks compare at the
#: dimensionality level; the concrete currency is resolved separately.
CURRENCY_TOKEN = "CURRENCY"  # noqa: S105 (a unit token, not a secret)


class Unit(enum.StrEnum):
    """The core vocabulary of unit tokens (GEP 10).

    One token = one meaning, independent of any other field. A bare token is
    *complete* as written; a ``…_FLOW`` token *needs a period*, supplied by
    the name suffix (columns/functions) or ``reference_period`` (parameters).
    Where both a stock and a flow of a quantity exist, the flow is marked
    ``…_FLOW`` and the stock is the bare token (:attr:`CURRENCY` /
    :attr:`CURRENCY_FLOW`).

    The full vocabulary is this enumeration plus the
    :class:`CurrencyUnitToken` declaration tokens that
    :func:`register_currency` derives from each registered concrete currency.
    The agnostic currency tokens here denote the *union* of the registered
    currencies; they are the only currency tokens columns and functions may
    declare — functions are currency-agnostic by design.

    YAML spells the identical strings (``unit: CURRENCY_FLOW``); Python code
    must use the members themselves (``unit=Unit.CURRENCY_FLOW``).
    """

    CURRENCY_FLOW = "CURRENCY_FLOW"
    """An amount of currency per period: wages, claims, benefits."""

    CURRENCY = "CURRENCY"
    """An amount of currency, full stop: wealth, asset thresholds. The complete
    (non-period) counterpart of :attr:`CURRENCY_FLOW` — a currency *stock* is this
    bare token, a currency *flow* is :attr:`CURRENCY_FLOW`. There is no
    ``CURRENCY_STOCK`` spelling: the ``…_FLOW`` suffix is the only flow marker."""

    DIMENSIONLESS = "DIMENSIONLESS"
    """A plain dimensionless number: a share (a Steuersatz), a rate, a head
    count. The complete (non-period) counterpart of :attr:`DIMENSIONLESS_FLOW`.
    There is no ``DIMENSIONLESS_STOCK`` — a dimensionless level is the bare
    token. Replaces the former ``unit=None`` / ``unit: null`` spelling."""

    DIMENSIONLESS_FLOW = "DIMENSIONLESS_FLOW"
    """A dimensionless quantity *per period* (``1/period``): a count or a
    share per unit time — births per year, or the per-year change of a
    dimensionless factor (the pension Zugangsfaktor). The complete,
    non-period counterpart is :attr:`DIMENSIONLESS`."""

    YEARS = "YEARS"
    """A *duration* in years: an age, an age threshold, a number of years. A
    duration is the difference between two calendar points and is fully
    multiplicative (``YEARS * 2`` scales, ``YEARS`` converts to ``MONTHS``). The
    calendar *point* counterpart — a specific year on the calendar — is
    :attr:`CALENDAR_YEAR` (GEP 10)."""

    MONTHS = "MONTHS"
    """A *duration* in months: the span between two calendar months. The
    multiplicative duration counterpart of the :attr:`CALENDAR_MONTH` point."""

    DAYS = "DAYS"
    """A *duration* in days: the span between two calendar days. The
    multiplicative duration counterpart of the :attr:`CALENDAR_DAY` point."""

    CALENDAR_YEAR = "CALENDAR_YEAR"
    """A *point* on the calendar measured in years: a birth year, the policy
    year. An affine point, not a duration (GEP 10): two calendar years
    *subtract* to a :attr:`YEARS` duration (``policy_year - geburtsjahr``) and a
    :attr:`YEARS` duration *adds* to a calendar year, but two calendar years
    cannot be added and a calendar year cannot be scaled."""

    CALENDAR_MONTH = "CALENDAR_MONTH"
    """A *point* on the calendar measured in months. The month-axis counterpart
    of :attr:`CALENDAR_YEAR`; two subtract to a :attr:`MONTHS` duration. A
    *cyclic* month-of-year ordinal (``geburtsmonat`` 1-12) is not a calendar
    point but :attr:`DIMENSIONLESS` (GEP 10)."""

    CALENDAR_DAY = "CALENDAR_DAY"
    """A *point* on the calendar measured in days. The day-axis counterpart of
    :attr:`CALENDAR_YEAR`; two subtract to a :attr:`DAYS` duration."""

    HOURS_FLOW = "HOURS_FLOW"
    """Hours per period: working hours."""

    SQUARE_METERS = "SQUARE_METERS"
    """An area in square meters: dwelling size."""

    HECTARES = "HECTARES"
    """An area in hectares: land."""

    CURRENCY_PER_SQUARE_METER_FLOW = "CURRENCY_PER_SQUARE_METER_FLOW"
    """An amount of currency per square meter per period: rent caps."""


# A frozen dataclass, not a ``NamedTuple``: under ``from __future__ import
# annotations`` beartype's package claw mis-reads a ``NamedTuple`` field whose
# stringified annotation is not a bare identifier (here ``str | None``) as a
# forward reference and raises at import. A dataclass with the same fields is
# fine, and the type is only ever accessed by attribute, never as a tuple.
@dataclass(frozen=True, slots=True)
class _TokenResolution:
    """How a token resolves internally: pint base expression and kind."""

    base: str | None
    """The pint expression of the token's non-period part (``None`` for a
    dimensionless base)."""
    is_flow: bool
    """Whether a period must be supplied."""


#: Maps each token to its resolution. Internal — declarations never contain
#: pint syntax.
_TOKEN_BASE_AND_IS_FLOW: dict[Unit, _TokenResolution] = {
    Unit.CURRENCY_FLOW: _TokenResolution(base=CURRENCY_TOKEN, is_flow=True),
    Unit.CURRENCY: _TokenResolution(base=CURRENCY_TOKEN, is_flow=False),
    Unit.DIMENSIONLESS: _TokenResolution(base=None, is_flow=False),
    Unit.DIMENSIONLESS_FLOW: _TokenResolution(base=None, is_flow=True),
    Unit.YEARS: _TokenResolution(base="delta_calendar_year", is_flow=False),
    Unit.MONTHS: _TokenResolution(base="delta_calendar_month", is_flow=False),
    Unit.DAYS: _TokenResolution(base="delta_calendar_day", is_flow=False),
    Unit.CALENDAR_YEAR: _TokenResolution(base="calendar_year", is_flow=False),
    Unit.CALENDAR_MONTH: _TokenResolution(base="calendar_month", is_flow=False),
    Unit.CALENDAR_DAY: _TokenResolution(base="calendar_day", is_flow=False),
    Unit.HOURS_FLOW: _TokenResolution(base="hour", is_flow=True),
    Unit.SQUARE_METERS: _TokenResolution(base="meter ** 2", is_flow=False),
    Unit.HECTARES: _TokenResolution(base="hectare", is_flow=False),
    Unit.CURRENCY_PER_SQUARE_METER_FLOW: _TokenResolution(
        base=f"{CURRENCY_TOKEN} / meter ** 2", is_flow=True
    ),
}


#: The agnostic (currency-dimensioned) tokens of the core vocabulary. Each
#: gets a concrete per-currency variant on :func:`register_currency`;
#: parameters must declare the concrete variant (GEP 10).
_AGNOSTIC_CURRENCY_TOKENS: frozenset[Unit] = frozenset(
    token
    for token, resolution in _TOKEN_BASE_AND_IS_FLOW.items()
    if resolution.base is not None and CURRENCY_TOKEN in resolution.base
)


class CurrencyUnitToken(str):
    """A declaration token pinning down a concrete currency (GEP 10).

    Created by :func:`register_currency`, never directly:
    ``register_currency("DM", ...)`` derives one concrete variant per
    currency-dimensioned core token — ``DM``, ``DM_FLOW``,
    ``DM_PER_SQUARE_METER_FLOW``. For all dimensionality checks the token
    means exactly what its agnostic counterpart means — a registered
    currency *is a* ``CURRENCY``. In addition it names the concrete currency
    a parameter's numbers are denominated in, which the build-time conversion
    to the run currency reads off the declaration.

    Only *parameters* may declare these tokens. Columns and functions take
    :class:`Unit` members and thereby stay currency-agnostic.
    """

    __slots__ = ("agnostic", "currency")

    currency: str
    agnostic: Unit

    def __new__(cls, spelling: str, *, currency: str, agnostic: Unit) -> Self:
        self = super().__new__(cls, spelling)
        self.currency = currency
        self.agnostic = agnostic
        return self


def token_is_agnostic_currency(token: Unit | CurrencyUnitToken | None) -> bool:
    """Whether a token is a currency-dimensioned member of the core vocabulary.

    Parameters must not declare these once a concrete currency is registered
    (GEP 10): the declaration names the currency the numbers are written in,
    so it must be one of the concrete per-currency variants.
    """
    return isinstance(token, Unit) and token in _AGNOSTIC_CURRENCY_TOKENS


#: One declaration token per (registered currency, kind): ``DM``,
#: ``DM_FLOW``, … — populated by :func:`register_currency`, keyed by spelling.
_CURRENCY_UNIT_TOKENS: dict[str, CurrencyUnitToken] = {}


#: Any member of the full token vocabulary: the core enumeration or a
#: currency token derived from a registered currency.
UnitToken = Unit | CurrencyUnitToken


def unit_token_is_flow(token: Unit | CurrencyUnitToken) -> bool:
    """Whether a token denotes a per-period quantity (needs a period source)."""
    if isinstance(token, CurrencyUnitToken):
        token = token.agnostic
    return _TOKEN_BASE_AND_IS_FLOW[token].is_flow


def token_source_currency(token: Unit | CurrencyUnitToken | None) -> str | None:
    """The concrete currency a declaration token pins down, if any (GEP 10).

    ``None`` for the agnostic tokens, for non-currency tokens, and for
    dimensionless declarations.
    """
    return token.currency if isinstance(token, CurrencyUnitToken) else None


def unit_for_derived_node(
    token: Unit | CurrencyUnitToken | UnsetUnitType,
) -> Unit | UnsetUnitType:
    """The unit token a node *derived* from a source with this token carries.

    Derived nodes — time-conversion variants and aggregations — are functions,
    and a function is currency-agnostic by design (GEP 10): it computes on
    values already converted to the run currency. So a source that pins down a
    concrete currency (a parameter) hands on the agnostic counterpart of its
    currency token; every other token — a time, an area, dimensionless, an
    already-agnostic currency, or no declaration — passes through unchanged.
    """
    return token.agnostic if isinstance(token, CurrencyUnitToken) else token


def coerce_unit_token(
    value: str | Unit | CurrencyUnitToken,
    *,
    where: str,
) -> Unit | CurrencyUnitToken:
    """Coerce a YAML ``unit:`` value to a vocabulary token (GEP 10).

    A dimensionless quantity (a share, a rate, a head count) declares
    :attr:`Unit.DIMENSIONLESS` (``unit: DIMENSIONLESS``) like any other token;
    ``None`` is no longer a unit declaration (GEP 10) and reaching here with it
    is an internal bug. Any string must spell a member of the core enumeration
    or a currency token derived from a registered currency exactly; everything
    else — including pint syntax like ``"CURRENCY"`` or ``"CURRENCY / year"``,
    and the former ``"null"`` spelling — is rejected.

    Args:
        value: The raw declaration — a string from YAML or an already-coerced
            token.
        where: Identifier for error messages (e.g. the parameter's name).

    Raises:
        UnitDefinitionError: If the value is not part of the vocabulary.
    """
    if isinstance(value, Unit | CurrencyUnitToken):
        return value
    if value in _CURRENCY_UNIT_TOKENS:
        return _CURRENCY_UNIT_TOKENS[value]
    try:
        return Unit(value)
    except ValueError:
        raise UnitDefinitionError(
            f"{where}: invalid unit token {value!r}. A unit declaration must be "
            f"one of {', '.join([*Unit, *_CURRENCY_UNIT_TOKENS])} (GEP 10); use "
            f"DIMENSIONLESS for a dimensionless quantity."
        ) from None


class UnsetUnitType(enum.Enum):
    """The type of :data:`UNSET_UNIT`; a single-member enum so it type-checks."""

    TOKEN = enum.auto()

    def __repr__(self) -> str:
        return "UNSET_UNIT"


#: Sentinel distinguishing an *omitted* unit declaration from an explicit
#: dimensionless one (GEP 10): :attr:`Unit.DIMENSIONLESS` declares a
#: dimensionless quantity (a share, a rate, a head count); :data:`UNSET_UNIT`
#: means no declaration was made, which the mandatory-units check reports as an
#: error.
UNSET_UNIT = UnsetUnitType.TOKEN


def _build_registry() -> pint.UnitRegistry:
    """Create the module-level registry with the units TTSIM knows about.

    pint's defaults already provide the ``[time]`` units (``year``, ``month``,
    ``week``, ``day`` — with the per-year factors GETTSIM uses: 12, 365.25/7,
    365.25) and ``[length]``/``[area]`` units (``meter``, ``hectare``). We add:

    - ``CURRENCY`` as the reference unit of a new ``[currency]`` dimension;
    - ``quarter_year`` for the ``_q`` suffix (pint's built-in ``quarter`` is a
      unit of mass);
    - ``calendar_year`` / ``calendar_month`` / ``calendar_day`` as affine
      *point* units (GEP 10): a specific year/month/day on the calendar, as
      opposed to a *duration*. pint models a point as an offset unit, whose
      offset must be **non-zero** or pint silently treats it as a plain
      (multiplicative) unit and loses the affine algebra that forbids
      ``point + point``. The epoch is otherwise irrelevant — the dry-run only
      ever uses magnitude ``1.0`` and the runtime path is bare arrays — so we
      pick the 1900-01-01 epoch, aligned across the three axes. Subtracting two
      points yields pint's companion ``delta_calendar_*`` *duration* unit, which
      :attr:`Unit.YEARS` / :attr:`Unit.MONTHS` / :attr:`Unit.DAYS` resolve to
      (each is ratio 1 against ``year`` / ``month`` / ``day``, so existing
      duration declarations are unchanged).

    pint's remaining built-ins parse, but :func:`parse_unit` rejects every
    token outside :data:`_ALLOWED_UNIT_TOKENS`, so they cannot appear in a
    declaration.
    """
    ureg = pint.UnitRegistry()
    ureg.define(f"{CURRENCY_TOKEN} = [currency]")
    ureg.define("quarter_year = year / 4 = quarter_of_year")
    ureg.define("calendar_year = year; offset: 1900")
    ureg.define("calendar_month = month; offset: 22800")  # 1900 * 12
    ureg.define("calendar_day = day; offset: 693975")  # 1900 * 365.25
    return ureg


#: The single module-level registry. Downstream packages mutate it by calling
#: :func:`register_currency`; nothing else should call ``UNIT_REGISTRY.define``.
UNIT_REGISTRY = _build_registry()

#: Reference dimensionalities, computed once against the built registry. The
#: ``[currency]`` dimension and ``[time]`` (named via ``year``) are process
#: constants; the boundary helpers compare a unit's components against them to
#: pick out its currency / flow-period part.
_CURRENCY_DIMENSIONALITY = UNIT_REGISTRY.Quantity(1.0, CURRENCY_TOKEN).dimensionality
_TIME_DIMENSIONALITY = UNIT_REGISTRY.Quantity(1.0, "year").dimensionality

#: The unit tokens a declaration may combine (GEP 10): TTSIM rejects any unit
#: it does not know about. :func:`register_currency` adds each registered
#: concrete currency. ``meter`` is admitted for areas (``meter ** 2``).
_ALLOWED_UNIT_TOKENS: set[str] = {
    CURRENCY_TOKEN,
    "year",
    "quarter_year",
    "month",
    "week",
    "day",
    "hour",
    "meter",
    "hectare",
    # Calendar-point (affine) units and their companion durations (GEP 10).
    "calendar_year",
    "calendar_month",
    "calendar_day",
    "delta_calendar_year",
    "delta_calendar_month",
    "delta_calendar_day",
}

#: The name of the registered base currency, set by ``register_currency(...,
#: base=True)``. ``None`` until a downstream package registers one.
_base_currency: str | None = None

#: Tolerance for the magnitude part of a unit-equivalence comparison.
_REL_TOL = 1e-9


def base_currency() -> str | None:
    """Return the name of the registered base currency, or ``None``."""
    return _base_currency


def register_currency(
    name: str,
    *,
    base: bool = False,
    definition: str | None = None,
) -> None:
    """Register a concrete currency in the ``[currency]`` dimension.

    Downstream packages call this on import. Exactly one currency per process
    is the *base* currency (factor 1 against the abstract :data:`CURRENCY_TOKEN`
    reference); every other currency is defined relative to an already-known
    currency.

    Registration also derives the currency's *declaration tokens*
    (:class:`CurrencyUnitToken`): ``register_currency("DM", ...)`` makes
    ``DM``, ``DM_FLOW``, and ``DM_PER_SQUARE_METER_FLOW`` part of the
    unit vocabulary, so parameters can pin down the concrete currency their
    numbers are written in.

    Args:
        name: The currency's unit name (e.g. ``"euro"``, ``"DM"``).
        base: Whether this is the base currency. Mutually exclusive with
            ``definition``.
        definition: A pint-parseable definition relative to another currency
            (e.g. ``"euro / 1.95583"``). Mutually exclusive with ``base``.

    Raises:
        UnitDefinitionError: If the arguments are inconsistent, if a second base
            currency is registered, if the definition does not resolve to the
            ``[currency]`` dimension, or if a derived declaration token would
            collide with the core vocabulary.
    """
    global _base_currency  # noqa: PLW0603

    if base == (definition is not None):
        raise UnitDefinitionError(
            "register_currency requires exactly one of `base=True` or "
            f"`definition=...`; got base={base!r}, definition={definition!r}."
        )
    _fail_if_currency_token_collides(name)
    if base and _base_currency not in (None, name):
        raise UnitDefinitionError(
            f"Cannot register {name!r} as the base currency: "
            f"{_base_currency!r} is already the base currency."
        )

    currency_dim = UNIT_REGISTRY.Quantity(1.0, CURRENCY_TOKEN).dimensionality
    if name in UNIT_REGISTRY:
        # Idempotent re-registration (e.g. a re-imported module). Tolerate it
        # only if the existing definition is consistent with this call — same
        # dimension *and* same conversion factor against CURRENCY. Checking the
        # dimension alone would silently keep the old factor: re-registering
        # `castar / 4` as `castar / 5`, or a derived currency as `base=True`,
        # would be swallowed and invalidate every later conversion (GEP 10).
        existing_dim = UNIT_REGISTRY.Quantity(1.0, name).dimensionality
        if existing_dim != currency_dim:
            raise UnitDefinitionError(
                f"Cannot register currency {name!r}: a non-currency unit of "
                f"that name already exists ({existing_dim})."
            )
        existing_factor = UNIT_REGISTRY.Quantity(1.0, name).to(CURRENCY_TOKEN).magnitude
        if definition is not None:
            requested_factor = (
                UNIT_REGISTRY.parse_expression(definition).to(CURRENCY_TOKEN).magnitude
            )
        else:  # base currency: factor 1 against CURRENCY by definition.
            requested_factor = 1.0
        if not math.isclose(existing_factor, requested_factor, rel_tol=_REL_TOL):
            requested_desc = "base (factor 1)" if base else f"{definition!r}"
            raise UnitDefinitionError(
                f"Cannot re-register currency {name!r}: it already converts to "
                f"{existing_factor} {CURRENCY_TOKEN}, but this call ({requested_desc}) "
                f"requests {requested_factor}. A currency's factor against "
                f"{CURRENCY_TOKEN} must be consistent across registrations (GEP 10)."
            )
    else:
        UNIT_REGISTRY.define(
            f"{name} = {CURRENCY_TOKEN}" if base else f"{name} = {definition}"
        )
        if UNIT_REGISTRY.Quantity(1.0, name).dimensionality != currency_dim:
            raise UnitDefinitionError(
                f"Currency {name!r} defined as {definition!r} does not resolve "
                f"to the [currency] dimension."
            )

    if base:
        _base_currency = name
    _ALLOWED_UNIT_TOKENS.add(name)
    _register_currency_unit_tokens(name)


def _currency_token_spelling(name: str, agnostic: Unit) -> str:
    """The spelling of a currency's concrete variant of an agnostic token."""
    return f"{name.upper()}{str(agnostic).removeprefix(CURRENCY_TOKEN)}"


def _fail_if_currency_token_collides(name: str) -> None:
    """Reject a currency whose declaration tokens would collide (GEP 10)."""
    for agnostic in _AGNOSTIC_CURRENCY_TOKENS:
        spelling = _currency_token_spelling(name=name, agnostic=agnostic)
        if spelling in Unit.__members__:
            raise UnitDefinitionError(
                f"Cannot register currency {name!r}: its declaration token "
                f"{spelling!r} collides with the core unit vocabulary."
            )


def _register_currency_unit_tokens(name: str) -> None:
    """Derive a registered currency's declaration tokens (GEP 10).

    One concrete variant per currency-dimensioned core token, spelled by
    replacing the agnostic ``CURRENCY`` prefix with the upper-cased currency
    name: ``"SILVER_PENNY"`` yields ``SILVER_PENNY``,
    ``SILVER_PENNY_FLOW``, and ``SILVER_PENNY_PER_SQUARE_METER_FLOW``.
    Idempotent, mirroring :func:`register_currency`.
    """
    for agnostic in _AGNOSTIC_CURRENCY_TOKENS:
        spelling = _currency_token_spelling(name=name, agnostic=agnostic)
        if spelling not in _CURRENCY_UNIT_TOKENS:
            _CURRENCY_UNIT_TOKENS[spelling] = CurrencyUnitToken(
                spelling, currency=name, agnostic=agnostic
            )


def parse_unit(unit_str: str) -> pint.Unit:
    """Parse a pint unit string, enforcing the closed pint-token vocabulary.

    Internal (GEP 10): declarations are :class:`Unit` members, never pint
    syntax. This parser serves the *internal* pint surfaces — Layer-2 input
    tags, the framework date nodes, and the resolution machinery.

    Args:
        unit_str: A pint-parseable unit string combining only pint tokens
            TTSIM knows about. May use the :data:`CURRENCY_TOKEN` to denote
            the ``[currency]`` dimension (e.g. ``"CURRENCY"``,
            ``"CURRENCY / meter ** 2"``).

    Returns:
        The parsed pint unit.

    Raises:
        UnitDefinitionError: If the string cannot be parsed, involves a unit
            token TTSIM does not know about, or resolves to the dimensionless
            unit (declare ``DIMENSIONLESS`` instead).
    """
    if not isinstance(unit_str, str):
        raise UnitDefinitionError(
            f"A unit must be given as a string, got {unit_str!r}."
        )
    try:
        unit = UNIT_REGISTRY.parse_units(unit_str)
    except (pint.errors.PintError, AssertionError, ValueError, TypeError) as e:
        raise UnitDefinitionError(f"Could not parse unit {unit_str!r}: {e}") from e
    _fail_if_unit_tokens_are_unknown(unit=unit, unit_str=unit_str)
    if not to_units_container(unit):
        raise UnitDefinitionError(
            f"Unit {unit_str!r} resolves to the dimensionless unit. A "
            f"dimensionless quantity (a share, a rate, a head count) declares "
            f"`DIMENSIONLESS` (GEP 10)."
        )
    return unit


def _fail_if_unit_tokens_are_unknown(
    unit: pint.Unit,
    unit_str: str,
) -> None:
    """Reject any unit token TTSIM does not know about (GEP 10).

    The check operates on the *tokens* of the parsed unit, not its
    dimensionality, so pint built-ins that happen to live in an admissible
    dimension (``count``, ``percent``, prefixed units like ``kilometer``)
    are rejected too: there is exactly one way to write every unit.
    """
    offending = sorted(
        token for token in to_units_container(unit) if token not in _ALLOWED_UNIT_TOKENS
    )
    if offending:
        raise UnitDefinitionError(
            f"Unit {unit_str!r} involves unit token(s) TTSIM does not know "
            f"about: {', '.join(offending)}. Known units are "
            f"{', '.join(sorted(_ALLOWED_UNIT_TOKENS))} (GEP 10)."
        )


def _divide_by_period(non_time_unit: pint.Unit, period_pint_name: str) -> pint.Unit:
    """Return ``non_time_unit / period`` as a pint unit."""
    period = UNIT_REGISTRY.Quantity(1.0, period_pint_name)
    return (UNIT_REGISTRY.Quantity(1.0, non_time_unit) / period).units


def _token_base_unit(token: Unit | CurrencyUnitToken) -> pint.Unit:
    """The pint unit of a token's non-period part.

    A currency token resolves to the agnostic :data:`CURRENCY_TOKEN` unit:
    for dimensionality checks a registered currency means exactly what its
    agnostic counterpart means (the union semantics of GEP 10) — the concrete
    currency only drives the build-time conversion of the numbers.
    """
    if isinstance(token, CurrencyUnitToken):
        token = token.agnostic
    base = _TOKEN_BASE_AND_IS_FLOW[token].base
    return (
        UNIT_REGISTRY.dimensionless if base is None else UNIT_REGISTRY.parse_units(base)
    )


def resolve_column_unit(
    token: Unit | CurrencyUnitToken,
    time_unit_id: str | None,
) -> pint.Unit:
    """Resolve a column/function's full unit from its ``unit=`` token and suffix.

    The suffix ⟺ flow rule is checked in both directions (GEP 10): a time
    suffix (``_y``/``_q``/``_m``/``_w``/``_d``) requires a ``…_FLOW`` token
    and supplies its period, so ``betrag_m`` declared
    ``unit=Unit.CURRENCY_FLOW`` resolves to ``CURRENCY / month`` and the
    auto-generated ``betrag_y`` to ``CURRENCY / year``. A complete token
    (``CURRENCY``, ``YEARS``, ``DIMENSIONLESS``, …) forbids a suffix; a
    dimensionless quantity therefore declares :attr:`Unit.DIMENSIONLESS` and a
    dimensionless flow :attr:`Unit.DIMENSIONLESS_FLOW`.

    Columns and functions are currency-agnostic by design (GEP 10), so a
    concrete currency token (``SILVER_PENNY_FLOW``, ``DM``, …) is
    rejected here — only parameters pin down the currency their numbers are
    written in.

    Args:
        token: The ``unit=`` declaration — a :class:`Unit` member
            (:attr:`Unit.DIMENSIONLESS` for a share, a head count).
        time_unit_id: The GEP-1 time-unit suffix id, or ``None`` for a node
            without one.

    Returns:
        The resolved pint unit.

    Raises:
        UnitDefinitionError: If the suffix ⟺ flow rule is violated,
            ``time_unit_id`` is not a recognised suffix id, or the token
            pins down a concrete currency.
    """
    if isinstance(token, CurrencyUnitToken):
        raise UnitDefinitionError(
            f"Unit token {token} pins down a concrete currency, which only "
            f"parameters may do. Columns and functions are currency-agnostic "
            f"and declare {Unit.CURRENCY_FLOW} / {Unit.CURRENCY} "
            f"(GEP 10)."
        )
    return _resolve_token_via_suffix(token=token, time_unit_id=time_unit_id)


def _resolve_token_via_suffix(
    token: Unit | CurrencyUnitToken,
    time_unit_id: str | None,
) -> pint.Unit:
    """Resolve a token whose period (if any) comes from a name suffix (GEP 10).

    Shared by columns/functions (:func:`resolve_column_unit`) and scalar
    parameters (:func:`resolve_scalar_param_unit`): a ``…_FLOW`` token requires
    a time suffix on the name and is divided by its period; a complete token
    forbids one.
    """
    if time_unit_id is not None and time_unit_id not in TIME_UNIT_ID_TO_PINT_NAME:
        raise UnitDefinitionError(
            f"Unknown time-unit suffix id {time_unit_id!r}; expected one of "
            f"{', '.join(TIME_UNIT_ID_TO_PINT_NAME)}."
        )
    if unit_token_is_flow(token):
        if time_unit_id is None:
            raise UnitDefinitionError(
                f"Unit token {token} denotes a flow and requires a time-unit "
                f"suffix (_y/_q/_m/_w/_d) on the name to supply its period "
                f"(GEP 10)."
            )
        return _divide_by_period(
            non_time_unit=_token_base_unit(token),
            period_pint_name=TIME_UNIT_ID_TO_PINT_NAME[time_unit_id],
        )
    if time_unit_id is not None:
        raise UnitDefinitionError(
            f"Unit token {token} is complete as written, but the name carries "
            f"a time-unit suffix (_{time_unit_id}). A suffixed name denotes a "
            f"flow and requires a `…_FLOW` token (GEP 10)."
        )
    return _token_base_unit(token)


def resolve_scalar_param_unit(
    token: Unit | CurrencyUnitToken,
    time_unit_id: str | None,
) -> pint.Unit:
    """Resolve a scalar parameter's unit from its token and name suffix (GEP 10).

    A scalar parameter takes its period from a time suffix on its *name*, just
    like a column (GEP 1): ``lump_sum_deduction_y`` resolves a
    :attr:`Unit.CURRENCY_FLOW` declaration to ``CURRENCY / year``.
    ``reference_period`` plays no part — it is reserved for the period sources
    with no name to suffix (integer-keyed dict leaves, mapping parameter
    axes), and the caller rejects a scalar parameter that sets one. Unlike
    :func:`resolve_column_unit`, a concrete currency token is allowed:
    parameters pin down the currency their numbers are written in.
    """
    return _resolve_token_via_suffix(token=token, time_unit_id=time_unit_id)


def resolve_param_unit(
    token: Unit | CurrencyUnitToken,
    reference_period: str | None,
) -> pint.Unit:
    """Resolve a unit from its ``unit:`` token and a ``reference_period``.

    Used for the period sources that have no name to carry a suffix (GEP 10):
    integer-keyed dict leaves, uniformly-typed dict parameters, raw parameters,
    and mapping parameter axes. Scalar parameters take their period from a
    name suffix instead — see :func:`resolve_scalar_param_unit`.

    ``reference_period`` is *functional*: it is required by a ``…_FLOW`` token,
    whose period it supplies, and forbidden otherwise. In particular
    ``DIMENSIONLESS`` with a non-null ``reference_period`` is an error:
    ``DIMENSIONLESS`` is complete as written, and the per-period dimensionless
    quantity is :attr:`Unit.DIMENSIONLESS_FLOW`.

    Parameters may pin down a concrete currency (``SILVER_PENNY``,
    ``DM_FLOW``, …); such a token resolves exactly like its agnostic
    counterpart — the concrete currency drives the build-time conversion,
    not the dimensionality check.

    Args:
        token: The ``unit:`` declaration — a :class:`Unit` member
            (:attr:`Unit.DIMENSIONLESS` for a dimensionless parameter) or a
            :class:`CurrencyUnitToken`.
        reference_period: The ``reference_period`` label (``"Year"``,
            ``"Month"``, …), or ``None``.

    Returns:
        The resolved pint unit.

    Raises:
        UnitDefinitionError: If the flow ⟺ ``reference_period`` rule is
            violated or ``reference_period`` is not a recognised label.
    """
    if (
        reference_period is not None
        and reference_period not in REFERENCE_PERIOD_TO_PINT_NAME
    ):
        raise UnitDefinitionError(
            f"Unknown reference_period {reference_period!r}; expected one of "
            f"{', '.join(REFERENCE_PERIOD_TO_PINT_NAME)}."
        )
    if unit_token_is_flow(token):
        if reference_period is None:
            raise UnitDefinitionError(
                f"Unit token {token} denotes a flow and requires a non-null "
                f"`reference_period` to supply its period (GEP 10)."
            )
        return _divide_by_period(
            non_time_unit=_token_base_unit(token),
            period_pint_name=REFERENCE_PERIOD_TO_PINT_NAME[reference_period],
        )
    if reference_period is not None:
        raise UnitDefinitionError(
            f"Unit token {token} is complete as written and cannot be "
            f"combined with `reference_period: {reference_period}` (GEP 10)."
        )
    return _token_base_unit(token)


def _function_name(function: Callable[..., Any]) -> str:
    """A human-readable function name for error messages."""
    return getattr(function, "__qualname__", getattr(function, "__name__", "?"))


def units_are_equivalent(left: pint.Unit, right: pint.Unit) -> bool:
    """Whether two units are interchangeable on a DAG edge.

    Two units are equivalent iff they share a dimensionality *and* a magnitude
    (i.e. their ratio is dimensionless and equal to 1). This is stricter than
    pint's compatibility: ``euro / month`` and ``euro / year`` are both
    ``[currency] / [time]`` but are *not* equivalent (ratio 12), so a monthly
    node feeding a yearly consumer is caught.

    The base currency is defined as factor 1 against the :data:`CURRENCY_TOKEN`
    reference, so a ``"CURRENCY"`` declaration is equivalent to a value inferred
    in the base currency. Cross-currency magnitudes (e.g. ``DM``) are reconciled
    by the currency knob before this check runs (issue #120).

    A calendar-point (affine offset) unit cannot be divided, so equivalence of
    such units is decided by *identity* rather than by magnitude (GEP 10): two
    ``calendar_year`` points are equivalent, but a ``calendar_year`` point is
    *not* equivalent to a ``year`` / ``delta_calendar_year`` duration (the
    distinction S1 introduces) nor to a ``calendar_month`` point on another axis.
    """
    left_quantity = UNIT_REGISTRY.Quantity(1.0, left)
    right_quantity = UNIT_REGISTRY.Quantity(1.0, right)
    if left_quantity.dimensionality != right_quantity.dimensionality:
        return False
    try:
        ratio = (left_quantity / right_quantity).to_reduced_units()
    except pint.OffsetUnitCalculusError:
        return left == right
    return math.isclose(ratio.magnitude, 1.0, rel_tol=_REL_TOL)


def is_calendar_point_unit(unit: pint.Unit) -> bool:
    """Whether a resolved unit is an affine calendar *point* (GEP 10).

    A calendar point (``calendar_year`` and its month/day siblings) is a pint
    offset unit: it obeys affine algebra, not the magnitude algebra of a
    duration. Two points *subtract* to a duration and a duration *shifts* a
    point, but two points cannot be added, a point cannot be scaled, and points
    on different calendar axes cannot be combined — pint raises an
    :class:`pint.OffsetUnitCalculusError` on any of these.

    Callers that implement the affine ``+``/``-`` rules (the build-time dry-run)
    detect a point this way and delegate the operation to pint rather than to the
    magnitude-equivalence check, which would wrongly reject the valid
    ``point + duration``. Detection is by the very property that defines an
    offset unit: it cannot be divided by itself.
    """
    quantity = UNIT_REGISTRY.Quantity(1.0, unit)
    try:
        quantity / quantity
    except pint.OffsetUnitCalculusError:
        return True
    return False


def _as_unit(unit: str | pint.Unit) -> pint.Unit:
    """Coerce a declared unit (string or already-resolved pint unit) to a unit."""
    return parse_unit(unit) if isinstance(unit, str) else unit


def currency_conversion_factor(source_currency: str, run_currency: str) -> float:
    """Build-time factor converting a value from ``source_currency`` to the run one.

    Used to bake historical parameters denominated in their legal currency (e.g.
    DM) into the run currency at environment-build time (GEP 10). pint is the
    single source of truth for the rate. Both currencies must be registered.

    Raises:
        UnitDefinitionError: If either currency is unknown or not a currency.
    """
    for name in (source_currency, run_currency):
        if name not in UNIT_REGISTRY:
            raise UnitDefinitionError(
                f"Cannot convert currency: {name!r} is not a registered currency."
            )
    try:
        return UNIT_REGISTRY.Quantity(1.0, source_currency).to(run_currency).magnitude
    except pint.DimensionalityError as e:
        raise UnitDefinitionError(
            f"Cannot convert {source_currency!r} to {run_currency!r}: {e}"
        ) from e


def _currency_component_of(units: pint.Unit) -> pint.Unit | None:
    """Return the currency component of a (possibly composite) unit, or ``None``.

    Used at the Layer-2 input boundary to convert a pint-tagged column's
    currency to the run currency while leaving its period and area untouched:
    e.g. the ``DM`` in ``DM / month``.
    """
    for token in to_units_container(units):
        candidate = UNIT_REGISTRY.parse_units(token)
        if (
            UNIT_REGISTRY.Quantity(1.0, candidate).dimensionality
            == _CURRENCY_DIMENSIONALITY
        ):
            return candidate
    return None


def _flow_period_of(units: pint.Unit) -> pint.Unit | None:
    """Return a unit's flow period — its time component in the *denominator*.

    The ``month`` of ``CURRENCY / month``, the ``week`` of ``hour / week``. A
    *numerator* time unit (the ``year`` of an age, ``Unit.YEARS``) is not a flow
    period and is ignored, so an intrinsically-temporal column is not mistaken
    for a flow. Returns ``None`` for a unit with no per-period part.
    """
    for token, exponent in to_units_container(units).items():
        if isinstance(exponent, complex):  # pint exponents are real; narrow for ty
            continue
        candidate = UNIT_REGISTRY.parse_units(token)
        if (
            exponent < 0
            and UNIT_REGISTRY.Quantity(1.0, candidate).dimensionality
            == _TIME_DIMENSIONALITY
        ):
            return candidate
    return None


def unit_residual_excluding_currency_and_flow_period(units: pint.Unit) -> pint.Unit:
    """A unit with its currency component and flow period divided out (GEP 10).

    The two axes the Layer-2 input boundary does *not* require to match exactly:
    the currency is converted to the run currency at the boundary, and the flow
    period is screened against the column's name suffix. What remains — the
    numerator scale (area, intrinsic time, plain counts) — must match the
    declared unit *exactly*, so the input check compares the residuals of a tag
    and its declared unit for equivalence rather than mere dimensionality
    (a ``HECTARES`` column tagged ``m²`` shares the area dimension but is a
    10,000-fold level error).
    """
    currency = _currency_component_of(units)
    residual = units / currency if currency is not None else units
    period = _flow_period_of(residual)
    return residual * period if period is not None else residual


def _suffix_period_of(column_label: str | None) -> pint.Unit | None:
    """Return the flow period named by a column's GEP-1 time suffix.

    ``…_m`` → ``month``; a name with no time suffix → ``None``.
    """
    if column_label is None:
        return None
    match = _QNAME_TIME_SUFFIX_PATTERN.search(column_label)
    if match is None:
        return None
    return UNIT_REGISTRY.parse_units(
        TIME_UNIT_ID_TO_PINT_NAME[match.group("time_unit")]
    )


def _fail_if_tag_period_disagrees_with_suffix(
    units: pint.Unit, *, column_label: str | None
) -> None:
    """Strict period guard (GEP 10): a pint tag's flow period must match the
    column's GEP-1 time suffix exactly — including both absent.

    A ``_m`` column needs a ``/month`` tag; an unsuffixed column needs a tag with
    no period. This catches a contradictory period that would otherwise be
    stripped silently (e.g. a ``_m`` column tagged ``DM / year`` — a 12-fold error).
    """
    tag_period = _flow_period_of(units)
    suffix_period = _suffix_period_of(column_label)
    matches = (
        tag_period is None
        if suffix_period is None
        else tag_period is not None
        and units_are_equivalent(left=tag_period, right=suffix_period)
    )
    if matches:
        return
    where = f" on input column {column_label!r}" if column_label else ""
    tag_desc = f"the flow period '{tag_period}'" if tag_period else "no flow period"
    suffix_desc = f"'{suffix_period}'" if suffix_period else "none (no time suffix)"
    raise UnitConsistencyError(
        f"pint-tagged input{where} has {tag_desc}, but the column's time suffix "
        f"implies {suffix_desc}. A tag's period must match the column's suffix "
        f"exactly — a `_m` column needs a `/month` tag, an unsuffixed column a "
        f"tag with no period."
    )


def strip_input_quantity_at_boundary(
    quantity: Any,  # noqa: ANN401 (a pint Quantity wrapping an input column)
    *,
    run_currency: str | None,
    column_label: str | None = None,
) -> Any:  # noqa: ANN401
    """Convert a pint-tagged input column to the run currency, then strip it (GEP 10).

    A user *may* attach a pint ``Quantity`` to an input column. The tag may only
    combine units TTSIM knows about, and its flow period must match the column's
    GEP-1 time suffix exactly (a ``_m`` column needs a ``/month`` tag; an
    unsuffixed column needs a tag with no period). Its currency component is then
    *converted* to the run currency — so a DM-tagged column can feed a euro run,
    rescaled at the boundary — while its period and area are left untouched. A tag
    already in the run currency, or with no currency component, is stripped
    unchanged. The bare magnitude is returned for the numeric runtime path.

    The period check is the only mismatch the boundary can catch on its own: the
    column's *declared* unit (its dimension, the numerator) is not threaded here,
    so a wrong-dimension tag (a currency on an age column) is not caught — that is
    the deferred full-validation path.

    Raises:
        UnitDefinitionError: If the tag involves a unit token TTSIM does not
            know about.
        UnitConsistencyError: If the tag's flow period disagrees with the
            column's time suffix.
    """
    try:
        _fail_if_unit_tokens_are_unknown(
            unit=quantity.units, unit_str=str(quantity.units)
        )
    except UnitDefinitionError as e:
        where = f" on input column {column_label!r}" if column_label else ""
        raise UnitDefinitionError(f"pint-tagged input{where}: {e}") from e
    _fail_if_tag_period_disagrees_with_suffix(quantity.units, column_label=column_label)
    if run_currency is None:
        return quantity.magnitude
    source_currency = _currency_component_of(quantity.units)
    if source_currency is None:
        return quantity.magnitude
    run_unit = UNIT_REGISTRY.parse_units(run_currency)
    if source_currency == run_unit:
        return quantity.magnitude
    target = quantity.units / source_currency * run_unit
    return quantity.to(target).magnitude


def infer_function_unit(
    function: Callable[..., Any],
    input_units: Mapping[str, str | pint.Unit],
    *,
    non_unit_kwargs: Mapping[str, Any] | None = None,
) -> pint.Unit:
    """Infer the output unit of a scalar function body via a pint dry-run.

    Each declared input is wrapped in a representative ``Quantity(1.0, unit)``
    and the scalar body is executed in NumPy+pint. pint propagates units
    through the arithmetic and raises on a dimensionally invalid operation
    (e.g. adding a currency to a currency-per-area). The output unit is read
    off the result.

    This never runs on user data values and never under ``jit``.

    Args:
        function: The scalar function body to dry-run.
        input_units: Maps each unit-carrying parameter name to its declared
            unit string.
        non_unit_kwargs: Extra keyword arguments passed through verbatim (e.g.
            ``xnp`` for functions that take the array module). These are *not*
            wrapped in quantities.

    Returns:
        The inferred output unit. A function returning a bare (dimensionless)
        Python number yields the dimensionless unit.

    Raises:
        UnitInferenceError: If the body performs a dimensionally invalid
            operation or otherwise fails to dry-run.
    """
    func_name = _function_name(function)
    quantities: dict[str, Any] = {
        name: UNIT_REGISTRY.Quantity(1.0, _as_unit(unit))
        for name, unit in input_units.items()
    }
    if non_unit_kwargs:
        quantities.update(non_unit_kwargs)
    try:
        result = function(**quantities)
    except pint.DimensionalityError as e:
        raise UnitInferenceError(
            f"Dimensionally invalid operation while inferring the unit of "
            f"{func_name!r}: {e}"
        ) from e
    except Exception as e:
        raise UnitInferenceError(
            f"Could not dry-run {func_name!r} to infer its unit: "
            f"{type(e).__name__}: {e}"
        ) from e
    if isinstance(result, pint.Quantity):
        return result.units
    # A bare number carries no dimension.
    return UNIT_REGISTRY.dimensionless


def unit_for_aggregation(
    source_unit: Unit | UnsetUnitType,
    agg_type: AggType,
) -> Unit | UnsetUnitType:
    """Auto-assign the unit token of an aggregation node (GEP 10, #119).

    Parallels how GEP 4 resolves an aggregation's *type* from the source and the
    aggregation rule:

    - ``SUM`` / ``MEAN`` / ``MIN`` / ``MAX`` preserve the source token (a sum
      or average of currency flows is still a currency flow);
    - ``COUNT`` is a head count — a dimensionless integer, i.e.
      :attr:`Unit.DIMENSIONLESS` regardless of source;
    - ``ANY`` / ``ALL`` yield a boolean, which is a dimensionless quantity
      (GEP 10), i.e. :attr:`Unit.DIMENSIONLESS`.

    Args:
        source_unit: The source column's ``unit`` token — a :class:`Unit`
            member (:attr:`Unit.DIMENSIONLESS` for a dimensionless source) or
            :data:`UNSET_UNIT` if the source does not declare a unit.
        agg_type: The :class:`ttsim.tt.aggregation.AggType` of the aggregation.

    Returns:
        The auto-assigned unit token. :attr:`Unit.DIMENSIONLESS` for a
        ``COUNT`` head count or a boolean ``ANY`` / ``ALL`` result; otherwise
        the preserved source token (:data:`UNSET_UNIT` when ``SUM`` /
        ``MEAN`` / … preserve a source that itself lacks a declaration, which
        the mandatory-units check then reports against the source).
    """
    if agg_type in (AggType.COUNT, AggType.ANY, AggType.ALL):
        return Unit.DIMENSIONLESS
    # SUM, MEAN, MIN, MAX preserve the source unit.
    return source_unit


def fail_if_units_are_missing(
    units_by_qname: Mapping[str, Unit | CurrencyUnitToken | UnsetUnitType],
) -> None:
    """Data-independent check that every node declares a unit (GEP 10).

    Mandatory units parallel GEP 9's return-type enforcement: a missing unit is
    a definition error. :attr:`Unit.DIMENSIONLESS` is *not* missing — it
    declares a dimensionless quantity (GEP 10); a node without any declaration
    maps to :data:`UNSET_UNIT`. This is the leaf check that
    :func:`ttsim.interface_dag_elements.unit_checks.fail_if_environment_units_are_missing`
    runs over the whole assembled environment (wired in as a ``fail_if``).

    Raises:
        UnitDefinitionError: If any qualified name maps to :data:`UNSET_UNIT`.
    """
    missing = sorted(
        qname for qname, unit in units_by_qname.items() if unit is UNSET_UNIT
    )
    if missing:
        raise UnitDefinitionError(
            "The following nodes are missing a mandatory `unit=` declaration "
            f"(GEP 10; declare `unit=Unit.DIMENSIONLESS` / `unit: DIMENSIONLESS` "
            f"for a dimensionless quantity): {', '.join(missing)}."
        )
