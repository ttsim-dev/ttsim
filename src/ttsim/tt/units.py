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
(``DM_STOCK``, ``DM_FLOW``, …). The agnostic currency tokens
(:attr:`Unit.CURRENCY_STOCK` / :attr:`Unit.CURRENCY_FLOW`) denote the
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
"""

from __future__ import annotations

import enum
import math
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple, Self

import pint
from pint.util import to_units_container

from ttsim.exceptions import (
    UnitConsistencyError,
    UnitDefinitionError,
    UnitInferenceError,
)

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
#: resolve the currency tokens (``CURRENCY_FLOW``, ``CURRENCY_STOCK``, …)
#: before any concrete currency is registered. Checks compare at the
#: dimensionality level; the concrete currency is resolved separately.
CURRENCY_TOKEN = "CURRENCY"  # noqa: S105 (a unit token, not a secret)


class Unit(enum.StrEnum):
    """The core vocabulary of unit tokens (GEP 10).

    One token = one meaning, independent of any other field. A bare token is
    *complete* as written; a ``…_FLOW`` token *needs a period*, supplied by
    the name suffix (columns/functions) or ``reference_period`` (parameters).
    Where both kinds of a quantity exist, both are marked
    (:attr:`CURRENCY_STOCK` / :attr:`CURRENCY_FLOW`) — a bare ``CURRENCY``
    is deliberately unwritable, so no token can be misread as complete when
    it is not.

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

    CURRENCY_STOCK = "CURRENCY_STOCK"
    """An amount of currency, full stop: wealth, asset thresholds."""

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
    """A quantity measured in years: ages, age thresholds, calendar years."""

    HOURS_FLOW = "HOURS_FLOW"
    """Hours per period: working hours."""

    SQUARE_METERS = "SQUARE_METERS"
    """An area in square meters: dwelling size."""

    HECTARES = "HECTARES"
    """An area in hectares: land."""

    CURRENCY_PER_SQUARE_METER_FLOW = "CURRENCY_PER_SQUARE_METER_FLOW"
    """An amount of currency per square meter per period: rent caps."""


class _TokenResolution(NamedTuple):
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
    Unit.CURRENCY_STOCK: _TokenResolution(base=CURRENCY_TOKEN, is_flow=False),
    Unit.DIMENSIONLESS: _TokenResolution(base=None, is_flow=False),
    Unit.DIMENSIONLESS_FLOW: _TokenResolution(base=None, is_flow=True),
    Unit.YEARS: _TokenResolution(base="year", is_flow=False),
    Unit.HOURS_FLOW: _TokenResolution(base="hour", is_flow=True),
    Unit.SQUARE_METERS: _TokenResolution(base="meter ** 2", is_flow=False),
    Unit.HECTARES: _TokenResolution(base="hectare", is_flow=False),
    Unit.CURRENCY_PER_SQUARE_METER_FLOW: _TokenResolution(
        base=f"{CURRENCY_TOKEN} / meter ** 2", is_flow=True
    ),
}


class CurrencyUnitToken(str):
    """A declaration token pinning down a concrete currency (GEP 10).

    Created by :func:`register_currency`, never directly:
    ``register_currency("DM", ...)`` derives ``DM_STOCK`` and ``DM_FLOW``.
    For all dimensionality checks the token means exactly what its agnostic
    counterpart (:attr:`Unit.CURRENCY_STOCK` / :attr:`Unit.CURRENCY_FLOW`)
    means — a registered currency *is a* ``CURRENCY``. In addition it names
    the concrete currency a parameter's numbers are denominated in, which the
    build-time conversion to the run currency reads off the declaration.

    Only *parameters* may declare these tokens. Columns and functions take
    :class:`Unit` members and thereby stay currency-agnostic.
    """

    __slots__ = ("currency", "is_flow")

    currency: str
    is_flow: bool

    def __new__(cls, spelling: str, *, currency: str, is_flow: bool) -> Self:
        self = super().__new__(cls, spelling)
        self.currency = currency
        self.is_flow = is_flow
        return self


#: One declaration token per (registered currency, kind): ``DM_STOCK``,
#: ``DM_FLOW``, … — populated by :func:`register_currency`, keyed by spelling.
_CURRENCY_UNIT_TOKENS: dict[str, CurrencyUnitToken] = {}


#: Any member of the full token vocabulary: the core enumeration or a
#: currency token derived from a registered currency.
UnitToken = Unit | CurrencyUnitToken


def unit_token_is_flow(token: Unit | CurrencyUnitToken) -> bool:
    """Whether a token denotes a per-period quantity (needs a period source)."""
    if isinstance(token, CurrencyUnitToken):
        return token.is_flow
    return _TOKEN_BASE_AND_IS_FLOW[token].is_flow


def token_source_currency(token: Unit | CurrencyUnitToken | None) -> str | None:
    """The concrete currency a declaration token pins down, if any (GEP 10).

    ``None`` for the agnostic tokens, for non-currency tokens, and for
    dimensionless declarations.
    """
    return token.currency if isinstance(token, CurrencyUnitToken) else None


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


def _build_registry() -> pint.UnitRegistry:
    """Create the module-level registry with the units TTSIM knows about.

    pint's defaults already provide the ``[time]`` units (``year``, ``month``,
    ``week``, ``day`` — with the per-year factors GETTSIM uses: 12, 365.25/7,
    365.25) and ``[length]``/``[area]`` units (``meter``, ``hectare``). We add:

    - ``CURRENCY`` as the reference unit of a new ``[currency]`` dimension;
    - ``quarter_year`` for the ``_q`` suffix (pint's built-in ``quarter`` is a
      unit of mass).

    pint's remaining built-ins parse, but :func:`parse_unit` rejects every
    token outside :data:`_ALLOWED_UNIT_TOKENS`, so they cannot appear in a
    declaration.
    """
    ureg = pint.UnitRegistry()
    ureg.define(f"{CURRENCY_TOKEN} = [currency]")
    ureg.define("quarter_year = year / 4 = quarter_of_year")
    return ureg


#: The single module-level registry. Downstream packages mutate it by calling
#: :func:`register_currency`; nothing else should call ``UREG.define``.
UREG = _build_registry()

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
    ``DM_STOCK`` and ``DM_FLOW`` part of the unit vocabulary, so parameters
    can pin down the concrete currency their numbers are written in.

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

    currency_dim = UREG.Quantity(1.0, CURRENCY_TOKEN).dimensionality
    if name in UREG:
        # Idempotent re-registration (e.g. a re-imported module). Tolerate it
        # only if the existing definition is consistent with this call.
        existing_dim = UREG.Quantity(1.0, name).dimensionality
        if existing_dim != currency_dim:
            raise UnitDefinitionError(
                f"Cannot register currency {name!r}: a non-currency unit of "
                f"that name already exists ({existing_dim})."
            )
    else:
        UREG.define(f"{name} = {CURRENCY_TOKEN}" if base else f"{name} = {definition}")
        if UREG.Quantity(1.0, name).dimensionality != currency_dim:
            raise UnitDefinitionError(
                f"Currency {name!r} defined as {definition!r} does not resolve "
                f"to the [currency] dimension."
            )

    if base:
        _base_currency = name
    _ALLOWED_UNIT_TOKENS.add(name)
    _register_currency_unit_tokens(name)


def _fail_if_currency_token_collides(name: str) -> None:
    """Reject a currency whose declaration tokens would collide (GEP 10)."""
    for suffix in ("_STOCK", "_FLOW"):
        spelling = f"{name.upper()}{suffix}"
        if spelling in Unit.__members__:
            raise UnitDefinitionError(
                f"Cannot register currency {name!r}: its declaration token "
                f"{spelling!r} collides with the core unit vocabulary."
            )


def _register_currency_unit_tokens(name: str) -> None:
    """Derive a registered currency's declaration tokens (GEP 10).

    ``name`` is upper-cased and suffixed with the stock/flow kind markers:
    ``"silver_penny"`` yields ``SILVER_PENNY_STOCK`` and ``SILVER_PENNY_FLOW``.
    Idempotent, mirroring :func:`register_currency`.
    """
    for suffix, is_flow in (("_STOCK", False), ("_FLOW", True)):
        spelling = f"{name.upper()}{suffix}"
        if spelling not in _CURRENCY_UNIT_TOKENS:
            _CURRENCY_UNIT_TOKENS[spelling] = CurrencyUnitToken(
                spelling, currency=name, is_flow=is_flow
            )


def parse_unit(unit_str: str) -> pint.Unit:
    """Parse a ``unit=`` string into a pint unit, enforcing the closed vocab.

    Args:
        unit_str: A pint-parseable unit string combining only units TTSIM
            knows about. May use the :data:`CURRENCY_TOKEN` to denote the
            ``[currency]`` dimension (e.g. ``"CURRENCY"``,
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
        unit = UREG.parse_units(unit_str)
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
    period = UREG.Quantity(1.0, period_pint_name)
    return (UREG.Quantity(1.0, non_time_unit) / period).units


def _token_base_unit(token: Unit | CurrencyUnitToken) -> pint.Unit:
    """The pint unit of a token's non-period part.

    A currency token resolves to the agnostic :data:`CURRENCY_TOKEN` unit:
    for dimensionality checks a registered currency means exactly what its
    agnostic counterpart means (the union semantics of GEP 10) — the concrete
    currency only drives the build-time conversion of the numbers.
    """
    if isinstance(token, CurrencyUnitToken):
        return UREG.parse_units(CURRENCY_TOKEN)
    base = _TOKEN_BASE_AND_IS_FLOW[token].base
    return UREG.dimensionless if base is None else UREG.parse_units(base)


def resolve_column_unit(
    token: Unit | CurrencyUnitToken | None,
    time_unit_id: str | None,
) -> pint.Unit:
    """Resolve a column/function's full unit from its ``unit=`` token and suffix.

    The suffix ⟺ flow rule is checked in both directions (GEP 10): a time
    suffix (``_y``/``_q``/``_m``/``_w``/``_d``) requires a ``…_FLOW`` token
    and supplies its period, so ``betrag_m`` declared
    ``unit=Unit.CURRENCY_FLOW`` resolves to ``CURRENCY / month`` and the
    auto-generated ``betrag_y`` to ``CURRENCY / year``. A complete token
    (``CURRENCY_STOCK``, ``YEARS``, …) or a dimensionless declaration
    (``None``) forbids a suffix.

    Columns and functions are currency-agnostic by design (GEP 10), so a
    concrete currency token (``SILVER_PENNY_FLOW``, ``DM_STOCK``, …) is
    rejected here — only parameters pin down the currency their numbers are
    written in.

    Args:
        token: The ``unit=`` declaration — a :class:`Unit` member, or
            ``None`` for a dimensionless quantity (a share, a head count).
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
            f"and declare {Unit.CURRENCY_FLOW} / {Unit.CURRENCY_STOCK} "
            f"(GEP 10)."
        )
    if time_unit_id is not None and time_unit_id not in TIME_UNIT_ID_TO_PINT_NAME:
        raise UnitDefinitionError(
            f"Unknown time-unit suffix id {time_unit_id!r}; expected one of "
            f"{', '.join(TIME_UNIT_ID_TO_PINT_NAME)}."
        )
    if token is None:
        if time_unit_id is not None:
            raise UnitDefinitionError(
                f"A name with a time-unit suffix (_{time_unit_id}) denotes a "
                f"flow and requires a `…_FLOW` unit token; `unit=None` "
                f"declares a dimensionless quantity (a dimensionless flow is "
                f"`{Unit.DIMENSIONLESS_FLOW}`) (GEP 10)."
            )
        return UREG.dimensionless
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


def resolve_param_unit(
    token: Unit | CurrencyUnitToken | None,
    reference_period: str | None,
) -> pint.Unit:
    """Resolve a parameter's full unit from its ``unit:`` token and period.

    ``reference_period`` is *functional* (GEP 10): it is required by a
    ``…_FLOW`` token, whose period it supplies — the parameter analog of the
    name suffix — and forbidden otherwise. In particular ``unit: null`` with
    a non-null ``reference_period`` is an error: ``null`` always and only
    means dimensionless, and the per-period dimensionless quantity is
    :attr:`Unit.DIMENSIONLESS_FLOW`.

    Parameters may pin down a concrete currency (``SILVER_PENNY_STOCK``,
    ``DM_FLOW``, …); such a token resolves exactly like its agnostic
    counterpart — the concrete currency drives the build-time conversion,
    not the dimensionality check.

    Args:
        token: The ``unit:`` declaration — a :class:`Unit` member, a
            :class:`CurrencyUnitToken`, or ``None`` for a dimensionless
            parameter.
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
    if token is None:
        if reference_period is not None:
            raise UnitDefinitionError(
                f"`unit: null` declares a dimensionless parameter and cannot "
                f"be combined with `reference_period: {reference_period}`. A "
                f"dimensionless quantity per period declares "
                f"`unit: {Unit.DIMENSIONLESS_FLOW}` (GEP 10)."
            )
        return UREG.dimensionless
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
    """
    left_quantity = UREG.Quantity(1.0, left)
    right_quantity = UREG.Quantity(1.0, right)
    if left_quantity.dimensionality != right_quantity.dimensionality:
        return False
    ratio = (left_quantity / right_quantity).to_reduced_units()
    return math.isclose(ratio.magnitude, 1.0, rel_tol=_REL_TOL)


def infer_function_unit(
    function: Callable[..., Any],
    input_units: Mapping[str, str],
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
        name: UREG.Quantity(1.0, parse_unit(unit_str))
        for name, unit_str in input_units.items()
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
    return UREG.dimensionless


def fail_if_function_unit_is_inconsistent(
    *,
    function: Callable[..., Any],
    declared_unit: str,
    input_units: Mapping[str, str],
    function_name: str | None = None,
    non_unit_kwargs: Mapping[str, Any] | None = None,
) -> None:
    """Per-function body check: inferred output unit must match the declared one.

    This is the data-independent Layer-1 check of GEP 10, exercised here against
    crafted fixtures and a single mettsim function. Wiring it into the assembled
    environment as an always-on ``fail_if`` is issue #121.

    Raises:
        UnitInferenceError: If the body is dimensionally invalid.
        UnitConsistencyError: If the inferred unit does not match the declared
            unit.
    """
    name = function_name or _function_name(function)
    inferred = infer_function_unit(
        function=function, input_units=input_units, non_unit_kwargs=non_unit_kwargs
    )
    declared = parse_unit(declared_unit)
    if not units_are_equivalent(left=inferred, right=declared):
        raise UnitConsistencyError(
            f"Function {name!r} declares unit '{declared}' but its body infers "
            f"'{inferred}'."
        )
