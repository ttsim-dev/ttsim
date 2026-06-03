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

A declaration is one member of the closed :class:`Unit` enumeration —
``unit=Unit.CURRENCY_FLOW`` in code, ``unit: CURRENCY_FLOW`` in YAML.
Tokens come in two kinds: *flow* tokens (named ``…_FLOW``) denote a
per-period quantity and are completed by a period supplied by the name
suffix (columns/functions) or ``reference_period`` (parameters); all other
tokens are *complete* as written and admit no period source. Declarations
never contain pint syntax; internally each token resolves to a pint unit.
A dimensionless quantity (a share, a rate, a head count) declares *no* unit
(``unit=None`` in code, ``unit: null`` in YAML) and never combines with a
period source; the per-period dimensionless quantity is its own flow token
(:attr:`Unit.SHARE_FLOW`). New tokens require a GEP amendment and a TTSIM
PR — there is deliberately no registration API, which keeps the JSON schema
for the parameter YAMLs statically enumerable.

Currencies are registered by downstream packages via
:func:`register_currency`, which provides **conversion factors only** — it
does not extend the declaration vocabulary. The :data:`CURRENCY_TOKEN` (the
literal string ``"CURRENCY"``) is a real unit anchoring the ``[currency]``
dimension, so the currency tokens resolve regardless of whether a concrete
currency has been registered yet — checks compare at the dimensionality
level and the concrete currency is resolved separately (issue #120).
Concrete currencies appear only in Layer-2 input tags and the currency
machinery, never in a declaration.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Callable, Mapping
from typing import Any

import pint
from pint.util import to_units_container

from ttsim.exceptions import (
    UnitConsistencyError,
    UnitDefinitionError,
    UnitInferenceError,
)

#: The pint unit anchoring the ``[currency]`` dimension, used internally to
#: resolve the currency tokens (``CURRENCY_FLOW``, ``CURRENCY_STOCK``, …)
#: before any concrete currency is registered. Checks compare at the
#: dimensionality level; the concrete currency is resolved separately.
CURRENCY_TOKEN = "CURRENCY"  # noqa: S105 (a unit token, not a secret)


class Unit(enum.StrEnum):
    """The closed vocabulary of unit tokens (GEP 10).

    One token = one meaning, independent of any other field. A bare token is
    *complete* as written; a ``…_FLOW`` token *needs a period*, supplied by
    the name suffix (columns/functions) or ``reference_period`` (parameters).
    Where both kinds of a quantity exist, both are marked
    (:attr:`CURRENCY_STOCK` / :attr:`CURRENCY_FLOW`) — a bare ``CURRENCY``
    is deliberately unwritable, so no token can be misread as complete when
    it is not.

    YAML spells the identical strings (``unit: CURRENCY_FLOW``); Python code
    must use the members themselves (``unit=Unit.CURRENCY_FLOW``).
    """

    CURRENCY_FLOW = "CURRENCY_FLOW"
    """An amount of currency per period: wages, claims, benefits."""

    CURRENCY_STOCK = "CURRENCY_STOCK"
    """An amount of currency, full stop: wealth, asset thresholds."""

    SHARE_FLOW = "SHARE_FLOW"
    """A dimensionless share per period (``1/period``), e.g. a wealth-tax
    rate. Not to be confused with a plain share (a Steuersatz), which is
    dimensionless and declares ``unit=None``."""

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


#: Maps each token to ``(base, is_flow)``: the pint expression of its
#: non-period part (``None`` for a dimensionless base) and whether a period
#: must be supplied. Internal — declarations never contain pint syntax.
_TOKEN_BASE_AND_IS_FLOW: dict[Unit, tuple[str | None, bool]] = {
    Unit.CURRENCY_FLOW: (CURRENCY_TOKEN, True),
    Unit.CURRENCY_STOCK: (CURRENCY_TOKEN, False),
    Unit.SHARE_FLOW: (None, True),
    Unit.YEARS: ("year", False),
    Unit.HOURS_FLOW: ("hour", True),
    Unit.SQUARE_METERS: ("meter ** 2", False),
    Unit.HECTARES: ("hectare", False),
    Unit.CURRENCY_PER_SQUARE_METER_FLOW: (f"{CURRENCY_TOKEN} / meter ** 2", True),
}


def unit_token_is_flow(token: Unit) -> bool:
    """Whether a token denotes a per-period quantity (needs a period source)."""
    return _TOKEN_BASE_AND_IS_FLOW[token][1]


def coerce_unit_token(
    value: str | Unit | None,
    *,
    where: str,
) -> Unit | None:
    """Coerce a YAML ``unit:`` value to a :class:`Unit` member (GEP 10).

    ``None`` (``unit: null``) declares a dimensionless quantity and passes
    through. Any string must spell a member of the closed enumeration
    exactly; everything else — including pint syntax like ``"CURRENCY"`` or
    ``"CURRENCY / year"`` — is rejected.

    Args:
        value: The raw declaration (a string from YAML, an already-coerced
            member, or ``None``).
        where: Identifier for error messages (e.g. the parameter's name).

    Raises:
        UnitDefinitionError: If the value is not a member of the enumeration.
    """
    if value is None or isinstance(value, Unit):
        return value
    try:
        return Unit(value)
    except ValueError:
        raise UnitDefinitionError(
            f"{where}: invalid unit token {value!r}. A unit declaration must "
            f"be one of {', '.join(Unit)} — or null for a dimensionless "
            f"quantity (GEP 10)."
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

    Args:
        name: The currency's unit name (e.g. ``"euro"``, ``"DM"``).
        base: Whether this is the base currency. Mutually exclusive with
            ``definition``.
        definition: A pint-parseable definition relative to another currency
            (e.g. ``"euro / 1.95583"``). Mutually exclusive with ``base``.

    Raises:
        UnitDefinitionError: If the arguments are inconsistent, if a second base
            currency is registered, or if the definition does not resolve to the
            ``[currency]`` dimension.
    """
    global _base_currency  # noqa: PLW0603

    if base == (definition is not None):
        raise UnitDefinitionError(
            "register_currency requires exactly one of `base=True` or "
            f"`definition=...`; got base={base!r}, definition={definition!r}."
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
        if base and _base_currency not in (None, name):
            raise UnitDefinitionError(
                f"Cannot register {name!r} as the base currency: "
                f"{_base_currency!r} is already the base currency."
            )
        if base:
            _base_currency = name
        _ALLOWED_UNIT_TOKENS.add(name)
        return

    if base:
        if _base_currency is not None:
            raise UnitDefinitionError(
                f"Cannot register {name!r} as the base currency: "
                f"{_base_currency!r} is already the base currency."
            )
        UREG.define(f"{name} = {CURRENCY_TOKEN}")
        _base_currency = name
    else:
        UREG.define(f"{name} = {definition}")

    if UREG.Quantity(1.0, name).dimensionality != currency_dim:
        raise UnitDefinitionError(
            f"Currency {name!r} defined as {definition!r} does not resolve to "
            f"the [currency] dimension."
        )
    _ALLOWED_UNIT_TOKENS.add(name)


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
            unit (declare ``unit=None`` / ``unit: null`` instead).
    """
    if not isinstance(unit_str, str):
        raise UnitDefinitionError(
            f"A unit must be given as a string, got {unit_str!r}."
        )
    try:
        unit = UREG.parse_units(unit_str)
    except (pint.errors.PintError, AssertionError, ValueError, TypeError) as e:
        raise UnitDefinitionError(f"Could not parse unit {unit_str!r}: {e}") from e
    _fail_if_unit_tokens_are_unknown(unit, unit_str)
    if not to_units_container(unit):
        raise UnitDefinitionError(
            f"Unit {unit_str!r} resolves to the dimensionless unit. A "
            f"dimensionless quantity (a share, a rate, a head count) declares "
            f"no unit at all: `unit=None` in code, `unit: null` in YAML "
            f"(GEP 10)."
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
    func_name = getattr(function, "__qualname__", getattr(function, "__name__", "?"))
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
    name = function_name or getattr(
        function, "__qualname__", getattr(function, "__name__", "?")
    )
    inferred = infer_function_unit(
        function, input_units, non_unit_kwargs=non_unit_kwargs
    )
    declared = parse_unit(declared_unit)
    if not units_are_equivalent(inferred, declared):
        raise UnitConsistencyError(
            f"Function {name!r} declares unit '{declared}' but its body infers "
            f"'{inferred}'."
        )
