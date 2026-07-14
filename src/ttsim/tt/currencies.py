"""Currency registration and boundary conversion.

The registration and conversion *operations* of the ``[currency]`` dimension. The
currency vocabulary itself — the ``CURRENCY`` token, the ``Unit`` builder, and the state
a declaration is resolved against — lives in :mod:`ttsim.tt.units`; this module reads
and mutates that shared state.

A downstream package registers its currencies on import: one base currency, any number
of others defined relative to an already-registered one, and the dated
statutory-currency mapping.
"""

from __future__ import annotations

import datetime
import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager

import pint
from pint.util import to_units_container

from ttsim.exceptions import UnitDefinitionError
from ttsim.tt.units import (
    _ALLOWED_UNIT_TOKENS,
    _REL_TOL,
    CURRENCY_TOKEN,
    UNIT_REGISTRY,
    CompositeUnit,
    Unit,
    _registered_currencies,
)

_base_currency: list[str] = []

#: The statutory-currency mapping: ``(start_date, currency)`` pairs sorted by
#: start date, each applying from its start date until the next entry's. Empty
#: until a downstream package registers its mapping.
_statutory_currencies: list[tuple[datetime.date, str]] = []


def base_currency() -> str:
    """The registered base currency — the default data currency.

    A downstream package registers its base currency on import (gettsim's
    ``euro``, mettsim's ``CASTAR``), and user data is assumed to arrive in it —
    users need not pass ``data_currency=`` themselves.

    Raises:
        UnitDefinitionError: If no base currency is registered — the data
            currency must be a concrete currency (GEP 10).
    """
    if not _base_currency:
        raise UnitDefinitionError(
            "No base currency is registered, so the data currency is undefined. "
            "A package must register one with `register_currency(name=..., "
            "base=True)` before the system can run (GEP 10)."
        )
    return _base_currency[0]


def register_statutory_currencies(currency_by_start_date: Mapping[str, str]) -> None:
    """Declare the statutory currency at each start date.

    A downstream package calls this on import, after registering the currencies
    it references. Each entry applies from its start date (a dashed ISO string)
    until the next entry's start date; the computation for a policy date runs in
    the statutory currency at that date (:func:`statutory_currency_for_date`),
    and the build guard requires every parameter to be declared in it.

    Example:
        register_statutory_currencies({"1948-06-21": "DM", "2002-01-01": "EUR"})

    The mapping is mandatory: a run for a policy date with no registered
    statutory currency fails.

    Raises:
        UnitDefinitionError: If the mapping is empty, references an unregistered
            currency, or a different mapping is already registered.
    """
    if not currency_by_start_date:
        raise UnitDefinitionError(
            "register_statutory_currencies requires at least one entry; got an "
            "empty mapping (GEP 10)."
        )
    unregistered = sorted(
        {
            name
            for name in currency_by_start_date.values()
            if name not in _registered_currencies
        }
    )
    if unregistered:
        raise UnitDefinitionError(
            f"register_statutory_currencies references "
            f"{', '.join(repr(name) for name in unregistered)}, which "
            f"{'is' if len(unregistered) == 1 else 'are'} not registered. "
            f"Register every statutory currency with `register_currency` first "
            f"(GEP 10)."
        )
    entries = sorted(
        (datetime.date.fromisoformat(start_date), name)
        for start_date, name in currency_by_start_date.items()
    )
    if _statutory_currencies and _statutory_currencies != entries:
        raise UnitDefinitionError(
            f"A different statutory-currency mapping is already registered "
            f"({_statutory_currencies}). A process has a single statutory-"
            f"currency mapping (GEP 10)."
        )
    _statutory_currencies[:] = entries


def statutory_currency_for_date(policy_date: datetime.date) -> str:
    """The statutory currency at a given policy date.

    Raises:
        UnitDefinitionError: If no mapping is registered, or ``policy_date``
            lies before the mapping's first entry.
    """
    if not _statutory_currencies:
        raise UnitDefinitionError(
            "No statutory-currency mapping is registered, so the computation "
            "currency is undefined. A package must register one with "
            "`register_statutory_currencies({start_date: currency, ...})` "
            "before the system can run (GEP 10)."
        )
    for start_date, name in reversed(_statutory_currencies):
        if policy_date >= start_date:
            return name
    raise UnitDefinitionError(
        f"The statutory-currency mapping starts at "
        f"{_statutory_currencies[0][0].isoformat()}, so the statutory currency "
        f"at {policy_date.isoformat()} is undefined. Extend the mapping "
        f"registered with `register_statutory_currencies` (GEP 10)."
    )


def _fail_if_definition_references_no_registered_currency(
    name: str, definition: str
) -> None:
    """Reject a currency definition that does not chain to a registered currency.

    Every non-base currency is defined relative to exactly one already-registered
    concrete currency (``"CASTAR / 4"``). A definition against the abstract
    :data:`CURRENCY_TOKEN` reference alone, or against no currency at all, would
    start a second, unconnected base — which the single-base model forbids.

    Raises:
        UnitDefinitionError: If the definition references an unregistered unit,
            no registered currency, or more than one.
    """
    try:
        parsed_definition = UNIT_REGISTRY.parse_expression(definition)
    except pint.UndefinedUnitError as error:
        raise UnitDefinitionError(
            f"Currency {name!r} is defined as {definition!r}, which "
            f"references an unregistered unit. Define a currency relative "
            f"to an already-registered one (GEP 10)."
        ) from error
    referenced = sorted(
        str(token)
        for token in to_units_container(parsed_definition.units)
        if str(token) in _registered_currencies
    )
    if len(referenced) > 1:
        raise UnitDefinitionError(
            f"Currency {name!r} must be defined relative to exactly one "
            f"registered currency; {definition!r} references "
            f"{', '.join(referenced)} (GEP 10)."
        )
    if not referenced:
        raise UnitDefinitionError(
            f"Currency {name!r} defined as {definition!r} references no "
            f"registered currency. Define it relative to an already-registered "
            f"one (e.g. the base currency) (GEP 10)."
        )


@contextmanager
def isolated_currency_registration() -> Iterator[None]:
    """Restore the currency bookkeeping on exit (a test isolation tool).

    Registrations made inside the block do not leak: the currency set, the base
    currency, the statutory-currency mapping, and the token vocabulary are
    restored. The pint definitions
    created inside the block cannot be removed, but without the bookkeeping
    they are inert, and a later *consistent* re-registration is tolerated
    (:func:`register_currency`).
    """
    saved_currencies = set(_registered_currencies)
    saved_base = list(_base_currency)
    saved_statutory = list(_statutory_currencies)
    saved_tokens = set(_ALLOWED_UNIT_TOKENS)
    try:
        yield
    finally:
        _registered_currencies.clear()
        _registered_currencies.update(saved_currencies)
        _base_currency[:] = saved_base
        _statutory_currencies[:] = saved_statutory
        _ALLOWED_UNIT_TOKENS.clear()
        _ALLOWED_UNIT_TOKENS.update(saved_tokens)


def register_currency(
    name: str,
    *,
    base: bool = False,
    definition: str | None = None,
) -> None:
    """Register a concrete currency in the ``[currency]`` dimension.

    Downstream packages call this on import. Exactly one currency is the *base*
    currency (factor 1 against the abstract :data:`CURRENCY_TOKEN` reference);
    every other currency is defined relative to an already-known currency. All
    currencies are interconvertible (:func:`currency_conversion_factor`), and the
    base is the default data currency (the ``data_currency`` interface node).

    The registered currency becomes a valid compositional *base* — its
    upper-cased name (``register_currency("DM", ...)`` makes ``DM``,
    ``DM_PER_MONTH``, … parseable) — so parameters can pin down the concrete
    currency their numbers are written in.

    Args:
        name: The currency's unit name (e.g. ``"euro"``, ``"DM"``).
        base: Whether this is the base currency. Mutually exclusive with
            ``definition``.
        definition: A pint-parseable definition relative to an already-registered
            currency (e.g. ``"euro / 1.95583"``). Mutually exclusive with
            ``base``.

    Raises:
        UnitDefinitionError: If the arguments are inconsistent, if a different
            base currency is already registered, if the definition does not
            resolve to the ``[currency]`` dimension, or if it does not reference
            exactly one registered currency.
    """
    if base == (definition is not None):
        raise UnitDefinitionError(
            "register_currency requires exactly one of `base=True` or "
            f"`definition=...`; got base={base!r}, definition={definition!r}."
        )
    if base and _base_currency and _base_currency[0] != name:
        raise UnitDefinitionError(
            f"Cannot register {name!r} as the base currency: {_base_currency[0]!r} "
            f"is already the base. A process has a single base currency (GEP 10)."
        )
    if definition is not None:
        _fail_if_definition_references_no_registered_currency(
            name=name, definition=definition
        )

    currency_dim = UNIT_REGISTRY.Quantity(1.0, CURRENCY_TOKEN).dimensionality
    if name in UNIT_REGISTRY:
        # Idempotent re-registration (e.g. a re-imported module). Tolerate it
        # only if the existing definition is consistent with this call — same
        # dimension *and* same conversion factor: checking the dimension alone
        # would silently keep the old factor and invalidate later conversions.
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
        else:
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

    _ALLOWED_UNIT_TOKENS.add(name)
    _registered_currencies.add(name)
    if base:
        _base_currency[:] = [name]
    # Surface the concrete currency on the `Unit` builder (`Unit.EUR`, `Unit.DM`,
    # `Unit.SILVER_PENNY`) so it can tag a `UnitAnnotatedColumn` of input data.
    # A column/function declaration still rejects a concrete base
    # (`resolve_compositional_column_unit`); this only makes it reachable.
    setattr(Unit, name.upper(), CompositeUnit(base=name.upper()))


def currency_conversion_factor(source_currency: str, target_currency: str) -> float:
    """The factor converting a value from ``source_currency`` to ``target_currency``.

    Used only where data enters and leaves the computation: input columns are
    converted from the data currency to the computation currency, and
    currency-denominated results are converted back (GEP 10). pint is the
    single source of truth for the rate. Both currencies must be registered;
    all registered currencies are interconvertible.

    Raises:
        UnitDefinitionError: If either currency is unknown or not a currency.
    """
    for name in (source_currency, target_currency):
        if name not in _registered_currencies:
            raise UnitDefinitionError(
                f"Cannot convert currency: {name!r} is not a registered currency."
            )
    try:
        return (
            UNIT_REGISTRY.Quantity(1.0, source_currency).to(target_currency).magnitude
        )
    except pint.DimensionalityError as e:
        raise UnitDefinitionError(
            f"Cannot convert {source_currency!r} to {target_currency!r}: {e}"
        ) from e
