"""Concrete-currency registration and build-time conversion (GEP 10).

The registration and conversion *operations* of the ``[currency]`` dimension.
The currency vocabulary itself — the ``CURRENCY`` token, the ``Unit`` builder,
and the state a declaration is resolved against — lives in
:mod:`ttsim.tt.units`; this module reads and mutates that shared state.

Every run is denominated in a single concrete currency. A downstream package
registers its currencies on import: one base currency (factor 1 against the
abstract :data:`CURRENCY_TOKEN` reference) and any number of others defined
relative to an already-registered one. All currencies are interconvertible, so
conversion between any two is always well-defined.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
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

#: The registered base currency (a one-element list, not a bare ``str``, so it
#: can be mutated in place — a module-level ``global`` reassignment is banned by
#: the linter). Empty until a downstream package registers its base.
_base_currency: list[str] = []


def base_currency() -> str:
    """The registered base currency the whole run is denominated in.

    A downstream package registers its base currency on import (gettsim's
    ``euro``, mettsim's ``CASTAR``), and that is the run's default — users need
    not pass ``currency=`` themselves.

    Raises:
        UnitDefinitionError: If no base currency is registered — a run must be
            denominated in a concrete currency (GEP 10).
    """
    if not _base_currency:
        raise UnitDefinitionError(
            "No base currency is registered, so the run currency is undefined. "
            "A package must register one with `register_currency(name=..., "
            "base=True)` before the system can run (GEP 10)."
        )
    return _base_currency[0]


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
    currency, and the token vocabulary are restored. The pint definitions
    created inside the block cannot be removed, but without the bookkeeping
    they are inert, and a later *consistent* re-registration is tolerated
    (:func:`register_currency`).
    """
    saved_currencies = set(_registered_currencies)
    saved_base = list(_base_currency)
    saved_tokens = set(_ALLOWED_UNIT_TOKENS)
    try:
        yield
    finally:
        _registered_currencies.clear()
        _registered_currencies.update(saved_currencies)
        _base_currency[:] = saved_base
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
    base is the default run currency (the ``currency`` interface node).

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


def currency_conversion_factor(source_currency: str, run_currency: str) -> float:
    """Build-time factor converting a value from ``source_currency`` to the run one.

    Used to bake historical parameters denominated in their legal currency (e.g.
    DM) into the run currency at environment-build time. pint is the single
    source of truth for the rate. Both currencies must be registered; all
    registered currencies are interconvertible.

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
