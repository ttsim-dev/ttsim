"""Concrete-currency registration and build-time conversion (GEP 10).

The registration and conversion *operations* of the ``[currency]`` dimension.
The currency vocabulary itself — the ``CURRENCY`` token, the ``Unit`` builder,
and the state a declaration is resolved against — lives in
:mod:`ttsim.tt.units`; this module reads and mutates that shared state.
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
    _currency_family_root,
    _registered_currencies,
    CURRENCY_TOKEN,
    UNIT_REGISTRY,
    CompositeUnit,
    Unit,
    currency_family_root,
)


def base_currency() -> str | None:
    """The single registered base currency, or ``None`` if none is registered.

    With more than one currency *family* registered in the process — two
    packages imported into one test run — there is no meaningful process-wide
    base, so the run currency must be passed explicitly via ``currency=``.

    Raises:
        UnitDefinitionError: If base currencies of more than one family are
            registered.
    """
    roots = registered_base_currencies()
    if not roots:
        return None
    if len(roots) == 1:
        return roots[0]
    raise UnitDefinitionError(
        f"Base currencies of {len(roots)} different families are registered "
        f"({', '.join(roots)}), so there is no process-wide default currency. "
        f"Pass `currency=...` explicitly (GEP 10)."
    )


def registered_base_currencies() -> tuple[str, ...]:
    """The registered currency-family roots (the base currencies), sorted."""
    return tuple(sorted(set(_currency_family_root.values())))


def _definition_family_root(name: str, definition: str) -> str:
    """The family root of a currency defined relative to another.

    A definition referencing one registered currency joins that currency's
    family. A definition against the abstract :data:`CURRENCY_TOKEN` reference
    alone roots its *own* family: like a base currency it chains to no other,
    only at a factor other than 1.

    Raises:
        UnitDefinitionError: If the definition references an unregistered unit
            or more than one registered currency.
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
            f"Currency {name!r} must be defined relative to at most one "
            f"registered currency; {definition!r} references "
            f"{', '.join(referenced)} (GEP 10)."
        )
    if not referenced:
        return name
    return _currency_family_root[referenced[0]]


@contextmanager
def isolated_currency_registration() -> Iterator[None]:
    """Restore the currency bookkeeping on exit (a test isolation tool).

    Registrations made inside the block do not leak: the currency set, the
    family roots, and the token vocabulary are restored. The pint definitions
    created inside the block cannot be removed, but without the bookkeeping
    they are inert, and a later *consistent* re-registration is tolerated
    (:func:`register_currency`).
    """
    saved_currencies = set(_registered_currencies)
    saved_roots = dict(_currency_family_root)
    saved_tokens = set(_ALLOWED_UNIT_TOKENS)
    try:
        yield
    finally:
        _registered_currencies.clear()
        _registered_currencies.update(saved_currencies)
        _currency_family_root.clear()
        _currency_family_root.update(saved_roots)
        _ALLOWED_UNIT_TOKENS.clear()
        _ALLOWED_UNIT_TOKENS.update(saved_tokens)


def register_currency(
    name: str,
    *,
    base: bool = False,
    definition: str | None = None,
) -> None:
    """Register a concrete currency in the ``[currency]`` dimension.

    Downstream packages call this on import. Exactly one currency per *family*
    is the *base* currency (factor 1 against the abstract
    :data:`CURRENCY_TOKEN` reference); every other currency is defined relative
    to an already-known currency of the same family. Families from different
    packages coexist in one process — conversion is possible only within a
    family (:func:`currency_conversion_factor`), and the default run currency
    follows the policy objects in play (the ``currency`` interface node) once
    more than one family is registered.

    The registered currency becomes a valid compositional *base* — its
    upper-cased name (``register_currency("DM", ...)`` makes ``DM``,
    ``DM_PER_MONTH``, … parseable) — so parameters can pin down the concrete
    currency their numbers are written in.

    Args:
        name: The currency's unit name (e.g. ``"euro"``, ``"DM"``).
        base: Whether this is a base currency. Mutually exclusive with
            ``definition``.
        definition: A pint-parseable definition relative to another currency
            (e.g. ``"euro / 1.95583"``). Mutually exclusive with ``base``.

    Raises:
        UnitDefinitionError: If the arguments are inconsistent, if the
            definition does not resolve to the ``[currency]`` dimension, or if
            it does not reference exactly one registered currency.
    """
    if base == (definition is not None):
        raise UnitDefinitionError(
            "register_currency requires exactly one of `base=True` or "
            f"`definition=...`; got base={base!r}, definition={definition!r}."
        )
    family_root = (
        _definition_family_root(name=name, definition=definition)
        if definition is not None
        else name
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
        existing_root = _currency_family_root.get(name)
        if existing_root is not None and existing_root != family_root:
            raise UnitDefinitionError(
                f"Cannot re-register currency {name!r} into the family of "
                f"{family_root!r}: it belongs to the family of "
                f"{existing_root!r} (GEP 10)."
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
    _currency_family_root[name] = family_root
    # Surface the concrete currency on the `Unit` builder (`Unit.EUR`, `Unit.DM`,
    # `Unit.SILVER_PENNY`) so it can tag a `UnitAnnotatedColumn` of input data.
    # A column/function declaration still rejects a concrete base
    # (`resolve_compositional_column_unit`); this only makes it reachable.
    setattr(Unit, name.upper(), CompositeUnit(base=name.upper()))


def currency_conversion_factor(source_currency: str, run_currency: str) -> float:
    """Build-time factor converting a value from ``source_currency`` to the run one.

    Used to bake historical parameters denominated in their legal currency (e.g.
    DM) into the run currency at environment-build time. pint is the single
    source of truth for the rate. Both currencies must be registered and belong
    to the same *family*: two packages' families share no exchange rate, and
    their pint factors would relate them 1:1 through the abstract
    :data:`CURRENCY_TOKEN` reference — a silent wrong number, so it is rejected
    here.

    Raises:
        UnitDefinitionError: If either currency is unknown, not a currency, or
            of another family than the other.
    """
    for name in (source_currency, run_currency):
        if name not in UNIT_REGISTRY:
            raise UnitDefinitionError(
                f"Cannot convert currency: {name!r} is not a registered currency."
            )
    source_root = currency_family_root(source_currency)
    run_root = currency_family_root(run_currency)
    if source_root != run_root:
        raise UnitDefinitionError(
            f"No exchange rate connects {source_currency!r} (family of "
            f"{source_root!r}) and {run_currency!r} (family of {run_root!r}): "
            f"they were registered by different packages. Use a run currency "
            f"from the family the parameters are denominated in (GEP 10)."
        )
    try:
        return UNIT_REGISTRY.Quantity(1.0, source_currency).to(run_currency).magnitude
    except pint.DimensionalityError as e:
        raise UnitDefinitionError(
            f"Cannot convert {source_currency!r} to {run_currency!r}: {e}"
        ) from e


