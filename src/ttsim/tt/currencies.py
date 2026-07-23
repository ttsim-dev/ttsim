"""A policy system's currencies, and the boundary conversion between them.

A :class:`UnitSystem` is the value a policy package builds once, at import, and
hands to ``main(unit_system=...)``: its currencies, the dated statutory-currency
mapping, the grouping levels its declarations spell — and the pint registry all
of those are defined in.

The registry is per system, so two policy systems coexist in one process, each
with its own base currency. "Exactly one currency is the base" (GEP 10) holds
within a system, by construction: the base is a constructor argument, not
something a second import could contradict.

The unit *vocabulary* a declaration is spelled in — the ``CURRENCY`` token, the
``TTSIMUnit`` builder, the :class:`CompositeUnit` grammar — is shared and lives in
:mod:`ttsim.tt.units`.
"""

from __future__ import annotations

import dataclasses
import datetime
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from types import MappingProxyType
from typing import Any

import pint
from pint.util import to_units_container

from ttsim.exceptions import UnitDefinitionError
from ttsim.tt.grouping_levels import (
    define_grouping_level_dimensions,
    register_grouping_levels,
)
from ttsim.tt.units import (
    _ALLOWED_UNIT_TOKENS,
    _COMPOSITIONAL_BASE_TO_PINT,
    CURRENCY_TOKEN,
    CompositeUnit,
    TTSIMUnit,
    _registered_currencies,
    _unit_is_currency,
    build_registry,
)


@dataclasses.dataclass(frozen=True, eq=False, kw_only=True)
class UnitSystem:
    """The currencies, statutory-currency mapping, and registry of one policy system.

    A package builds one system and exports it as a singleton, so a system is
    identified by object identity: ``eq=False`` keeps the inherited
    identity-based ``__hash__``, which lets a system key an ``lru_cache`` (its
    ``statutory_currencies`` mapping is otherwise unhashable).

    A policy package declares its system once and exports it::

        UNIT_SYSTEM = UnitSystem(
            base_currency="EUR",
            other_currencies={"DM": "EUR / 1.95583"},
            statutory_currencies={"0001-01-01": "DM", "2002-01-01": "EUR"},
            grouping_levels=["hh", "bg", "fg"],
        )

    All of a system's currencies are interconvertible
    (:meth:`currency_conversion_factor`); a currency of *another* system is not,
    and is rejected rather than silently taken to be worth the same.

    Raises:
        UnitDefinitionError: If a currency name clashes with a unit the shared
            vocabulary already defines or differs only in case from another
            registered currency, if a definition does not resolve to the
            ``[currency]`` dimension or does not reference exactly one of this
            system's currencies, or if the statutory-currency mapping is empty,
            is keyed by anything other than an ISO date, or names a currency the
            system does not have.
    """

    base_currency: str
    """The system's unit of account, and the default data currency. Defined as
    factor 1 against the abstract ``[currency]`` reference; every other currency
    is defined relative to it or to another already-defined one."""

    statutory_currencies: Mapping[str, str]
    """The currency statutes denominate their numbers in, keyed by the dashed ISO
    start date it applies from (until the next entry's). Mandatory: a run for a
    policy date with no statutory currency fails."""

    other_currencies: Mapping[str, str] = dataclasses.field(
        default_factory=lambda: MappingProxyType({})
    )
    """Each further currency, mapped to a pint-parseable definition relative to
    an already-defined currency of this system (``{"DM": "EUR / 1.95583"}``).
    Definitions are applied in order, so one may reference an earlier one."""

    grouping_levels: Sequence[str] = ()
    """The group levels this system's declarations spell (``["hh", "bg"]``). The
    individual ``person`` leaf is always present. A build discovers further
    levels from the policy environment's ``*_id`` columns and registers them
    then."""

    registry: pint.UnitRegistry = dataclasses.field(init=False, repr=False)
    """The system's own pint registry: the shared vocabulary plus this system's
    currency definitions and grouping-level dimensions."""

    currencies: frozenset[str] = dataclasses.field(init=False)
    """Every currency name this system defines — the base and the others."""

    statutory_currency_by_start_date: tuple[tuple[datetime.date, str], ...] = (
        dataclasses.field(init=False, repr=False)
    )
    """:attr:`statutory_currencies` parsed and sorted by start date."""

    field_units_by_class: dict[type, dict[str, Any] | None] = dataclasses.field(
        init=False, repr=False, default_factory=dict
    )
    """Memo of the resolved unit annotations of each parameter dataclass the
    unit check has seen, keyed by class. Each value maps a field name to what its
    pluck yields — a resolved ``pint.Unit``, a nested dataclass ``type``, or a
    schedule-field marker built in :mod:`ttsim.interface_dag_elements.unit_checks`
    (kept loose here so this module owns no private name from that layer). The
    units are this system's registry's, so the memo is the system's."""

    def __post_init__(self) -> None:
        registry = build_registry()
        object.__setattr__(self, "registry", registry)
        object.__setattr__(
            self,
            "currencies",
            frozenset({self.base_currency, *self.other_currencies}),
        )
        self._define_currencies()
        object.__setattr__(
            self,
            "statutory_currency_by_start_date",
            self._parsed_statutory_currencies(),
        )
        # Registry-local, and the last step that can reject anything; running it
        # here keeps every global mutation below unfailable.
        define_grouping_level_dimensions(names=self.grouping_levels, registry=registry)
        self._publish_currencies()
        register_grouping_levels(names=self.grouping_levels, registry=registry)

    def currency_conversion_factor(
        self, *, source_currency: str, target_currency: str
    ) -> float:
        """The factor converting ``source_currency`` into ``target_currency``.

        Used only where data enters and leaves the computation: input columns are
        converted from the data currency to the computation currency, and
        currency-denominated results are converted back (GEP 10). pint is the
        single source of truth for the rate. Both currencies must belong to this
        system; all of a system's currencies are interconvertible.

        Raises:
            UnitDefinitionError: If either currency is not one of this system's.
        """
        for name in (source_currency, target_currency):
            if name not in self.currencies:
                raise UnitDefinitionError(
                    f"Cannot convert currency: {name!r} is not a registered "
                    f"currency of this policy system. Its currencies are "
                    f"{', '.join(sorted(self.currencies))} (GEP 10)."
                )
        return (
            self.registry.Quantity(1.0, source_currency).to(target_currency).magnitude
        )

    def statutory_currency_for_date(self, policy_date: datetime.date) -> str:
        """The statutory currency at a given policy date.

        Raises:
            UnitDefinitionError: If ``policy_date`` lies before the mapping's
                first entry.
        """
        for start_date, name in reversed(self.statutory_currency_by_start_date):
            if policy_date >= start_date:
                return name
        raise UnitDefinitionError(
            f"The statutory-currency mapping starts at "
            f"{self.statutory_currency_by_start_date[0][0].isoformat()}, so the "
            f"statutory currency at {policy_date.isoformat()} is undefined. "
            f"Extend the mapping this policy system declares (GEP 10)."
        )

    def _define_currencies(self) -> None:
        """Define the base and every other currency in the system's registry.

        The base is factor 1 against the abstract :data:`CURRENCY_TOKEN`
        reference; every other currency is defined relative to an
        already-defined one, so all of them chain back to the base and are
        interconvertible.

        Definitions land in the system's own registry, which is discarded with a
        system whose construction fails; :meth:`_publish_currencies` does the
        process-global half once every check has passed.
        """
        self._define_one_currency(name=self.base_currency, definition=CURRENCY_TOKEN)
        defined = {self.base_currency}
        for name, definition in self.other_currencies.items():
            self._fail_if_definition_references_no_known_currency(
                name=name, definition=definition, defined=defined
            )
            self._define_one_currency(name=name, definition=definition)
            defined.add(name)
        for name in self.currencies:
            self._fail_if_builder_base_is_taken(name=name)

    def _publish_currencies(self) -> None:
        """Widen the process-global vocabulary by this system's currencies.

        Runs only once every check has passed — including the statutory-mapping
        one — so a system whose construction raises leaves the vocabulary as it
        found it.
        """
        for name in self.currencies:
            _ALLOWED_UNIT_TOKENS.add(name)
            _registered_currencies.add(name)
            # Surface the concrete currency on the `TTSIMUnit` builder (`TTSIMUnit.EUR`,
            # `TTSIMUnit.DM`, `TTSIMUnit.SILVER_PENNY`) so it can tag a
            # `UnitAnnotatedColumn` of input data. A column/function declaration
            # still rejects a concrete base (`resolve_compositional_column_unit`);
            # this only makes it reachable.
            setattr(TTSIMUnit, name.upper(), CompositeUnit(base=name.upper()))

    def _fail_if_builder_base_is_taken(self, name: str) -> None:
        """Reject a currency whose `TTSIMUnit` base another unit already owns.

        A currency reaches the builder namespace under its upper-cased name, and
        :func:`ttsim.tt.units.parse_compositional_unit` matches a base against that
        same upper-cased form. Two names that differ only in case would therefore
        share one base — one silently shadowing the other on the builder, and the
        shared base naming neither of them unambiguously. The clash counts whether
        the other name belongs to this system or to one already registered.
        """
        base = name.upper()
        shadowed = sorted(
            other
            for other in _registered_currencies | self.currencies
            if other != name and other.upper() == base
        )
        if shadowed:
            raise UnitDefinitionError(
                f"Cannot register currency {name!r}: currency "
                f"{', '.join(repr(other) for other in shadowed)} already claims the "
                f"unit base {base!r}. Currency names must differ by more than case "
                f"(GEP 10)."
            )
        if base in _COMPOSITIONAL_BASE_TO_PINT:
            raise UnitDefinitionError(
                f"Cannot register currency {name!r}: {base!r} is a non-currency unit "
                f"base. Pick a name outside the shared unit vocabulary (GEP 10)."
            )

    def _define_one_currency(self, name: str, definition: str) -> None:
        """Define one currency in the registry, checking it lands in ``[currency]``."""
        if name in self.registry:
            raise UnitDefinitionError(
                f"Cannot define currency {name!r}: a unit of that name already "
                f"exists ({self.registry.Quantity(1.0, name).dimensionality}). "
                f"Pick a name outside the shared unit vocabulary (GEP 10)."
            )
        self.registry.define(f"{name} = {definition}")
        if not _unit_is_currency(self.registry.parse_units(name)):
            raise UnitDefinitionError(
                f"Currency {name!r} defined as {definition!r} does not resolve "
                f"to the [currency] dimension."
            )

    def _fail_if_definition_references_no_known_currency(
        self, name: str, definition: str, defined: set[str]
    ) -> None:
        """Reject a currency definition that does not chain to a known currency.

        Every non-base currency is defined relative to exactly one currency this
        system has already defined (``"CASTAR / 4"``). A definition against the
        abstract :data:`CURRENCY_TOKEN` reference alone, or against no currency
        at all, would start a second, unconnected base — which the single-base
        model forbids.

        Raises:
            UnitDefinitionError: If the definition references a unit the registry
                does not know, no currency of this system, or more than one.
        """
        try:
            parsed = self.registry.parse_expression(definition)
        except pint.UndefinedUnitError as error:
            raise UnitDefinitionError(
                f"Currency {name!r} is defined as {definition!r}, which "
                f"references a unit this policy system does not define. Define a "
                f"currency relative to one of its own (GEP 10)."
            ) from error
        referenced = sorted(
            str(token)
            for token in to_units_container(parsed.units)
            if str(token) in defined
        )
        if len(referenced) > 1:
            raise UnitDefinitionError(
                f"Currency {name!r} must be defined relative to exactly one "
                f"currency of this policy system; {definition!r} references "
                f"{', '.join(referenced)} (GEP 10)."
            )
        if not referenced:
            raise UnitDefinitionError(
                f"Currency {name!r} defined as {definition!r} references no "
                f"currency of this policy system. Define it relative to one "
                f"already defined (e.g. the base currency) (GEP 10)."
            )

    def _parsed_statutory_currencies(self) -> tuple[tuple[datetime.date, str], ...]:
        """Parse and sort the statutory-currency mapping.

        Raises:
            UnitDefinitionError: If the mapping is empty, is keyed by anything other
                than an ISO date, or names a currency this system does not define.
        """
        if not self.statutory_currencies:
            raise UnitDefinitionError(
                "`statutory_currencies` requires at least one entry; got an "
                "empty mapping (GEP 10)."
            )
        unknown = sorted(set(self.statutory_currencies.values()) - self.currencies)
        if unknown:
            raise UnitDefinitionError(
                f"`statutory_currencies` references "
                f"{', '.join(repr(name) for name in unknown)}, which "
                f"{'is' if len(unknown) == 1 else 'are'} not a currency of this "
                f"policy system. Declare every statutory currency as the "
                f"`base_currency` or in `other_currencies` (GEP 10)."
            )
        return tuple(sorted(self._parsed_start_dates()))

    def _parsed_start_dates(self) -> Iterator[tuple[datetime.date, str]]:
        """Each start date parsed, rejecting any spelling but dashed ``YYYY-MM-DD``.

        `date.fromisoformat` also accepts the basic (``20200101``) and week-date
        (``2021-W01-1``) forms, so the round-trip comparison — not the parse alone
        — is what pins the one documented spelling.
        """
        for start_date, name in self.statutory_currencies.items():
            try:
                parsed = datetime.date.fromisoformat(start_date)
            except ValueError as error:
                raise UnitDefinitionError(
                    self._bad_start_date_message(start_date=start_date, name=name)
                ) from error
            if start_date != parsed.isoformat():
                raise UnitDefinitionError(
                    self._bad_start_date_message(start_date=start_date, name=name)
                )
            yield parsed, name

    @staticmethod
    def _bad_start_date_message(start_date: str, name: str) -> str:
        return (
            f"`statutory_currencies` is keyed by the dashed ISO date a currency "
            f"becomes statutory; {start_date!r} (mapped to {name!r}) is not one. "
            f"Spell it YYYY-MM-DD (GEP 10)."
        )


@contextmanager
def isolated_currency_registration() -> Iterator[None]:
    """Restore the shared unit vocabulary on exit (a test isolation tool).

    A :class:`UnitSystem` keeps its currency definitions and level dimensions to
    itself, but it also widens the process-global vocabulary — the currency
    *names* a declaration may spell, the pint tokens a unit may combine, and the
    concrete currency bases injected on :class:`TTSIMUnit` — so that a
    `CompositeUnit` can be classified without a system in scope. This block
    restores all three, so a system built inside it leaves the vocabulary as it
    found it.
    """
    saved_currencies = set(_registered_currencies)
    saved_tokens = set(_ALLOWED_UNIT_TOKENS)
    saved_bases = set(vars(TTSIMUnit))
    try:
        yield
    finally:
        _registered_currencies.clear()
        _registered_currencies.update(saved_currencies)
        _ALLOWED_UNIT_TOKENS.clear()
        _ALLOWED_UNIT_TOKENS.update(saved_tokens)
        for base in set(vars(TTSIMUnit)) - saved_bases:
            delattr(TTSIMUnit, base)
