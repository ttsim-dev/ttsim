"""The pint-based unit framework.

Establishes the closed unit vocabulary that checks the dimensional soundness of
the taxes-and-transfers DAG, plus the build-time machinery that runs the check.

The vocabulary here is spelled in :class:`CompositeUnit` values, which carry no
registry reference — ``TTSIMUnit.CURRENCY.PER_MONTH`` is a pure value. Resolving one
to a :class:`pint.Unit` needs a registry, which a policy system owns
(:class:`ttsim.tt.currencies.UnitSystem`); every helper below that resolves,
parses, or compares takes that registry explicitly.

pint is a build-time tool only — it never wraps a live array (a
:class:`pint.Quantity` is not a JAX pytree and does not trace under ``jit``). It
serves two build-time jobs:

- the dry-run dimensionality check (:mod:`ttsim.interface_dag_elements.unit_checks`);
- sourcing the time- and currency-conversion factors baked into the numeric
  workers.

The runtime path stays pure arrays, single currency, JAX-safe.

Every declaration is a fully-spelled :class:`CompositeUnit` — a base optionally
divided by a physical denominator (an area or working hours), a period, and a
grouping level, in that canonical order. It
has two round-tripping spellings (via :func:`parse_compositional_unit` /
:func:`str`):

- fluent, off the :class:`TTSIMUnit` namespace
  (``TTSIMUnit.CURRENCY.PER_MONTH.PER_BG``);
- flat canonical string, in YAML (``CURRENCY_PER_MONTH_PER_BG``).

The base is ``CURRENCY`` on columns and functions; on parameters it is a
registered concrete currency (``SILVER_PENNY``, ``DM``, …) that also names the
currency the numbers are written in, so the build-time guard can hold it
against the policy date's statutory currency. For dimensionality a concrete
currency means exactly what ``CURRENCY`` means.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, TypeVar

import pint
from pint.util import to_units_container

from ttsim.exceptions import (
    UnitConsistencyError,
    UnitDefinitionError,
)
from ttsim.tt.aggregation import AggType

_PERIODS: tuple[tuple[str, str, str], ...] = (
    ("y", "YEAR", "year"),
    ("q", "QUARTER", "quarter_year"),
    ("m", "MONTH", "month"),
    ("w", "WEEK", "week"),
    ("d", "DAY", "day"),
)
TIME_UNIT_ID_TO_PINT_NAME = {suffix_id: pint for suffix_id, _, pint in _PERIODS}
TIME_UNIT_ID_TO_PERIOD_TOKEN = {suffix_id: token for suffix_id, token, _ in _PERIODS}
_PERIOD_TOKEN_TO_PINT = {token: pint for _, token, pint in _PERIODS}

_QNAME_TIME_SUFFIX_PATTERN = re.compile(
    rf"_(?P<time_unit>[{''.join(TIME_UNIT_ID_TO_PINT_NAME)}])$"
)

CURRENCY_TOKEN = "CURRENCY"  # noqa: S105 (a unit token, not a secret)

_GROUPING_LEVEL_PREFIX = "grouping_level_"

PERSON_LEVEL = "person"

_PER = "_PER_"

_AREA_TOKEN_TO_PINT: dict[str, str] = {
    "SQUARE_METER": "meter ** 2",
    "HOURS": "working_hour",
}

_COMPOSITIONAL_BASE_TO_PINT: dict[str, str | None] = {
    "DIMENSIONLESS": None,
    "PERSON_COUNT": f"{_GROUPING_LEVEL_PREFIX}{PERSON_LEVEL}",
    "HOURS": "working_hour",
    "SQUARE_METER": "meter ** 2",
    "HECTARE": "hectare",
    "YEARS": "delta_calendar_year",
    "MONTHS": "delta_calendar_month",
    "DAYS": "delta_calendar_day",
    "CALENDAR_YEAR": "calendar_year",
    "CALENDAR_MONTH": "calendar_month",
    "CALENDAR_DAY": "calendar_day",
}

#: The grouping levels the fluent builder offers a ``per_<level>`` attribute for
#: (e.g. ``per_bg``). Populated per package by
#: :func:`register_unit_builder_levels`; the generic :meth:`CompositeUnit.PER_LEVEL`
#: works for any level regardless.
_unit_builder_levels: set[str] = set()


@dataclass(frozen=True)
class CompositeUnit:
    """A fully-spelled compositional unit — *the* declaration type.

    A base divided by at most one *physical denominator* (an area or working
    hours), one *period*, and one *level*, held in that canonical order; the
    builder methods enforce the order, so a non-canonical chain
    (``.PER_BG.PER_MONTH``) is a definition error. Two round-tripping spellings
    (via :func:`str`):

    - fluent, off a base (``TTSIMUnit.CURRENCY.PER_MONTH.PER_BG``);
    - flat canonical string, parsed by :func:`parse_compositional_unit`
      (``"CURRENCY_PER_MONTH_PER_BG"``).

    It resolves to a pint unit via :func:`resolve_compositional_unit`.
    """

    base: str
    area: str | None = None
    period: str | None = None
    level: str | None = None

    if TYPE_CHECKING:
        # Per-level builder steps are added at runtime, so tell `ty` any builder
        # attribute yields a CompositeUnit. Runtime keeps strict lookup, so an
        # unregistered `per_<level>` still raises AttributeError.
        def __getattr__(self, name: str) -> CompositeUnit: ...

    def __str__(self) -> str:
        parts = [self.base, self.area, self.period, self.level]
        return _PER.join(part for part in parts if part is not None)

    def __repr__(self) -> str:
        return f"CompositeUnit({self})"

    @property
    def is_flow(self) -> bool:
        """Whether this unit is a flow — i.e. has a period denominator."""
        return self.period is not None

    def _with_area(self, area: str) -> CompositeUnit:
        if self.area is not None or self.period is not None or self.level is not None:
            raise UnitDefinitionError(
                f"Cannot add the physical denominator '{area}' to '{self}': the "
                f"canonical order is base _PER_ <area or hours> _PER_ <period> "
                f"_PER_ <level>, with at most one physical denominator (GEP 10)."
            )
        return replace(self, area=area)

    def _with_period(self, period: str) -> CompositeUnit:
        if self.period is not None or self.level is not None:
            raise UnitDefinitionError(
                f"Cannot add period '{period}' to '{self}': a period must precede "
                f"the level and there is at most one period (GEP 10)."
            )
        return replace(self, period=period)

    def _with_level(self, level: str) -> CompositeUnit:
        if self.level is not None:
            raise UnitDefinitionError(
                f"Cannot add level '{level}' to '{self}': a unit carries at most "
                f"one grouping level (GEP 10)."
            )
        if level.lower() == PERSON_LEVEL:
            raise UnitDefinitionError(
                f"Cannot spell the person leaf on '{self}': the individual level is "
                f"implied, never spelled — drop '_PER_PERSON'. Only group levels are "
                f"spelled (GEP 10)."
            )
        return replace(self, level=level.upper())

    @property
    def PER_SQUARE_METER(self) -> CompositeUnit:  # noqa: N802 (DSL: mirrors the token)
        """This unit per square meter (the area denominator)."""
        return self._with_area("SQUARE_METER")

    @property
    def PER_HOURS(self) -> CompositeUnit:  # noqa: N802 (DSL: mirrors the token)
        """This unit per working hour (``TTSIMUnit.CURRENCY.PER_HOURS``, a wage floor).

        Shares the physical-denominator slot with the area: a price is per at
        most one physical thing.
        """
        return self._with_area("HOURS")

    @property
    def PER_MONTH(self) -> CompositeUnit:  # noqa: N802 (DSL: mirrors the token)
        """This unit per month."""
        return self._with_period("MONTH")

    @property
    def PER_YEAR(self) -> CompositeUnit:  # noqa: N802 (DSL: mirrors the token)
        """This unit per year."""
        return self._with_period("YEAR")

    @property
    def PER_QUARTER(self) -> CompositeUnit:  # noqa: N802 (DSL: mirrors the token)
        """This unit per quarter."""
        return self._with_period("QUARTER")

    @property
    def PER_WEEK(self) -> CompositeUnit:  # noqa: N802 (DSL: mirrors the token)
        """This unit per week."""
        return self._with_period("WEEK")

    @property
    def PER_DAY(self) -> CompositeUnit:  # noqa: N802 (DSL: mirrors the token)
        """This unit per day."""
        return self._with_period("DAY")

    def PER_LEVEL(self, name: str) -> CompositeUnit:  # noqa: N802 (DSL: mirrors token)
        """This unit per grouping level ``name`` (e.g. ``"bg"``)."""
        return self._with_level(name)


def _classify_denominator(token: str) -> str:
    """Classify a denominator token as ``"area"``, ``"period"``, or ``"level"``.

    Area and period are closed vocabularies; everything else is taken to be a
    grouping level, validated against the registered levels at resolution time.
    """
    if token in _AREA_TOKEN_TO_PINT:
        return "area"
    if token in _PERIOD_TOKEN_TO_PINT:
        return "period"
    return "level"


def _is_currency_base(base: str) -> bool:
    """Whether a base token denotes a currency (agnostic or concrete)."""
    if base == CURRENCY_TOKEN:
        return True
    return any(base == name.upper() for name in _registered_currency_names())


def _registered_currency_names() -> set[str]:
    """The concrete currencies registered so far (their pint unit names)."""
    return set(_registered_currencies)


def parse_compositional_unit(spelling: str) -> CompositeUnit:
    """Parse a flat canonical compositional spelling.

    ``"CURRENCY_PER_MONTH_PER_BG"`` → ``CompositeUnit(base="CURRENCY",
    period="MONTH", level="BG")``. The denominators must appear in canonical
    order (``base _PER_ <area> _PER_ <period> _PER_ <level>``) with at most one
    per kind; a non-canonical or repeated denominator is rejected, so there is
    exactly one spelling per unit. The base may be the agnostic
    :data:`CURRENCY_TOKEN`, a registered concrete currency, or any non-currency
    compositional base. A bare base (``DIMENSIONLESS``, ``SILVER_PENNY``) is a
    complete unit with no denominators.

    Raises:
        UnitDefinitionError: If the spelling is empty, names an unknown base, or
            violates the canonical order / one-per-kind rules.
    """
    if not spelling:
        raise UnitDefinitionError("Empty compositional unit spelling (GEP 10).")
    base, *denominators = spelling.split(_PER)
    if base not in _COMPOSITIONAL_BASE_TO_PINT and not _is_currency_base(base):
        raise UnitDefinitionError(
            f"Unknown compositional base {base!r} in {spelling!r}. A base is the "
            f"agnostic '{CURRENCY_TOKEN}', a registered currency, or one of "
            f"{', '.join(sorted(_COMPOSITIONAL_BASE_TO_PINT))} (GEP 10)."
        )
    unit = CompositeUnit(base=base)
    for token in denominators:
        kind = _classify_denominator(token)
        if kind == "area":
            unit = unit._with_area(token)  # noqa: SLF001
        elif kind == "period":
            unit = unit._with_period(token)  # noqa: SLF001
        else:
            unit = unit._with_level(token)  # noqa: SLF001
    return unit


#: Compositional bases that carry a grouping level by default (currency is added
#: separately); every other base is level-less. Booleans also carry a level but are
#: dimensionless, so they ride the ``is_boolean`` flag rather than appear here. See
#: GEP 10 for which quantities carry a level and why.
_LEVEL_CARRYING_BASES: frozenset[str] = frozenset(
    {"PERSON_COUNT", "SQUARE_METER", "HECTARE", "HOURS"}
)


def base_is_level_carrying(base: str) -> bool:
    """Whether a compositional base carries a grouping level by default."""
    return _is_currency_base(base) or base in _LEVEL_CARRYING_BASES


def resolve_compositional_unit(
    unit: CompositeUnit, *, registry: pint.UnitRegistry, with_level: bool = True
) -> pint.Unit:
    """Resolve a compositional unit to its pint unit in ``registry``.

    Each denominator divides the base in turn:

    - a period by its pint period;
    - a physical denominator by its pint unit (``meter ** 2``, ``working_hour``);
    - a level by its grouping-level dimension.

    A currency base resolves to the agnostic :data:`CURRENCY_TOKEN` dimension —
    for dimensionality a concrete currency means exactly what ``CURRENCY`` means
    (the concrete currency drives the build-time conversion, not the check).

    Raises:
        UnitDefinitionError: If a level denominator names a grouping level the
            registry does not define.
    """
    if _is_currency_base(unit.base):
        resolved = registry.parse_units(CURRENCY_TOKEN)
    elif unit.base == "PERSON_COUNT":
        # Via the helper so an un-built registry fails loudly.
        resolved = _grouping_level_unit(name=PERSON_LEVEL, registry=registry)
    else:
        base = _COMPOSITIONAL_BASE_TO_PINT[unit.base]
        resolved = (
            registry.dimensionless if base is None else registry.parse_units(base)
        )
    if unit.area is not None:
        resolved = _divide_by_period(
            non_time_unit=resolved,
            period_pint_name=_AREA_TOKEN_TO_PINT[unit.area],
            registry=registry,
        )
    if unit.period is not None:
        resolved = _divide_by_period(
            non_time_unit=resolved,
            period_pint_name=_PERIOD_TOKEN_TO_PINT[unit.period],
            registry=registry,
        )
    if with_level and unit.level is not None:
        resolved = divide_by_grouping_level(
            unit=resolved, level=unit.level.lower(), registry=registry
        )
    return resolved


def _attach_implied_person_leaf(
    resolved: pint.Unit,
    unit: CompositeUnit,
    registry: pint.UnitRegistry,
    *,
    is_boolean: bool = False,
) -> pint.Unit:
    """Attach the implied ``[person]`` leaf where the vocabulary calls for it.

    An omitted group level means a person property; whether that attaches the
    ``[person]`` leaf is fixed per spelling (GEP 10):

    - a boolean always carries its level;
    - a level-carrying base (currency, area, working hours, the ``[person]``
      count) carries the leaf — unless a *physical denominator* (an area or
      working hours) makes the unit a price or a density, owned by nobody: a
      rent cap (``CURRENCY_PER_SQUARE_METER_PER_MONTH``) or a wage floor
      (``CURRENCY_PER_HOURS``) stays leaf-less so it cancels cleanly against
      the physical quantity;
    - an intensive base (a duration, a share, a calendar point) stays bare.
    """
    if unit.level is not None:
        return resolved
    if is_boolean or (base_is_level_carrying(unit.base) and unit.area is None):
        return divide_by_grouping_level(
            unit=resolved, level=PERSON_LEVEL, registry=registry
        )
    return resolved


def resolve_compositional_column_unit(
    unit: CompositeUnit,
    *,
    time_unit_id: str | None,
    grouping_level: str,
    where: str,
    registry: pint.UnitRegistry,
    is_boolean: bool = False,
) -> pint.Unit:
    """Resolve a column/function's compositional unit, validating the name suffix.

    Two rules tie the unit to the name (GEP 10):

    - the spelled period must match the name's time suffix;
    - the grouping level is **declared, not read off the suffix** — a quantity
      carries a group level iff it is a property of the group as a whole. A
      spelled group level must equal the name's aggregation suffix, but may be
      omitted at *any* suffix.

    An omitted group level makes the column a *person* property: the implied
    person leaf for a level-carrying base (currency, area, working hours, the
    ``[person]`` count) or a boolean, simply level-less for an intensive base (a
    duration, a share, a calendar point). So ``betrag_m_hh`` may declare
    ``CURRENCY_PER_MONTH_PER_HH`` (the household's amount) or ``CURRENCY_PER_MONTH``
    (each member's, constant within the household), never ``…_PER_BG``.

    A concrete-currency base is rejected: a function does not care which
    currency it computes in — its values are denominated in whatever the
    statutory currency of the policy date is — so only parameters and rounding
    specs pin down concrete currencies.

    Raises:
        UnitDefinitionError: If the base pins a concrete currency, or the spelled
            period or level disagrees with the name suffix.
    """
    if token_source_currency(unit) is not None:
        raise UnitDefinitionError(
            f"{where}: a column/function pins the concrete currency {unit.base!r}. "
            f"A function runs in the statutory currency of the policy date, "
            f"whichever that is: declare the agnostic {CURRENCY_TOKEN} and leave "
            f"concrete currencies to parameters and rounding specs (GEP 10)."
        )
    expected_period = (
        TIME_UNIT_ID_TO_PERIOD_TOKEN[time_unit_id] if time_unit_id is not None else None
    )
    if unit.period != expected_period:
        raise UnitDefinitionError(
            f"{where}: the unit spells period {unit.period!r} but the name's time "
            f"suffix implies {expected_period!r}; they must agree (GEP 10)."
        )
    if unit.level is not None and unit.level != grouping_level.upper():
        raise UnitDefinitionError(
            f"{where}: the unit spells level {unit.level!r} but the name's "
            f"aggregation suffix implies {grouping_level.upper()!r}; a spelled "
            f"group level must not contradict the suffix (GEP 10)."
        )
    resolved = resolve_compositional_unit(unit=unit, registry=registry, with_level=True)
    return _attach_implied_person_leaf(
        resolved=resolved, unit=unit, registry=registry, is_boolean=is_boolean
    )


def unit_with_rebased_period(unit: CompositeUnit, time_unit_id: str) -> CompositeUnit:
    """Re-base a flow unit to a new period.

    A time-conversion variant of a flow (``betrag_m`` → ``betrag_y``) carries the
    same quantity per a *different* period, so only the period denominator
    changes: ``CURRENCY_PER_MONTH`` → ``CURRENCY_PER_YEAR``. A non-flow unit (no
    period) is returned unchanged — there is nothing to re-base.
    """
    if unit.period is None:
        return unit
    return replace(unit, period=TIME_UNIT_ID_TO_PERIOD_TOKEN[time_unit_id])


def resolve_compositional_param_unit(
    unit: CompositeUnit,
    *,
    registry: pint.UnitRegistry,
    time_unit_id: str | None = None,
    where: str,
) -> pint.Unit:
    """Resolve a parameter's compositional unit.

    A parameter spells its period and any *group* level (it has no name suffix to
    read them off); an omitted group level is the individual level, exactly as
    for a column:

    - ``SILVER_PENNY_PER_MONTH`` — a per-person amount;
    - ``SILVER_PENNY_PER_FAM`` — a per-family amount;
    - a level-less base (``DIMENSIONLESS``, ``YEARS``) stays level-less.

    A concrete-currency base is allowed — parameters pin the currency their
    numbers are written in. A scalar parameter additionally takes a time suffix
    on its *name*; where one is present the spelled period must agree with it.

    Raises:
        UnitDefinitionError: If a present name time suffix disagrees with the
            spelled period.
    """
    if time_unit_id is not None:
        expected_period = TIME_UNIT_ID_TO_PERIOD_TOKEN[time_unit_id]
        if unit.period != expected_period:
            raise UnitDefinitionError(
                f"{where}: the unit spells period {unit.period!r} but the name's "
                f"time suffix implies {expected_period!r}; they must agree (GEP 10)."
            )
    resolved = resolve_compositional_unit(unit=unit, registry=registry, with_level=True)
    return _attach_implied_person_leaf(resolved=resolved, unit=unit, registry=registry)


def _resolve_agnostic_body_unit(
    unit: CompositeUnit, *, registry: pint.UnitRegistry, where: str, what: str
) -> pint.Unit:
    """Resolve a code-side compositional unit with no name to validate against.

    Shared by the :func:`cast_ttsim_unit` target and a parameter dataclass field
    annotation: both state a full unit whose spelled period and group level stand
    as given, both use the agnostic currency (the concrete denomination lives in
    the parameter YAML), and both resolve like a column declaration.

    Raises:
        UnitDefinitionError: If the base pins a concrete currency.
    """
    if token_source_currency(unit) is not None:
        raise UnitDefinitionError(
            f"{where}: {what} pins the concrete currency {unit.base!r}; declare "
            f"the agnostic {CURRENCY_TOKEN} — only parameters and rounding specs "
            f"pin down concrete currencies (GEP 10)."
        )
    resolved = resolve_compositional_unit(unit=unit, registry=registry, with_level=True)
    return _attach_implied_person_leaf(resolved=resolved, unit=unit, registry=registry)


def resolve_compositional_cast_unit(
    unit: CompositeUnit, *, registry: pint.UnitRegistry, where: str
) -> pint.Unit:
    """Resolve the target unit of a :func:`cast_ttsim_unit` call inside a body."""
    return _resolve_agnostic_body_unit(
        unit=unit, registry=registry, where=where, what="a cast inside a body"
    )


def resolve_compositional_field_unit(
    unit: CompositeUnit, *, registry: pint.UnitRegistry, where: str
) -> pint.Unit:
    """Resolve a parameter dataclass field annotation's compositional unit (GEP 10)."""
    return _resolve_agnostic_body_unit(
        unit=unit, registry=registry, where=where, what="a field annotation"
    )


class _UnitNamespaceMeta(type):
    """Metaclass for :class:`TTSIMUnit` so ``ty`` accepts dynamically-added bases.

    Concrete currency bases (``TTSIMUnit.EUR``, ``TTSIMUnit.DM``,
    ``TTSIMUnit.SILVER_PENNY``)
    are injected onto :class:`TTSIMUnit` by :func:`register_currency` at registration
    time — they cannot be hard-wired class attributes because the currency
    vocabulary is discovered per package. At runtime an injected base is a real
    attribute, so this metaclass adds no ``__getattr__``; under type checking it
    declares one so ``TTSIMUnit.EUR`` type-checks (mirroring
    :class:`CompositeUnit`'s builder-step hint).
    """

    if TYPE_CHECKING:

        def __getattr__(cls, name: str) -> CompositeUnit: ...


class TTSIMUnit(metaclass=_UnitNamespaceMeta):
    """The builder namespace of unit *bases*.

    Each attribute is a bare :class:`CompositeUnit` — ``TTSIMUnit.CURRENCY`` *is*
    ``CompositeUnit(base="CURRENCY")`` — that heads a ``.per_*`` builder chain
    enforcing the canonical order ``base _PER_ <area> _PER_ <period> _PER_
    <level>`` (``TTSIMUnit.CURRENCY.PER_MONTH.PER_BG``).

    The agnostic currency base ``CURRENCY`` lives here permanently; each concrete
    currency base (``EUR``, ``DM``, ``SILVER_PENNY``) is injected by
    :func:`register_currency` when its package registers it, so concrete bases
    can tag a :class:`UnitAnnotatedColumn` of input data (``TTSIMUnit.EUR.PER_MONTH``)
    even though a column/function declaration must stay agnostic.
    """

    CURRENCY = CompositeUnit(base=CURRENCY_TOKEN)
    """An amount of currency (agnostic): wages, claims, benefits, wealth. A
    period denominator makes it a flow (``TTSIMUnit.CURRENCY.PER_MONTH``)."""

    DIMENSIONLESS = CompositeUnit(base="DIMENSIONLESS")
    """A plain dimensionless number: a share, a rate. A boolean declares
    ``DIMENSIONLESS`` too — bare for a person-level indicator, its group level
    spelled for a group one (``TTSIMUnit.DIMENSIONLESS.PER_FAM``)."""

    PERSON_COUNT = CompositeUnit(base="PERSON_COUNT")
    """The individual (leaf) count base — the numerator of a head count
    (``[person]``). With a level denominator it is a head count per group:
    ``TTSIMUnit.PERSON_COUNT.PER_BG`` resolves to ``[person] / [bg]``."""

    HOURS = CompositeUnit(base="HOURS")
    """Working hours (the isolated ``[hours]`` dimension). A period denominator
    re-bases them (``TTSIMUnit.HOURS.PER_WEEK``)."""

    SQUARE_METER = CompositeUnit(base="SQUARE_METER")
    """An area in square meters; also the lone *area* denominator
    (``TTSIMUnit.CURRENCY.PER_SQUARE_METER``)."""

    HECTARE = CompositeUnit(base="HECTARE")
    """An area in hectares: land."""

    YEARS = CompositeUnit(base="YEARS")
    """A *duration* in years: an age, an age threshold. The calendar *point*
    counterpart is :attr:`CALENDAR_YEAR`."""

    MONTHS = CompositeUnit(base="MONTHS")
    """A *duration* in months. The point counterpart is :attr:`CALENDAR_MONTH`."""

    DAYS = CompositeUnit(base="DAYS")
    """A *duration* in days. The point counterpart is :attr:`CALENDAR_DAY`."""

    CALENDAR_YEAR = CompositeUnit(base="CALENDAR_YEAR")
    """A *point* on the calendar measured in years: a birth year, the policy
    year. Two calendar years subtract to a :attr:`YEARS` duration."""

    CALENDAR_MONTH = CompositeUnit(base="CALENDAR_MONTH")
    """A *point* on the calendar measured in months."""

    CALENDAR_DAY = CompositeUnit(base="CALENDAR_DAY")
    """A *point* on the calendar measured in days."""


@dataclass(frozen=True)
class UnitAnnotatedColumn:
    """A column of data paired with the :class:`CompositeUnit` it is measured in.

    The leaf type of the unit-annotated input *and* result trees (GEP 10). On the
    way in, the user hand-authors one per column —
    ``UnitAnnotatedColumn(values=[2000.0, 0.0], unit=TTSIMUnit.EUR.PER_MONTH)`` — and
    a currency column must name a **concrete** currency (``TTSIMUnit.EUR``,
    ``TTSIMUnit.DM``), exactly as a parameter does; the agnostic
    ``TTSIMUnit.CURRENCY`` is rejected at the boundary. On the way out, each result
    leaf is wrapped the same
    way, its ``unit`` the node's resolved unit — a column's in the concrete data
    currency, a parameter's in its statutory currency.

    The wrapper is resolved to bare arrays (and its unit validated) at the
    build-time boundary; it never flows through the numeric runtime.

    Args:
        values: The column data — any leaf the ordinary input tree accepts (a
            list, a numpy/JAX array, a ``pd.Series``); canonicalized downstream.
        unit: The column's compositional unit, built off :class:`TTSIMUnit`.
    """

    values: Any
    unit: CompositeUnit


_CastValueT = TypeVar("_CastValueT")


def cast_ttsim_unit(
    value: _CastValueT,
    unit: str | CompositeUnit,  # noqa: ARG001
) -> _CastValueT:
    """Re-tag ``value`` with ``unit`` for the build-time unit check (GEP 10).

    The expression-level escape hatch of the dry-run. Like ``typing.cast``, it
    is the identity at run time — ``value`` comes back unchanged, scalar or
    column, so the numeric path and JAX tracing are untouched. Only the dry-run
    gives the call meaning: the stand-in flowing through it is re-tagged with the
    stated unit wholesale — dimension, flow period, and grouping level, resolved
    like a declaration (currency-agnostic, the person leaf implied). The rest of
    the body stays checked, and every override is visible at the expression that
    needs it.

    Use it where a single operation is dimensionally irregular but deliberate:

    - policy-mandated cross-level arithmetic (a group extreme against a person
      threshold, a group share times a group total);
    - a granularity conversion on the calendar axes;
    - a genuine dimensioned constant that cannot be promoted to a parameter.

    Args:
        value: The expression to re-tag; returned unchanged.
        unit: The stated unit — built off :class:`TTSIMUnit` or the flat
            compositional spelling.

    Returns:
        ``value``, unchanged.
    """
    return value


#: Sentinel with two readings, told apart by the node type (GEP 10):
#:
#: - on a ``@param_function``, ``unit=UNSET_UNIT`` is the *explicit* declaration
#:   of a structured output (a dataclass of related parameters, a converter-built
#:   schedule) that is not a quantity: the body is not dry-run and consumers
#:   state each plucked value's unit with ``cast_ttsim_unit``;
#: - everywhere else a real unit is mandatory, so the sentinel only survives as a
#:   dataclass field default (the ``unit`` field needs one for field-ordering),
#:   marking an *omitted* declaration that the mandatory-units check reports.
#:
#: It is a :class:`CompositeUnit` so the field type is clean; its base never
#: resolves and it is only ever compared by identity.
UNSET_UNIT: CompositeUnit = CompositeUnit(base="__UNSET__")


def token_is_agnostic_currency(token: CompositeUnit | None) -> bool:
    """Whether a unit is a currency-dimensioned *agnostic* declaration.

    Parameters must not declare these once a concrete currency is registered:
    the declaration names the currency the numbers are written in, so it must
    pin down a concrete currency. A compositional unit is agnostic when its base
    is the agnostic :data:`CURRENCY_TOKEN`.
    """
    return isinstance(token, CompositeUnit) and token.base == CURRENCY_TOKEN


def token_source_currency(token: CompositeUnit | None) -> str | None:
    """The concrete currency a declaration pins down, if any.

    ``None`` for the agnostic base, for non-currency declarations, and for
    dimensionless ones. A compositional unit pins down a currency when its base
    is a registered concrete currency.
    """
    if not isinstance(token, CompositeUnit):
        return None
    return next(
        (name for name in _registered_currency_names() if name.upper() == token.base),
        None,
    )


def replace_concrete_with_agnostic_currency(token: CompositeUnit) -> CompositeUnit:
    """The unit a node *derived* from a source with this unit carries.

    Derived nodes — time-conversion variants and aggregations — are functions,
    and a function never pins down a concrete currency: it runs in the
    statutory currency of the policy date, whichever that is. So a source that
    pins one down (a parameter) hands on the agnostic counterpart of its
    declaration; every other unit passes through unchanged.
    """
    return (
        replace(token, base=CURRENCY_TOKEN)
        if token_source_currency(token) is not None
        else token
    )


def coerce_to_composite_unit(
    value: str | CompositeUnit,
    *,
    where: str,
) -> CompositeUnit:
    """Coerce a YAML ``unit:`` value to a :class:`CompositeUnit`.

    A string is a *compositional* spelling (``CURRENCY_PER_MONTH_PER_BG``,
    ``PERSON_COUNT_PER_BG``, ``SILVER_PENNY_PER_YEAR``, or a bare base
    ``DIMENSIONLESS``) parsed into a :class:`CompositeUnit`; an already-coerced
    :class:`CompositeUnit` passes through. Everything else — pint syntax like
    ``"CURRENCY / year"`` or the former ``"null"`` spelling — is rejected.

    Args:
        value: The raw declaration — a string from YAML or an already-coerced
            unit.
        where: Identifier for error messages (e.g. the parameter's name).

    Raises:
        UnitDefinitionError: If the value is not part of the vocabulary.
    """
    if isinstance(value, CompositeUnit):
        return value
    if isinstance(value, str):
        try:
            return parse_compositional_unit(value)
        except UnitDefinitionError:
            pass
    raise UnitDefinitionError(
        f"{where}: invalid unit declaration {value!r}. A unit must be a "
        f"compositional spelling (e.g. CURRENCY_PER_MONTH_PER_BG, "
        f"PERSON_COUNT_PER_BG), a bare base (e.g. CURRENCY, SILVER_PENNY), or "
        f"DIMENSIONLESS for a "
        f"dimensionless quantity (GEP 10)."
    )


def build_registry() -> pint.UnitRegistry:
    """Create a registry holding the units TTSIM knows about.

    One registry per policy system (:class:`ttsim.tt.currencies.UnitSystem`),
    which then defines its own currencies and grouping levels into it. The
    vocabulary built here is the part every system shares.

    pint's defaults already provide the ``[time]`` units (``year``, ``month``,
    ``week``, ``day`` — with the per-year factors GETTSIM uses: 12, 365.25/7,
    365.25) and ``[length]``/``[area]`` units (``meter``, ``hectare``). We add:

    - ``CURRENCY`` as the reference unit of a new ``[currency]`` dimension;
    - ``working_hour`` as the reference unit of a new ``[hours]`` dimension,
      isolated from pint's ``[time]`` ``hour``: ``working_hour / week``
      is then ``[hours] / [time]`` rather than the bare number ``[time] /
      [time]``, so working hours cannot be confused with — or added to —
      a share, and the only conversion possible is re-basing the *period*
      denominator. pint's ``hour`` is left untouched (``day = 24 · hour``) but
      is not an admissible token;
    - ``quarter_year`` for the ``_q`` suffix (pint's built-in ``quarter`` is a
      unit of mass);
    - ``calendar_year`` / ``calendar_month`` / ``calendar_day`` as affine
      *point* units: a specific year/month/day on the calendar, as
      opposed to a *duration*. pint models a point as an offset unit, whose
      offset must be **non-zero** or pint silently treats it as a plain
      (multiplicative) unit and loses the affine algebra that forbids
      ``point + point``. The epoch is otherwise irrelevant — the dry-run only
      ever uses magnitude ``1.0`` and the runtime path is bare arrays — so we
      pick the 1900-01-01 epoch, aligned across the three axes. Subtracting two
      points yields pint's companion ``delta_calendar_*`` *duration* unit, which
      :attr:`TTSIMUnit.YEARS` / :attr:`TTSIMUnit.MONTHS` /
      :attr:`TTSIMUnit.DAYS` resolve to (each is ratio 1 against
      ``year`` / ``month`` / ``day``).

    pint's remaining built-ins parse, but :func:`parse_unit` rejects every
    token outside :data:`_ALLOWED_UNIT_TOKENS`, so they cannot appear in a
    declaration.
    """
    ureg = pint.UnitRegistry()
    ureg.define(f"{CURRENCY_TOKEN} = [currency]")
    ureg.define("working_hour = [hours]")
    ureg.define("quarter_year = year / 4 = quarter_of_year")
    ureg.define("calendar_year = year; offset: 1900")
    ureg.define("calendar_month = month; offset: 22800")  # 1900 * 12
    ureg.define("calendar_day = day; offset: 693975")  # 1900 * 365.25
    return ureg


#: The dimension names :func:`build_registry` mints for currency and takes from
#: pint for time. Every registry spells them the same way, and a pint
#: dimensionality compares by content, so the boundary helpers pick a unit's
#: currency / flow-period component out by matching against these directly —
#: no registry needed.
_CURRENCY_DIMENSIONALITY: Mapping[str, Any] = {"[currency]": 1}
_TIME_DIMENSIONALITY: Mapping[str, Any] = {"[time]": 1}


def _unit_is_currency(unit: pint.Unit) -> bool:
    """Whether a pint unit is exactly one power of the ``[currency]`` dimension."""
    return dict(unit.dimensionality) == _CURRENCY_DIMENSIONALITY


def _unit_is_time(unit: pint.Unit) -> bool:
    """Whether a pint unit is exactly one power of the ``[time]`` dimension."""
    return dict(unit.dimensionality) == _TIME_DIMENSIONALITY


#: The unit tokens a declaration may combine: TTSIM rejects any unit it does not
#: know about. Each policy system adds its concrete currencies and grouping
#: levels. The set is global and additive: a token is a *name*, and two systems'
#: names are disjoint but for the shared vocabulary, so admitting the union costs
#: only precision in the error message for a token of the wrong system — which
#: the system's own registry then rejects anyway.
#: ``meter`` is admitted for areas (``meter ** 2``).
_ALLOWED_UNIT_TOKENS: set[str] = {
    CURRENCY_TOKEN,
    "year",
    "quarter_year",
    "month",
    "week",
    "day",
    "working_hour",
    "meter",
    "hectare",
    # Calendar-point (affine) units and their companion durations.
    "calendar_year",
    "calendar_month",
    "calendar_day",
    "delta_calendar_year",
    "delta_calendar_month",
    "delta_calendar_day",
}

#: The concrete currencies of every policy system built so far (their pint unit
#: names). A currency is a valid compositional *base* (its upper-cased name); on
#: a parameter the base also names the currency the numbers are written in. Names
#: only — which currencies are *interconvertible* is a per-system question the
#: system's registry answers.
_registered_currencies: set[str] = set()

#: Tolerance for the magnitude part of a unit-equivalence comparison.
_REL_TOL = 1e-9


def _grouping_level_unit_name(name: str) -> str:
    """The internal pint unit name anchoring a grouping level's dimension."""
    return f"{_GROUPING_LEVEL_PREFIX}{name}"


def registered_grouping_levels(registry: pint.UnitRegistry) -> set[str]:
    """The grouping levels a registry defines a dimension for.

    The bare names (``"person"``, ``"hh"``, …). ``person`` — the individual leaf,
    doubling as the ``[person]`` count dimension — is present once any level has
    been registered. The set is discovered per build from the policy
    environment's ``*_id`` columns; ttsim ships no fixed list.
    """
    return {
        name.removeprefix(_GROUPING_LEVEL_PREFIX)
        for name in registry
        if name.startswith(_GROUPING_LEVEL_PREFIX)
    }


def _fail_if_grouping_level_is_unknown(name: str, registry: pint.UnitRegistry) -> None:
    """Reject a grouping level the registry defines no dimension for."""
    if _grouping_level_unit_name(name) not in registry:
        known = (
            ", ".join(sorted(registered_grouping_levels(registry)))
            or "(none registered)"
        )
        raise UnitDefinitionError(
            f"Unknown grouping level {name!r}; expected one of {known}. Grouping "
            f"levels are discovered per build from the `*_id` columns and "
            f"registered via register_grouping_levels (GEP 10)."
        )


def _grouping_level_unit(name: str, registry: pint.UnitRegistry) -> pint.Unit:
    """The pint unit of a registered grouping level.

    Raises:
        UnitDefinitionError: If the level has not been registered.
    """
    _fail_if_grouping_level_is_unknown(name=name, registry=registry)
    return registry.parse_units(_grouping_level_unit_name(name))


def divide_by_grouping_level(
    unit: pint.Unit, level: str, registry: pint.UnitRegistry
) -> pint.Unit:
    """Return ``unit`` divided by a grouping level's unit.

    A leveled quantity carries its level as a denominator, exactly as a flow
    carries its period as one: ``CURRENCY / month`` at level ``hh`` becomes
    ``CURRENCY / month / [hh]``, and at the individual level ``person`` it becomes
    ``CURRENCY / month / [person]``. The ``[person]`` denominator of a person-level
    quantity is the same dimension as a head count's ``[person]`` numerator, so the
    two cancel.

    The division is *unit* arithmetic, not quantity arithmetic: attaching a level
    is index bookkeeping, so it must stay legal for offset (calendar-point) bases,
    which pint forbids in quantity multiplication.

    Raises:
        UnitDefinitionError: If the level has not been registered.
    """
    return unit / _grouping_level_unit(name=level, registry=registry)


def grouping_level_count_unit(
    target_level: str, registry: pint.UnitRegistry
) -> pint.Unit:
    """The unit of a head count over ``target_level``.

    A head count is the ``[person]`` *count* dimension over the group it counts
    within: counting persons in a household is ``[person] / [hh]`` — persons per
    household. ``COUNT`` aggregations mint this, and it is the conversion factor
    that bridges levels (``[person]/[hh] · CURRENCY/[hh] = CURRENCY/[person]``).

    Raises:
        UnitDefinitionError: If ``person`` or ``target_level`` is not registered.
    """
    return _grouping_level_unit(
        name=PERSON_LEVEL, registry=registry
    ) / _grouping_level_unit(name=target_level, registry=registry)


def parse_unit(unit_str: str, registry: pint.UnitRegistry) -> pint.Unit:
    """Parse a pint unit string, enforcing the closed pint-token vocabulary.

    Internal: declarations are :class:`CompositeUnit`\\ s, never pint syntax. This
    parser serves the internal pint surfaces — input tags, the framework date
    nodes, and the resolution machinery.

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
        unit = registry.parse_units(unit_str)
    except (pint.errors.PintError, AssertionError, ValueError, TypeError) as e:
        raise UnitDefinitionError(f"Could not parse unit {unit_str!r}: {e}") from e
    _fail_if_unit_tokens_are_unknown(unit=unit, unit_str=unit_str)
    if not to_units_container(unit):
        raise UnitDefinitionError(
            f"TTSIMUnit {unit_str!r} resolves to the dimensionless unit. A "
            f"dimensionless quantity (a share, a rate, a head count) declares "
            f"`DIMENSIONLESS` (GEP 10)."
        )
    return unit


def _fail_if_unit_tokens_are_unknown(
    unit: pint.Unit,
    unit_str: str,
) -> None:
    """Reject any unit token TTSIM does not know about.

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
            f"TTSIMUnit {unit_str!r} involves unit token(s) TTSIM does not know "
            f"about: {', '.join(offending)}. Known units are "
            f"{', '.join(sorted(_ALLOWED_UNIT_TOKENS))} (GEP 10)."
        )


def _divide_by_period(
    non_time_unit: pint.Unit, period_pint_name: str, registry: pint.UnitRegistry
) -> pint.Unit:
    """Return ``non_time_unit / period`` as a pint unit."""
    period = registry.Quantity(1.0, period_pint_name)
    return (registry.Quantity(1.0, non_time_unit) / period).units


def units_are_equivalent(
    left: pint.Unit, right: pint.Unit, registry: pint.UnitRegistry
) -> bool:
    """Whether two units are interchangeable on a DAG edge.

    Equivalent iff they share a dimensionality *and* a magnitude (their ratio is
    dimensionless and equal to 1) — stricter than pint's compatibility:
    ``euro / month`` and ``euro / year`` are both ``[currency] / [time]`` but are
    *not* equivalent (ratio 12), so a monthly node feeding a yearly consumer is
    caught.

    Two wrinkles:

    - the base currency is factor 1 against the :data:`CURRENCY_TOKEN` reference,
      so a ``"CURRENCY"`` declaration is equivalent to a value inferred in the
      base currency; cross-currency magnitudes (e.g. ``DM``) are reconciled by
      the build-time currency conversion before this check runs;
    - a calendar-point (affine offset) unit cannot be divided, so such units are
      compared by *identity*, not magnitude: two ``calendar_year`` points are
      equivalent, but a ``calendar_year`` point is *not* equivalent to a
      ``year`` / ``delta_calendar_year`` duration nor to a ``calendar_month``
      point on another axis.
    """
    left_quantity = registry.Quantity(1.0, left)
    right_quantity = registry.Quantity(1.0, right)
    if left_quantity.dimensionality != right_quantity.dimensionality:
        return False
    try:
        ratio = (left_quantity / right_quantity).to_reduced_units()
    except pint.OffsetUnitCalculusError:
        return left == right
    return math.isclose(ratio.magnitude, 1.0, rel_tol=_REL_TOL)


def is_calendar_point_unit(unit: pint.Unit, registry: pint.UnitRegistry) -> bool:
    """Whether a resolved unit is an affine calendar *point*.

    A calendar point (``calendar_year`` and its month/day siblings) is a pint
    offset unit: it obeys affine algebra, not the magnitude algebra of a
    duration. pint raises an :class:`pint.OffsetUnitCalculusError` on any illegal
    operation, so:

    - two points *subtract* to a duration, and a duration *shifts* a point;
    - two points cannot be added, a point cannot be scaled, and points on
      different calendar axes cannot be combined.

    Callers that implement the affine ``+``/``-`` rules (the build-time dry-run)
    detect a point this way and delegate the operation to pint rather than to the
    magnitude-equivalence check, which would wrongly reject the valid
    ``point + duration``. Detection is by the very property that defines an offset
    unit: it cannot be divided by itself.
    """
    quantity = registry.Quantity(1.0, unit)
    try:
        quantity / quantity
    except pint.OffsetUnitCalculusError:
        return True
    return False


def _currency_component_of(
    units: pint.Unit, registry: pint.UnitRegistry
) -> pint.Unit | None:
    """Return the currency component of a (possibly composite) unit, or ``None``.

    Used at the input boundary to convert a pint-tagged column's currency to the
    data currency while leaving its period and area untouched: e.g. the ``DM`` in
    ``DM / month``.
    """
    for token in to_units_container(units):
        candidate = registry.parse_units(token)
        if _unit_is_currency(candidate):
            return candidate
    return None


def unit_has_currency_component(units: pint.Unit, registry: pint.UnitRegistry) -> bool:
    """Whether a (possibly composite) unit carries a currency component."""
    return _currency_component_of(units=units, registry=registry) is not None


def unit_has_agnostic_currency_component(
    units: pint.Unit, registry: pint.UnitRegistry
) -> bool:
    """Whether a unit's currency component is the agnostic ``CURRENCY`` token.

    Distinguishes the two currency spellings when results are returned: a
    column resolves to the agnostic ``CURRENCY`` (it is computed in the
    computation currency and converted to the data currency), a parameter to
    its concrete statutory currency (never converted, labelled as declared) —
    GEP 10.
    """
    component = _currency_component_of(units=units, registry=registry)
    return component is not None and component == registry.parse_units(CURRENCY_TOKEN)


#: The dimensionality-key prefix of a grouping-level dimension: the internal pint
#: unit name :data:`_GROUPING_LEVEL_PREFIX` wrapped in pint's ``[…]`` dimension
#: brackets (e.g. ``[grouping_level_hh]``).
_GROUPING_LEVEL_DIM_PREFIX = f"[{_GROUPING_LEVEL_PREFIX}"


def _grouping_levels_with_exponent(unit: pint.Unit) -> Iterator[tuple[str, Any]]:
    """Yield ``(level_name, exponent)`` for each grouping-level dimension of a unit.

    A negative exponent is a denominator level (``/[hh]``), a positive one a
    numerator level (a ``[person]`` head count). Non-grouping dimensions and
    pint's (never-occurring here) complex exponents are skipped.
    """
    for dimension, exponent in unit.dimensionality.items():
        if isinstance(exponent, complex):  # pint exponents are real; narrow for ty
            continue
        if dimension.startswith(_GROUPING_LEVEL_DIM_PREFIX):
            yield dimension[len(_GROUPING_LEVEL_DIM_PREFIX) : -1], exponent


def _unit_without_grouping_levels(
    unit: pint.Unit, registry: pint.UnitRegistry
) -> pint.Unit:
    """A unit with every grouping level (numerator and denominator) divided out.

    Stripping a level is index bookkeeping on the unit's container, not quantity
    arithmetic, so it stays legal for offset (calendar-point) bases.
    """
    container = to_units_container(unit)
    level_keys = [k for k in container if k.startswith(_GROUPING_LEVEL_PREFIX)]
    return registry.Unit(container.remove(level_keys))


def _unit_level_denominator(unit: pint.Unit) -> str | None:
    """The grouping level a resolved unit carries as a denominator.

    A leveled quantity carries its level as a ``/[level]`` denominator (negative
    exponent in the dimensionality), exactly as a flow carries its period. Returns
    the level name (``"hh"``, ``"person"``, …) found in the denominator, or
    ``None`` for a level-less unit. A head count's ``[person]`` *numerator*
    (positive exponent) is not a denominator level and is ignored, so a
    ``[person]/[hh]`` count reports ``"hh"`` — its index level.
    """
    return next(
        (
            name
            for name, exponent in _grouping_levels_with_exponent(unit)
            if exponent < 0
        ),
        None,
    )


def _substitute_currency(
    units: pint.Unit, currency: str, registry: pint.UnitRegistry
) -> pint.Unit:
    """Swap a unit's currency component for ``currency``; a no-op if it has none.

    The one currency move input and output handling share: the period, area
    and levels are left untouched. For results ``currency`` is the data
    currency (:func:`output_unit_in_data_currency`); for tagged input data it
    is the tag's concrete currency (:func:`input_strip_unit`).
    """
    component = _currency_component_of(units=units, registry=registry)
    if component is None:
        return units
    return units / component * registry.parse_units(currency)


def input_target_unit_in_data_currency(
    units: pint.Unit, data_currency: str, registry: pint.UnitRegistry
) -> pint.Unit:
    """Restate the unit of an input column requested as a target.

    Such a column is returned exactly as provided — in the data currency —
    whatever currency its declaration pins down: agnostic (an ordinary column)
    or concrete (a data override of a parameter, declared ``DM_PER_MONTH`` but
    holding the user's euro values). The label follows the value, so any
    currency component is substituted with the data currency (GEP 10).
    """
    return _substitute_currency(units=units, currency=data_currency, registry=registry)


def output_unit_in_data_currency(
    units: pint.Unit, data_currency: str, registry: pint.UnitRegistry
) -> pint.Unit:
    """Restate a computed result column's resolved unit in the data currency.

    A computed column is converted to the data currency before being returned,
    so its label swaps the agnostic ``CURRENCY`` component for it
    (``CURRENCY / month`` → ``euro / month``) while period and area are left
    untouched. A unit with no agnostic currency component — ``year``,
    ``hectare``, dimensionless — is returned unchanged. Requested parameters do
    not pass through here: their value is never converted, so they keep their
    statutory currency (:func:`param_unit_in_computation_currency`).
    """
    if not unit_has_agnostic_currency_component(units=units, registry=registry):
        return units
    return _substitute_currency(units=units, currency=data_currency, registry=registry)


def param_unit_in_computation_currency(
    units: pint.Unit, computation_currency: str, registry: pint.UnitRegistry
) -> pint.Unit:
    """Restate a requested parameter's resolved unit in its statutory currency.

    A parameter keeps its statutory value — never converted to the data
    currency — and that value is denominated in the computation currency (the
    build guard pins every parameter's currency to it). So the resolved
    agnostic ``CURRENCY`` component is spelled in the computation currency, not
    the data currency; a non-currency unit is returned unchanged (GEP 10).
    """
    return _substitute_currency(
        units=units, currency=computation_currency, registry=registry
    )


#: Reverse of the forward token→pint maps, for :func:`composite_from_resolved_unit`.
_PINT_NAME_TO_PERIOD_TOKEN = {v: k for k, v in _PERIOD_TOKEN_TO_PINT.items()}
_PINT_NAME_TO_BASE_TOKEN = {
    pint_name: token
    for token, pint_name in _COMPOSITIONAL_BASE_TO_PINT.items()
    if pint_name is not None
}


def composite_from_resolved_unit(
    units: pint.Unit, registry: pint.UnitRegistry
) -> CompositeUnit:
    """Reconstruct the compositional spelling of a *resolved* pint unit.

    The output-side inverse of :func:`resolve_compositional_unit`: it labels a
    result-tree leaf with a :class:`CompositeUnit`, so the result tree is the
    same shape as the input tree (GEP 10). A resolved unit obeys the grammar, so
    each component maps back to one slot:

    - the currency / count / area / duration numerator → the base;
    - the flow period → the period;
    - the spelled group level → the level.

    The implied person leaf (a ``grouping_level_person`` denominator) is dropped,
    as a spelling never spells it; a person-leaf *numerator* is the
    ``PERSON_COUNT`` base of a head count.

    Apply it to a unit already restated in the data currency
    (:func:`output_unit_in_data_currency`) so the base is a concrete currency
    (``EUR``), never the agnostic ``CURRENCY``.
    """
    currency = _currency_component_of(units=units, registry=registry)
    period = _flow_period_of(units=units, registry=registry)
    base = str(currency).upper() if currency is not None else "DIMENSIONLESS"
    area: str | None = None
    level: str | None = None
    for name, exponent in _grouping_levels_with_exponent(units):
        if name == PERSON_LEVEL and exponent > 0:
            base = "PERSON_COUNT"
        elif name != PERSON_LEVEL and exponent < 0:
            level = name.upper()
        # A `grouping_level_person` denominator is the implied leaf — dropped.
    physical = to_units_container(
        _unit_without_grouping_levels(unit=units, registry=registry)
    )
    for token, exponent in physical.items():
        if isinstance(exponent, complex):  # pint exponents are real; narrow for ty
            continue
        if token == "meter":  # noqa: S105 (a pint unit token, not a secret)
            base = "SQUARE_METER" if exponent > 0 and currency is None else base
            area = "SQUARE_METER" if exponent < 0 else area
        elif token == "working_hour" and exponent < 0:  # noqa: S105
            area = "HOURS"
        elif currency is None and exponent > 0 and token in _PINT_NAME_TO_BASE_TOKEN:
            base = _PINT_NAME_TO_BASE_TOKEN[token]
    return CompositeUnit(
        base=base,
        area=area,
        period=_PINT_NAME_TO_PERIOD_TOKEN[str(period)] if period is not None else None,
        level=level,
    )


def _flow_period_of(units: pint.Unit, registry: pint.UnitRegistry) -> pint.Unit | None:
    """Return a unit's flow period — its time component in the *denominator*.

    The ``month`` of ``CURRENCY / month``, the ``week`` of ``working_hour /
    week``. A
    *numerator* time unit (the ``year`` of an age, ``TTSIMUnit.YEARS``) is not a flow
    period and is ignored, so an intrinsically-temporal column is not mistaken
    for a flow. Returns ``None`` for a unit with no per-period part.
    """
    for token, exponent in to_units_container(units).items():
        if isinstance(exponent, complex):  # pint exponents are real; narrow for ty
            continue
        candidate = registry.parse_units(token)
        if exponent < 0 and _unit_is_time(candidate):
            return candidate
    return None


def unit_residual_excluding_currency_and_flow_period(
    units: pint.Unit, registry: pint.UnitRegistry
) -> pint.Unit:
    """A unit's *measurement* residual: currency, flow period, and levels removed.

    The input check screens measurement (the numerator scale — area, intrinsic
    time, plain counts) on its own axis, leaving the other three to the boundary:
    the currency is converted at the boundary, the flow period is screened
    against the column's name suffix, and the **grouping level** is screened
    against the column's declared level (the level is declared, not read off
    the suffix — GEP 10). So this
    divides out all three and the input check compares the residuals of a tag and
    its declared unit for equivalence rather than mere dimensionality (a
    ``HECTARE`` column tagged ``m²`` shares the area dimension but is a
    10,000-fold level error).
    """
    currency = _currency_component_of(units=units, registry=registry)
    residual = units / currency if currency is not None else units
    period = _flow_period_of(units=residual, registry=registry)
    residual = residual * period if period is not None else residual
    return _unit_without_grouping_levels(unit=residual, registry=registry)


def _suffix_period_of(
    column_label: str | None, registry: pint.UnitRegistry
) -> pint.Unit | None:
    """Return the flow period named by a column's GEP-1 time suffix.

    ``…_m`` → ``month``; a name with no time suffix → ``None``.
    """
    if column_label is None:
        return None
    match = _QNAME_TIME_SUFFIX_PATTERN.search(column_label)
    if match is None:
        return None
    return registry.parse_units(TIME_UNIT_ID_TO_PINT_NAME[match.group("time_unit")])


def _fail_if_tag_period_disagrees_with_suffix(
    units: pint.Unit, *, column_label: str | None, registry: pint.UnitRegistry
) -> None:
    """Strict period guard: a pint tag's flow period must match the column's
    GEP-1 time suffix exactly — including both absent.

    A ``_m`` column needs a ``/month`` tag; an unsuffixed column needs a tag with
    no period. This catches a contradictory period that would otherwise be
    stripped silently (e.g. a ``_m`` column tagged ``DM / year`` — a 12-fold error).
    """
    tag_period = _flow_period_of(units=units, registry=registry)
    suffix_period = _suffix_period_of(column_label=column_label, registry=registry)
    matches = (
        tag_period is None
        if suffix_period is None
        else tag_period is not None
        and units_are_equivalent(
            left=tag_period, right=suffix_period, registry=registry
        )
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


def input_strip_unit(unit: CompositeUnit, registry: pint.UnitRegistry) -> pint.Unit:
    """The concrete pint unit used to strip a :class:`UnitAnnotatedColumn`.

    Resolves the tag with its concrete currency and flow period — the two axes the
    boundary acts on: the currency is converted at the boundary and the period
    is screened against the name suffix. Grouping levels do not affect the
    magnitude and are omitted, so this needs no registered level dimension.
    """
    resolved = resolve_compositional_unit(
        unit=unit, registry=registry, with_level=False
    )
    concrete = token_source_currency(unit)
    return (
        resolved
        if concrete is None
        else _substitute_currency(units=resolved, currency=concrete, registry=registry)
    )


def strip_input_quantity_at_boundary(
    quantity: Any,  # noqa: ANN401 (a pint Quantity wrapping an input column)
    *,
    data_currency: str,
    registry: pint.UnitRegistry,
    column_label: str | None = None,
) -> Any:  # noqa: ANN401
    """Convert a pint-tagged input column to the data currency, then strip it.

    A user *may* attach a pint ``Quantity`` to an input column. At the boundary:

    - the tag may only combine units TTSIM knows about;
    - its flow period must match the column's GEP-1 time suffix exactly (a
      ``_m`` column needs a ``/month`` tag; an unsuffixed column a tag with no
      period);
    - its currency component is *converted* to the data currency — the tag
      overrides the blanket "untagged data is in the data currency" assumption
      per column, so a DM-tagged column can ride along EUR data — while period
      and area are left untouched; a tag already in the data currency, or with
      no currency component, is stripped unchanged.

    The bare magnitude is returned. The uniform crossing from the data currency
    into the computation currency happens later, in ``processed_data``, for
    tagged and untagged columns alike (GEP 10).

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
    _fail_if_tag_period_disagrees_with_suffix(
        units=quantity.units, column_label=column_label, registry=registry
    )
    source_currency = _currency_component_of(units=quantity.units, registry=registry)
    if source_currency is None:
        return quantity.magnitude
    data_currency_unit = registry.parse_units(data_currency)
    if source_currency == data_currency_unit:
        return quantity.magnitude
    target = quantity.units / source_currency * data_currency_unit
    return quantity.to(target).magnitude


def head_count_from_boolean_sum(
    agg_type: AggType, *, source_is_boolean: bool
) -> AggType:
    """Normalise a ``SUM`` over a boolean to a ``COUNT`` for unit purposes.

    Summing a boolean counts the persons its flag is true for, so its unit is a
    head count's — the same :attr:`TTSIMUnit.PERSON_COUNT` a ``COUNT`` mints. Every
    other aggregation keeps its own type. This is the single source of truth used
    by both the declared-token minter (:func:`unit_for_aggregation`) and the
    resolved-unit deriver
    (:func:`ttsim.interface_dag_elements.unit_checks._resolve_agg_by_group_unit`),
    so the two cannot drift.
    """
    if agg_type is AggType.SUM and source_is_boolean:
        return AggType.COUNT
    return agg_type


def unit_for_aggregation(
    source_unit: CompositeUnit,
    agg_type: AggType,
    target_level: str = PERSON_LEVEL,
    *,
    source_is_boolean: bool = False,
) -> CompositeUnit:
    """Auto-assign the *declared* unit of an aggregation node.

    The single source of truth for an automatically added aggregation's token
    (``my_col`` → ``my_col_hh``); author-written ``@agg_by_group_function`` /
    ``@agg_by_p_id_function`` nodes declare their unit explicitly (GEP 10):

    - a **head count** — ``COUNT``, or a ``SUM`` over a boolean source
      (``source_is_boolean``, counting the persons its flag is true for) — is the
      ``[person]`` count base at its target level: :attr:`TTSIMUnit.PERSON_COUNT` per
      ``target_level``;
    - ``SUM`` / ``MIN`` / ``MAX`` over a non-boolean source are properties of the
      **target** group whatever the source's base (GEP 10): they keep the
      source's physical token and take the target level — spelled for a group
      (``CURRENCY_PER_MONTH_PER_FAM``, ``MONTHS_PER_FG`` for an ``_fg`` extreme of
      a bare duration), stripped for the person-level result of an
      ``agg_by_p_id`` (the person leaf is implied, never spelled);
    - ``MEAN`` is the exception — a per-head average belongs to the **person**
      (``MEAN = SUM / COUNT`` cancels the group), so it takes the individual
      spelling: the source token with any group level stripped;
    - ``ANY`` / ``ALL`` yield a boolean, a leveled dimensionless quantity at the
      target level: bare :attr:`TTSIMUnit.DIMENSIONLESS` for an individual result,
      ``DIMENSIONLESS_PER_<target_level>`` for a group one.

    A ``PERSON_COUNT`` head count at the individual :data:`PERSON_LEVEL` (an
    ``agg_by_p_id`` ``COUNT``) is the bare :attr:`TTSIMUnit.PERSON_COUNT`, which
    resolves to ``[person] / [person]`` = dimensionless.

    Args:
        source_unit: The source column's ``unit`` — a :class:`CompositeUnit`
            (:data:`UNSET_UNIT` if the source does not declare one).
        agg_type: The :class:`ttsim.tt.aggregation.AggType` of the aggregation.
        target_level: The group level the node aggregates to (read off its name
            suffix); :data:`PERSON_LEVEL` for an individual-level result.
        source_is_boolean: Whether the aggregated source column is boolean — a
            ``SUM`` over it is then minted as a head count (see
            :func:`head_count_from_boolean_sum`).

    Returns:
        The auto-assigned unit. ``PERSON_COUNT_PER_<target_level>`` for a ``COUNT`` head
        count (the bare :attr:`TTSIMUnit.PERSON_COUNT` at the individual level),
        ``DIMENSIONLESS_PER_<target_level>`` for a boolean ``ANY`` / ``ALL`` result
        (bare :attr:`TTSIMUnit.DIMENSIONLESS` at the individual level); otherwise the
        source token at the target (``SUM`` / ``MIN`` / ``MAX``) or individual
        (``MEAN``) level (:data:`UNSET_UNIT` when the source itself lacks a
        declaration, which the mandatory-units check then reports against the
        source).
    """
    agg_type = head_count_from_boolean_sum(
        agg_type=agg_type, source_is_boolean=source_is_boolean
    )
    if agg_type is AggType.COUNT:
        return (
            TTSIMUnit.PERSON_COUNT
            if target_level == PERSON_LEVEL
            else TTSIMUnit.PERSON_COUNT.PER_LEVEL(target_level)
        )
    if agg_type in (AggType.ANY, AggType.ALL):
        return (
            TTSIMUnit.DIMENSIONLESS
            if target_level == PERSON_LEVEL
            else TTSIMUnit.DIMENSIONLESS.PER_LEVEL(target_level)
        )
    if source_unit is UNSET_UNIT:
        return source_unit
    if agg_type is AggType.MEAN or target_level == PERSON_LEVEL:
        return replace(source_unit, level=None)
    return replace(source_unit, level=target_level.upper())


def resolved_unit_for_aggregation(
    *,
    agg_type: AggType,
    target_level: str,
    registry: pint.UnitRegistry,
    source_unit: pint.Unit | None = None,
    source_level: str | None = None,
) -> pint.Unit:
    """The resolved unit of an aggregation node, level-aware.

    The level-aware counterpart of :func:`unit_for_aggregation`: it operates on
    *resolved* pint units (the physical token combined with its flow period and
    grouping level) and is where a grouping level is minted, swapped, or
    preserved. ``target_level`` is the group being aggregated *to* (read off the
    aggregation suffix); ``source_level`` is the source column's own level
    (``None`` for a level-less source such as an age).

    - ``SUM`` / ``MIN`` / ``MAX`` results are properties of the **target** group
      whatever the source's base (GEP 10): the source level (if any) is swapped
      for the target and a level-less source *acquires* it — an ``_hh`` sum of a
      person income is ``CURRENCY/[hh]``, an ``_fg`` min of a bare age is
      ``MONTHS/[fg]``. At the person target (an ``agg_by_p_id`` node) the result
      is a person property: implied person leaf for a leveled source, bare for a
      level-less one.
    - ``MEAN`` is the exception: a per-head average belongs to the **person**
      (``MEAN = SUM / COUNT``, and leveling it to the target would break
      ``mean · count = sum``), so the source level is swapped for the person
      leaf; a level-less source and the dimensionless truth value of a boolean
      (whose mean is a share) stay bare.
    - ``COUNT`` mints a head count ``[person] / [target]`` — persons per target
      group — independent of the source.
    - ``ANY`` / ``ALL`` yield a boolean *at the target level* — ``1 / [target]``
      — so a group-level indicator carries the level its name claims.

    ``COUNT`` and ``ANY`` / ``ALL`` are independent of the source, so their
    ``source_unit`` / ``source_level`` default to ``None`` and are ignored.

    Args:
        agg_type: The :class:`ttsim.tt.aggregation.AggType` of the aggregation.
        target_level: The group level being aggregated to (e.g. ``"hh"``).
        source_unit: The source column's resolved pint unit. Required for the
            value aggregations ``SUM`` / ``MEAN`` / ``MIN`` / ``MAX``; ignored
            (and ``None``) for ``COUNT`` / ``ANY`` / ``ALL``.
        source_level: The source column's grouping level (e.g. ``"person"``), or
            ``None`` if the source carries no level (or is ignored, as above).

    Returns:
        The aggregation node's resolved pint unit.

    Raises:
        UnitDefinitionError: If ``target_level`` or ``source_level`` names an
            unregistered grouping level.
        ValueError: If a value aggregation is requested without a ``source_unit``.
    """
    if agg_type in (AggType.ANY, AggType.ALL):
        return divide_by_grouping_level(
            unit=registry.dimensionless, level=target_level, registry=registry
        )
    if agg_type is AggType.COUNT:
        return grouping_level_count_unit(target_level=target_level, registry=registry)
    if source_unit is None:
        msg = (
            f"A value aggregation ({agg_type}) needs a source_unit; only "
            f"COUNT / ANY / ALL are source-independent."
        )
        raise ValueError(msg)
    # Cancel the source's own level first (unit arithmetic — offset-safe for
    # calendar-point bases) so the bare base can take the result's level.
    stripped = (
        source_unit
        if source_level is None
        else source_unit * _grouping_level_unit(name=source_level, registry=registry)
    )
    if agg_type is AggType.MEAN:
        if source_level is None or not to_units_container(stripped):
            return stripped
        return divide_by_grouping_level(
            unit=stripped, level=PERSON_LEVEL, registry=registry
        )
    if target_level == PERSON_LEVEL and source_level is None:
        return stripped
    return divide_by_grouping_level(
        unit=stripped, level=target_level, registry=registry
    )


def fail_if_units_are_missing(
    units_by_qname: Mapping[str, CompositeUnit],
) -> None:
    """Data-independent check that every node declares a unit.

    A missing unit is a definition error. :attr:`TTSIMUnit.DIMENSIONLESS` is *not*
    missing — it declares a dimensionless quantity; a node without any declaration
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
            f"(GEP 10; declare `unit=TTSIMUnit.DIMENSIONLESS` / `unit: DIMENSIONLESS` "
            f"for a dimensionless quantity): {', '.join(missing)}."
        )
