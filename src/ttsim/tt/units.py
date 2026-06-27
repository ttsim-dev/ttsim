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

Compositional units (GEP 10)
----------------------------

Every declaration is a fully-spelled *compositional* unit
(:class:`CompositeUnit`): a base optionally divided by an area, a period, and a
grouping level, in that canonical order. Code authors build one fluently off
the :class:`Unit` namespace (``Unit.CURRENCY.PER_MONTH.PER_BG``); YAML spells
the flat canonical string (``CURRENCY_PER_MONTH_PER_BG``), and the two
round-trip through :func:`parse_compositional_unit` and :func:`str`. A bare base
is a complete unit in its own right (``Unit.CURRENCY`` ==
``CompositeUnit(base="CURRENCY")``, ``unit: DIMENSIONLESS``).

The base is currency-agnostic ``CURRENCY`` on columns and functions — they are
currency-agnostic by design — and a registered *concrete* currency
(``SILVER_PENNY``, ``DM``, …) on parameters, which additionally names the
currency the numbers are written in so the build-time conversion to the run
currency can read it off. For dimensionality a concrete currency means exactly
what ``CURRENCY`` means.

The :data:`CURRENCY_TOKEN` (the literal string ``"CURRENCY"``) is a real unit
anchoring the ``[currency]`` dimension, so a currency unit resolves regardless
of whether a concrete currency has been registered yet — checks compare at the
dimensionality level and the concrete currency is resolved separately
(issue #120).

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

import math
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

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


#: The pint unit anchoring the ``[currency]`` dimension, used internally to
#: resolve a currency unit before any concrete currency is registered. Checks
#: compare at the dimensionality level; the concrete currency is resolved
#: separately.
CURRENCY_TOKEN = "CURRENCY"  # noqa: S105 (a unit token, not a secret)

#: The internal pint-unit-name prefix for a grouping-level dimension (GEP 10).
#: A grouping level (``hh``, ``bg``, … and the individual ``person``) is its own
#: base dimension; its pint unit and dimension are prefixed so a bare level name
#: cannot collide with a pint built-in (``hour``) or a currency token: ``hh``
#: alone is pint's hour unit, ``grouping_level_hh`` is free.
_GROUPING_LEVEL_PREFIX = "grouping_level_"

#: The individual (leaf) grouping level — the entity identified by ``p_id``
#: (GEP 10). It always exists and doubles as the ``[person]`` *count* dimension:
#: a person-level quantity's ``/[person]`` denominator and a head count's
#: ``[person]`` numerator are the same dimension, so they cancel.
PERSON_LEVEL = "person"


# ===========================================================================
# Compositional units (GEP 10)
# ===========================================================================
#
# A unit is ``base (_PER_ <denominator>)*`` with denominators in a fixed
# canonical order ``base _PER_ <area> _PER_ <period> _PER_ <level>``. The flat
# string (``CURRENCY_PER_MONTH_PER_BG``) is what YAML declares and what
# :func:`str` of a built unit produces, so the fluent ``.py`` builder
# (``Unit.CURRENCY.PER_MONTH.PER_BG``) and the YAML spelling round-trip through
# the same parser and formatter. Every compositional unit resolves to a pint
# unit: a period divides by its pint period, an area by ``meter ** 2``, a level
# by its grouping-level dimension.

#: The separator joining a base and its denominators in the flat spelling.
_PER = "_PER_"

#: The closed *period* denominators and the pint unit each names. Having a
#: period denominator is what makes a unit a *flow* (GEP 10).
_PERIOD_TOKEN_TO_PINT: dict[str, str] = {
    "MONTH": "month",
    "YEAR": "year",
    "QUARTER": "quarter_year",
    "WEEK": "week",
    "DAY": "day",
}

#: The closed *area* denominators (physical) and the pint unit each names.
_AREA_TOKEN_TO_PINT: dict[str, str] = {
    "SQUARE_METER": "meter ** 2",
}

#: The non-currency compositional *bases* and the pint base each resolves to
#: (``None`` is the dimensionless base). The currency bases — agnostic
#: :data:`CURRENCY_TOKEN` and the registered concrete currencies — are handled
#: separately because they share the ``[currency]`` dimension. ``PERSON``
#: resolves to the ``[person]`` leaf level (registered with the grouping levels),
#: so it is also handled separately in :func:`resolve_compositional_unit`.
_COMPOSITIONAL_BASE_TO_PINT: dict[str, str | None] = {
    "DIMENSIONLESS": None,
    "PERSON": f"{_GROUPING_LEVEL_PREFIX}{PERSON_LEVEL}",
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
    """A fully-spelled compositional unit (GEP 10).

    Built fluently off a base (``Unit.CURRENCY.PER_MONTH.PER_BG``) or parsed
    from the flat canonical string (``parse_compositional_unit`` of
    ``"CURRENCY_PER_MONTH_PER_BG"``); the two round-trip through :func:`str`.
    The denominators are held in canonical order — at most one *area*, one
    *period*, one *level* — and the builder methods enforce that order, so a
    non-canonical chain (``.PER_BG.PER_MONTH``) is a definition error.

    A :class:`CompositeUnit` is *the* declaration type; it resolves to a pint
    unit via :func:`resolve_compositional_unit`.
    """

    base: str
    area: str | None = None
    period: str | None = None
    level: str | None = None

    if TYPE_CHECKING:
        # The per-level builder steps (``per_bg``, ``per_fam``, …) are added at
        # runtime by :func:`register_unit_builder_levels` for each grouping level
        # the build discovers (the level vocabulary is open, GEP 10). They cannot
        # be hard-declared, so this tells ``ty`` and editors that any builder
        # attribute yields a :class:`CompositeUnit`. Runtime keeps strict
        # lookup — an unregistered ``per_<level>`` (or a typo) raises
        # ``AttributeError`` — because the block is type-checking only.
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

    @property
    def carries_level(self) -> bool:
        """Whether this unit carries a grouping-level denominator."""
        return self.level is not None

    # ---- fluent builder steps (canonical order enforced) ----

    def _with_area(self, area: str) -> CompositeUnit:
        if self.area is not None or self.period is not None or self.level is not None:
            raise UnitDefinitionError(
                f"Cannot add area '{area}' to '{self}': the canonical order is "
                f"base _PER_ <area> _PER_ <period> _PER_ <level> (GEP 10)."
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
        return replace(self, level=level.upper())

    @property
    def PER_SQUARE_METER(self) -> CompositeUnit:  # noqa: N802 (DSL: mirrors the token)
        """This unit per square meter (the area denominator)."""
        return self._with_area("SQUARE_METER")

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
    grouping level, validated against the registered levels at resolution time
    (the level vocabulary is open and discovered per build, GEP 10).
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
    """Parse a flat canonical compositional spelling (GEP 10).

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


#: Maps a GEP-1 time-unit suffix id to the compositional *period* token, so a
#: column's spelled period can be checked against its name suffix (GEP 10).
TIME_UNIT_ID_TO_PERIOD_TOKEN: dict[str, str] = {
    "y": "YEAR",
    "q": "QUARTER",
    "m": "MONTH",
    "w": "WEEK",
    "d": "DAY",
}

#: The compositional bases that are *extensive* — they carry a grouping level by
#: default (summing a level's members sums them): currency, area, and the
#: ``[person]`` count. An unsuffixed extensive column is at :data:`PERSON_LEVEL`;
#: an intensive base (duration, calendar point, dimensionless) carries no level.
_EXTENSIVE_COMPOSITIONAL_BASES: frozenset[str] = frozenset(
    {"PERSON", "SQUARE_METER", "HECTARE"}
)


def composite_base_is_extensive(base: str) -> bool:
    """Whether a compositional base carries a grouping level by default (GEP 10)."""
    return _is_currency_base(base) or base in _EXTENSIVE_COMPOSITIONAL_BASES


def resolve_compositional_unit(
    unit: CompositeUnit, *, with_level: bool = True
) -> pint.Unit:
    """Resolve a compositional unit to its pint unit (GEP 10).

    The denominators divide the base in turn: a period divides by its pint
    period, an area by ``meter ** 2``, a level by its grouping-level dimension.

    A currency base resolves to the agnostic :data:`CURRENCY_TOKEN` dimension —
    for dimensionality a concrete currency means exactly what ``CURRENCY`` means
    (the concrete currency drives the build-time conversion, not the check).

    Raises:
        UnitDefinitionError: If a level denominator names an unregistered
            grouping level.
    """
    if _is_currency_base(unit.base):
        resolved = UNIT_REGISTRY.parse_units(CURRENCY_TOKEN)
    elif unit.base == "PERSON":
        # The `[person]` leaf count dimension, registered with the grouping
        # levels — go through the helper so an un-built registry fails loudly.
        resolved = _grouping_level_unit(PERSON_LEVEL)
    else:
        base = _COMPOSITIONAL_BASE_TO_PINT[unit.base]
        resolved = (
            UNIT_REGISTRY.dimensionless
            if base is None
            else UNIT_REGISTRY.parse_units(base)
        )
    if unit.area is not None:
        resolved = _divide_by_period(
            resolved, period_pint_name=_AREA_TOKEN_TO_PINT[unit.area]
        )
    if unit.period is not None:
        resolved = _divide_by_period(
            resolved, period_pint_name=_PERIOD_TOKEN_TO_PINT[unit.period]
        )
    if with_level and unit.level is not None:
        resolved = divide_by_grouping_level(unit=resolved, level=unit.level.lower())
    return resolved


def resolve_compositional_column_unit(
    unit: CompositeUnit,
    *,
    time_unit_id: str | None,
    grouping_level: str,
    where: str,
) -> pint.Unit:
    """Resolve a column/function's compositional unit, validating the name suffix.

    Convention A (GEP 10): a column spells its period and any *group* level, which
    must match the name's suffix; the *person* leaf level is implied (not spelled)
    and added here for an extensive base. So ``betrag_m`` declares
    ``CURRENCY_PER_MONTH`` (resolved ``CURRENCY / month / [person]``), ``betrag_m_hh``
    declares ``CURRENCY_PER_MONTH_PER_HH``, and a level-less ``MIN``-of-age at
    ``_fg`` declares ``YEARS`` (the ``fg`` suffix is a mere index level).

    Columns are currency-agnostic, so a concrete-currency base is rejected.

    Raises:
        UnitDefinitionError: If the base pins a concrete currency, the spelled
            period/level disagrees with the name suffix, or an extensive group
            column fails to spell its level.
    """
    if token_source_currency(unit) is not None:
        raise UnitDefinitionError(
            f"{where}: a column/function pins the concrete currency {unit.base!r}; "
            f"columns are currency-agnostic and must use {CURRENCY_TOKEN} (GEP 10)."
        )
    expected_period = (
        TIME_UNIT_ID_TO_PERIOD_TOKEN[time_unit_id] if time_unit_id is not None else None
    )
    if unit.period != expected_period:
        raise UnitDefinitionError(
            f"{where}: the unit spells period {unit.period!r} but the name's time "
            f"suffix implies {expected_period!r}; they must agree (GEP 10)."
        )
    is_group_level = grouping_level != PERSON_LEVEL
    if unit.level is not None and unit.level != grouping_level.upper():
        raise UnitDefinitionError(
            f"{where}: the unit spells level {unit.level!r} but the name's "
            f"aggregation suffix implies {grouping_level.upper()!r} (GEP 10)."
        )
    if unit.level is None and is_group_level and composite_base_is_extensive(unit.base):
        raise UnitDefinitionError(
            f"{where}: an extensive quantity at the {grouping_level!r} level must "
            f"spell that level (e.g. ..._PER_{grouping_level.upper()}); the person "
            f"leaf is the only implied level (GEP 10)."
        )
    resolved = resolve_compositional_unit(unit, with_level=True)
    # Add the implied person leaf level for an extensive base with no spelled
    # level — the unsuffixed-name case (group columns spell their level above).
    if unit.level is None and composite_base_is_extensive(unit.base):
        resolved = divide_by_grouping_level(unit=resolved, level=grouping_level)
    return resolved


def composite_with_rebased_period(
    unit: CompositeUnit, time_unit_id: str
) -> CompositeUnit:
    """Re-base a flow compositional unit to a new period (GEP 10).

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
    time_unit_id: str | None = None,
    where: str,
) -> pint.Unit:
    """Resolve a parameter's compositional unit (GEP 10).

    A parameter spells its unit *fully* — period and level both in the string —
    because (unlike a column) it has no name suffix to imply a level from: there
    is no person default, a parameter is level-agnostic unless it spells a level
    (``SILVER_PENNY_PER_PERSON`` for a per-person threshold,
    ``SILVER_PENNY_PER_FAM`` for a per-family one, ``SILVER_PENNY`` for a
    level-agnostic amount). A concrete-currency base is allowed — parameters pin
    the currency their numbers are written in. A scalar parameter additionally
    takes a time suffix on its *name*; where one is present the spelled period
    must agree with it.

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
    return resolve_compositional_unit(unit, with_level=True)


def register_unit_builder_levels(names: Iterable[str]) -> None:
    """Give the fluent builder a ``per_<level>`` attribute for each level (GEP 10).

    The level vocabulary is open and discovered per build, so the builder cannot
    hard-wire the level step the way it does the closed area/period steps. Each
    package registers its levels (at import, before its declarations run) and
    thereby earns the sugar ``Unit.PERSON.PER_BG`` / ``…\u200b.PER_MONTH.PER_BG``
    alongside the always-available generic :meth:`CompositeUnit.PER_LEVEL`. The
    person leaf is always registered. Idempotent.

    The bases on the :class:`Unit` namespace are themselves
    :class:`CompositeUnit`\\ s, so adding the ``per_<level>`` property to
    :class:`CompositeUnit` makes ``Unit.PERSON.PER_BG`` work too — there is one
    place to extend. ``ty`` and editors see the sugar through
    :class:`CompositeUnit`'s type-checking-only ``__getattr__``.
    """
    for name in (PERSON_LEVEL, *names):
        if name in _unit_builder_levels:
            continue
        _unit_builder_levels.add(name)
        setattr(
            CompositeUnit,
            f"PER_{name.upper()}",
            property(lambda self, level=name: self.PER_LEVEL(level)),
        )


class Unit:
    """The builder namespace of unit *bases* (GEP 10).

    Each attribute is a bare :class:`CompositeUnit` — ``Unit.CURRENCY`` *is*
    ``CompositeUnit(base="CURRENCY")`` — that heads a ``.per_*`` builder chain
    enforcing the canonical order ``base _PER_ <area> _PER_ <period> _PER_
    <level>`` (``Unit.CURRENCY.PER_MONTH.PER_BG``). The area and period steps are
    closed sets and so are real, autocompleting properties on
    :class:`CompositeUnit`; the per-level step is discovered per build, so
    :func:`register_unit_builder_levels` adds a ``per_<level>`` attribute for each
    registered level alongside the always-available generic
    :meth:`CompositeUnit.PER_LEVEL`.

    Only the agnostic currency base ``CURRENCY`` lives here; concrete currency
    bases (``SILVER_PENNY``, ``DM``) are spelled directly in parameter YAML and
    reached via :func:`parse_compositional_unit`, never off this namespace.
    """

    CURRENCY = CompositeUnit(base=CURRENCY_TOKEN)
    """An amount of currency (agnostic): wages, claims, benefits, wealth. A
    period denominator makes it a flow (``Unit.CURRENCY.PER_MONTH``)."""

    DIMENSIONLESS = CompositeUnit(base="DIMENSIONLESS")
    """A plain dimensionless number: a share, a rate. A boolean declares
    ``DIMENSIONLESS`` too and takes its level from the name suffix (GEP 10)."""

    PERSON = CompositeUnit(base="PERSON")
    """The individual (leaf) count base — the numerator of a head count
    (``[person]``). With a level denominator it is a head count per group:
    ``Unit.PERSON.PER_BG`` resolves to ``[person] / [bg]``."""

    HOURS = CompositeUnit(base="HOURS")
    """Working hours (the isolated ``[hours]`` dimension). A period denominator
    re-bases them (``Unit.HOURS.PER_WEEK``)."""

    SQUARE_METER = CompositeUnit(base="SQUARE_METER")
    """An area in square meters; also the lone *area* denominator
    (``Unit.CURRENCY.PER_SQUARE_METER``)."""

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


#: Sentinel distinguishing an *omitted* unit declaration from an explicit one
#: (GEP 10). Units are mandatory in the public decorators, so this is
#: unreachable through them; it survives only as a dataclass field default (the
#: ``unit`` field needs one for field-ordering) that the mandatory-units check
#: reports. A :class:`CompositeUnit` so the field type is cleanly
#: :class:`CompositeUnit`; its base never resolves and it is only ever compared
#: by identity.
UNSET_UNIT: CompositeUnit = CompositeUnit(base="__UNSET__")


def token_is_agnostic_currency(token: CompositeUnit | None) -> bool:
    """Whether a unit is a currency-dimensioned *agnostic* declaration (GEP 10).

    Parameters must not declare these once a concrete currency is registered:
    the declaration names the currency the numbers are written in, so it must
    pin down a concrete currency. A compositional unit is agnostic when its base
    is the agnostic :data:`CURRENCY_TOKEN`.
    """
    return isinstance(token, CompositeUnit) and token.base == CURRENCY_TOKEN


def token_source_currency(token: CompositeUnit | None) -> str | None:
    """The concrete currency a declaration pins down, if any (GEP 10).

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


def unit_for_derived_node(token: CompositeUnit) -> CompositeUnit:
    """The unit a node *derived* from a source with this unit carries (GEP 10).

    Derived nodes — time-conversion variants and aggregations — are functions,
    and a function is currency-agnostic by design: it computes on values already
    converted to the run currency. So a source that pins down a concrete currency
    (a parameter) hands on the agnostic counterpart of its declaration; every
    other unit passes through unchanged.
    """
    return (
        replace(token, base=CURRENCY_TOKEN)
        if token_source_currency(token) is not None
        else token
    )


def coerce_unit_token(
    value: str | CompositeUnit,
    *,
    where: str,
) -> CompositeUnit:
    """Coerce a YAML ``unit:`` value to a :class:`CompositeUnit` (GEP 10).

    A string is a *compositional* spelling (``CURRENCY_PER_MONTH_PER_BG``,
    ``PERSON_PER_BG``, ``SILVER_PENNY_PER_YEAR``, or a bare base ``DIMENSIONLESS``)
    parsed into a :class:`CompositeUnit`; an already-coerced
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
        f"compositional spelling (e.g. CURRENCY_PER_MONTH_PER_BG, PERSON_PER_BG), "
        f"a bare base (e.g. CURRENCY, SILVER_PENNY), or DIMENSIONLESS for a "
        f"dimensionless quantity (GEP 10)."
    )


def _build_registry() -> pint.UnitRegistry:
    """Create the module-level registry with the units TTSIM knows about.

    pint's defaults already provide the ``[time]`` units (``year``, ``month``,
    ``week``, ``day`` — with the per-year factors GETTSIM uses: 12, 365.25/7,
    365.25) and ``[length]``/``[area]`` units (``meter``, ``hectare``). We add:

    - ``CURRENCY`` as the reference unit of a new ``[currency]`` dimension;
    - ``working_hour`` as the reference unit of a new ``[hours]`` dimension,
      isolated from pint's ``[time]`` ``hour`` (GEP 10): ``working_hour / week``
      is then ``[hours] / [time]`` rather than the bare number ``[time] /
      [time]``, so working hours can no longer be confused with — or added to —
      a share, and the only conversion possible is re-basing the *period*
      denominator. pint's ``hour`` is left untouched (``day = 24 · hour``) but
      is not an admissible token;
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
      (each is ratio 1 against ``year`` / ``month`` / ``day``).

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
    "working_hour",
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

#: The concrete currencies registered so far (their pint unit names). A currency
#: is a valid compositional *base* (its upper-cased name); on a parameter the
#: base also names the currency the numbers are written in (GEP 10).
_registered_currencies: set[str] = set()

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

    The registered currency becomes a valid compositional *base* — its
    upper-cased name (``register_currency("DM", ...)`` makes ``DM``,
    ``DM_PER_MONTH``, … parseable) — so parameters can pin down the concrete
    currency their numbers are written in.

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
    _registered_currencies.add(name)


#: The grouping levels registered so far (the bare names, e.g. ``"person"``,
#: ``"hh"``), populated by :func:`register_grouping_levels`. ``person`` (the
#: individual leaf, doubling as the ``[person]`` count dimension) is always
#: present once any level has been registered. The set is discovered per build
#: from the policy environment's ``*_id`` columns; ttsim ships no fixed list.
_registered_grouping_levels: set[str] = set()


def _grouping_level_unit_name(name: str) -> str:
    """The internal pint unit name anchoring a grouping level's dimension."""
    return f"{_GROUPING_LEVEL_PREFIX}{name}"


def register_grouping_levels(names: Iterable[str]) -> None:
    """Register grouping levels as base dimensions in the registry (GEP 10).

    Each grouping level — the individual ``person`` (the leaf, identified by
    ``p_id``) and one per ``*_id`` group column (``hh``, ``bg``, ``fg``, …) — is
    its *own* pint base dimension with no conversion to any other: a household
    holds a variable number of persons, so the levels are not units of one shared
    dimension (the way ``month`` and ``year`` are units of ``[time]``) but
    distinct, non-interconvertible base dimensions. The level set is not fixed; it
    is discovered per build from the policy environment's ``*_id`` columns, so the
    orchestration (issue #119) calls this with the levels it found.

    ``person`` (the leaf level) is always registered: it doubles as the
    ``[person]`` *count* dimension, the conversion factor between levels that lets
    head counts and per-person amounts cancel.

    Each level is defined under an internal :data:`_GROUPING_LEVEL_PREFIX`-prefixed
    pint name anchoring a fresh base dimension and added to the closed pint-token
    vocabulary, so ``CURRENCY / month`` divided by a level resolves and the
    level's token is admissible in an internal pint surface. Re-registering an
    already-known level is a tolerated no-op (e.g. a re-imported module), mirroring
    :func:`register_currency`.

    Args:
        names: The grouping-level names to register (e.g. ``["hh", "bg"]``).
            ``person`` is added unconditionally.
    """
    for name in (PERSON_LEVEL, *names):
        if name in _registered_grouping_levels:
            continue
        unit_name = _grouping_level_unit_name(name)
        UNIT_REGISTRY.define(f"{unit_name} = [{unit_name}]")
        _ALLOWED_UNIT_TOKENS.add(unit_name)
        _registered_grouping_levels.add(name)
    # Give the fluent builder a `per_<level>` attribute for each level too, so
    # `Unit.PERSON.PER_BG` works once the build has discovered `bg` (GEP 10).
    # Packages that use the builder at import time call
    # `register_unit_builder_levels` directly, before their declarations run.
    register_unit_builder_levels(names)


def _fail_if_grouping_level_is_unknown(name: str) -> None:
    """Reject a grouping-level name that has not been registered (GEP 10)."""
    if name not in _registered_grouping_levels:
        known = ", ".join(sorted(_registered_grouping_levels)) or "(none registered)"
        raise UnitDefinitionError(
            f"Unknown grouping level {name!r}; expected one of {known}. Grouping "
            f"levels are discovered per build from the `*_id` columns and "
            f"registered via register_grouping_levels (GEP 10)."
        )


def _grouping_level_unit(name: str) -> pint.Unit:
    """The pint unit of a registered grouping level (GEP 10).

    Raises:
        UnitDefinitionError: If the level has not been registered.
    """
    _fail_if_grouping_level_is_unknown(name)
    return UNIT_REGISTRY.parse_units(_grouping_level_unit_name(name))


def divide_by_grouping_level(unit: pint.Unit, level: str) -> pint.Unit:
    """Return ``unit`` divided by a grouping level's unit (GEP 10).

    A leveled quantity carries its level as a denominator, exactly as a flow
    carries its period as one: ``CURRENCY / month`` at level ``hh`` becomes
    ``CURRENCY / month / [hh]``, and at the individual level ``person`` it becomes
    ``CURRENCY / month / [person]``. The ``[person]`` denominator of a person-level
    quantity is the same dimension as a head count's ``[person]`` numerator, so the
    two cancel.

    Raises:
        UnitDefinitionError: If the level has not been registered.
    """
    level_quantity = UNIT_REGISTRY.Quantity(1.0, _grouping_level_unit(level))
    return (UNIT_REGISTRY.Quantity(1.0, unit) / level_quantity).units


def grouping_level_count_unit(target_level: str) -> pint.Unit:
    """The unit of a head count over ``target_level`` (GEP 10).

    A head count is the ``[person]`` *count* dimension over the group it counts
    within: counting persons in a household is ``[person] / [hh]`` — persons per
    household. ``COUNT`` aggregations mint this, and it is the conversion factor
    that bridges levels (``[person]/[hh] · CURRENCY/[hh] = CURRENCY/[person]``).

    Raises:
        UnitDefinitionError: If ``person`` or ``target_level`` is not registered.
    """
    person = UNIT_REGISTRY.Quantity(1.0, _grouping_level_unit(PERSON_LEVEL))
    target = UNIT_REGISTRY.Quantity(1.0, _grouping_level_unit(target_level))
    return (person / target).units


def parse_unit(unit_str: str) -> pint.Unit:
    """Parse a pint unit string, enforcing the closed pint-token vocabulary.

    Internal (GEP 10): declarations are :class:`CompositeUnit`\\ s, never pint
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


def _token_base_unit(token: CompositeUnit) -> pint.Unit:
    """The pint unit of a unit's physical kind — its period and grouping level
    stripped (its area kept).

    A currency base resolves to the agnostic :data:`CURRENCY_TOKEN` unit: for
    dimensionality a registered currency means exactly what its agnostic
    counterpart means (the union semantics of GEP 10) — the concrete currency
    only drives the build-time conversion of the numbers.
    """
    return resolve_compositional_unit(replace(token, period=None, level=None))


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


def output_unit_in_run_currency(
    units: pint.Unit, run_currency: str | None
) -> pint.Unit:
    """Restate a resolved (agnostic) DAG unit in the concrete run currency (GEP 10).

    Output columns are *computed* in the run currency, so labelling them is pure
    naming, not conversion: the agnostic ``CURRENCY`` component of the resolved
    unit is swapped for the run currency (``CURRENCY / month`` → ``euro / month``)
    while period and area are left untouched. A unit with no currency component
    (``year``, ``hectare``, dimensionless) is returned unchanged.

    This is the inverse of the currency move
    :func:`strip_input_quantity_at_boundary` makes on the way in.

    Raises:
        UnitDefinitionError: If the unit is currency-dimensioned but no run
            currency is set — a run with currency quantities must pin one down.
    """
    currency_component = _currency_component_of(units)
    if currency_component is None:
        return units
    if run_currency is None:
        raise UnitDefinitionError(
            f"Cannot restate '{units}' without a run currency: a run that uses "
            f"currency quantities must pin down a concrete currency (GEP 10)."
        )
    return units / currency_component * UNIT_REGISTRY.parse_units(run_currency)


def _flow_period_of(units: pint.Unit) -> pint.Unit | None:
    """Return a unit's flow period — its time component in the *denominator*.

    The ``month`` of ``CURRENCY / month``, the ``week`` of ``working_hour /
    week``. A
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
    (a ``HECTARE`` column tagged ``m²`` shares the area dimension but is a
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
    source_unit: CompositeUnit,
    agg_type: AggType,
) -> CompositeUnit:
    """Auto-assign the unit of an aggregation node (GEP 10, #119).

    Parallels how GEP 4 resolves an aggregation's *type* from the source and the
    aggregation rule:

    - ``SUM`` / ``MEAN`` / ``MIN`` / ``MAX`` preserve the source unit (a sum
      or average of currency flows is still a currency flow);
    - ``COUNT`` is a head count — :attr:`Unit.PERSON` (the ``[person]`` count
      base) regardless of source. For an ``agg_by_group`` node the resolved unit
      is recomputed level-aware as ``[person] / [target]`` (the token is a
      placeholder there); for an ``agg_by_p_id`` node the token *is* what
      resolves, through the column path, to ``[person] / [person]`` =
      dimensionless — a head count per individual, which is exactly a plain
      number (GEP 10);
    - ``ANY`` / ``ALL`` yield a boolean, which is a dimensionless quantity
      (GEP 10), i.e. :attr:`Unit.DIMENSIONLESS`.

    Args:
        source_unit: The source column's ``unit`` — a :class:`CompositeUnit`
            (:data:`UNSET_UNIT` if the source does not declare one).
        agg_type: The :class:`ttsim.tt.aggregation.AggType` of the aggregation.

    Returns:
        The auto-assigned unit. :attr:`Unit.PERSON` for a ``COUNT`` head count,
        :attr:`Unit.DIMENSIONLESS` for a boolean ``ANY`` / ``ALL`` result;
        otherwise the preserved source unit (:data:`UNSET_UNIT` when ``SUM`` /
        ``MEAN`` / … preserve a source that itself lacks a declaration, which the
        mandatory-units check then reports against the source).
    """
    if agg_type is AggType.COUNT:
        return Unit.PERSON
    if agg_type in (AggType.ANY, AggType.ALL):
        return Unit.DIMENSIONLESS
    # SUM, MEAN, MIN, MAX preserve the source unit.
    return source_unit


def resolved_unit_for_aggregation(
    *,
    source_unit: pint.Unit,
    agg_type: AggType,
    target_level: str,
    source_level: str | None,
) -> pint.Unit:
    """The resolved unit of an aggregation node, level-aware (GEP 10, #119).

    The level-aware counterpart of :func:`unit_for_aggregation`: it operates on
    *resolved* pint units (the physical token combined with its flow period and
    grouping level) and is where a grouping level is minted, swapped, or
    preserved. The orchestration (issue #119) threads the ``target_level`` (the
    group being aggregated *to*, read off the aggregation suffix) and the
    ``source_level`` (the source column's own level, ``None`` for a level-less
    source such as an age).

    - ``SUM`` is the extensive aggregation: it preserves the physical token and
      *swaps* the source level for the target level in the denominator — summing
      per-person amounts to the household gives ``CURRENCY/month/[hh]``. A
      level-less source (``source_level=None``) is summed as-is.
    - ``COUNT`` mints a head count ``[person] / [target]`` — persons per target
      group, the ``[person]`` count dimension over the target level — independent
      of the source.
    - ``MEAN`` / ``MIN`` / ``MAX`` pick a representative member's value, the same
      kind of quantity as the source, so they *preserve* the source unit verbatim,
      level-ness and all: the min of person incomes stays ``CURRENCY/[person]``,
      the min of a level-less age stays ``MONTHS``. They do *not* swap to the
      target level.
    - ``ANY`` / ``ALL`` yield a boolean *at the target level* — ``1 / [target]``
      (a ``DIMENSIONLESS_PER_<target>`` quantity, GEP 10) — so a group-level
      indicator carries the level its name claims.

    Args:
        source_unit: The source column's resolved pint unit.
        agg_type: The :class:`ttsim.tt.aggregation.AggType` of the aggregation.
        target_level: The group level being aggregated to (e.g. ``"hh"``).
        source_level: The source column's grouping level (e.g. ``"person"``), or
            ``None`` if the source carries no level.

    Returns:
        The aggregation node's resolved pint unit.

    Raises:
        UnitDefinitionError: If ``target_level`` or ``source_level`` names an
            unregistered grouping level.
    """
    if agg_type in (AggType.ANY, AggType.ALL):
        # A boolean aggregation mints a boolean *at its target level* (GEP 10):
        # ``alle_erwachsen_fam`` is a fam-level truth value, ``1 / [fam]``, the
        # same shape as a ``DIMENSIONLESS_PER_FAM`` declaration. The level lets
        # the result check catch a fam-named predicate computed at another level.
        return divide_by_grouping_level(
            unit=UNIT_REGISTRY.dimensionless, level=target_level
        )
    if agg_type is AggType.COUNT:
        return grouping_level_count_unit(target_level=target_level)
    if agg_type is AggType.SUM:
        if source_level is None:
            return source_unit
        numerator = UNIT_REGISTRY.Quantity(1.0, source_unit) * UNIT_REGISTRY.Quantity(
            1.0, _grouping_level_unit(source_level)
        )
        return divide_by_grouping_level(unit=numerator.units, level=target_level)
    # MEAN, MIN, MAX preserve the source unit verbatim (level-ness and all).
    return source_unit


def fail_if_units_are_missing(
    units_by_qname: Mapping[str, CompositeUnit],
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
