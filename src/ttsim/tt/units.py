"""The pint-based unit framework.

Establishes the unit vocabulary that checks the dimensional soundness of the
taxes-and-transfers DAG, plus the build-time machinery that runs the check.

Two kinds of object are called a "unit" here, and every identifier in this package names
which one it means:

- a **TTSIM unit** (``ttsim_unit``) is a :class:`CompositeUnit`, the token a policy
  author *declares*. It is a plain frozen dataclass, needs no registry, and exists
  before any build starts.
- a **pint unit** (``pint_unit``) is a :class:`pint.Unit`, what a TTSIM unit *resolves
  to* against a :class:`pint.UnitRegistry`. It carries the dimensions the check does
  arithmetic on and lives only at build/check time.

Every declaration is a fully-spelled :class:`CompositeUnit` — a base optionally divided
by a physical denominator (an area or working hours), a period, and a grouping level, in
that canonical order. It has two round-tripping spellings (via :func:`parse_ttsim_unit`
/ :func:`str`):

- fluent, off the :class:`TTSIMUnit` namespace
  (``TTSIMUnit.CURRENCY.PER_MONTH.PER_BG``);
- flat canonical string, in YAML (``CURRENCY_PER_MONTH_PER_BG``).
"""

from __future__ import annotations

import dataclasses
import datetime
import math
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar

import pint
from pint.util import to_units_container
from typing_extensions import TypeIs

from ttsim.exceptions import (
    UnitConsistencyError,
    UnitDefinitionError,
)
from ttsim.interface_dag_elements.shared import (
    get_re_pattern_for_all_time_units_and_groupings,
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

_PER = "_PER_"

_AREA_TOKEN_TO_PINT: dict[str, str] = {
    "SQUARE_METER": "meter ** 2",
    "HOURS": "working_hour",
}

_TTSIM_UNIT_BASE_TO_PINT: dict[str, str | None] = {
    "DIMENSIONLESS": None,
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
#: :func:`register_unit_builder_levels`
_unit_builder_levels: set[str] = set()


_CastValueT = TypeVar("_CastValueT")


@dataclass(frozen=True)
class CompositeUnit:
    """A fully-spelled TTSIM unit — *the* declaration type.

    A base divided by at most one *physical denominator* (an area or working
    hours), one *period*, and one *level*, held in that canonical order; the
    builder methods enforce the order, so a non-canonical chain
    (``.PER_BG.PER_MONTH``) is a definition error. Two round-tripping spellings
    (via :func:`str`):

    - fluent, off a base (``TTSIMUnit.CURRENCY.PER_MONTH.PER_BG``);
    - flat canonical string, parsed by :func:`parse_ttsim_unit`
      (``"CURRENCY_PER_MONTH_PER_BG"``).

    It resolves to a :class:`pint.Unit` via :func:`resolve_ttsim_unit`.
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
                f"_PER_ <level>, with at most one physical denominator    ."
            )
        return replace(self, area=area)

    def _with_period(self, period: str) -> CompositeUnit:
        if self.period is not None or self.level is not None:
            raise UnitDefinitionError(
                f"Cannot add period '{period}' to '{self}': a period must precede "
                f"the level and there is at most one period    ."
            )
        return replace(self, period=period)

    def _with_level(self, level: str) -> CompositeUnit:
        if self.level is not None:
            raise UnitDefinitionError(
                f"Cannot add level '{level}' to '{self}': a unit carries at most "
                f"one grouping level    ."
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
    return any(base == name.upper() for name in _registered_currencies)


def parse_ttsim_unit(spelling: str) -> CompositeUnit:
    """The parser: a flat canonical spelling in, a TTSIM unit out.

    ``"CURRENCY_PER_MONTH_PER_BG"`` → ``CompositeUnit(base="CURRENCY",
    period="MONTH", level="BG")``. A pure ``str -> CompositeUnit`` function with
    no registry and no error context; :func:`ttsim_unit_from_yaml_value` is the
    boundary that wraps it for values arriving from YAML.

    Raises:
        UnitDefinitionError: If the spelling is empty, names an unknown base, or
            violates the canonical order / one-per-kind rules.
    """
    if not spelling:
        raise UnitDefinitionError("Empty compositional unit spelling (GEP 10).")
    base, *denominators = spelling.split(_PER)
    if base not in _TTSIM_UNIT_BASE_TO_PINT and not _is_currency_base(base):
        raise UnitDefinitionError(
            f"Unknown compositional base {base!r} in {spelling!r}. A base is the "
            f"agnostic '{CURRENCY_TOKEN}', a registered currency, or one of "
            f"{', '.join(sorted(_TTSIM_UNIT_BASE_TO_PINT))} (GEP 10)."
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


def resolve_ttsim_unit(
    unit: CompositeUnit, *, registry: pint.UnitRegistry, with_level: bool = True
) -> pint.Unit:
    """Resolve a TTSIM unit to its pint unit in ``registry``.

    The plain, unvalidated bridge between the two layers; the ``resolve_*_for_*``
    wrappers add the layer-specific declaration rules on top of it.

    Raises:
        UnitDefinitionError: If a level denominator names a grouping level the
            registry does not define.
    """
    if _is_currency_base(unit.base):
        resolved = registry.parse_units(CURRENCY_TOKEN)
    else:
        base = _TTSIM_UNIT_BASE_TO_PINT[unit.base]
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


def resolve_ttsim_unit_for_column(
    unit: CompositeUnit,
    *,
    time_unit_id: str | None,
    grouping_level: str | None,
    where: str,
    registry: pint.UnitRegistry,
) -> pint.Unit:
    """Resolve a column/function's TTSIM unit, validating it against the name.

    Two rules are enforced (GEP 10):
    - a present name time suffix must agree with the spelled period;
    - concrete currencies are rejected for columns/functions, only the agnostic
        ``CURRENCY`` is allowed (only parameters and rounding specs pin down concrete
        currencies).

    Raises:
        UnitDefinitionError: If the base pins a concrete currency, or the spelled
            period or level disagrees with the name suffix.
    """
    if ttsim_unit_currency(unit) is not None:
        raise UnitDefinitionError(
            f"{where}: a column/function pins the concrete currency {unit.base!r}. "
            f"A function runs in the statutory currency of the policy date, "
            f"whichever that is: declare the agnostic {CURRENCY_TOKEN} and leave "
            f"concrete currencies to parameters and rounding specs."
        )
    expected_period = (
        TIME_UNIT_ID_TO_PERIOD_TOKEN[time_unit_id] if time_unit_id is not None else None
    )
    if unit.period != expected_period:
        raise UnitDefinitionError(
            f"{where}: the unit spells period {unit.period!r} but the name's time "
            f"suffix implies {expected_period!r}; they must agree."
        )
    expected_level = grouping_level.upper() if grouping_level is not None else None
    if unit.level is not None and unit.level != expected_level:
        raise UnitDefinitionError(
            f"{where}: the unit spells group level {unit.level!r} but the name's "
            f"aggregation suffix implies {expected_level or 'no level (bare)'}; a "
            f"spelled group level must match the suffix, or be omitted for a bare "
            f"(per-person / level-neutral) quantity."
        )
    return resolve_ttsim_unit(unit=unit, registry=registry, with_level=True)


def replace_time_unit(unit: CompositeUnit, time_unit_id: str) -> CompositeUnit:
    """Replace a flow unit's period with the requested time unit.

    Return a non-flow unit unchanged.
    """
    if unit.period is None:
        return unit
    return replace(unit, period=TIME_UNIT_ID_TO_PERIOD_TOKEN[time_unit_id])


def resolve_ttsim_unit_for_param(
    unit: CompositeUnit,
    *,
    registry: pint.UnitRegistry,
    time_unit_id: str | None = None,
    where: str,
) -> pint.Unit:
    """Resolve a parameter's TTSIM unit.

    A parameter spells its period and its level (it has no name suffix to read
    them off); an omitted level is **bare**, exactly as for a column:

    - ``SILVER_PENNY_PER_MONTH`` — a per-person amount (bare);
    - ``SILVER_PENNY_PER_FAM`` — a per-family amount;
    - a bare base (``DIMENSIONLESS``, ``YEARS``, a rate) stays bare.

    Raises:
        UnitDefinitionError: If a present name time suffix disagrees with the
            spelled period.
    """
    if time_unit_id is not None:
        expected_period = TIME_UNIT_ID_TO_PERIOD_TOKEN[time_unit_id]
        if unit.period != expected_period:
            raise UnitDefinitionError(
                f"{where}: the unit spells period {unit.period!r} but the name's "
                f"time suffix implies {expected_period!r}; they must agree."
            )
    return resolve_ttsim_unit(unit=unit, registry=registry, with_level=True)


def resolve_agnostic_ttsim_unit(
    unit: CompositeUnit, *, registry: pint.UnitRegistry, where: str, what: str
) -> pint.Unit:
    """Resolve a currency-agnostic TTSIM unit declared in code, not on a DAG node.

    Such a declaration hangs off code, not off a named DAG node: a
    :func:`cast_ttsim_unit` call in a body, or a field of a schedule's
    :class:`InputOutputUnits`. With no node name there is no GEP-1 name suffix to
    cross-check the spelled period against, so the period is taken as declared. A
    concrete currency is rejected — only parameters and rounding specs pin a currency
    down; code-side declarations spell the agnostic ``CURRENCY``.

    Args:
        unit: The declared TTSIM unit. registry: The pint registry to resolve against.
        where: The location prefix of an error message (e.g. the qname). what: How to
        name the declaration in an error message (e.g.
            ``"a cast inside a body"``).

    Returns:
        The resolved pint unit, including any grouping level.

    Raises:
        UnitDefinitionError: If the base pins a concrete currency.
    """
    if ttsim_unit_currency(unit) is not None:
        raise UnitDefinitionError(
            f"{where}: {what} pins the concrete currency {unit.base!r}; declare "
            f"the agnostic {CURRENCY_TOKEN} — only parameters and rounding specs "
            f"pin down concrete currencies (GEP 10)."
        )
    return resolve_ttsim_unit(unit=unit, registry=registry, with_level=True)


class _UnitNamespaceMeta(type):
    """Metaclass for :class:`TTSIMUnit` so ``ty`` accepts dynamically-added bases.

    Concrete currency bases (``TTSIMUnit.EUR``, ``TTSIMUnit.DM``,
    ``TTSIMUnit.SILVER_PENNY``) are injected onto :class:`TTSIMUnit` by
    :class:`~ttsim.tt.units.UnitSystem` as it defines its currencies — they cannot
    be hard-wired class attributes because the currency vocabulary is discovered per
    package. At runtime an injected base is a real attribute, so this metaclass adds no
    ``__getattr__``; under type checking it declares one so ``TTSIMUnit.EUR``
    type-checks (mirroring :class:`CompositeUnit`'s builder-step hint).
    """

    if TYPE_CHECKING:

        def __getattr__(cls, name: str) -> CompositeUnit: ...


class TTSIMUnit(metaclass=_UnitNamespaceMeta):
    """The builder namespace of unit *bases*."""

    CURRENCY = CompositeUnit(base=CURRENCY_TOKEN)
    """An amount of currency (agnostic): wages, claims, benefits, wealth."""

    DIMENSIONLESS = CompositeUnit(base="DIMENSIONLESS")
    """A plain dimensionless number: a share, a rate, a head count, a boolean."""

    HOURS = CompositeUnit(base="HOURS")
    """Working hours (the isolated ``[hours]`` dimension). Not a member of the [time]
    dimension to allow for [hours] / [time] units."""

    SQUARE_METER = CompositeUnit(base="SQUARE_METER")
    """An area in square meters."""

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

    Args:
        values: The column data — any leaf the ordinary input tree accepts (a
            list, a numpy/JAX array, a ``pd.Series``); canonicalized downstream.
        unit: The column's TTSIM unit, built off :class:`TTSIMUnit`.
    """

    values: Any
    unit: CompositeUnit


if TYPE_CHECKING:
    UserNestedUnitAnnotatedData: TypeAlias = Mapping[
        str, "UnitAnnotatedColumn | UserNestedUnitAnnotatedData"
    ]
    """Tree mapping TTSIM paths to unit-annotated columns.

    The leaf type is what separates it from `ttsim.typing.UserNestedData`, whose
    leaves are bare columns.
    """
else:
    # Widened to one level for beartype, like `NestedData` in `ttsim.typing`: the
    # recursive alias's stringified inner name is not resolvable at runtime.
    UserNestedUnitAnnotatedData = Mapping[str, UnitAnnotatedColumn | Mapping]


def cast_ttsim_unit(
    value: _CastValueT,
    unit: str | CompositeUnit,  # noqa: ARG001
) -> _CastValueT:
    """Re-tag ``value`` with ``unit`` for the build-time unit check.

    The expression-level escape hatch of the unit check. Like ``typing.cast``, it
    is the identity at run time — ``value`` comes back unchanged, scalar or
    column, so the numeric path and JAX tracing are untouched.

    Use it where a single operation is dimensionally irregular but deliberate:

    - policy-mandated cross-level arithmetic (a group extreme against a person
      threshold, a group share times a group total);
    - a granularity conversion on the calendar axes;
    - a genuine dimensioned constant that cannot be promoted to a parameter.

    Args:
        value: The expression to re-tag; returned unchanged.
        unit: The stated unit — built off :class:`TTSIMUnit` or the flat
            TTSIM unit spelling.

    Returns:
        ``value``, unchanged.
    """
    return value


@dataclass(frozen=True)
class _UnsetUnit:
    """Sentinel type for an unspecified quantity unit."""


UNSET_UNIT = _UnsetUnit()
UnitDeclaration: TypeAlias = CompositeUnit | _UnsetUnit


def is_unset_unit(value: object) -> TypeIs[_UnsetUnit]:
    """Return whether `value` represents an unspecified unit declaration.

    Discriminate by sentinel type so copied and deserialized declarations retain
    their meaning without relying on process-local object identity.
    """
    return isinstance(value, _UnsetUnit)


@dataclass(frozen=True)
class InputOutputUnits:
    """The two axes of a schedule builder's output.

    A ``@param_function`` whose body builds a schedule — a
    :class:`~ttsim.tt.param_objects.PiecewisePolynomialParamValue` or a
    :class:`~ttsim.tt.param_objects.ConsecutiveIntLookupTableParamValue` — is a
    *function between quantities*, not a single quantity, so it declares the domain and
    range of that function instead of one ``unit=``:
    ``unit=InputOutputUnits(input_unit=TTSIMUnit.CURRENCY,
    output_unit=TTSIMUnit.CURRENCY.PER_YEAR)``. The framework screens each ``look_up`` /
    ``piecewise_polynomial`` argument against ``input_unit`` and hands the consumer the
    ``output_unit``. A schedule-typed field of a parameter dataclass states the same
    pair in its ``Annotated[…]`` metadata.

    A multi-dimensional :class:`ConsecutiveIntLookupTableParamValue` is keyed by several
    axes at once — e.g. ``look_up(anzahl_personen_hh, mietstufe_hh)`` — whose units may
    differ. Pass a **tuple** of :class:`CompositeUnit` as ``input_unit`` to screen each
    ``look_up`` argument against the corresponding axis positionally; the call must
    supply exactly as many arguments as declared axes. A tuple is only for lookup tables
    — ``piecewise_polynomial`` takes one domain argument, so a tuple on a piecewise
    builder is a contract error.

    Both axes are currency-agnostic TTSIM units, exactly like a column/function
    declaration: a schedule builder runs in the statutory currency of the policy date,
    whichever that is.
    """

    input_unit: CompositeUnit | tuple[CompositeUnit, ...]
    """The unit each domain argument (``look_up`` index, ``piecewise_polynomial``
    ``x``) is screened against — a single :class:`CompositeUnit` applied to every
    argument, or a tuple screened positionally (argument ``i`` against axis
    ``i``)."""
    output_unit: CompositeUnit
    """The unit the schedule produces at every call site."""


def ttsim_unit_from_yaml_value(
    value: str | CompositeUnit,
    *,
    where: str,
) -> CompositeUnit:
    """The YAML boundary: turn a raw ``unit:`` value into a TTSIM unit.

    This is the boundary wrapper around :func:`parse_ttsim_unit`, not a second
    parser. It admits the two shapes a declaration can arrive in and attaches a
    ``where`` context to every failure:

    - a string spelling (``CURRENCY_PER_MONTH_PER_BG``, ``DIMENSIONLESS_PER_BG``,
      ``SILVER_PENNY_PER_YEAR``, or a bare base ``DIMENSIONLESS``), handed to
      :func:`parse_ttsim_unit`;
    - an already-built :class:`CompositeUnit`, passed through untouched.

    Everything else — pint syntax like ``"CURRENCY / year"``, ``None``, a YAML
    ``null`` — is rejected.

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
            return parse_ttsim_unit(value)
        except UnitDefinitionError:
            pass
    raise UnitDefinitionError(
        f"{where}: invalid unit declaration {value!r}. A unit must be a "
        "compositional spelling (e.g. CURRENCY_PER_MONTH_PER_BG, DIMENSIONLESS_PER_BG)."
    )


def build_registry() -> pint.UnitRegistry:
    """Create a registry holding the units TTSIM knows about.

    One registry per policy system (:class:`ttsim.tt.units.UnitSystem`), which then
    defines its own currencies and grouping levels into it. The vocabulary built here is
    the part every system shares.

    pint's defaults already provide the ``[time]`` units (``year``, ``month``, ``week``,
    ``day`` — with the per-year factors GETTSIM uses: 12, 365.25/7, 365.25) and
    ``[length]``/``[area]`` units (``meter``, ``hectare``). We add:

    - ``CURRENCY`` as the reference unit of a new ``[currency]`` dimension;
    - ``working_hour`` as the reference unit of a new ``[hours]`` dimension, isolated
      from pint's ``[time]`` ``hour``: ``working_hour / week`` is then ``[hours] /
      [time]`` rather than the bare number ``[time] / [time]``, so working hours cannot
      be confused with — or added to — a share, and the only conversion possible is
      re-basing the *period* denominator.
    - ``quarter_year`` for the ``_q`` suffix (pint's built-in ``quarter`` is a unit of
      mass);
    - ``calendar_year`` / ``calendar_month`` / ``calendar_day`` as affine *point* units.
      Subtracting two points yields pint's companion ``delta_calendar_*`` *duration*
      unit, which :attr:`TTSIMUnit.YEARS` / :attr:`TTSIMUnit.MONTHS` /
      :attr:`TTSIMUnit.DAYS` resolve to (each is ratio 1 against ``year`` / ``month`` /
      ``day``).

    pint's remaining built-ins parse, but :func:`parse_unit` rejects every token outside
    :data:`_ALLOWED_UNIT_TOKENS`, so they cannot appear in a declaration.
    """
    ureg = pint.UnitRegistry()
    ureg.define(f"{CURRENCY_TOKEN} = [currency]")
    ureg.define("working_hour = [hours]")
    ureg.define("quarter_year = year / 4 = quarter_of_year")
    ureg.define("calendar_year = year; offset: 1900")
    ureg.define("calendar_month = month; offset: 22800")  # 1900 * 12
    ureg.define("calendar_day = day; offset: 693975")  # 1900 * 365.25
    return ureg


#: The affine calendar-point units :func:`build_registry` defines — the only
#: offset units any TTSIM registry contains.
_CALENDAR_POINT_UNIT_NAMES = frozenset(
    {"calendar_year", "calendar_month", "calendar_day"}
)


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


_registered_currencies: set[str] = set()

#: Tolerance for the magnitude part of a unit-equivalence comparison.
_REL_TOL = 1e-9


def _grouping_level_unit_name(name: str) -> str:
    """The internal pint unit name anchoring a grouping level's dimension."""
    return f"{_GROUPING_LEVEL_PREFIX}{name}"


def registered_grouping_levels(registry: pint.UnitRegistry) -> set[str]:
    """The grouping levels a registry defines a dimension for."""
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
            f"Unknown grouping level {name!r}; expected one of {known}. A "
            f"grouping level exists for each `*_id` column of the policy "
            f"environment (GEP 10)."
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
    ``CURRENCY / month / [hh]``. An individual quantity carries no level — it is
    bare (``CURRENCY / month``). A head count over the same group is ``1 / [hh]``,
    so dividing a per-``[hh]`` amount by it cancels the ``[hh]`` down to a bare
    per-person amount.

    The division is *unit* arithmetic, not quantity arithmetic: attaching a level
    is index bookkeeping, so it must stay legal for offset (calendar-point) bases,
    which pint forbids in quantity multiplication.

    Raises:
        UnitDefinitionError: If the level has not been registered.
    """
    return unit / _grouping_level_unit(name=level, registry=registry)


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
    """Reject any unit token TTSIM does not know about."""
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
    if left == right:
        return True
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

    Callers that implement the affine ``+``/``-`` rules (the build-time unit check)
    detect a point this way and delegate the operation to pint rather than to the
    magnitude-equivalence check, which would wrongly reject the valid
    ``point + duration``. Detection is by the very property that defines an offset
    unit: it cannot be divided by itself.

    The only offset units :func:`build_registry` ever defines are the three
    :data:`_CALENDAR_POINT_UNIT_NAMES`, so a unit spelling none of them is
    resolved without the (comparatively expensive) pint probe — this predicate
    sits on the unit check's ``+``/``-`` path, where almost every operand is a
    plain magnitude unit.
    """
    if _CALENDAR_POINT_UNIT_NAMES.isdisjoint(to_units_container(unit)):
        return False
    quantity = registry.Quantity(1.0, unit)
    try:
        quantity / quantity
    except pint.OffsetUnitCalculusError:
        return True
    return False


def ttsim_unit_currency(ttsim_unit: CompositeUnit | None) -> str | None:
    """The concrete currency a TTSIM unit pins down, if any."""
    if not isinstance(ttsim_unit, CompositeUnit):
        return None
    return next(
        (name for name in _registered_currencies if name.upper() == ttsim_unit.base),
        None,
    )


def _pint_unit_currency(
    units: pint.Unit, registry: pint.UnitRegistry
) -> pint.Unit | None:
    """The currency component of a (possibly composite) pint unit, or ``None``."""
    for token in to_units_container(units):
        candidate = registry.parse_units(token)
        if _unit_is_currency(candidate):
            return candidate
    return None


def ttsim_unit_has_currency(ttsim_unit: CompositeUnit | None) -> bool:
    """Whether a TTSIM unit's base is a currency (agnostic or concrete)."""
    return (
        ttsim_unit_has_agnostic_currency(ttsim_unit)
        or ttsim_unit_currency(ttsim_unit) is not None
    )


def pint_unit_has_currency(units: pint.Unit, registry: pint.UnitRegistry) -> bool:
    """Whether a (possibly composite) pint unit carries a currency component."""
    return _pint_unit_currency(units=units, registry=registry) is not None


def ttsim_unit_has_agnostic_currency(ttsim_unit: CompositeUnit | None) -> bool:
    """Whether a TTSIM unit's base is the agnostic ``CURRENCY``."""
    return isinstance(ttsim_unit, CompositeUnit) and ttsim_unit.base == CURRENCY_TOKEN


def pint_unit_has_agnostic_currency(
    units: pint.Unit, registry: pint.UnitRegistry
) -> bool:
    """Whether a pint unit's currency component is the agnostic ``CURRENCY``.

    Distinguishes the two currency spellings when results are returned: a
    column resolves to the agnostic ``CURRENCY`` (it is computed in the
    computation currency and converted to the data currency), a parameter to
    its concrete statutory currency (never converted, labelled as declared) —
    GEP 10.
    """
    component = _pint_unit_currency(units=units, registry=registry)
    return component is not None and component == registry.parse_units(CURRENCY_TOKEN)


def ttsim_unit_with_agnostic_currency(ttsim_unit: CompositeUnit) -> CompositeUnit:
    """The TTSIM unit a node *derived* from a source with this unit carries.

    A concrete currency base is replaced by the agnostic ``CURRENCY``; every
    other base, and the area, period and level, are left untouched.
    """
    return (
        replace(ttsim_unit, base=CURRENCY_TOKEN)
        if ttsim_unit_currency(ttsim_unit) is not None
        else ttsim_unit
    )


def _pint_unit_with_currency(
    units: pint.Unit, currency: str, registry: pint.UnitRegistry
) -> pint.Unit:
    """Swap a pint unit's currency component for ``currency``; a no-op if it has none.

    The one currency move input and output handling share: the period, area
    and levels are left untouched. For results ``currency`` is the data
    currency (:func:`output_unit_in_data_currency`); for tagged input data it
    is the tag's concrete currency (:func:`resolve_ttsim_unit_for_input`).
    """
    component = _pint_unit_currency(units=units, registry=registry)
    if component is None:
        return units
    return units / component * registry.parse_units(currency)


#: The dimensionality-key prefix of a grouping-level dimension: the internal pint
#: unit name :data:`_GROUPING_LEVEL_PREFIX` wrapped in pint's ``[…]`` dimension
#: brackets (e.g. ``[grouping_level_hh]``).
_GROUPING_LEVEL_DIM_PREFIX = f"[{_GROUPING_LEVEL_PREFIX}"


def _grouping_levels_with_exponent(unit: pint.Unit) -> Iterator[tuple[str, Any]]:
    """Yield ``(level_name, exponent)`` for each grouping-level dimension of a unit.

    A negative exponent is a denominator level (``/[hh]``); grouping levels only
    ever appear as denominators. Non-grouping dimensions and pint's
    (never-occurring here) complex exponents are skipped.
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
    the level name (``"hh"``, ``"bg"``, …) found in the denominator, or ``None``
    for a bare unit. A head count ``1 / [hh]`` reports ``"hh"`` — its index level.
    """
    return next(
        (
            name
            for name, exponent in _grouping_levels_with_exponent(unit)
            if exponent < 0
        ),
        None,
    )


def input_target_unit_in_data_currency(
    units: pint.Unit, data_currency: str, registry: pint.UnitRegistry
) -> pint.Unit:
    """Restate the unit of an input column requested as a target.

    Such a column is returned exactly as provided — in the data currency —
    whatever currency its declaration pins down: agnostic (an ordinary column)
    or concrete (a data override of a parameter, declared ``DM_PER_MONTH`` but
    holding the user's euro values). The label follows the value, so any
    currency component is substituted with the data currency.
    """
    return _pint_unit_with_currency(
        units=units, currency=data_currency, registry=registry
    )


def output_unit_in_data_currency(
    units: pint.Unit, data_currency: str, registry: pint.UnitRegistry
) -> pint.Unit:
    """Restate a computed result column's resolved unit in the data currency."""
    if not pint_unit_has_agnostic_currency(units=units, registry=registry):
        return units
    return _pint_unit_with_currency(
        units=units, currency=data_currency, registry=registry
    )


def param_unit_in_computation_currency(
    units: pint.Unit, computation_currency: str, registry: pint.UnitRegistry
) -> pint.Unit:
    """Restate a requested parameter's resolved unit in its statutory currency."""
    return _pint_unit_with_currency(
        units=units, currency=computation_currency, registry=registry
    )


#: Reverse of the forward token→pint maps, for :func:`ttsim_unit_from_pint_unit`.
_PINT_NAME_TO_PERIOD_TOKEN = {v: k for k, v in _PERIOD_TOKEN_TO_PINT.items()}
_PINT_NAME_TO_BASE_TOKEN = {
    pint_name: token
    for token, pint_name in _TTSIM_UNIT_BASE_TO_PINT.items()
    if pint_name is not None
}


def ttsim_unit_from_pint_unit(
    units: pint.Unit, registry: pint.UnitRegistry
) -> CompositeUnit:
    """Reconstruct the TTSIM unit spelling of a resolved pint unit.

    The output-side inverse of :func:`resolve_ttsim_unit`: it labels a result-tree leaf
    with a :class:`CompositeUnit`, so the result tree is the same shape as the input
    tree. A resolved unit obeys the grammar, so each component maps back to one slot:

    - the currency / area / duration numerator → the base;
    - the flow period → the period;
    - the spelled group level → the level.

    Apply it to a unit already restated in the data currency
    (:func:`output_unit_in_data_currency`) so the base is a concrete currency (``EUR``),
    never the agnostic ``CURRENCY``.
    """
    currency = _pint_unit_currency(units=units, registry=registry)
    period = _flow_period_of(units=units, registry=registry)
    base = str(currency).upper() if currency is not None else "DIMENSIONLESS"
    area: str | None = None
    level: str | None = None
    for name, exponent in _grouping_levels_with_exponent(units):
        if exponent < 0:
            # A denominator level (a group level) is spelled.
            level = name.upper()
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
    """Return a unit's flow period — its time component in the *denominator*."""
    for token, exponent in to_units_container(units).items():
        if isinstance(exponent, complex):  # pint exponents are real; narrow for ty
            continue
        candidate = registry.parse_units(token)
        if exponent < 0 and _unit_is_time(candidate):
            return candidate
    return None


def _suffix_period_of(
    column_label: str | None, registry: pint.UnitRegistry
) -> pint.Unit | None:
    """Return the flow period named by a column's GEP-1 time suffix.

    Example:
        >>> _suffix_period_of(column_label="income_m", registry=ureg)
        <Unit('month')>
        >>> _suffix_period_of(column_label="wealth", registry=ureg)
        None
    """
    if column_label is None:
        return None
    match = get_re_pattern_for_all_time_units_and_groupings(
        time_units=tuple(TIME_UNIT_ID_TO_PINT_NAME),
        grouping_levels=tuple(sorted(registered_grouping_levels(registry))),
    ).fullmatch(column_label)
    if match is None or not match.group("time_unit"):
        return None
    return registry.parse_units(TIME_UNIT_ID_TO_PINT_NAME[match.group("time_unit")])


def _fail_if_tag_period_disagrees_with_suffix(
    units: pint.Unit, *, column_label: str | None, registry: pint.UnitRegistry
) -> None:
    """Strict period guard: a pint tag's flow period must match the column's
    GEP-1 time suffix exactly — including both absent."""
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


def resolve_ttsim_unit_for_input(
    unit: CompositeUnit, registry: pint.UnitRegistry
) -> pint.Unit:
    """The concrete pint unit used to strip a :class:`UnitAnnotatedColumn`.

    Resolves the tag with its concrete currency and flow period — the two axes the
    boundary acts on: the currency is converted at the boundary and the period
    is screened against the name suffix. Grouping levels do not affect the
    magnitude and are omitted, so this needs no registered level dimension.
    """
    resolved = resolve_ttsim_unit(unit=unit, registry=registry, with_level=False)
    concrete = ttsim_unit_currency(unit)
    return (
        resolved
        if concrete is None
        else _pint_unit_with_currency(
            units=resolved, currency=concrete, registry=registry
        )
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
    into the computation currency happens right after, in
    ``input_data_in_computation_currency``, for tagged and untagged columns
    alike.

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
    source_currency = _pint_unit_currency(units=quantity.units, registry=registry)
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
    """Normalise a ``SUM`` over a boolean to a ``COUNT`` for unit purposes."""
    if agg_type is AggType.SUM and source_is_boolean:
        return AggType.COUNT
    return agg_type


def unit_for_aggregation(
    source_unit: UnitDeclaration,
    agg_type: AggType,
    target_level: str | None = None,
    *,
    source_is_boolean: bool = False,
) -> UnitDeclaration:
    """Auto-assign the *declared* unit of an aggregation node.

    The single source of truth for an automatically added aggregation's token
    (``my_col`` → ``my_col_hh``); author-written ``@agg_by_group_function`` /
    ``@agg_by_p_id_function`` nodes declare their unit explicitly (GEP 10).
    ``target_level`` is the group the node aggregates *to*, or ``None`` for an
    individual (bare) result — an ``agg_by_p_id`` node:

    - a **head count** — ``COUNT``, or a ``SUM`` over a boolean source
      (``source_is_boolean``, counting the persons its flag is true for) — is the
      dimensionless :attr:`TTSIMUnit.DIMENSIONLESS` per ``target_level``
      (``DIMENSIONLESS_PER_HH`` = ``1 / [hh]``); bare at an individual target;
    - ``SUM`` / ``MIN`` / ``MAX`` over a non-boolean source are properties of the
      **target** group whatever the source's base (GEP 10): they keep the
      source's physical token and take the target level — spelled for a group
      (``CURRENCY_PER_MONTH_PER_FAM``, ``MONTHS_PER_FG`` for an ``_fg`` extreme of
      a bare duration), and **bare** for an individual ``agg_by_p_id`` result;
    - ``MEAN`` is a per-head average that belongs to the **individual**
      (``MEAN = SUM / COUNT`` cancels the group), so its result is **bare** — the
      source's group level, if any, is dropped;
    - ``ANY`` / ``ALL`` yield a boolean, a dimensionless quantity at the target
      level (``DIMENSIONLESS_PER_<target_level>``); bare at an individual target.

    An individual result carries no grouping level, so an ``agg_by_p_id``
    ``COUNT`` is bare ``DIMENSIONLESS``.

    Args:
        source_unit: The source column's ``unit`` — a :class:`CompositeUnit`
            (:data:`UNSET_UNIT` if the source does not declare one).
        agg_type: The :class:`ttsim.tt.aggregation.AggType` of the aggregation.
        target_level: The group level the node aggregates to (read off its name
            suffix); ``None`` for an individual (bare) result.
        source_is_boolean: Whether the aggregated source column is boolean — a
            ``SUM`` over it is then minted as a head count (see
            :func:`head_count_from_boolean_sum`).

    Returns:
        The auto-assigned unit. ``DIMENSIONLESS`` (per target) for a ``COUNT``
        head count and for a boolean ``ANY`` / ``ALL`` result; otherwise the
        source token at the target (``SUM`` / ``MIN`` / ``MAX``) or bare
        (``MEAN``, and any individual ``agg_by_p_id``)
        (:data:`UNSET_UNIT` when the source itself lacks a declaration, which the
        mandatory-units check then reports against the source).
    """
    agg_type = head_count_from_boolean_sum(
        agg_type=agg_type, source_is_boolean=source_is_boolean
    )
    if agg_type in (AggType.COUNT, AggType.ANY, AggType.ALL):
        base = TTSIMUnit.DIMENSIONLESS
        return base if target_level is None else base.PER_LEVEL(target_level)
    if is_unset_unit(source_unit):
        return source_unit
    if agg_type is AggType.MEAN or target_level is None:
        # A MEAN is a per-head average, and an agg_by_p_id result is an individual
        # property: both are bare, so the source's group level is dropped.
        return replace(source_unit, level=None)
    return replace(source_unit, level=target_level.upper())


def resolved_unit_for_aggregation(
    *,
    agg_type: AggType,
    target_level: str | None,
    registry: pint.UnitRegistry,
    source_unit: pint.Unit | None = None,
    source_level: str | None = None,
) -> pint.Unit:
    """The resolved unit of an aggregation node, level-aware.

    The level-aware counterpart of :func:`unit_for_aggregation`: it operates on
    *resolved* pint units (the physical token combined with its flow period and
    grouping level) and is where a grouping level is minted, swapped, or
    preserved. ``target_level`` is the group being aggregated *to* (read off the
    aggregation suffix), or ``None`` for an individual (bare) result — an
    ``agg_by_p_id`` node; ``source_level`` is the source column's own level
    (``None`` for a bare source such as an age).

    - ``SUM`` / ``MIN`` / ``MAX`` results are properties of the **target** group
      whatever the source's base (GEP 10): the source level (if any) is swapped
      for the target and a bare source *acquires* it — an ``_hh`` sum of a person
      income is ``CURRENCY/[hh]``, an ``_fg`` min of a bare age is ``MONTHS/[fg]``.
      At an individual target (an ``agg_by_p_id`` node) the result is **bare**.
    - ``MEAN`` is a per-head average that belongs to the **individual**
      (``MEAN = SUM / COUNT``, and leveling it to the target would break
      ``mean · count = sum``), so the result is **bare** — the source level, if
      any, is dropped.
    - ``COUNT`` mints a head count ``1 / [target]`` — persons per target group —
      independent of the source; bare at an individual target.
    - ``ANY`` / ``ALL`` yield a boolean *at the target level* — ``1 / [target]``
      — so a group-level indicator carries the level its name claims; bare at an
      individual target.

    ``COUNT`` and ``ANY`` / ``ALL`` are independent of the source, so their
    ``source_unit`` / ``source_level`` default to ``None`` and are ignored.

    Args:
        agg_type: The :class:`ttsim.tt.aggregation.AggType` of the aggregation.
        target_level: The group level being aggregated to (e.g. ``"hh"``), or
            ``None`` for an individual (bare) result.
        source_unit: The source column's resolved pint unit. Required for the
            value aggregations ``SUM`` / ``MEAN`` / ``MIN`` / ``MAX``; ignored
            (and ``None``) for ``COUNT`` / ``ANY`` / ``ALL``.
        source_level: The source column's grouping level (e.g. ``"hh"``), or
            ``None`` if the source carries no level (or is ignored, as above).

    Returns:
        The aggregation node's resolved pint unit.

    Raises:
        UnitDefinitionError: If ``target_level`` or ``source_level`` names an
            unregistered grouping level.
        ValueError: If a value aggregation is requested without a ``source_unit``.
    """
    if agg_type in (AggType.ANY, AggType.ALL, AggType.COUNT):
        if target_level is None:
            return registry.dimensionless
        return divide_by_grouping_level(
            unit=registry.dimensionless, level=target_level, registry=registry
        )
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
    # A MEAN is bare (individual), and an agg_by_p_id result (``target_level is
    # None``) is bare too: the level-stripped source stands as-is.
    if agg_type is AggType.MEAN or target_level is None:
        return stripped
    return divide_by_grouping_level(
        unit=stripped, level=target_level, registry=registry
    )


def fail_if_units_are_missing(
    units_by_qname: Mapping[str, UnitDeclaration],
) -> None:
    """Data-independent check that every node declares a unit.

    Raises:
        UnitDefinitionError: If any qualified name maps to :data:`UNSET_UNIT`.
    """
    missing = sorted(
        qname for qname, unit in units_by_qname.items() if is_unset_unit(unit)
    )
    if missing:
        raise UnitDefinitionError(
            "The following nodes are missing a mandatory `unit=` declaration "
            f"(GEP 10; declare `unit=TTSIMUnit.DIMENSIONLESS` / `unit: DIMENSIONLESS` "
            f"for a dimensionless quantity): {', '.join(missing)}."
        )


def register_unit_builder_levels(names: Iterable[str]) -> None:
    """Give the fluent builder a ``per_<level>`` attribute for each level."""
    names = list(names)
    fail_if_grouping_level_names_are_invalid(names=names)
    for name in names:
        if name in _unit_builder_levels:
            continue
        _unit_builder_levels.add(name)
        setattr(
            CompositeUnit,
            f"PER_{name.upper()}",
            property(lambda self, level=name: self.PER_LEVEL(level)),
        )


def fail_if_grouping_level_names_are_invalid(names: Iterable[str]) -> None:
    """Reject a grouping-level name the builder cannot own.

    A level claims the builder step ``PER_<NAME>`` on :class:`CompositeUnit`,
    which is a process-global class shared by every system. Two kinds of
    names are therefore refused:

    - any name whose step is already one of the closed area/period steps, since
      a ``month`` level would turn ``PER_MONTH`` from a flow period into a
      grouping level for every declaration in the process;
    - any name that is not lower-case, because a level is registered verbatim but
      resolved lower-cased (:func:`ttsim.tt.units.resolve_ttsim_unit`), so
      ``"HH"`` would register a level that ``.PER_HH`` cannot resolve.

    Raises:
        UnitDefinitionError: If any name is refused.
    """
    for name in names:
        if name != name.lower():
            raise UnitDefinitionError(
                f"Grouping level {name!r} must be lower-case: a level is registered "
                f"under the name given but resolved lower-cased, so {name!r} would "
                f"register a level that `.PER_{name.upper()}` cannot resolve. Spell it "
                f"{name.lower()!r} (GEP 10)."
            )
        step = f"PER_{name.upper()}"
        if name not in _unit_builder_levels and hasattr(CompositeUnit, step):
            raise UnitDefinitionError(
                f"Grouping level {name!r} would claim the builder step {step}, which "
                f"is already a unit denominator. Rename the group so its level does "
                f"not collide with an area or period (GEP 10)."
            )


def register_grouping_levels(names: Iterable[str], registry: pint.UnitRegistry) -> None:
    """Register grouping levels as base dimensions in a policy system's registry.

    Args:
        names: The grouping-level names to register (e.g. ``["hh", "bg"]``). registry:
        The policy system's registry to define the dimensions in.
    """
    names = list(names)
    define_grouping_level_dimensions(names=names, registry=registry)
    for name in names:
        _ALLOWED_UNIT_TOKENS.add(_grouping_level_unit_name(name))
    # Packages that use the builder at import time call
    # `register_unit_builder_levels` directly, before their declarations run.
    register_unit_builder_levels(names)


def define_grouping_level_dimensions(
    names: Iterable[str], registry: pint.UnitRegistry
) -> None:
    """Define each grouping level's base dimension in ``registry``."""
    names = list(names)
    fail_if_grouping_level_names_are_invalid(names=names)
    for name in names:
        unit_name = _grouping_level_unit_name(name)
        if unit_name not in registry:
            registry.define(f"{unit_name} = [{unit_name}]")


@dataclasses.dataclass(frozen=True, eq=False, kw_only=True)
class UnitSystem:
    """The currencies, statutory-currency mapping, and registry of one policy system.

    A package builds one system and exports it as a singleton, so a system is identified
    by object identity: ``eq=False`` keeps the inherited identity-based ``__hash__``,
    which lets a system key an ``lru_cache`` (its ``statutory_currencies`` mapping is
    otherwise unhashable).

    A policy package declares its system once and exports it::

        UNIT_SYSTEM = UnitSystem(
            base_currency="EUR", other_currencies={"DM": "EUR / 1.95583"},
            statutory_currencies={"0001-01-01": "DM", "2002-01-01": "EUR"},
        )

    All of a system's currencies are interconvertible
    (:meth:`currency_conversion_factor`).

    Raises:
        UnitDefinitionError: If a currency name is unusable as a ``TTSIMUnit``
            base — it claims the agnostic ``CURRENCY`` base, names a non-currency base
            of the shared vocabulary, differs from another registered currency only in
            case, or spells the ``_PER_`` denominator delimiter — if a definition does
            not resolve to the ``[currency]`` dimension or does not reference exactly
            one of this system's currencies, or if the statutory-currency mapping is
            empty, is keyed by anything other than an ISO date, or names a currency the
            system does not have.
    """

    base_currency: str
    """The system's unit of account, and the default data currency. Defined as
    factor 1 against the abstract ``[currency]`` reference; every other currency
    is defined relative to it or to another already-defined one."""

    other_currencies: Mapping[str, str] = dataclasses.field(
        default_factory=lambda: MappingProxyType({})
    )
    """Each further currency, mapped to a pint-parseable definition relative to
    an already-defined currency of this system (``{"DM": "EUR / 1.95583"}``).
    Definitions are applied in order, so one may reference an earlier one."""

    statutory_currencies: Mapping[str, str]
    """The currency statutes denominate their numbers in, keyed by the dashed ISO
    start date it applies from (until the next entry's)."""

    registry: pint.UnitRegistry = dataclasses.field(init=False, repr=False)
    """The system's own pint registry: the shared vocabulary plus this system's
    currency definitions."""

    currencies: frozenset[str] = dataclasses.field(init=False)
    """Every currency name this system defines — the base and the others."""

    statutory_currency_by_start_date: tuple[tuple[datetime.date, str], ...] = (
        dataclasses.field(init=False, repr=False)
    )
    """:attr:`statutory_currencies` parsed and sorted by start date."""

    def __post_init__(self) -> None:
        # Snapshot the caller's containers: everything derived below (the parsed
        # dates, the currency set, the registry) would otherwise silently disagree
        # with them if the caller mutated theirs afterwards.
        object.__setattr__(
            self,
            "statutory_currencies",
            MappingProxyType(dict(self.statutory_currencies)),
        )
        object.__setattr__(
            self, "other_currencies", MappingProxyType(dict(self.other_currencies))
        )
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
        self._publish_currencies()

    def currency_conversion_factor(
        self, *, source_currency: str, target_currency: str
    ) -> float:
        """The factor converting ``source_currency`` into ``target_currency``.

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
            f"Extend the mapping this policy system declares."
        )

    def _define_currencies(self) -> None:
        """Define the base and every other currency in the system's registry.

        The base is factor 1 against the abstract :data:`CURRENCY_TOKEN` reference;
        every other currency is defined relative to an already-defined one, so all of
        them chain back to the base and are interconvertible.
        """
        self._define_one_currency(name=self.base_currency, definition=CURRENCY_TOKEN)
        defined = {self.base_currency}
        for name, definition in self.other_currencies.items():
            self._fail_if_definition_references_no_known_currency(
                name=name, definition=definition, defined=defined
            )
            self._define_one_currency(name=name, definition=definition)
            defined.add(name)
        claimed = set(_registered_currencies)
        for name in (self.base_currency, *self.other_currencies):
            self._fail_if_name_is_unusable_as_a_unit_base(name=name, claimed=claimed)
            claimed.add(name)

    def _publish_currencies(self) -> None:
        """Widen the process-global vocabulary by this system's currencies."""
        for name in self.currencies:
            _ALLOWED_UNIT_TOKENS.add(name)
            _registered_currencies.add(name)
            # Surface the concrete currency on the `TTSIMUnit` builder (`TTSIMUnit.EUR`,
            # `TTSIMUnit.DM`, `TTSIMUnit.SILVER_PENNY`) so it can tag a
            # `UnitAnnotatedColumn` of input data. A column/function declaration
            # still rejects a concrete base (`resolve_ttsim_unit_for_column`);
            # this only makes it reachable.
            setattr(TTSIMUnit, name.upper(), CompositeUnit(base=name.upper()))

    def _fail_if_name_is_unusable_as_a_unit_base(
        self, name: str, claimed: set[str]
    ) -> None:
        """Reject a currency name the ``TTSIMUnit`` base namespace cannot carry.

        A currency reaches the builder namespace under its upper-cased name, and
        :func:`ttsim.tt.units.parse_ttsim_unit` matches a base against that same
        upper-cased form. Four names are refused:

        - the agnostic :data:`CURRENCY_TOKEN`, which every currency-agnostic
          declaration spells;
        - a non-currency base of the shared vocabulary (``HOURS``, ``HECTARE``,
          ``DIMENSIONLESS``, …), which the currency would shadow for every
          policy package in the process;
        - a name differing only in case from a currency in ``claimed`` — the
          process-global registry widened by this system's earlier currencies:
          both project onto one base, so one shadows the other on the builder
          and :func:`ttsim.tt.units.ttsim_unit_currency` resolves the base back
          to either of them, letting a conversion pick the wrong exchange rate;
        - a name spelling the :data:`ttsim.tt.units._PER` denominator delimiter,
          whose base would parse back as a base plus a denominator, so the token
          would not round-trip through its canonical string or YAML form.
        """
        base = name.upper()
        if base == CURRENCY_TOKEN:
            raise UnitDefinitionError(
                f"Cannot register currency {name!r}: {CURRENCY_TOKEN!r} is the "
                f"agnostic currency base every currency-agnostic declaration "
                f"spells, so a concrete currency claiming it would make every such "
                f"declaration name that currency. Pick another name (GEP 10)."
            )
        if base in _TTSIM_UNIT_BASE_TO_PINT:
            raise UnitDefinitionError(
                f"Cannot register currency {name!r}: {base!r} is a non-currency "
                f"unit base the shared unit vocabulary already owns, so registering "
                f"it would silently shadow that base for every policy package in "
                f"the process. Pick another name (GEP 10)."
            )
        if _PER in base:
            head, denominator = base.split(_PER, 1)
            raise UnitDefinitionError(
                f"Cannot register currency {name!r}: {_PER!r} separates a unit from "
                f"its denominators, so the base {base!r} would parse back as "
                f"{head!r} denominated by {denominator!r} instead of round-tripping "
                f"as one base. Pick a name without it (GEP 10)."
            )
        shadowed = sorted(
            other for other in claimed if other != name and other.upper() == base
        )
        if shadowed:
            raise UnitDefinitionError(
                f"Cannot register currency {name!r}: currency "
                f"{', '.join(repr(other) for other in shadowed)} already claims the "
                f"unit base {base!r}, so one would silently shadow the other on the "
                f"`TTSIMUnit` builder and a conversion could pick either one's "
                f"exchange rate. Currency names must differ by more than case "
                f"(GEP 10)."
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
