"""The pint-based unit framework.

Establishes the unit vocabulary that checks the dimensional soundness of the
taxes-and-transfers DAG, plus the build-time machinery that runs the check. GEP 10
specifies the vocabulary and the declaration rules; error messages point back to it.

Two kinds of object are called a "unit" here, and every identifier in this package names
which one it means:

- a **TTSIM unit** (``ttsim_unit``) is a :class:`CompositeUnit`, the token a policy
  author *declares*. It is a plain frozen dataclass, needs no registry, and exists
  before any build starts.
- a **pint unit** (``pint_unit``) is a :class:`pint.Unit`, what a TTSIM unit *resolves
  to* against a :class:`pint.UnitRegistry`. It carries the dimensions the check does
  arithmetic on and lives only at build/check time.

Every declaration is a fully-spelled :class:`CompositeUnit` — a base optionally divided
by an area, by working hours, by a period, and by a grouping level, in that canonical
order. It has two round-tripping spellings (via :func:`ttsim_unit_from_string` /
:func:`str`):

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
from enum import Enum, auto
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar

import pint
from pint.util import to_units_container

from ttsim.exceptions import (
    UnitConsistencyError,
    UnitDefinitionError,
)
from ttsim.interface_dag_elements.shared import (
    get_re_pattern_for_all_time_units_and_groupings,
)
from ttsim.tt.aggregation import AggType
from ttsim.typing import OrderedQNames

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

CURRENCY_TOKEN = "CURRENCY"  # noqa: S105 (a unit token, not a secret)

_GROUPING_LEVEL_PREFIX = "grouping_level_"

#: The dimensionality-key prefix of a grouping-level dimension: the internal pint
#: unit name :data:`_GROUPING_LEVEL_PREFIX` wrapped in pint's ``[…]`` dimension
#: brackets (e.g. ``[grouping_level_hh]``).
_GROUPING_LEVEL_DIM_PREFIX = f"[{_GROUPING_LEVEL_PREFIX}"

_PER = "_PER_"

#: A whole identifier in a currency ``value`` expression (``"CASTAR / 4"`` →
#: ``CASTAR``). Matching whole tokens keeps a currency whose name is a substring
#: of another's (``STAR`` in ``CASTAR``) from reading as a reference to it.
_UNIT_TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

_AREA_TOKEN_TO_PINT: dict[str, str] = {"SQUARE_METER": "meter ** 2"}

_HOURS_TOKEN_TO_PINT: dict[str, str] = {"HOURS": "working_hour"}

_TTSIM_UNIT_BASE_TO_PINT: dict[str, str | None] = {
    "DIMENSIONLESS": None,
    "HOURS": "working_hour",
    "SQUARE_METER": "meter ** 2",
    "HECTARE": "hectare",
    "YEARS": "delta_calendar_year",
    "QUARTERS": "calendar_quarter_duration",
    "MONTHS": "delta_calendar_month",
    "DAYS": "delta_calendar_day",
    "CALENDAR_YEAR": "calendar_year",
    "CALENDAR_QUARTER": "calendar_quarter",
    "CALENDAR_MONTH": "calendar_month",
    "CALENDAR_DAY": "calendar_day",
}

#: Reverse of the forward token→pint maps, for :func:`ttsim_unit_from_pint_unit`.
_PINT_NAME_TO_PERIOD_TOKEN = {v: k for k, v in _PERIOD_TOKEN_TO_PINT.items()}

_PINT_NAME_TO_BASE_TOKEN = {
    pint_name: token
    for token, pint_name in _TTSIM_UNIT_BASE_TO_PINT.items()
    if pint_name is not None
}

#: The affine calendar-point unit :func:`build_registry` defines.
_CALENDAR_POINT_UNIT_NAMES = frozenset({"calendar_year"})

#: Calendar positions within a larger unit. They are separate dimensions so Pint
#: cannot silently treat February as two months or day 31 as a 31-day duration.
_CALENDAR_ORDINAL_UNIT_NAMES = frozenset(
    {"calendar_quarter", "calendar_month", "calendar_day"}
)

#: The dimension names :func:`build_registry` mints for currency and takes from
#: pint for time. Every registry spells them the same way, and a pint
#: dimensionality compares by content, so the boundary helpers pick a unit's
#: currency / flow-period component out by matching against these directly —
#: no registry needed.
_CURRENCY_DIMENSIONALITY: Mapping[str, Any] = {"[currency]": 1}

_TIME_DIMENSIONALITY: Mapping[str, Any] = {"[time]": 1}

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
    "calendar_quarter",
    "calendar_month",
    "calendar_day",
    "delta_calendar_year",
    "calendar_quarter_duration",
    "delta_calendar_quarter",
    "delta_calendar_month",
    "delta_calendar_day",
}

_registered_currencies: set[str] = set()

#: The grouping levels the fluent builder offers a ``per_<level>`` attribute for
#: (e.g. ``per_bg``). Populated per package by
#: :func:`register_unit_builder_levels`
_unit_builder_levels: set[str] = set()

#: Tolerance for the magnitude part of a unit-equivalence comparison.
_REL_TOL = 1e-9

_CastValueT = TypeVar("_CastValueT")


@dataclass(frozen=True)
class CompositeUnit:
    """A fully-spelled TTSIM unit — *the* declaration type.

    A base divided by at most one *area* **or** one span of *working hours*, one
    *period*, and one *level*, held in that canonical order; the builder methods
    enforce the order and the area/hours exclusion, so a non-canonical chain
    (``.PER_BG.PER_MONTH``) or a doubly-denominated one
    (``.PER_SQUARE_METER.PER_HOURS``) is a definition error. Two round-tripping
    spellings (via :func:`str`):

    - fluent, off a base (``TTSIMUnit.CURRENCY.PER_MONTH.PER_BG``);
    - flat canonical string, parsed by :func:`ttsim_unit_from_string`
      (``"CURRENCY_PER_MONTH_PER_BG"``).

    It resolves to a :class:`pint.Unit` via :func:`pint_unit_from_ttsim_unit`.
    """

    base: str
    area: str | None = None
    hours: str | None = None
    period: str | None = None
    level: str | None = None

    if TYPE_CHECKING:
        # Per-level builder steps are added at runtime, so tell `ty` any builder
        # attribute yields a CompositeUnit. Runtime keeps strict lookup, so an
        # unregistered `per_<level>` still raises AttributeError.
        def __getattr__(self, name: str) -> CompositeUnit: ...

    def __str__(self) -> str:
        parts = [self.base, self.area, self.hours, self.period, self.level]
        return _PER.join(part for part in parts if part is not None)

    def __repr__(self) -> str:
        return f"CompositeUnit({self})"

    @property
    def is_flow(self) -> bool:
        """Whether this unit is a flow — i.e. has a period denominator."""
        return self.period is not None

    def _with_area(self, area: str) -> CompositeUnit:
        if self.hours is not None:
            raise UnitDefinitionError(
                f"Cannot add the area '{area}' to '{self}': a unit is denominated "
                f"by an area or by working hours, never by both."
            )
        if self.area is not None or self.period is not None or self.level is not None:
            raise UnitDefinitionError(
                f"Cannot add the area '{area}' to '{self}': the canonical order is "
                f"base _PER_ <area> _PER_ <hours> _PER_ <period> _PER_ <level>, "
                f"with at most one area."
            )
        return replace(self, area=area)

    def _with_hours(self, hours: str) -> CompositeUnit:
        if self.area is not None:
            raise UnitDefinitionError(
                f"Cannot add working hours '{hours}' to '{self}': a unit is "
                f"denominated by an area or by working hours, never by both."
            )
        if self.hours is not None or self.period is not None or self.level is not None:
            raise UnitDefinitionError(
                f"Cannot add working hours '{hours}' to '{self}': the canonical "
                f"order is base _PER_ <area> _PER_ <hours> _PER_ <period> _PER_ "
                f"<level>, with at most one hours denominator."
            )
        return replace(self, hours=hours)

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

        Mutually exclusive with the area: a price is per an area or per working
        hours, never per both.
        """
        return self._with_hours("HOURS")

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

    QUARTERS = CompositeUnit(base="QUARTERS")
    """A *duration* in quarters. The point counterpart is
    :attr:`CALENDAR_QUARTER`."""

    MONTHS = CompositeUnit(base="MONTHS")
    """A *duration* in months. The point counterpart is :attr:`CALENDAR_MONTH`."""

    DAYS = CompositeUnit(base="DAYS")
    """A *duration* in days. The point counterpart is :attr:`CALENDAR_DAY`."""

    CALENDAR_YEAR = CompositeUnit(base="CALENDAR_YEAR")
    """A *point* on the calendar measured in years: a birth year, the policy
    year. Two calendar years subtract to a :attr:`YEARS` duration."""

    CALENDAR_QUARTER = CompositeUnit(base="CALENDAR_QUARTER")
    """A *point* on the calendar measured in quarters."""

    CALENDAR_MONTH = CompositeUnit(base="CALENDAR_MONTH")
    """A *point* on the calendar measured in months."""

    CALENDAR_DAY = CompositeUnit(base="CALENDAR_DAY")
    """A *point* on the calendar measured in days."""


@dataclass(frozen=True)
class UnitAnnotatedColumn:
    """A column of data paired with the :class:`CompositeUnit` it is measured in."""

    values: Any
    """Any leaf the ordinary input tree accepts (a list, a numpy/JAX array, a
    ``pd.Series``); canonicalized downstream."""
    unit: CompositeUnit
    """The column's TTSIM unit, built off :class:`TTSIMUnit`."""


@dataclass(frozen=True)
class UnsetUnit:
    """Sentinel type for an unspecified quantity unit.

    Test for it with ``isinstance(value, UnsetUnit)``, never with
    ``value is UNSET_UNIT``: ttsim cloudpickles DAG-built objects, and a
    round-trip mints a fresh instance, so object identity does not survive it
    while the type does.
    """


UNSET_UNIT = UnsetUnit()


UnitDeclaration: TypeAlias = CompositeUnit | UnsetUnit


class QuantityKind(Enum):
    """Narrow semantic evidence used by grouping-level unit checks.

    These labels do not add dimensions to Pint. They record only the two cases in
    which a dimensionless group denominator is meaningful: a head count and a
    yes/no indicator. Everything else deliberately remains generic.
    """

    GENERIC = auto()
    COUNT = auto()
    INDICATOR = auto()


@dataclass(frozen=True)
class InputOutputUnits:
    """The two axes of a schedule builder's output.

    A ``@param_function`` whose body builds a
    :class:`~ttsim.tt.param_objects.PiecewisePolynomialParamValue` or a
    :class:`~ttsim.tt.param_objects.ConsecutiveIntLookupTableParamValue` is a *function
    between quantities*, so it declares that function's domain and range instead of one
    ``unit=``: ``unit=InputOutputUnits(input_unit=TTSIMUnit.CURRENCY,
    output_unit=TTSIMUnit.CURRENCY.PER_YEAR)``. A schedule-typed field of a parameter
    dataclass states the same pair in its ``Annotated[…]`` metadata.

    A multi-dimensional :class:`ConsecutiveIntLookupTableParamValue` is keyed by several
    axes at once — ``look_up(anzahl_personen_hh, mietstufe_hh)`` — whose units may
    differ, so ``input_unit`` may be a **tuple** screened positionally; the call must
    supply exactly as many arguments as declared axes. ``piecewise_polynomial`` takes
    one domain argument, so a tuple on a piecewise builder is a contract error.

    Both axes are currency-agnostic, exactly like a column/function declaration.
    """

    input_unit: CompositeUnit | tuple[CompositeUnit, ...]
    """The unit each domain argument (``look_up`` index, ``piecewise_polynomial``
    ``x``) is screened against — a single :class:`CompositeUnit` applied to every
    argument, or a tuple screened positionally (argument ``i`` against axis
    ``i``)."""
    output_unit: CompositeUnit
    """The unit the schedule produces at every call site."""
    input_kind: QuantityKind | tuple[QuantityKind, ...] = QuantityKind.GENERIC
    """Semantic evidence for each input axis.

    A tuple must parallel a tuple-valued ``input_unit``. Authors need to set this
    only for a dimensionless axis carrying a grouping level; ordinary schedule
    axes stay :attr:`QuantityKind.GENERIC`.
    """
    output_kind: QuantityKind = QuantityKind.GENERIC
    """Semantic evidence for the output axis, under the same narrow rule."""

    def __post_init__(self) -> None:
        if isinstance(self.input_kind, tuple) and (
            not isinstance(self.input_unit, tuple)
            or len(self.input_kind) != len(self.input_unit)
        ):
            raise UnitDefinitionError(
                "InputOutputUnits: tuple-valued `input_kind` must have exactly one "
                "entry for each tuple-valued `input_unit` axis (GEP 10)."
            )


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

    ``unit`` is built off :class:`TTSIMUnit` or spelled flat.
    """
    return value


def ttsim_unit_from_string(spelling: str) -> CompositeUnit:
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
        elif kind == "hours":
            unit = unit._with_hours(token)  # noqa: SLF001
        elif kind == "period":
            unit = unit._with_period(token)  # noqa: SLF001
        else:
            unit = unit._with_level(token)  # noqa: SLF001
    return unit


def ttsim_unit_from_yaml_value(
    value: str | CompositeUnit,
    *,
    where: str,
) -> CompositeUnit:
    """The YAML boundary: turn a raw ``unit:`` value into a TTSIM unit.

    This is the boundary wrapper around :func:`ttsim_unit_from_string`, not a second
    parser. It admits the two shapes a declaration can arrive in and attaches a
    ``where`` context to every failure:

    - a string spelling (``CURRENCY_PER_MONTH_PER_BG``, ``DIMENSIONLESS_PER_BG``,
      ``SILVER_PENNY_PER_YEAR``, or a bare base ``DIMENSIONLESS``), handed to
      :func:`ttsim_unit_from_string`;
    - an already-built :class:`CompositeUnit`, passed through untouched.

    Everything else — pint syntax like ``"CURRENCY / year"``, ``None``, a YAML
    ``null`` — is rejected.

    Raises:
        UnitDefinitionError: If the value is not part of the vocabulary.
    """
    if isinstance(value, CompositeUnit):
        return value
    if isinstance(value, str):
        try:
            return ttsim_unit_from_string(value)
        except UnitDefinitionError:
            pass
    raise UnitDefinitionError(
        f"{where}: invalid unit declaration {value!r}. A unit must be a "
        "compositional spelling (e.g. CURRENCY_PER_MONTH_PER_BG, DIMENSIONLESS_PER_BG)."
    )


def ttsim_unit_from_pint_unit(
    units: pint.Unit, registry: pint.UnitRegistry
) -> CompositeUnit:
    """Reconstruct the TTSIM unit spelling of a resolved pint unit.

    The output-side inverse of :func:`pint_unit_from_ttsim_unit`: it labels a
    result-tree leaf with a :class:`CompositeUnit`, so the result tree is the same
    shape as the input tree. A resolved unit obeys the grammar, so each component
    maps back to one slot:

    - the currency / area / duration numerator → the base;
    - the flow period → the period;
    - the spelled group level → the level.

    Apply it to a unit already restated in a concrete currency
    (:func:`pint_unit_with_currency`) so the base is that currency (``EUR``), never the
    agnostic ``CURRENCY``.
    """
    currency = _pint_unit_currency(units=units, registry=registry)
    period = _flow_period_of(units=units, registry=registry)
    base = str(currency).upper() if currency is not None else "DIMENSIONLESS"
    area: str | None = None
    hours: str | None = None
    level: str | None = None
    for name, exponent in _grouping_levels_with_exponent(units):
        if exponent < 0:
            # A denominator level (a group level) is spelled.
            level = name.upper()
    without_levels = to_units_container(
        _unit_without_grouping_levels(unit=units, registry=registry)
    )
    for token, exponent in without_levels.items():
        if isinstance(exponent, complex):  # pint exponents are real; narrow for ty
            continue
        if token == "meter":  # noqa: S105 (a pint unit token, not a secret)
            base = "SQUARE_METER" if exponent > 0 and currency is None else base
            area = "SQUARE_METER" if exponent < 0 else area
        elif token == "working_hour" and exponent < 0:  # noqa: S105
            hours = "HOURS"
        elif currency is None and exponent > 0 and token in _PINT_NAME_TO_BASE_TOKEN:
            base = _PINT_NAME_TO_BASE_TOKEN[token]
    return CompositeUnit(
        base=base,
        area=area,
        hours=hours,
        period=_PINT_NAME_TO_PERIOD_TOKEN[str(period)] if period is not None else None,
        level=level,
    )


def pint_unit_from_ttsim_unit(
    unit: CompositeUnit, *, registry: pint.UnitRegistry, with_level: bool = True
) -> pint.Unit:
    """Resolve a TTSIM unit to its pint unit in ``registry``.

    The plain, unvalidated bridge between the two layers; the
    ``pint_unit_from_ttsim_unit_for_*`` resolvers add the layer-specific
    declaration rules on top of it.

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
    if unit.hours is not None:
        resolved = _divide_by_period(
            non_time_unit=resolved,
            period_pint_name=_HOURS_TOKEN_TO_PINT[unit.hours],
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


def pint_unit_from_ttsim_unit_for_column(
    unit: CompositeUnit,
    *,
    name: str | None,
    grouping_levels: OrderedQNames,
    where: str,
    registry: pint.UnitRegistry,
) -> pint.Unit:
    """Resolve a column/function's TTSIM unit.

    One of three layer-specific resolvers (column, parameter, input), differing
    only in the currency rule and the name cross-check.

    Currency: only the agnostic ``CURRENCY`` is allowed. A column runs in the
    statutory currency of the policy date, whichever that is; concrete currencies
    belong to parameters and rounding specs.

    Name: ``name`` is the node's leaf name, whose GEP-1 suffixes must agree with
    the spelled period *and* the spelled grouping level. ``name=None`` says the
    declaration hangs off code rather than a named DAG node — a
    :func:`cast_ttsim_unit` call in a body, a field of an
    :class:`InputOutputUnits`, an aggregation checked against its derivation — so
    period and level are taken as spelled, and ``grouping_levels`` is irrelevant.

    ``grouping_levels`` are the policy environment's levels, the alternation
    ``name``'s aggregation suffix is read against.

    Raises:
        UnitDefinitionError: If the base pins a concrete currency, or the spelled
            period or level disagrees with the name's suffixes.
    """
    if ttsim_unit_currency(unit) is not None:
        raise UnitDefinitionError(
            f"{where}: a column/function pins the concrete currency {unit.base!r}. "
            f"A function runs in the statutory currency of the policy date, "
            f"whichever that is: declare the agnostic {CURRENCY_TOKEN} and leave "
            f"concrete currencies to parameters and rounding specs (GEP 10)."
        )
    if name is not None:
        time_unit_id, grouping_level = _name_suffixes(
            name=name, grouping_levels=grouping_levels
        )
        _fail_if_period_disagrees_with_name(
            unit=unit,
            time_unit_id=time_unit_id,
            where=where,
            error_class=UnitDefinitionError,
        )
        expected_level = grouping_level.upper() if grouping_level is not None else None
        if unit.level is not None and unit.level != expected_level:
            raise UnitDefinitionError(
                f"{where}: the unit spells group level {unit.level!r} but the name's "
                f"aggregation suffix implies {expected_level or 'no level (bare)'}; a "
                f"spelled group level must match the suffix, or be omitted for a bare "
                f"(per-person / level-neutral) quantity."
            )
    return pint_unit_from_ttsim_unit(unit=unit, registry=registry, with_level=True)


def pint_unit_from_ttsim_unit_for_param(
    unit: CompositeUnit,
    *,
    name: str | None,
    grouping_levels: OrderedQNames,
    where: str,
    registry: pint.UnitRegistry,
) -> pint.Unit:
    """Resolve a parameter's TTSIM unit.

    One of three layer-specific resolvers (column, parameter, input), differing
    only in the currency rule and the name cross-check.

    Currency: only a concrete currency is allowed. A parameter's numbers are
    statutory magnitudes, which are written in one particular currency; the
    agnostic ``CURRENCY`` leaves that unstated.

    Name: ``name`` is the parameter's leaf name (or a dict parameter's leaf key).
    A *present* GEP-1 time suffix must agree with the spelled period; an unsuffixed
    name constrains nothing, since GEP-1's suffix rule governs DAG columns, not
    parameters. A parameter spells its own grouping level, so the level is never
    cross-checked. ``name=None`` says there is no name at all — a schedule axis, an
    aggregation's declaration — and ``grouping_levels`` is irrelevant.

    ``grouping_levels`` are the policy environment's levels; they delimit where
    ``name``'s time suffix ends, so a name whose level suffix is unknown reads as
    unsuffixed.

    Raises:
        UnitDefinitionError: If the base is the agnostic currency, or a present
            name time suffix disagrees with the spelled period.
    """
    if ttsim_unit_has_agnostic_currency(unit):
        suffixes = str(unit).removeprefix(CURRENCY_TOKEN)
        raise UnitDefinitionError(
            f"{where}: parameters must pin down the concrete currency their "
            f"numbers are written in; the agnostic unit {unit} is not "
            f"allowed here. Declare the statutory currency at the parameter's "
            f"dates, e.g. DM{suffixes} or EUR{suffixes} (GEP 10)."
        )
    if name is not None:
        time_unit_id, _ = _name_suffixes(name=name, grouping_levels=grouping_levels)
        if time_unit_id is not None:
            _fail_if_period_disagrees_with_name(
                unit=unit,
                time_unit_id=time_unit_id,
                where=where,
                error_class=UnitDefinitionError,
            )
    return pint_unit_from_ttsim_unit(unit=unit, registry=registry, with_level=True)


def pint_unit_from_ttsim_unit_for_input(
    unit: CompositeUnit,
    *,
    name: str | None,
    grouping_levels: OrderedQNames,
    where: str,
    registry: pint.UnitRegistry,
) -> pint.Unit:
    """Resolve the TTSIM unit a user tagged an input column with.

    One of three layer-specific resolvers (column, parameter, input), differing
    only in the currency rule and the name cross-check.

    Currency: this resolver accepts either spelling — a concrete currency states
    what the column's numbers *are* and overrides the blanket "untagged data is in
    the data currency" assumption per column, the agnostic ``CURRENCY`` resolves
    to that assumption. The public contract is narrower: the boundary check
    :func:`ttsim.interface_dag_elements.fail_if.input_currency_is_not_concrete`
    additionally requires a currency tag to name a concrete currency, since the
    data itself is denominated in one.

    Name: ``name`` is the column's qualified name, whose GEP-1 time suffix must
    agree with the spelled period — a ``_m`` column needs a ``PER_MONTH`` tag, an
    unsuffixed column a tag with no period. ``name=None`` takes the period as
    spelled, and ``grouping_levels`` is irrelevant.

    ``grouping_levels`` are the policy environment's levels; they delimit where
    ``name``'s time suffix ends. Grouping levels do not affect the magnitude, so
    the resolved unit drops the level: this needs no registered level dimension.

    Raises:
        UnitConsistencyError: If the spelled period disagrees with the name's
            time suffix.
    """
    if name is not None:
        time_unit_id, _ = _name_suffixes(name=name, grouping_levels=grouping_levels)
        _fail_if_period_disagrees_with_name(
            unit=unit,
            time_unit_id=time_unit_id,
            where=where,
            error_class=UnitConsistencyError,
        )
    resolved = pint_unit_from_ttsim_unit(unit=unit, registry=registry, with_level=False)
    concrete = ttsim_unit_currency(unit)
    return (
        resolved
        if concrete is None
        else pint_unit_with_currency(
            units=resolved, currency=concrete, registry=registry
        )
    )


def pint_unit_from_string(unit_str: str, registry: pint.UnitRegistry) -> pint.Unit:
    """Parse a pint unit string, enforcing the closed pint-token vocabulary.

    Internal: declarations are :class:`CompositeUnit`\\ s, never pint syntax. This
    parser serves the framework date nodes, whose units are spelled as pint
    strings in :data:`FRAMEWORK_DATE_NODE_UNITS`.

    ``unit_str`` may use the :data:`CURRENCY_TOKEN` to denote the ``[currency]``
    dimension (``"CURRENCY"``, ``"CURRENCY / meter ** 2"``).

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


def replace_time_unit(unit: CompositeUnit, time_unit_id: str) -> CompositeUnit:
    """Replace a flow unit's period with the requested time unit.

    Return a non-flow unit unchanged.
    """
    if unit.period is None:
        return unit
    return replace(unit, period=TIME_UNIT_ID_TO_PERIOD_TOKEN[time_unit_id])


def ttsim_unit_with_agnostic_currency(ttsim_unit: CompositeUnit) -> CompositeUnit:
    """The TTSIM unit a node *derived* from a source with this unit carries.

    A concrete currency base is replaced by the agnostic ``CURRENCY``; every
    other base, and the area, hours, period and level, are left untouched.
    """
    return (
        replace(ttsim_unit, base=CURRENCY_TOKEN)
        if ttsim_unit_currency(ttsim_unit) is not None
        else ttsim_unit
    )


def pint_unit_with_currency(
    units: pint.Unit, currency: str, registry: pint.UnitRegistry
) -> pint.Unit:
    """Swap a pint unit's currency component for ``currency``; a no-op if it has none.

    The one currency move input and output handling share: the period, area,
    hours and levels are left untouched. For results ``currency`` is the data
    currency (a computed or input column) or the computation currency (a
    requested parameter); for annotated input data it is the tag's concrete
    currency.
    """
    component = _pint_unit_currency(units=units, registry=registry)
    if component is None:
        return units
    return units / component * registry.parse_units(currency)


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

    The division is *unit* arithmetic, not quantity arithmetic, so it stays legal
    for offset (calendar-point) bases.

    Raises:
        UnitDefinitionError: If the level has not been registered.
    """
    return unit / _grouping_level_unit(name=level, registry=registry)


def ttsim_unit_currency(ttsim_unit: CompositeUnit | None) -> str | None:
    """The concrete currency a TTSIM unit pins down, if any."""
    if not isinstance(ttsim_unit, CompositeUnit):
        return None
    return next(
        (name for name in _registered_currencies if name.upper() == ttsim_unit.base),
        None,
    )


def ttsim_unit_has_currency(ttsim_unit: CompositeUnit | None) -> bool:
    """Whether a TTSIM unit's base is a currency (agnostic or concrete)."""
    return (
        ttsim_unit_has_agnostic_currency(ttsim_unit)
        or ttsim_unit_currency(ttsim_unit) is not None
    )


def ttsim_unit_has_agnostic_currency(ttsim_unit: CompositeUnit | None) -> bool:
    """Whether a TTSIM unit's base is the agnostic ``CURRENCY``."""
    return isinstance(ttsim_unit, CompositeUnit) and ttsim_unit.base == CURRENCY_TOKEN


def pint_unit_has_currency(units: pint.Unit, registry: pint.UnitRegistry) -> bool:
    """Whether a (possibly composite) pint unit carries a currency component."""
    return _pint_unit_currency(units=units, registry=registry) is not None


def pint_unit_has_agnostic_currency(
    units: pint.Unit, registry: pint.UnitRegistry
) -> bool:
    """Whether a pint unit's currency component is the agnostic ``CURRENCY``.

    Distinguishes the two currency spellings when results are returned: a
    column resolves to the agnostic ``CURRENCY`` (it is computed in the
    computation currency and converted to the data currency), a parameter to
    its concrete statutory currency (never converted, labelled as declared).
    """
    component = _pint_unit_currency(units=units, registry=registry)
    return component is not None and component == registry.parse_units(CURRENCY_TOKEN)


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

    A calendar year is a Pint offset unit: it obeys affine algebra, not the
    magnitude algebra of a duration. Pint raises an
    :class:`pint.OffsetUnitCalculusError` on an illegal operation, so:

    - two points *subtract* to a duration, and a duration *shifts* a point;
    - two points cannot be added, a point cannot be scaled, and points on
      different calendar axes cannot be combined.

    Detection is by the property that defines an offset unit: it cannot be divided
    by itself. Only :data:`_CALENDAR_POINT_UNIT_NAMES` contains offset units,
    so a unit spelling none of them skips the pint probe.
    """
    if _CALENDAR_POINT_UNIT_NAMES.isdisjoint(to_units_container(unit)):
        return False
    quantity = registry.Quantity(1.0, unit)
    try:
        quantity / quantity
    except pint.OffsetUnitCalculusError:
        return True
    return False


def is_calendar_ordinal_unit(unit: pint.Unit) -> bool:
    """Whether ``unit`` is a quarter-, month-, or day-within-period ordinal."""
    return not _CALENDAR_ORDINAL_UNIT_NAMES.isdisjoint(to_units_container(unit))


def input_column_in_data_currency(
    values: Any,  # noqa: ANN401 (an input column: a list, an array, a `pd.Series`)
    *,
    unit: CompositeUnit,
    data_currency: str,
    grouping_levels: OrderedQNames,
    registry: pint.UnitRegistry,
    column_label: str,
) -> Any:  # noqa: ANN401
    """Convert a unit-annotated input column into the data currency.

    The user boundary for :class:`UnitAnnotatedColumn` data. The tag is a TTSIM
    unit, so pint appears here only to compute the conversion factor, and bare
    magnitudes come back out:

    - the tag's period is screened against the column's GEP-1 time suffix by
      :func:`pint_unit_from_ttsim_unit_for_input`;
    - a concrete currency is *converted* to the data currency — the tag overrides
      the blanket "untagged data is in the data currency" assumption per column,
      so a DM-tagged column can ride along EUR data — while period, area, hours
      and level are left untouched;
    - a tag already in the data currency, and a tag with no currency at all, pass
      their values through unchanged.

    The crossing from the data currency into the computation currency happens right
    after, in ``input_data_in_computation_currency``, for annotated and plain columns
    alike. The column's *declared* unit is not threaded here, so a wrong-dimension
    tag (a currency on an age column) is caught by the full input-unit check.

    Raises:
        UnitConsistencyError: If the tag's period disagrees with the column's
            time suffix.
    """
    resolved = pint_unit_from_ttsim_unit_for_input(
        unit=unit,
        name=column_label,
        grouping_levels=grouping_levels,
        where=f"Input column {column_label!r}",
        registry=registry,
    )
    source_currency = _pint_unit_currency(units=resolved, registry=registry)
    if source_currency is None:
        return values
    data_currency_unit = registry.parse_units(data_currency)
    if source_currency == data_currency_unit:
        return values
    target = resolved / source_currency * data_currency_unit
    return registry.Quantity(values, resolved).to(target).magnitude


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
    - ``calendar_year`` as an affine *point* unit. Subtracting two years yields
      the companion ``delta_calendar_year`` duration;
    - ``calendar_quarter`` / ``calendar_month`` / ``calendar_day`` as independent
      ordinal dimensions. They can be ordered on the same scale, but are not
      durations and support no general arithmetic.

    pint's remaining built-ins parse, but :func:`pint_unit_from_string` rejects every
    token outside :data:`_ALLOWED_UNIT_TOKENS`, so none can appear in a declaration.
    """
    ureg = pint.UnitRegistry()
    ureg.define(f"{CURRENCY_TOKEN} = [currency]")
    ureg.define("working_hour = [hours]")
    ureg.define("quarter_year = year / 4 = quarter_of_year")
    ureg.define("calendar_quarter_duration = year / 4 = delta_calendar_quarter")
    ureg.define("delta_calendar_month = month")
    ureg.define("delta_calendar_day = day")
    # Pint needs a nonzero offset to distinguish calendar points, such as year 1999,
    # from durations, such as 3 years. TTSIM never uses the offset's numeric value,
    # so use 1 consistently for all calendar units.
    ureg.define("calendar_year = year; offset: 1")
    ureg.define("calendar_quarter = [calendar_quarter_ordinal]")
    ureg.define("calendar_month = [calendar_month_ordinal]")
    ureg.define("calendar_day = [calendar_day_ordinal]")
    return ureg


def registered_grouping_levels(registry: pint.UnitRegistry) -> set[str]:
    """The grouping levels a registry defines a dimension for."""
    return {
        name.removeprefix(_GROUPING_LEVEL_PREFIX)
        for name in registry
        if name.startswith(_GROUPING_LEVEL_PREFIX)
    }


def register_grouping_levels(names: Iterable[str], registry: pint.UnitRegistry) -> None:
    """Register grouping levels (``["hh", "bg"]``) as base dimensions of a registry."""
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

    A level claims the builder step ``PER_<NAME>`` on :class:`CompositeUnit`, which is
    a process-global class shared by every system. Two kinds of names are refused:

    - a name whose step is already one of the closed area/hours/period steps;
    - a name that is not lower-case, since a level is registered verbatim but resolved
      lower-cased.

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
                f"not collide with an area, working hours, or a period (GEP 10)."
            )


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
    ``@agg_by_p_id_function`` nodes declare their unit explicitly.
    ``target_level`` is the group the node aggregates *to*, or ``None`` for an
    individual (bare) result — an ``agg_by_p_id`` node:

    - a **head count** — ``COUNT``, or a ``SUM`` over a boolean source
      (``source_is_boolean``, counting the persons its flag is true for) — is the
      dimensionless :attr:`TTSIMUnit.DIMENSIONLESS` per ``target_level``
      (``DIMENSIONLESS_PER_HH`` = ``1 / [hh]``); bare at an individual target;
    - ``SUM`` / ``MIN`` / ``MAX`` over a non-boolean source are properties of the
      **target** group whatever the source's base: they keep the
      source's physical token and take the target level — spelled for a group
      (``CURRENCY_PER_MONTH_PER_FAM``, ``MONTHS_PER_FG`` for an ``_fg`` extreme of
      a bare duration), and **bare** for an individual ``agg_by_p_id`` result;
    - ``MEAN`` is a statistic of the **target group**, like ``MIN`` and ``MAX``;
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
        source token at the target (``SUM`` / ``MEAN`` / ``MIN`` / ``MAX``) or
        bare for an individual ``agg_by_p_id``
        (:data:`UNSET_UNIT` when the source itself lacks a declaration, which the
        mandatory-units check then reports against the source).
    """
    agg_type = head_count_from_boolean_sum(
        agg_type=agg_type, source_is_boolean=source_is_boolean
    )
    if agg_type in (AggType.COUNT, AggType.ANY, AggType.ALL):
        base = TTSIMUnit.DIMENSIONLESS
        return base if target_level is None else base.PER_LEVEL(target_level)
    if isinstance(source_unit, UnsetUnit):
        return source_unit
    if target_level is None:
        # An agg_by_p_id result is an individual property and therefore bare.
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
      whatever the source's base: the source level (if any) is swapped
      for the target and a bare source *acquires* it — an ``_hh`` sum of a person
      income is ``CURRENCY/[hh]``, an ``_fg`` min of a bare age is ``MONTHS/[fg]``.
      At an individual target (an ``agg_by_p_id`` node) the result is **bare**.
    - ``MEAN`` is a statistic of the **target group**, like ``MIN`` and ``MAX``.
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
    # An agg_by_p_id result (``target_level is None``) is bare.
    if target_level is None:
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
        qname for qname, unit in units_by_qname.items() if isinstance(unit, UnsetUnit)
    )
    if missing:
        raise UnitDefinitionError(
            "The following nodes are missing a mandatory `unit=` declaration "
            f"(GEP 10; declare `unit=TTSIMUnit.DIMENSIONLESS` / `unit: DIMENSIONLESS` "
            f"for a dimensionless quantity): {', '.join(missing)}."
        )


@dataclasses.dataclass(frozen=True)
class Currency:
    """One currency of a :class:`UnitSystem`, named by its key in that mapping."""

    value: str | None = None
    """A pint-parseable definition relative to a currency defined earlier in the
    system's mapping (``"EUR / 1.95583"``). ``None`` marks the base currency, which
    is defined as factor 1 against the abstract ``[currency]`` reference."""

    statutory_from: str | None = None
    """The dashed ISO date from which statutes denominate their numbers in this
    currency, until the next currency's date. ``None`` means this currency is never
    the statutory one."""


@dataclasses.dataclass(frozen=True, eq=False, kw_only=True)
class UnitSystem:
    """The currencies and registry of one policy system.

    A package builds one system and exports it as a singleton, so a system is identified
    by object identity: ``eq=False`` keeps the inherited identity-based ``__hash__``,
    which lets a system key an ``lru_cache`` (its ``currencies`` mapping is otherwise
    unhashable).

    A policy package declares its system once and exports it::

        UNIT_SYSTEM = UnitSystem(
            currencies={
                "EUR": Currency(statutory_from="2002-01-01"),
                "DM": Currency(value="EUR / 1.95583", statutory_from="0001-01-01"),
            },
        )

    All of a system's currencies are interconvertible
    (:meth:`currency_conversion_factor`).

    Raises:
        UnitDefinitionError: If the currencies do not form a single base plus a chain
            of ``value``\\ s referencing earlier entries, if a name is unusable as a
            ``TTSIMUnit`` base, or if no currency ever becomes statutory.
    """

    currencies: Mapping[str, Currency]
    """Every currency this system defines, in definition order: exactly one base
    (no ``value``), then each further currency defined relative to one already
    named above it."""

    registry: pint.UnitRegistry = dataclasses.field(init=False, repr=False)
    """The system's own pint registry: the shared vocabulary plus this system's
    currency definitions."""

    statutory_currency_by_start_date: tuple[tuple[datetime.date, str], ...] = (
        dataclasses.field(init=False, repr=False)
    )
    """The currencies declaring a ``statutory_from``, parsed and sorted by date."""

    @property
    def base_currency(self) -> str:
        """The system's unit of account, and the default data currency.

        Raises:
            UnitDefinitionError: If not exactly one currency states no ``value``.
        """
        bases = [
            name for name, currency in self.currencies.items() if currency.value is None
        ]
        if len(bases) != 1:
            raise UnitDefinitionError(
                f"A policy system needs exactly one currency without a `value` — its "
                f"base currency, defined as factor 1 against the abstract "
                f"[currency] reference; every other currency states a `value` "
                f"relative to one defined before it. Got "
                f"{', '.join(repr(name) for name in bases) or 'none'} (GEP 10)."
            )
        return bases[0]

    def __post_init__(self) -> None:
        # Snapshot the caller's mapping: everything derived below (the parsed dates,
        # the registry) would otherwise silently disagree with it if the caller
        # mutated theirs afterwards.
        object.__setattr__(self, "currencies", MappingProxyType(dict(self.currencies)))
        registry = build_registry()
        object.__setattr__(self, "registry", registry)
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
        every other currency is defined relative to one named earlier in
        :attr:`currencies`, so all of them chain back to the base and are
        interconvertible.
        """
        self.base_currency  # noqa: B018 (fail early if the base is ambiguous)
        defined: set[str] = set()
        for name, currency in self.currencies.items():
            if currency.value is None:
                self._define_one_currency(name=name, definition=CURRENCY_TOKEN)
            else:
                self._fail_if_definition_references_no_known_currency(
                    name=name, definition=currency.value, defined=defined
                )
                self._define_one_currency(name=name, definition=currency.value)
            defined.add(name)
        claimed = set(_registered_currencies)
        for name in self.currencies:
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
            # still rejects a concrete base (`pint_unit_from_ttsim_unit_for_column`);
            # this only makes it reachable.
            setattr(TTSIMUnit, name.upper(), CompositeUnit(base=name.upper()))

    def _fail_if_name_is_unusable_as_a_unit_base(
        self, name: str, claimed: set[str]
    ) -> None:
        """Reject a currency name the ``TTSIMUnit`` base namespace cannot carry.

        A currency reaches the builder namespace under its upper-cased name, and
        :func:`ttsim.tt.units.ttsim_unit_from_string` matches a base against that same
        upper-cased form. Four names are refused: the agnostic :data:`CURRENCY_TOKEN`,
        a non-currency base of the shared vocabulary, a name differing only in case
        from one in ``claimed`` (the currencies already registered process-wide), and a
        name spelling the :data:`ttsim.tt.units._PER` denominator delimiter.
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
        """Reject a currency ``value`` that does not chain to a known currency.

        Every non-base currency states a ``value`` relative to exactly one currency
        named earlier in :attr:`currencies` (``"CASTAR / 4"``). A value against the
        abstract :data:`CURRENCY_TOKEN` reference alone, against no currency at all,
        or against one named further down the mapping would start a second,
        unconnected base — which the single-base model forbids.

        Raises:
            UnitDefinitionError: If the value references a unit the registry does
                not know, no currency defined before this one, or more than one.
        """
        referenced_tokens = set(_UNIT_TOKEN_PATTERN.findall(definition))
        too_late = sorted(
            other
            for other in self.currencies
            if other != name and other not in defined and other in referenced_tokens
        )
        if too_late:
            raise UnitDefinitionError(
                f"Currency {name!r} is defined as {definition!r}, which references "
                f"{', '.join(repr(other) for other in too_late)} — declared further "
                f"down this policy system's `currencies` mapping. A `value` may only "
                f"reference a currency declared before it; reorder the mapping "
                f"(GEP 10)."
            )
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
        """Parse and sort the currencies declaring a ``statutory_from``.

        Raises:
            UnitDefinitionError: If no currency ever becomes statutory, a
                ``statutory_from`` is anything other than a dashed ISO date, or
                two currencies claim the same ``statutory_from``.
        """
        parsed = tuple(sorted(self._parsed_start_dates()))
        if not parsed:
            raise UnitDefinitionError(
                "At least one currency must declare a `statutory_from` date; none "
                "of this policy system's currencies does (GEP 10)."
            )
        self._fail_if_start_dates_are_contested(parsed)
        return parsed

    @staticmethod
    def _fail_if_start_dates_are_contested(
        parsed: tuple[tuple[datetime.date, str], ...],
    ) -> None:
        """Reject a ``statutory_from`` date more than one currency claims.

        The statutory currency at a policy date is looked up by date, so two
        currencies becoming statutory on the same day leave it undetermined.
        """
        by_date: dict[datetime.date, list[str]] = {}
        for start_date, name in parsed:
            by_date.setdefault(start_date, []).append(name)
        contested = {
            start_date: names for start_date, names in by_date.items() if len(names) > 1
        }
        if contested:
            spelled = "; ".join(
                f"{start_date.isoformat()}: {', '.join(names)}"
                for start_date, names in sorted(contested.items())
            )
            raise UnitDefinitionError(
                f"Each `statutory_from` date names the one currency that becomes "
                f"statutory on it, but several currencies claim the same date "
                f"({spelled}). Give each of them its own date (GEP 10)."
            )

    def _parsed_start_dates(self) -> Iterator[tuple[datetime.date, str]]:
        """Each ``statutory_from`` parsed, rejecting any spelling but ``YYYY-MM-DD``.

        `date.fromisoformat` also accepts the basic (``20200101``) and week-date
        (``2021-W01-1``) forms, so the round-trip comparison — not the parse alone
        — is what pins the one documented spelling.
        """
        for name, currency in self.currencies.items():
            start_date = currency.statutory_from
            if start_date is None:
                continue
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
            f"`statutory_from` is the dashed ISO date a currency becomes statutory; "
            f"{start_date!r} (on currency {name!r}) is not one. Spell it "
            f"YYYY-MM-DD (GEP 10)."
        )


def _classify_denominator(token: str) -> str:
    """Classify a denominator token as ``"area"``, ``"hours"``, ``"period"``, or
    ``"level"``.

    Area, hours and period are closed vocabularies; everything else is taken to be
    a grouping level, validated against the registered levels at resolution time.
    """
    if token in _AREA_TOKEN_TO_PINT:
        return "area"
    if token in _HOURS_TOKEN_TO_PINT:
        return "hours"
    if token in _PERIOD_TOKEN_TO_PINT:
        return "period"
    return "level"


def _is_currency_base(base: str) -> bool:
    """Whether a base token denotes a currency (agnostic or concrete)."""
    if base == CURRENCY_TOKEN:
        return True
    return any(base == name.upper() for name in _registered_currencies)


def _fail_if_period_disagrees_with_name(
    unit: CompositeUnit,
    *,
    time_unit_id: str | None,
    where: str,
    error_class: type[UnitDefinitionError | UnitConsistencyError],
) -> None:
    """Reject a spelled period that disagrees with a name's GEP-1 time suffix.

    Strict in both directions: an unsuffixed name must carry no period, and a
    suffixed one exactly the period the suffix names. The column and input
    resolvers apply it to every name; the parameter resolver only to a suffixed
    one, since a parameter's name is not governed by GEP-1.
    """
    expected_period = (
        TIME_UNIT_ID_TO_PERIOD_TOKEN[time_unit_id] if time_unit_id is not None else None
    )
    if unit.period != expected_period:
        raise error_class(
            f"{where}: the unit spells period {unit.period!r} but the name's time "
            f"suffix implies {expected_period!r}; they must agree."
        )


def _name_suffixes(
    name: str, grouping_levels: OrderedQNames
) -> tuple[str | None, str | None]:
    """The time-unit id and grouping level a GEP-1 name spells, each or both ``None``.

    ``grouping_levels`` are the policy environment's levels, passed in explicitly:
    the alternation a name is matched against must be the same at the input
    boundary — which runs before any registry is populated — as during
    environment resolution.

    Example:
        >>> _name_suffixes(name="income_m_hh", grouping_levels=("hh",))
        ('m', 'hh')
        >>> _name_suffixes(name="wealth", grouping_levels=("hh",))
        (None, None)
    """
    match = get_re_pattern_for_all_time_units_and_groupings(
        time_units=tuple(TIME_UNIT_ID_TO_PINT_NAME),
        grouping_levels=tuple(grouping_levels),
    ).fullmatch(name)
    if match is None:
        return None, None
    return (match.group("time_unit") or None, match.group("grouping") or None)


def _unit_is_currency(unit: pint.Unit) -> bool:
    """Whether a pint unit is exactly one power of the ``[currency]`` dimension."""
    return dict(unit.dimensionality) == _CURRENCY_DIMENSIONALITY


def _unit_is_time(unit: pint.Unit) -> bool:
    """Whether a pint unit is exactly one power of the ``[time]`` dimension."""
    return dict(unit.dimensionality) == _TIME_DIMENSIONALITY


def _pint_unit_currency(
    units: pint.Unit, registry: pint.UnitRegistry
) -> pint.Unit | None:
    """The currency component of a (possibly composite) pint unit, or ``None``."""
    for token in to_units_container(units):
        candidate = registry.parse_units(token)
        if _unit_is_currency(candidate):
            return candidate
    return None


def _grouping_level_unit_name(name: str) -> str:
    """The internal pint unit name anchoring a grouping level's dimension."""
    return f"{_GROUPING_LEVEL_PREFIX}{name}"


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


def _flow_period_of(units: pint.Unit, registry: pint.UnitRegistry) -> pint.Unit | None:
    """Return a unit's flow period — its time component in the *denominator*."""
    for token, exponent in to_units_container(units).items():
        if isinstance(exponent, complex):  # pint exponents are real; narrow for ty
            continue
        candidate = registry.parse_units(token)
        if exponent < 0 and _unit_is_time(candidate):
            return candidate
    return None
