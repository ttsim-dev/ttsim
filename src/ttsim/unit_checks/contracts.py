"""Unit contracts read off declarations and type annotations.

Resolves what a node's *type* promises before any value exists: a param
function's return annotation (a schedule type, a parameter dataclass, a flat
mapping of one), the units annotated on a parameter dataclass's fields, and the
two axes an ``InputOutputUnit`` declares. Nothing here touches the environment's
values or runs a body.

The other three stages build on this one: :mod:`ttsim.unit_checks.resolution`
asks whether a param function returns a schedule, and
:mod:`ttsim.unit_checks.execution` turns the resolved field kinds into the
stand-ins a body sees. :mod:`ttsim.unit_checks.declarations` runs the
``InputOutputUnit`` contract check defined here as part of its environment-wide
verification.
"""

from __future__ import annotations

import dataclasses
import re
import sys
from collections.abc import Mapping
from typing import (
    Any,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

import pint
from dags import get_annotations

from ttsim.exceptions import (
    UnitDefinitionError,
)
from ttsim.tt.column_objects_param_function import (
    ParamFunction,
)
from ttsim.tt.currencies import UnitSystem
from ttsim.tt.param_objects import (
    ConsecutiveIntLookupTableParamValue,
    ParamMappingObject,
    PiecewisePolynomialParamValue,
)
from ttsim.tt.units import (
    UNSET_UNIT,
    CompositeUnit,
    InputOutputUnit,
    resolve_agnostic_ttsim_unit,
    resolve_ttsim_unit_for_param,
    ttsim_unit_has_currency,
)
from ttsim.typing import (
    SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
)

#: The unqualified return-annotation names governed by the ``InputOutputUnit``
#: contract: a param function returning one of these declares both axes.
_SCHEDULE_RETURN_TYPE_NAMES = frozenset(
    {"PiecewisePolynomialParamValue", "ConsecutiveIntLookupTableParamValue"}
)


#: The schedule value classes a unit-annotated parameter-dataclass field may carry;
#: matched by identity/subclass (annotations resolve to real types via
#: ``get_type_hints``), so the field's declared axes reach its ``look_up``
#: consumers.
_SCHEDULE_VALUE_TYPES = (
    ConsecutiveIntLookupTableParamValue,
    PiecewisePolynomialParamValue,
)


def _schedule_param_function_contract_errors(
    env: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
) -> list[str]:
    """Check every param function's ``unit=`` against its return annotation (GEP 10).

    A schedule builder is a function between quantities, declared with
    ``unit=InputOutputUnit(...)``; a scalar/structured param function is not. The
    contract, keyed off the decorator and the return type, is broken — reported at
    build time — when a param function:

    - declares ``unit=InputOutputUnit(...)`` but is not annotated as returning a
      ``PiecewisePolynomialParamValue`` or a ``ConsecutiveIntLookupTableParamValue``
      (only a schedule has two axes);
    - is annotated as returning one of those schedule types but declares a quantity
      ``unit=`` or ``unit=UNSET_UNIT`` instead of ``unit=InputOutputUnit(...)``;
    - builds a ``ConsecutiveIntLookupTableParamValue`` from an ``InputOutputUnit``
      whose ``input_unit`` is (or, for a tuple, contains) a currency (a lookup
      table is keyed by consecutive integers, so no input axis is ever a currency);
    - builds a ``PiecewisePolynomialParamValue`` from a tuple ``input_unit``
      (piecewise takes one domain argument; a tuple is only for a multi-dimensional
      lookup table);
    - declares ``unit=InputOutputUnit(...)`` but leaves ``verify_units`` at its
      default ``True`` (a schedule builder's body builds a table, not a scalar, so
      it cannot be unit-verified — the skip must be stated explicitly).
    """
    errors: list[str] = []
    for qname, obj in env.items():
        if not isinstance(obj, ParamFunction):
            continue
        declares_io = isinstance(obj.unit, InputOutputUnit)
        return_type_name = _return_annotation_name(obj.function)
        returns_schedule = return_type_name in _SCHEDULE_RETURN_TYPE_NAMES
        if declares_io and not returns_schedule:
            errors.append(
                f"{qname}: declares `unit=InputOutputUnit(...)`, which states a "
                f"schedule's two axes, but is not annotated as returning a "
                f"PiecewisePolynomialParamValue or a "
                f"ConsecutiveIntLookupTableParamValue (GEP 10)."
            )
            continue
        if returns_schedule and not declares_io:
            errors.append(
                f"{qname}: is annotated as returning a schedule "
                f"(PiecewisePolynomialParamValue / "
                f"ConsecutiveIntLookupTableParamValue), so it must declare its two "
                f"axes with `unit=InputOutputUnit(input_unit=…, output_unit=…)`, not "
                f"`unit={_spell_ttsim_unit(obj.unit)}` (GEP 10)."
            )
            continue
        if not declares_io:
            continue
        io_unit = obj.unit
        builds_lookup_table = return_type_name == "ConsecutiveIntLookupTableParamValue"
        input_axes = (
            io_unit.input_unit
            if isinstance(io_unit.input_unit, tuple)
            else (io_unit.input_unit,)
        )
        if not builds_lookup_table and isinstance(io_unit.input_unit, tuple):
            errors.append(
                f"{qname}: declares a tuple `input_unit` but builds a piecewise "
                f"polynomial, which takes a single domain argument; a tuple of "
                f"positional axes is only for a multi-dimensional lookup table "
                f"(GEP 10)."
            )
        if builds_lookup_table and any(
            ttsim_unit_has_currency(cast("CompositeUnit", axis)) for axis in input_axes
        ):
            errors.append(
                f"{qname}: builds a lookup table but declares a currency "
                f"`input_unit={io_unit.input_unit}`; a lookup table is keyed by "
                f"consecutive integers, so no input axis is ever a currency "
                f"(GEP 10)."
            )
        if obj.verify_units:
            errors.append(
                f"{qname}: declares `unit=InputOutputUnit(...)` but leaves "
                f"`verify_units=True`; a schedule builder's body builds a table, not "
                f"a scalar, so it cannot be unit-verified — state "
                f"`verify_units=False` explicitly (GEP 10)."
            )
    return errors


def _fail_if_structured_field_annotations_are_invalid(
    env: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
    unit_system: UnitSystem,
) -> None:
    """Resolve every structured param function's field annotations at build time.

    A ``@param_function(unit=UNSET_UNIT)`` builds a structured value whose units
    live in its return type's field annotations. :func:`_structured_field_kinds`
    raises a :class:`UnitDefinitionError` on a malformed one — a concrete-currency
    pin, a unit on a container field, conflicting tokens — so running it here for
    every such producer's return dataclass (and every nested dataclass it
    references) catches the error whether or not a body ever plucks the field.

    Raises:
        UnitDefinitionError: If any parameter-dataclass field annotation is
            malformed.
    """
    visited: set[type] = set()
    for obj in env.values():
        if not isinstance(obj, ParamFunction) or obj.unit is not UNSET_UNIT:
            continue
        cls, item_cls = _resolved_return_structure(obj.function)
        for start in (cls, item_cls):
            if start is not None:
                _resolve_structured_field_annotations(
                    cls=start, unit_system=unit_system, visited=visited
                )


def _resolve_structured_field_annotations(
    cls: type,
    unit_system: UnitSystem,
    visited: set[type],
) -> None:
    """Walk a parameter dataclass tree, resolving each class's field annotations.

    :func:`_structured_field_kinds` raises on a malformed annotation; a
    nested-dataclass field (its kind is a ``type``) recurses, while a unit or
    schedule-field kind is terminal. ``visited`` guards a dataclass that
    references itself.

    A dataclass reachable only through a *nested* mapping-of-dataclass field
    (``foo: dict[str, Inner]``) is not walked: such a field is opaque at pluck
    time too (only a producer's top-level return resolves an ``item_cls``), so
    the eager walk and the unit check agree on what is reachable.
    """
    if cls in visited:
        return
    visited.add(cls)
    kinds = _structured_field_kinds(cls=cls, unit_system=unit_system)
    if kinds is None:
        return
    for kind in kinds.values():
        if isinstance(kind, type):
            _resolve_structured_field_annotations(
                cls=kind, unit_system=unit_system, visited=visited
            )


def _structured_field_kinds(
    cls: type, unit_system: UnitSystem
) -> dict[str, pint.Unit | type | _ScheduleFieldKind] | None:
    """Resolve a parameter dataclass's field annotations for the unit check.

    Maps each field to what its pluck yields:

    - an ``Annotated[<scalar>, TTSIMUnit…]`` field → the resolved unit;
    - an ``Annotated[<schedule type>, InputOutputUnit(...)]`` field → a
      :class:`_ScheduleFieldKind` carrying the schedule's input and output axes;
    - a nested-dataclass field → its class (whose plucks resolve recursively);
    - anything else (a bare scalar, dict, array) is absent — the pluck stays
      opaque and is cast at the site (GEP 10).

    A field whose annotation does not resolve at runtime (a name imported only
    under ``TYPE_CHECKING``) is skipped individually, so its pluck stays opaque
    while its siblings are still resolved and validated. ``None`` comes back only
    when the class carries no resolvable annotation at all.

    Memoized per class on ``unit_system``: a resolved unit belongs to that
    system's registry, so the memo cannot be shared across systems.

    Raises:
        UnitDefinitionError: If a field annotates several units, mismatches the
            marker to the field kind (a bare ``CompositeUnit`` on a schedule field,
            an ``InputOutputUnit`` on a scalar field), annotates a non-scalar,
            non-schedule field, or pins a concrete currency.
    """
    memo = unit_system.field_units_by_class
    if cls in memo:
        return memo[cls]
    hints = _resolvable_type_hints(cls=cls)
    if not hints:
        memo[cls] = None
        return None
    kinds: dict[str, pint.Unit | type | _ScheduleFieldKind] = {}
    for field in dataclasses.fields(cls):
        hint = hints.get(field.name, field.type)
        metadata = getattr(hint, "__metadata__", ())
        composite_tokens = [t for t in metadata if isinstance(t, CompositeUnit)]
        io_tokens = [t for t in metadata if isinstance(t, InputOutputUnit)]
        base = get_args(hint)[0] if hasattr(hint, "__metadata__") else hint
        where = f"Field '{cls.__name__}.{field.name}'"
        is_schedule = isinstance(base, type) and issubclass(base, _SCHEDULE_VALUE_TYPES)
        if len(composite_tokens) + len(io_tokens) > 1:
            spelled = [str(t) for t in composite_tokens] + [
                "InputOutputUnit(...)" for _ in io_tokens
            ]
            raise UnitDefinitionError(
                f"{where}: annotates {len(spelled)} units ({', '.join(spelled)}); a "
                f"field states exactly one (GEP 10)."
            )
        if is_schedule:
            kinds[field.name] = _schedule_field_kind(
                base=cast("type", base),
                io_tokens=io_tokens,
                composite_tokens=composite_tokens,
                unit_system=unit_system,
                where=where,
            )
        elif io_tokens:
            raise UnitDefinitionError(
                f"{where}: annotates `InputOutputUnit(...)`, which declares a "
                f"schedule's two axes, but the field is not a lookup/piecewise "
                f"value; a scalar field states a single unit (GEP 10)."
            )
        elif composite_tokens:
            resolved = resolve_agnostic_ttsim_unit(
                unit=composite_tokens[0],
                registry=unit_system.registry,
                where=where,
                what="the declaration",
            )
            if base in (int, float, bool):
                kinds[field.name] = resolved
            else:
                raise UnitDefinitionError(
                    f"{where}: a single unit annotation must sit on a scalar field "
                    f"(int/float/bool); this structured or container field has no "
                    f"single unit — cast at the pluck instead (GEP 10)."
                )
        elif isinstance(base, type) and dataclasses.is_dataclass(base):
            kinds[field.name] = base
    memo[cls] = kinds
    return kinds


@dataclasses.dataclass(frozen=True)
class _ScheduleFieldKind:
    """A parameter dataclass field typed as a schedule and carrying its two axes.

    A field annotated ``Annotated[<schedule type>, InputOutputUnit(...)]`` declares
    the schedule's INPUT and OUTPUT axes at the field, exactly as a
    schedule-returning ``@param_function`` declares them in its ``unit=`` — so a
    pluck of the field yields a :class:`_UnitCheckSchedule` that screens each
    ``look_up`` / ``piecewise_polynomial`` argument against ``input_unit`` and
    produces ``output_unit`` (GEP 10).
    """

    input_unit: pint.Unit | tuple[pint.Unit, ...]
    """The resolved input axis (or positional axes) each domain argument is
    screened against."""
    output_unit: pint.Unit
    """The resolved unit the schedule produces."""


def _schedule_field_kind(
    base: type,
    io_tokens: list[InputOutputUnit],
    composite_tokens: list[CompositeUnit],
    unit_system: UnitSystem,
    where: str,
) -> _ScheduleFieldKind:
    """Resolve a schedule-typed field's declared axes into a kind.

    A schedule field states its two axes with exactly one ``InputOutputUnit`` in
    its ``Annotated[...]`` metadata. A bare ``CompositeUnit`` there declares a
    single quantity, which a schedule (a function between quantities) is not, so
    it is rejected with a pointer to ``InputOutputUnit`` (GEP 10).

    The same type-specific axis rules the decorator enforces hold here, keyed off
    the field's schedule type: a ``PiecewisePolynomialParamValue`` field takes one
    domain argument, so a tuple ``input_unit`` is rejected; a
    ``ConsecutiveIntLookupTableParamValue`` field is keyed by consecutive integers,
    so no ``input_unit`` axis may be a currency.

    Raises:
        UnitDefinitionError: If the field carries a bare ``CompositeUnit`` marker,
            no ``InputOutputUnit`` at all, a tuple ``input_unit`` on a piecewise
            field, or a currency axis on a lookup-table field.
    """
    if composite_tokens:
        raise UnitDefinitionError(
            f"{where}: a schedule field (a lookup/piecewise value) is a function "
            f"between quantities, so it declares both axes with "
            f"`InputOutputUnit(input_unit=…, output_unit=…)`, not the single unit "
            f"`{composite_tokens[0]}` (GEP 10)."
        )
    if not io_tokens:
        raise UnitDefinitionError(
            f"{where}: a schedule field (a lookup/piecewise value) must annotate "
            f"`InputOutputUnit(input_unit=…, output_unit=…)` declaring its two axes "
            f"(GEP 10)."
        )
    io_token = io_tokens[0]
    input_axes = (
        io_token.input_unit
        if isinstance(io_token.input_unit, tuple)
        else (io_token.input_unit,)
    )
    if issubclass(base, PiecewisePolynomialParamValue) and isinstance(
        io_token.input_unit, tuple
    ):
        raise UnitDefinitionError(
            f"{where}: declares a tuple `input_unit` but is a piecewise polynomial, "
            f"which takes a single domain argument; a tuple of positional axes is "
            f"only for a multi-dimensional lookup table (GEP 10)."
        )
    if issubclass(base, ConsecutiveIntLookupTableParamValue) and any(
        ttsim_unit_has_currency(cast("CompositeUnit", axis)) for axis in input_axes
    ):
        raise UnitDefinitionError(
            f"{where}: is a lookup table keyed by consecutive integers, so no "
            f"`input_unit` axis may be a currency (got "
            f"`input_unit={io_token.input_unit}`); the integer keys are never "
            f"rescaled between currencies (GEP 10)."
        )
    return _ScheduleFieldKind(
        input_unit=_resolve_input_axes(
            input_unit=io_token.input_unit, registry=unit_system.registry, where=where
        ),
        output_unit=resolve_agnostic_ttsim_unit(
            unit=io_token.output_unit,
            registry=unit_system.registry,
            where=where,
            what="the declaration",
        ),
    )


def _resolve_input_axes(
    input_unit: CompositeUnit | tuple[CompositeUnit, ...],
    registry: pint.UnitRegistry,
    where: str,
) -> pint.Unit | tuple[pint.Unit, ...]:
    """Resolve a schedule declaration's input axis or axes (GEP 10).

    A single :class:`CompositeUnit` resolves to one pint unit, screened against
    every ``look_up`` argument; a tuple resolves to a tuple of pint units screened
    positionally (argument ``i`` against axis ``i``). Each axis is agnostic — a
    concrete currency is rejected element-wise, exactly as for a scalar field.
    """
    if isinstance(input_unit, tuple):
        return tuple(
            resolve_agnostic_ttsim_unit(
                unit=cast("CompositeUnit", axis),
                registry=registry,
                where=where,
                what="the declaration",
            )
            for axis in input_unit
        )
    return resolve_agnostic_ttsim_unit(
        unit=input_unit, registry=registry, where=where, what="the declaration"
    )


def _resolve_schedule_input_unit(
    obj: ParamMappingObject, registry: pint.UnitRegistry
) -> pint.Unit | None:
    """The resolved ``input_unit`` of a schedule/lookup parameter, or ``None``.

    Resolved the same way as the ``output_unit`` the environment exposes, so a
    concrete-currency input axis and an agnostic ``CURRENCY`` consumer argument
    compare as equivalent. ``None`` when the parameter left ``input_unit`` unset.
    """
    if obj.input_unit is UNSET_UNIT:
        return None
    return resolve_ttsim_unit_for_param(
        unit=cast("CompositeUnit", obj.input_unit),
        registry=registry,
        where="A schedule input axis",
    )


def _returns_a_schedule(obj: ParamFunction) -> bool:
    """Whether a param function is annotated as returning a schedule/lookup value."""
    return _return_annotation_name(obj.function) in _SCHEDULE_RETURN_TYPE_NAMES


def _resolved_return_structure(func: Any) -> tuple[type | None, type | None]:  # noqa: ANN401
    """The ``(dataclass, mapping_value_dataclass)`` a param function's return names.

    Exactly one element is non-``None`` when the return annotation is a dataclass
    or a flat mapping to one; both are ``None`` otherwise. A resolved annotation
    object and its stringified form (PEP 563 / ``from __future__ import
    annotations``) resolve identically, so the two forms never disagree. Anything
    unresolvable yields ``(None, None)``: the output stays opaque and plucks are
    cast at the site (GEP 10).
    """
    annotation = get_annotations(func, default="").get("return", "")
    if isinstance(annotation, type):
        return _dataclass_or_none(annotation), None
    if not isinstance(annotation, str):
        return None, _mapping_value_dataclass(annotation)
    if not annotation:
        return None, None
    match = _MAPPING_OF_DATACLASS_RE.match(annotation.strip())
    if match is not None:
        return None, _resolve_dotted_dataclass(name=match.group("value"), func=func)
    return _resolve_dotted_dataclass(name=annotation, func=func), None


def _return_annotation_name(func: Any) -> str:  # noqa: ANN401
    """The unqualified name of a function's return annotation.

    Annotations are strings under ``from __future__ import annotations``; the
    beartype claw may resolve them to live types. Either way the unqualified
    name identifies the schedule types the axes contract asks for.
    """
    annotation = get_annotations(func, default="").get("return", "")
    name = (
        annotation
        if isinstance(annotation, str)
        else getattr(annotation, "__name__", "")
    )
    return name.rsplit(".", maxsplit=1)[-1]


#: A ``dict``/``Mapping`` return annotation — optionally module-qualified
#: (``typing.Dict``, ``cabc.Mapping``) — whose value is a (possibly dotted)
#: dataclass name, e.g. ``dict[str, SatzMitAltersgrenzen]``. Only a *flat* mapping
#: to a dataclass matches; the key type is irrelevant so anything (a bracketed
#: ``tuple[int, int]`` included) is accepted there, while a nested-container value
#: (``dict[str, dict[int, ...]]``) fails to match and stays opaque (GEP 10).
_MAPPING_OF_DATACLASS_RE = re.compile(
    r"^(?:[\w.]+\.)?(?:dict|Dict|Mapping|MutableMapping)"
    r"\[.+,\s*(?P<value>[\w.]+)\s*\]$"
)


def _mapping_value_dataclass(annotation: Any) -> type | None:  # noqa: ANN401
    """The value dataclass of a resolved ``Mapping[..., <dataclass>]``, or ``None``."""
    origin = get_origin(annotation)
    if not (isinstance(origin, type) and issubclass(origin, Mapping)):
        return None
    args = get_args(annotation)
    return _dataclass_or_none(args[-1]) if args else None


def _resolve_dotted_dataclass(name: str, func: Any) -> type | None:  # noqa: ANN401
    """Resolve a (possibly dotted) dataclass name through the function's module."""
    obj: Any = sys.modules.get(getattr(func, "__module__", ""))
    for part in name.split("."):
        obj = getattr(obj, part, None)
    return _dataclass_or_none(obj)


def _dataclass_or_none(obj: Any) -> type | None:  # noqa: ANN401
    """``obj`` if it is a dataclass type, else ``None``."""
    return obj if isinstance(obj, type) and dataclasses.is_dataclass(obj) else None


def _resolvable_type_hints(cls: type) -> dict[str, Any]:
    """The class's type hints, dropping only those that do not resolve.

    ``get_type_hints`` is all-or-nothing: one name visible only under
    ``TYPE_CHECKING`` makes it raise for the whole class. Falling back to a
    per-field resolution keeps a single unresolvable annotation from disabling
    the unit check on every other field.

    Each class in the MRO is resolved against its *own* defining module, base
    before derived, so an inherited field keeps the meaning it has where it was
    declared and a derived class still shadows it. A field whose own annotation
    does not resolve is dropped rather than left with an inherited stand-in.

    Every field goes back through ``get_type_hints``, one at a time, so a nested
    forward reference (``list["Foo"]``) resolves as fully as it would have in the
    whole-class call — evaluating the annotation string alone would leave the
    inner name a `ForwardRef` and the field wrongly opaque.
    """
    try:
        return get_type_hints(cls, include_extras=True)
    except _UNRESOLVABLE_ANNOTATION_ERRORS:
        pass
    hints: dict[str, Any] = {}
    for klass in reversed(cls.__mro__):
        module = sys.modules.get(getattr(klass, "__module__", ""))
        namespace = {**(vars(module) if module else {}), **vars(klass)}
        for name, annotation in klass.__dict__.get("__annotations__", {}).items():
            resolved = _resolve_one_annotation(
                name=name, annotation=annotation, namespace=namespace
            )
            if resolved is _UNRESOLVABLE:
                hints.pop(name, None)
            else:
                hints[name] = resolved
    return hints


#: Distinguishes an annotation that cannot be resolved from one resolving to `None`.
_UNRESOLVABLE = object()

#: What resolving an annotation raises when the annotation cannot be resolved. The
#: whole-class attempt and the per-field fallback share the set, so a class that the
#: fallback could still salvage never escapes as an error from the first attempt.
_UNRESOLVABLE_ANNOTATION_ERRORS = (NameError, AttributeError, SyntaxError, TypeError)


def _resolve_one_annotation(
    name: str,
    annotation: Any,  # noqa: ANN401
    namespace: dict[str, Any],
) -> Any:  # noqa: ANN401
    """One field's annotation resolved, or :data:`_UNRESOLVABLE`."""
    holder = type("_SingleAnnotation", (), {"__annotations__": {name: annotation}})
    try:
        return get_type_hints(holder, globalns=namespace, include_extras=True)[name]
    except _UNRESOLVABLE_ANNOTATION_ERRORS:
        return _UNRESOLVABLE


def _spell_ttsim_unit(ttsim_unit: Any) -> str:  # noqa: ANN401
    """Spell a declared TTSIM unit for an error message."""
    if ttsim_unit is UNSET_UNIT:
        return "unset"
    return str(ttsim_unit)
