"""Validate unit declarations across an assembled policy environment.

GEP 10 specifies the declaration rules this module enforces.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import re
import sys
from collections.abc import Mapping
from types import ModuleType
from typing import (
    Any,
    NamedTuple,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

import dags.tree as dt
import pint
from dags import get_annotations

from ttsim.exceptions import (
    UnitConsistencyError,
    UnitDefinitionError,
)
from ttsim.interface_dag_elements.shared import (
    FRAMEWORK_PARTIAL_ARGUMENTS,
    get_re_pattern_for_all_time_units_and_groupings,
)
from ttsim.time_converters import TIME_UNIT_IDS_TO_LABELS
from ttsim.tt._function_rewriting import recompile_with_logical_ops_as_calls
from ttsim.tt.aggregation import AggType
from ttsim.tt.column_objects_param_function import (
    AggByGroupFunction,
    ColumnFunction,
    ColumnObject,
    ParamFunction,
    PolicyFunction,
    PolicyInput,
    qname_is_person_pointer,
)
from ttsim.tt.param_objects import (
    ConsecutiveIntLookupTableParamValue,
    ParamMappingObject,
    ParamObject,
    PiecewisePolynomialParamValue,
    RawParam,
    ScalarParam,
)
from ttsim.tt.type_resolution import (
    BOOL_KINDS,
    TypeResolutionError,
    resolve_kind_of_annotation,
    resolve_kind_of_column_function,
)
from ttsim.tt.units import (
    _GROUPING_LEVEL_PREFIX,
    UNSET_UNIT,
    CompositeUnit,
    InputOutputUnits,
    UnitAnnotatedColumn,
    UnitDeclaration,
    UnitSystem,
    UnsetUnit,
    _flow_period_of,
    _grouping_levels_with_exponent,
    _pint_unit_currency,
    _unit_level_denominator,
    _unit_without_grouping_levels,
    fail_if_units_are_missing,
    head_count_from_boolean_sum,
    pint_unit_from_string,
    pint_unit_from_ttsim_unit,
    pint_unit_from_ttsim_unit_for_column,
    pint_unit_from_ttsim_unit_for_param,
    pint_unit_has_currency,
    register_grouping_levels,
    resolved_unit_for_aggregation,
    ttsim_unit_currency,
    ttsim_unit_has_agnostic_currency,
    ttsim_unit_has_currency,
    ttsim_unit_with_agnostic_currency,
    units_are_equivalent,
)
from ttsim.typing import (
    OrderedQNames,
    SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
    UserNestedUnitAnnotatedData,
)

#: The unqualified return-annotation names governed by the ``InputOutputUnits``
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


def _unit_inference_module() -> ModuleType:
    """Load the body-inference engine after unit-validation helpers are defined."""
    return importlib.import_module("ttsim._unit_inference")


def _schedule_param_function_contract_errors(
    env: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
) -> list[str]:
    """Check every param function's ``unit=`` against its return annotation.

    A schedule builder is a function between quantities, declared with
    ``unit=InputOutputUnits(...)``; a scalar/structured param function is not. The
    contract, keyed off the decorator and the return type, is broken — reported at
    build time — when a param function:

    - declares ``unit=InputOutputUnits(...)`` but is not annotated as returning a
      ``PiecewisePolynomialParamValue`` or a ``ConsecutiveIntLookupTableParamValue``
      (only a schedule has two axes);
    - is annotated as returning one of those schedule types but declares a quantity
      ``unit=`` or ``unit=UNSET_UNIT`` instead of ``unit=InputOutputUnits(...)``;
    - builds a ``ConsecutiveIntLookupTableParamValue`` from an ``InputOutputUnits``
      whose ``input_unit`` is (or, for a tuple, contains) a currency (a lookup
      table is keyed by consecutive integers, so no input axis is ever a currency);
    - builds a ``PiecewisePolynomialParamValue`` from a tuple ``input_unit``
      (piecewise takes one domain argument; a tuple is only for a multi-dimensional
      lookup table);
    - declares ``unit=InputOutputUnits(...)`` but leaves ``verify_units`` at its
      default ``True`` (a schedule builder's body builds a table, not a scalar, so
      it cannot be unit-verified — the skip must be stated explicitly).
    """
    errors: list[str] = []
    for qname, obj in env.items():
        if not isinstance(obj, ParamFunction):
            continue
        declares_io = isinstance(obj.unit, InputOutputUnits)
        return_type_name = _return_annotation_name(obj.function)
        returns_schedule = return_type_name in _SCHEDULE_RETURN_TYPE_NAMES
        if declares_io and not returns_schedule:
            errors.append(
                f"{qname}: declares `unit=InputOutputUnits(...)`, which states a "
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
                f"axes with `unit=InputOutputUnits(input_unit=…, output_unit=…)`, not "
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
                f"{qname}: declares `unit=InputOutputUnits(...)` but leaves "
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
        if not isinstance(obj, ParamFunction) or not isinstance(obj.unit, UnsetUnit):
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
    - an ``Annotated[<schedule type>, InputOutputUnits(...)]`` field → a
      :class:`_ScheduleFieldKind` carrying the schedule's input and output axes;
    - a nested-dataclass field → its class (whose plucks resolve recursively);
    - anything else (a bare scalar, dict, array) is absent — the pluck stays
      opaque and is cast at the site.

    A field whose annotation does not resolve at runtime (a name imported only
    under ``TYPE_CHECKING``) is skipped individually, so its pluck stays opaque
    while its siblings are still resolved and validated. ``None`` comes back only
    when the class carries no resolvable annotation at all.

    Raises:
        UnitDefinitionError: If a field annotates several units, mismatches the
            marker to the field kind (a bare ``CompositeUnit`` on a schedule field,
            an ``InputOutputUnits`` on a scalar field), annotates a non-scalar,
            non-schedule field, or pins a concrete currency.
    """
    hints = _resolvable_type_hints(cls=cls)
    if not hints:
        return None
    kinds: dict[str, pint.Unit | type | _ScheduleFieldKind] = {}
    for field in dataclasses.fields(cast("Any", cls)):
        hint = hints.get(field.name, field.type)
        metadata = getattr(hint, "__metadata__", ())
        composite_tokens = [t for t in metadata if isinstance(t, CompositeUnit)]
        io_tokens = [t for t in metadata if isinstance(t, InputOutputUnits)]
        base = get_args(hint)[0] if hasattr(hint, "__metadata__") else hint
        where = f"Field '{cls.__name__}.{field.name}'"
        is_schedule = isinstance(base, type) and issubclass(base, _SCHEDULE_VALUE_TYPES)
        if len(composite_tokens) + len(io_tokens) > 1:
            spelled = [str(t) for t in composite_tokens] + [
                "InputOutputUnits(...)" for _ in io_tokens
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
                f"{where}: annotates `InputOutputUnits(...)`, which declares a "
                f"schedule's two axes, but the field is not a lookup/piecewise "
                f"value; a scalar field states a single unit (GEP 10)."
            )
        elif composite_tokens:
            resolved = pint_unit_from_ttsim_unit_for_column(
                unit=composite_tokens[0],
                name=None,
                grouping_levels=(),
                where=where,
                registry=unit_system.registry,
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
    return kinds


@dataclasses.dataclass(frozen=True)
class _ScheduleFieldKind:
    """A parameter dataclass field typed as a schedule and carrying its two axes.

    A field annotated ``Annotated[<schedule type>, InputOutputUnits(...)]`` declares
    the schedule's INPUT and OUTPUT axes at the field, exactly as a
    schedule-returning ``@param_function`` declares them in its ``unit=`` — so a
    pluck of the field yields a :class:`_UnitCheckSchedule` that screens each
    ``look_up`` / ``piecewise_polynomial`` argument against ``input_unit`` and
    produces ``output_unit``.
    """

    input_unit: pint.Unit | tuple[pint.Unit, ...]
    """The resolved input axis (or positional axes) each domain argument is
    screened against."""
    output_unit: pint.Unit
    """The resolved unit the schedule produces."""


def _schedule_field_kind(
    base: type,
    io_tokens: list[InputOutputUnits],
    composite_tokens: list[CompositeUnit],
    unit_system: UnitSystem,
    where: str,
) -> _ScheduleFieldKind:
    """Resolve a schedule-typed field's declared axes into a kind.

    A schedule field states its two axes with exactly one ``InputOutputUnits`` in
    its ``Annotated[...]`` metadata. A bare ``CompositeUnit`` there declares a
    single quantity, which a schedule (a function between quantities) is not, so
    it is rejected with a pointer to ``InputOutputUnits``.

    The same type-specific axis rules the decorator enforces hold here, keyed off
    the field's schedule type: a ``PiecewisePolynomialParamValue`` field takes one
    domain argument, so a tuple ``input_unit`` is rejected; a
    ``ConsecutiveIntLookupTableParamValue`` field is keyed by consecutive integers,
    so no ``input_unit`` axis may be a currency.

    Raises:
        UnitDefinitionError: If the field carries a bare ``CompositeUnit`` marker,
            no ``InputOutputUnits`` at all, a tuple ``input_unit`` on a piecewise
            field, or a currency axis on a lookup-table field.
    """
    if composite_tokens:
        raise UnitDefinitionError(
            f"{where}: a schedule field (a lookup/piecewise value) is a function "
            f"between quantities, so it declares both axes with "
            f"`InputOutputUnits(input_unit=…, output_unit=…)`, not the single unit "
            f"`{composite_tokens[0]}` (GEP 10)."
        )
    if not io_tokens:
        raise UnitDefinitionError(
            f"{where}: a schedule field (a lookup/piecewise value) must annotate "
            f"`InputOutputUnits(input_unit=…, output_unit=…)` declaring its two axes "
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
        output_unit=pint_unit_from_ttsim_unit_for_column(
            unit=io_token.output_unit,
            name=None,
            grouping_levels=(),
            where=where,
            registry=unit_system.registry,
        ),
    )


def _resolve_input_axes(
    input_unit: CompositeUnit | tuple[CompositeUnit, ...],
    registry: pint.UnitRegistry,
    where: str,
) -> pint.Unit | tuple[pint.Unit, ...]:
    """Resolve a schedule declaration's input axis or axes.

    A single :class:`CompositeUnit` resolves to one pint unit, screened against
    every ``look_up`` argument; a tuple resolves to a tuple of pint units screened
    positionally (argument ``i`` against axis ``i``). Each axis is agnostic — a
    concrete currency is rejected element-wise, exactly as for a scalar field.
    """
    if isinstance(input_unit, tuple):
        return tuple(
            pint_unit_from_ttsim_unit_for_column(
                unit=cast("CompositeUnit", axis),
                name=None,
                grouping_levels=(),
                where=where,
                registry=registry,
            )
            for axis in input_unit
        )
    return pint_unit_from_ttsim_unit_for_column(
        unit=input_unit,
        name=None,
        grouping_levels=(),
        where=where,
        registry=registry,
    )


def _resolve_schedule_input_unit(
    obj: ParamMappingObject, registry: pint.UnitRegistry
) -> pint.Unit | None:
    """The resolved ``input_unit`` of a schedule/lookup parameter, or ``None``.

    Resolved the same way as the ``output_unit`` the environment exposes, so a
    concrete-currency input axis and an agnostic ``CURRENCY`` consumer argument
    compare as equivalent. ``None`` when the parameter left ``input_unit`` unset.
    """
    if isinstance(obj.input_unit, UnsetUnit):
        return None
    return pint_unit_from_ttsim_unit_for_param(
        unit=cast("CompositeUnit", obj.input_unit),
        name=None,
        grouping_levels=(),
        where="A schedule input axis",
        registry=registry,
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
    cast at the site.
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
#: (``dict[str, dict[int, ...]]``) fails to match and stays opaque.
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
        annotations = inspect.get_annotations(klass, eval_str=False)
        for name, annotation in annotations.items():
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
    if isinstance(ttsim_unit, UnsetUnit):
        return "unset"
    return str(ttsim_unit)


FRAMEWORK_DATE_NODE_UNITS: Mapping[str, str] = {
    "policy_year": "calendar_year",
    "policy_month": "dimensionless",
    "policy_day": "dimensionless",
    "evaluation_year": "calendar_year",
    "evaluation_month": "dimensionless",
    "evaluation_day": "dimensionless",
}


def resolve_environment_units(
    env: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
    grouping_levels: OrderedQNames,
    unit_system: UnitSystem,
) -> dict[str, pint.Unit | dict[str | int, Any]]:
    """Resolve the complete unit of every annotated node in the environment."""
    registry = unit_system.registry
    register_grouping_levels(names=grouping_levels, registry=registry)
    pattern = get_re_pattern_for_all_time_units_and_groupings(
        time_units=tuple(TIME_UNIT_IDS_TO_LABELS),
        grouping_levels=grouping_levels,
    )
    resolved: dict[str, pint.Unit | dict[str | int, Any]] = {
        # `pint_unit_from_string` guides declarations to the DIMENSIONLESS token, so the
        # framework-internal ordinal spelling resolves directly.
        qname: (
            _dimensionless_unit(registry)
            if unit == "dimensionless"
            else pint_unit_from_string(unit_str=unit, registry=registry)
        )
        for qname, unit in FRAMEWORK_DATE_NODE_UNITS.items()
        if qname in env
    }
    for qname, obj in env.items():
        if not _is_checkable_node(qname=qname, obj=obj):
            continue
        if isinstance(obj, ParamObject):
            leaf_name = dt.tree_path_from_qname(qname)[-1]
            match = pattern.fullmatch(leaf_name)
            name_time_unit_id = match.group("time_unit") if match else None
            param_unit = _resolve_param_object_unit(
                qname=qname,
                obj=obj,
                grouping_levels=grouping_levels,
                registry=registry,
                name_time_unit_id=name_time_unit_id,
            )
            if param_unit is not None:
                resolved[qname] = param_unit
        elif isinstance(obj, AggByGroupFunction):
            agg_unit = _resolve_agg_by_group_unit(
                qname=qname,
                obj=obj,
                env=env,
                pattern=pattern,
                grouping_levels=grouping_levels,
                registry=registry,
            )
            if agg_unit is not None:
                resolved[qname] = agg_unit
        elif isinstance(obj, ParamFunction) and _returns_a_schedule(obj):
            token = getattr(obj, "unit", UNSET_UNIT)
            if isinstance(token, InputOutputUnits):
                # A schedule-returning param function declares its schedule's two
                # axes with `unit=InputOutputUnits(...)`; the environment-level
                # resolved unit is the OUTPUT axis (what `look_up` /
                # `piecewise_polynomial` consumers receive), resolved agnostically
                # like a field annotation — concrete currencies rejected, no
                # name-suffix rules.
                resolved[qname] = pint_unit_from_ttsim_unit_for_column(
                    unit=token.output_unit,
                    name=None,
                    grouping_levels=grouping_levels,
                    where=f"Schedule param function {qname!r}",
                    registry=registry,
                )
        else:  # ColumnObject | scalar ParamFunction
            token = getattr(obj, "unit", UNSET_UNIT)
            if isinstance(token, CompositeUnit):
                resolved[qname] = _resolve_leveled_column_unit(
                    token=token,
                    leaf_name=dt.tree_path_from_qname(qname)[-1],
                    grouping_levels=grouping_levels,
                    registry=registry,
                )
    return resolved


def _resolve_leveled_column_unit(
    token: CompositeUnit,
    leaf_name: str,
    grouping_levels: OrderedQNames,
    registry: pint.UnitRegistry,
) -> pint.Unit:
    """Resolve a column/function's full unit, including its grouping level."""
    return pint_unit_from_ttsim_unit_for_column(
        unit=token,
        name=leaf_name,
        grouping_levels=grouping_levels,
        where="A column/function",
        registry=registry,
    )


def _resolve_agg_by_group_unit(
    qname: str,
    obj: AggByGroupFunction,
    env: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
    pattern: re.Pattern[str],
    grouping_levels: OrderedQNames,
    registry: pint.UnitRegistry,
) -> pint.Unit | None:
    """Resolve a group-aggregation node's unit, level-aware.

    The aggregation is where a grouping level is minted, swapped, or preserved.
    The *target* level is the node's own aggregation suffix (an ``_hh`` node
    aggregates to ``[hh]``). The rule depends on
    :attr:`AggByGroupFunction.agg_type`:

    - a **head count** — ``COUNT``, or a ``SUM`` over a *boolean* source (counting
      the persons the indicator is true for) — mints ``1 / [target]``;
    - ``SUM`` / ``MIN`` / ``MAX`` resolve to the **target** level whatever the
      source (a bare source acquires it); ``MEAN`` resolves to **bare** — a
      per-head average belongs to the individual;
    - ``ANY`` / ``ALL`` yield a dimensionless boolean at the target level.

    The value source is the function's summed/averaged argument, read off the
    signature rather than by stripping the name suffix — a hand-written
    aggregation (``number_of_adults_fam`` sums ``adult``, not ``number_of_adults``)
    resolves correctly.
    """
    match = pattern.fullmatch(dt.tree_path_from_qname(qname)[-1])
    if not obj.verify_units:
        # An opted-out aggregation resolves from its own declared unit, exactly
        # like a plain column.
        token = getattr(obj, "unit", UNSET_UNIT)
        return (
            None
            if isinstance(token, UnsetUnit)
            else _resolve_leveled_column_unit(
                token=cast("CompositeUnit", token),
                leaf_name=dt.tree_path_from_qname(qname)[-1],
                grouping_levels=grouping_levels,
                registry=registry,
            )
        )
    target_level = _suffix_grouping_level(match)
    agg_type = obj.agg_type
    # COUNT and ANY/ALL are independent of the source's unit, so resolve them
    # before touching the source: well-defined even when the source declares none.
    if agg_type in (AggType.COUNT, AggType.ANY, AggType.ALL):
        return resolved_unit_for_aggregation(
            agg_type=agg_type, target_level=target_level, registry=registry
        )
    sources = {
        p
        for p in inspect.signature(obj.function).parameters
        if not p.endswith("_id")
        and not qname_is_person_pointer(p)
        and p not in FRAMEWORK_PARTIAL_ARGUMENTS
    }
    if len(sources) != 1:
        return None
    source_qname = sources.pop()
    source_obj = env.get(source_qname)
    source_token = getattr(source_obj, "unit", UNSET_UNIT)
    if isinstance(source_token, UnsetUnit):
        return None
    source_is_boolean = node_is_boolean(qname=source_qname, obj=source_obj)
    # A SUM over a boolean is a head count of the persons it is true for — the
    # same unit a COUNT mints, so resolve it as one (shared rule with the
    # declared-token minter, `unit_for_aggregation`).
    if (
        head_count_from_boolean_sum(
            agg_type=agg_type, source_is_boolean=source_is_boolean
        )
        is AggType.COUNT
    ):
        return resolved_unit_for_aggregation(
            agg_type=AggType.COUNT, target_level=target_level, registry=registry
        )
    source_unit = _resolve_leveled_column_unit(
        token=cast("CompositeUnit", source_token),
        leaf_name=dt.tree_path_from_qname(source_qname)[-1],
        grouping_levels=grouping_levels,
        registry=registry,
    )
    # The source's grouping level, read off its declared token (equivalently, its
    # resolved denominator — the two agree now that no level cancels).
    source_level = _composite_token_level(cast("CompositeUnit", source_token))
    return resolved_unit_for_aggregation(
        source_unit=source_unit,
        agg_type=agg_type,
        target_level=target_level,
        source_level=source_level,
        registry=registry,
    )


def _resolve_param_object_unit(
    qname: str,
    obj: ParamObject,
    grouping_levels: OrderedQNames,
    registry: pint.UnitRegistry,
    name_time_unit_id: str | None = None,
) -> pint.Unit | dict[str | int, Any] | None:
    """Resolve a parameter's declared TTSIM unit to a pint unit.

    Every parameter spells its unit fully: a **scalar** additionally
    takes a time suffix on its *name*, which must agree with the spelled period
    (``lump_sum_deduction_y`` declaring ``CURRENCY_PER_YEAR``). A **dict** or
    **require_converter** parameter with heterogeneous leaves declares a
    per-leaf ``unit:`` mapping (see :func:`_resolve_unit_mapping`), resolving to
    a nested dict of pint units mirroring the value structure; a uniformly typed
    one declares one unit for the whole structure. Mapping parameters
    (schedules, lookup tables) declare per-axis units instead — see
    :func:`_resolve_param_mapping_object_units`. Returns ``None`` for an
    unannotated parameter — the mandatory-units check reports it.
    """
    if isinstance(obj, ParamMappingObject):
        return _resolve_param_mapping_object_units(
            qname=qname,
            obj=obj,
            name_time_unit_id=name_time_unit_id,
            grouping_levels=grouping_levels,
            registry=registry,
        )
    if isinstance(obj.unit, UnsetUnit):
        return None
    if isinstance(obj.unit, Mapping):
        return _resolve_unit_mapping(
            qname=qname,
            unit_mapping=cast("Mapping[str | int, Any]", obj.unit),
            grouping_levels=grouping_levels,
            registry=registry,
        )
    # A scalar parameter takes its period from a time suffix on its name; a
    # dict/raw parameter has no single name to suffix.
    return pint_unit_from_ttsim_unit_for_param(
        unit=cast("CompositeUnit", obj.unit),
        name=dt.tree_path_from_qname(qname)[-1]
        if isinstance(obj, ScalarParam)
        else None,
        grouping_levels=grouping_levels,
        where=f"Parameter {qname!r}",
        registry=registry,
    )


def _resolve_param_mapping_object_units(
    qname: str,
    obj: ParamMappingObject,
    name_time_unit_id: str | None,
    grouping_levels: OrderedQNames,
    registry: pint.UnitRegistry,
) -> pint.Unit | None:
    """Resolve a mapping parameter's per-axis unit declarations.

    A schedule or lookup table is a function between quantities: it declares
    ``input_unit:`` and ``output_unit:`` instead of ``unit:``. Each axis
    is a fully-spelled :class:`CompositeUnit`, so its period and level are in the
    string. A time suffix on the parameter's *name* describes what it yields, so
    it must coincide with a flow ``output_unit``.

    The environment-level resolved unit is the *output* unit (what consumers
    receive); the input unit is validated here and consumed by the build-time
    currency conversion. Returns ``None`` if the output unit is unset — the
    mandatory-units check reports it.
    """
    tokens: dict[str, UnitDeclaration] = {}
    for axis, raw in (("input_unit", obj.input_unit), ("output_unit", obj.output_unit)):
        if isinstance(raw, UnsetUnit):
            tokens[axis] = UNSET_UNIT
            continue
        where = f"Parameter {qname!r}, {axis}"
        if isinstance(raw, Mapping):
            raise UnitDefinitionError(
                f"{where}: per-axis declarations are single units, not "
                f"mappings (GEP 10)."
            )
        tokens[axis] = cast("CompositeUnit", raw)
    output_token = tokens["output_unit"]
    if name_time_unit_id is not None:
        _fail_if_name_suffix_disagrees_with_output_axis(
            qname=qname,
            output_token=output_token,
            name_time_unit_id=name_time_unit_id,
            grouping_levels=grouping_levels,
            registry=registry,
        )
    input_token = tokens["input_unit"]
    if not isinstance(input_token, UnsetUnit):
        pint_unit_from_ttsim_unit_for_param(
            unit=input_token,
            name=None,
            grouping_levels=grouping_levels,
            where=f"Parameter {qname!r}, input_unit",
            registry=registry,
        )
    if isinstance(output_token, UnsetUnit):
        return None
    return pint_unit_from_ttsim_unit_for_param(
        unit=output_token,
        name=None,
        grouping_levels=grouping_levels,
        where=f"Parameter {qname!r}, output_unit",
        registry=registry,
    )


def _fail_if_name_suffix_disagrees_with_output_axis(
    qname: str,
    output_token: Any,  # noqa: ANN401
    name_time_unit_id: str,
    grouping_levels: OrderedQNames,
    registry: pint.UnitRegistry,
) -> None:
    """Check the name-suffix ⟺ flow-output coincidence rules.

    A name time suffix denotes a flow, so the ``output_unit`` must be a flow and
    its spelled period must agree with the suffix (validated by resolving the
    output unit against the suffix).
    """
    if isinstance(output_token, UnsetUnit) or not output_token.is_flow:
        raise UnitDefinitionError(
            f"Parameter {qname!r}: the name carries a time-unit suffix "
            f"(_{name_time_unit_id}), which denotes a flow, but "
            f"`output_unit:` is {_spell_ttsim_unit(output_token)} (GEP 10)."
        )
    pint_unit_from_ttsim_unit_for_param(
        unit=output_token,
        name=dt.tree_path_from_qname(qname)[-1],
        grouping_levels=grouping_levels,
        where=f"Parameter {qname!r}",
        registry=registry,
    )


def _resolve_unit_mapping(
    qname: str,
    unit_mapping: Mapping[str | int, Any],
    grouping_levels: OrderedQNames,
    registry: pint.UnitRegistry,
) -> dict[str | int, Any]:
    """Resolve a per-leaf ``unit:`` mapping to pint units.

    Each leaf is a fully-spelled :class:`CompositeUnit` (``DIMENSIONLESS`` for a
    dimensionless leaf), so its period and level are in the string; a leaf-key
    time suffix, if any, must agree with the spelled period. Nested mappings
    recurse, mirroring the value structure.
    """
    resolved: dict[str | int, Any] = {}
    for key, token in unit_mapping.items():
        if isinstance(token, Mapping):
            resolved[key] = _resolve_unit_mapping(
                qname=qname,
                unit_mapping=token,
                grouping_levels=grouping_levels,
                registry=registry,
            )
            continue
        where = f"Parameter {qname!r}, unit of leaf {key!r}"
        resolved[key] = pint_unit_from_ttsim_unit_for_param(
            unit=cast("CompositeUnit", token),
            name=str(key),
            grouping_levels=grouping_levels,
            where=where,
            registry=registry,
        )
    return resolved


def _is_checkable_node(qname: str, obj: Any) -> bool:  # noqa: ANN401
    """Whether a node's unit is resolved from its own declaration here.

    Every column object and parameter qualifies — including identifiers and
    boolean nodes (dimensionless quantities that declare ``DIMENSIONLESS``) and
    ``@group_creation_function`` group ids (auto-assigned ``DIMENSIONLESS``).
    The framework date nodes are the one case handled elsewhere: their unit
    comes from :data:`FRAMEWORK_DATE_NODE_UNITS`, not an author declaration, so
    they are resolved there and skipped here.
    """
    if not isinstance(obj, ColumnObject | ParamFunction | ParamObject):
        return False
    return qname not in FRAMEWORK_DATE_NODE_UNITS


def node_is_boolean(qname: str, obj: Any) -> bool:  # noqa: ANN401
    """Whether a node's output is boolean."""
    try:
        if isinstance(obj, PolicyInput):
            kind = resolve_kind_of_annotation(annotation=obj.data_type, node_name=qname)
        elif isinstance(obj, ColumnObject | ParamFunction):
            kind = resolve_kind_of_column_function(func=obj.function, node_name=qname)
        else:
            return False
    except TypeResolutionError:
        return False
    return kind in BOOL_KINDS


def _suffix_grouping_level(match: re.Match[str] | None) -> str | None:
    """The grouping level named by a name's aggregation suffix."""
    if match is None:
        return None
    return match.group("grouping") or None


def _composite_token_level(token: CompositeUnit) -> str | None:
    """The grouping level a TTSIM unit spells, or ``None`` if bare."""
    return token.level.lower() if token.level is not None else None


def _has_grouping_level_numerator(unit: pint.Unit) -> bool:
    """Whether a unit carries a grouping level as a *numerator*."""
    return any(exponent > 0 for _, exponent in _grouping_levels_with_exponent(unit))


class BooleanLevel(NamedTuple):
    """A unit's classification as a (possibly leveled) boolean."""

    is_boolean: bool
    """Whether the unit is a truth value at all."""
    level: str | None
    """The grouping level the truth value is measured per, `None` for a
    level-less boolean. Meaningless when `is_boolean` is False."""


def _as_boolean_level(unit: pint.Unit, registry: pint.UnitRegistry) -> BooleanLevel:
    """Classify a unit as a (possibly leveled) boolean and read its level.

    A boolean is a truth value: dimensionless apart from at most a single grouping
    level it is measured *per* — ``1 / [fam]`` for a fam-level indicator, plain
    dimensionless for a level-less share/flag. A unit with physical content
    (currency, area, a duration) or a grouping-level *numerator* (``[hh]``) is
    *not* a boolean. A head count is ``1 / [hh]``, so it is indistinguishable
    from a leveled boolean here — both are the plain number over their group.

    ``1 / [fam]`` → ``(True, "fam")``; a plain ``1`` → ``(True, None)``;
    ``EUR_PER_MONTH`` or ``[hh]`` → ``(False, None)``.
    """
    if _has_grouping_level_numerator(unit):
        return BooleanLevel(is_boolean=False, level=None)
    if not registry.Quantity(
        1.0, _unit_without_grouping_levels(unit=unit, registry=registry)
    ).dimensionless:
        return BooleanLevel(is_boolean=False, level=None)
    return BooleanLevel(is_boolean=True, level=_unit_level_denominator(unit))


def _boolean_quantity(level: str | None, registry: pint.UnitRegistry) -> pint.Quantity:
    """A representative boolean ``Quantity`` at ``level`` — ``1 / [level]``.

    ``_boolean_quantity("fam")`` is ``1 / [fam]`` (a fam-level indicator);
    ``_boolean_quantity(None)`` is a plain dimensionless ``1`` (a level-less flag).
    """
    truth = registry.Quantity(1.0, "")
    if level is None:
        return truth
    return truth / registry.Quantity(1.0, f"{_GROUPING_LEVEL_PREFIX}{level}")


def _combined_boolean_level(left: str | None, right: str | None) -> str | None:
    """Combine two boolean levels for a logical operator.

    Equal levels are kept; any mismatch downcasts to the individual level, which
    is **bare** (``None``).

    Two fam-level indicators give ``"fam"``; the mixed
    ``wealth_fam >= threshold_fam or wealth_kin >= threshold_kin`` combines a fam-
    and a kin-level operand, so the result is bare (``None``).
    """
    return left if left == right else None


def _dimensionless_unit(registry: pint.UnitRegistry) -> pint.Unit:
    """The dimensionless unit, used when reporting a logical op's bare operand."""
    return registry.dimensionless


def fail_if_environment_units_are_missing(
    env: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
    grouping_levels: OrderedQNames,  # noqa: ARG001  (kept for symmetry of the two checks)
) -> None:
    """Mandatory-units check over a fully assembled environment.

    Every active node must declare a unit:

    - a dict or require_converter parameter with per-leaf units must cover every
      leaf of the value active at the policy date;
    - a ``@param_function`` declaring ``unit=UNSET_UNIT`` (a structured value) or
      ``unit=InputOutputUnits(...)`` (a schedule builder) is exempt — its output is
      not a single quantity, and its units live in the field annotations or the
      declaration itself, so neither is an omission;
    - a rounding spec on a currency-valued function must declare its own unit (its
      magnitudes are statutory numbers in a concrete currency, like a
      parameter's); a missing one is reported as ``<qname> (rounding_spec)``.

    Raises:
        UnitDefinitionError: If any node (or per-leaf-mapping leaf) lacks a unit
            declaration.
    """
    units_by_qname: dict[str, UnitDeclaration] = {}
    for qname, obj in env.items():
        if not isinstance(obj, ColumnObject | ParamFunction | ParamObject):
            continue
        if qname in FRAMEWORK_DATE_NODE_UNITS:
            continue
        declared_unit = getattr(obj, "unit", UNSET_UNIT)
        if isinstance(obj, ParamFunction) and (
            isinstance(declared_unit, UnsetUnit | InputOutputUnits)
        ):
            # A structured value (`UNSET_UNIT`) states its units in the return
            # type's field annotations; a schedule builder (`InputOutputUnits`)
            # states them in the declaration itself.
            continue
        if isinstance(obj, ParamMappingObject | RawParam) and (
            isinstance(obj, ParamMappingObject)
            or not isinstance(obj.input_unit, UnsetUnit)
            or not isinstance(obj.output_unit, UnsetUnit)
        ):
            # A schedule/lookup or a per-axis require_converter declares per-axis
            # units instead of a single `unit:`.
            units_by_qname[f"{qname} (input_unit)"] = cast(
                "CompositeUnit", obj.input_unit
            )
            units_by_qname[f"{qname} (output_unit)"] = cast(
                "CompositeUnit", obj.output_unit
            )
            continue
        if isinstance(obj, ParamObject) and isinstance(declared_unit, Mapping):
            value = getattr(obj, "value", None)
            value_tree = value if isinstance(value, Mapping) else {}
            units_by_leaf = dt.flatten_to_qnames(
                cast("Mapping[str, Any]", declared_unit)
            )
            for leaf_qname in dt.flatten_to_qnames(value_tree):
                # A flat int-keyed dict (GEP 3 allows them, e.g. a Satz keyed by
                # child count) leaves its key an int, which has no `.split`; the
                # per-leaf unit mapping is keyed the same way, so look it up with
                # the original key and stringify only for the display path.
                leaf_path = dt.tree_path_from_qname(str(leaf_qname))
                display = f"{qname}[{']['.join(leaf_path)}]"
                # A leaf absent from the mapping defaults to :data:`UNSET_UNIT`,
                # which the mandatory-units check reports.
                units_by_qname[display] = units_by_leaf.get(leaf_qname, UNSET_UNIT)
        else:
            units_by_qname[qname] = cast("CompositeUnit", declared_unit)
            rounding_spec = getattr(obj, "rounding_spec", None)
            if (
                rounding_spec is not None
                and rounding_spec.unit is None
                and ttsim_unit_has_agnostic_currency(
                    cast("CompositeUnit", declared_unit)
                )
            ):
                units_by_qname[f"{qname} (rounding_spec)"] = UNSET_UNIT
    fail_if_units_are_missing(units_by_qname)


def fail_if_environment_units_are_inconsistent(
    env: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
    grouping_levels: OrderedQNames,
    unit_system: UnitSystem,
    resolved_pint_units: dict[str, pint.Unit | dict[str | int, Any]] | None = None,
) -> None:
    """Conservative body/edge verification over an assembled environment.

    Each kind of node is checked against its declaration:

    - a ``@policy_function`` / ``@param_function`` **body** is unit-checked on
      representative values built from its producers' resolved units (the DAG edges) —
      see the module docstring for the conservative rules and the branch-exploration
      strategy;
    - an **aggregation** has no scalar body but *derives* a unit from its source and
      agg_type; its declared token is checked against that derivation, the same
      declared-vs-produced contract a body is held to;
    - a **rounding spec** is checked against its function's unit
      (:func:`_rounding_spec_declaration_inconsistency`), and every **param function**
      against the schedule contract keyed off its ``unit=`` and return annotation
      (:func:`_schedule_param_function_contract_errors`).

    Time-conversion variants and group-creation functions are unit-assigned by
    construction and need no check.

    A structured value a ``@param_function`` builds carries its units solely from its
    type's field annotations (:func:`_structured_field_kinds`); the source parameter's
    per-leaf ``unit:`` mapping drives that raw value's currency conversion and documents
    its input, but is not cross-checked against the built object — a converter may
    legitimately transform units.

    Raises:
        UnitConsistencyError: If any body infers a concrete unit that disagrees
            with its declaration, an aggregation's declared unit disagrees with what it
            derives, or a schedule builder breaks the ``InputOutputUnits`` contract
            (declaration vs. return annotation vs. ``verify_units``). All offending
            nodes are reported together.
        UnitDefinitionError: If an ``InputOutputUnits`` axis or a
            parameter-dataclass field annotation is invalid.
    """
    inference = _unit_inference_module()
    registry = unit_system.registry
    _fail_if_structured_field_annotations_are_invalid(env=env, unit_system=unit_system)
    if resolved_pint_units is None:
        resolved_pint_units = resolve_environment_units(
            env=env, grouping_levels=grouping_levels, unit_system=unit_system
        )
    representative_values = inference._representative_values_by_qname(  # noqa: SLF001
        env=env, resolved_pint_units=resolved_pint_units, unit_system=unit_system
    )
    boolean_nodes = {
        qname
        for qname, obj in env.items()
        if isinstance(obj, ColumnObject | ParamFunction)
        and node_is_boolean(qname=qname, obj=obj)
    }
    errors: list[str] = _aggregation_declaration_errors(
        env=env, resolved_pint_units=resolved_pint_units, registry=registry
    )
    errors.extend(_rounding_spec_declaration_errors(env=env))
    errors.extend(_schedule_param_function_contract_errors(env=env))
    errors.extend(
        _body_verification_errors(
            env=env,
            resolved_pint_units=resolved_pint_units,
            representative_values=representative_values,
            boolean_nodes=boolean_nodes,
            unit_system=unit_system,
        )
    )
    if errors:
        raise UnitConsistencyError(
            "Environment unit-consistency check failed:\n  " + "\n  ".join(errors)
        )


def _measurement_unit(units: pint.Unit, registry: pint.UnitRegistry) -> pint.Unit:
    """Return a unit without currency, flow-period, or grouping-level components."""
    currency = _pint_unit_currency(units=units, registry=registry)
    residual = units / currency if currency is not None else units
    period = _flow_period_of(units=residual, registry=registry)
    residual = residual * period if period is not None else residual
    return _unit_without_grouping_levels(unit=residual, registry=registry)


def fail_if_input_units_are_inconsistent(
    input_ttsim_units: Mapping[str, CompositeUnit],
    resolved_pint_units: Mapping[str, Any],
    unit_system: UnitSystem,
    declared_ttsim_units: Mapping[str, CompositeUnit] | None = None,
) -> None:
    """Fail if an input column's tag disagrees with the unit declared for it.

    Two units meet here: the input column's **tag** (the unit the user attaches
    to the data, ``UnitAnnotatedColumn(..., unit=TTSIMUnit.EUR.PER_MONTH)``) and the
    unit **declared for that column in the policy environment** (its ``unit=`` /
    ``unit:``). They must agree, checked on three axes:

    - **currency presence** — the boundary only rescales a column it thinks is
      currency, so a ``DM`` tag on a ``DIMENSIONLESS`` column (or the converse)
      would silently rescale — or skip rescaling — the data; both sides must agree
      on whether a currency component is present;
    - **grouping level** — the tag spells its group level (``EUR.PER_MONTH.PER_BG``)
      or omits it for a bare (per-person / level-neutral) quantity, and this must
      equal the level the declared unit carries (the level is declared, not read
      off the suffix);
    - **measurement** — with currency (converted at the boundary) and flow period
      (screened against the name suffix by the dedicated period guard) factored
      out, the remaining numerator scale must match *exactly*: a ``HECTARES``
      column tagged ``m²`` (a 10,000-fold error) or a ``YEARS`` age tagged
      ``month`` is rejected here rather than silently mis-stripped.

    Raises:
        UnitConsistencyError: If any tagged column disagrees with the unit
            declared for it. All offending columns are reported together.
    """
    registry = unit_system.registry
    errors: list[str] = []
    for qname, tag_token in input_ttsim_units.items():
        tag = pint_unit_from_ttsim_unit(
            unit=tag_token, registry=registry, with_level=True
        )
        expected = resolved_pint_units.get(qname)
        if not isinstance(expected, pint.Unit):
            # No scalar declared unit (absent, or a dict parameter); nothing to check.
            continue
        if pint_unit_has_currency(
            units=tag, registry=registry
        ) != pint_unit_has_currency(units=expected, registry=registry):
            errors.append(
                f"  {qname}: tagged '{tag}' but declared '{expected}' — one carries "
                "a currency and the other does not."
            )
            continue
        tag_level = _composite_token_level(tag_token)
        declared_token = (declared_ttsim_units or {}).get(qname)
        expected_level = (
            _composite_token_level(declared_token)
            if declared_token is not None
            else _unit_level_denominator(expected)
        )
        if tag_level != expected_level:
            errors.append(
                f"  {qname}: tag is at the {tag_level or 'bare'!r} level but "
                f"the column is at the {expected_level or 'bare'!r} level."
            )
            continue
        tag_residual = _measurement_unit(units=tag, registry=registry)
        expected_residual = _measurement_unit(units=expected, registry=registry)
        if not units_are_equivalent(
            left=tag_residual, right=expected_residual, registry=registry
        ):
            errors.append(
                f"  {qname}: tagged '{tag}', which is not equivalent to the declared "
                f"unit '{expected}'."
            )
    if errors:
        raise UnitConsistencyError(
            "Input unit annotations are inconsistent with the DAG's declared "
            "units:\n" + "\n".join(sorted(errors))
        )


def flatten_unit_annotated_input_tree(
    tree: UserNestedUnitAnnotatedData,
) -> dict[tuple[str, ...], UnitAnnotatedColumn]:
    """Flatten the unit-annotated input tree, rejecting any bare leaf.

    Raises:
        UnitConsistencyError: If any leaf is not a ``UnitAnnotatedColumn``.
    """
    flat = dt.flatten_to_tree_paths(tree)
    fail_if_not_all_leaves_are_unit_annotated_columns(flat=flat)
    return cast("dict[tuple[str, ...], UnitAnnotatedColumn]", flat)


def fail_if_not_all_leaves_are_unit_annotated_columns(
    flat: Mapping[tuple[str, ...], Any],
) -> None:
    """Reject a unit-annotated input tree with any bare (untagged) leaf.

    Every leaf of the unit-annotated input tree must be a :class:`UnitAnnotatedColumn`.
    Reach the tree through :func:`flatten_unit_annotated_input_tree` rather than calling
    this directly: every consumer dereferences ``.unit`` or ``.values``, so the check
    belongs to the flattening step that hands those leaves out.

    Raises:
        UnitConsistencyError: If any leaf is not a ``UnitAnnotatedColumn``.
    """
    untagged = sorted(
        dt.qname_from_tree_path(path)
        for path, value in flat.items()
        if not isinstance(value, UnitAnnotatedColumn)
    )
    if untagged:
        raise UnitConsistencyError(
            "input_data__tree_with_unit_annotations requires every leaf to be a "
            "UnitAnnotatedColumn (GEP 10), but these are bare: "
            f"{', '.join(untagged)}. Tag a dimensionless column (an id, a boolean) "
            "with `UnitAnnotatedColumn(values=arr, unit=TTSIMUnit.DIMENSIONLESS)`, or "
            "pass untagged data via input_data__tree."
        )


def _aggregation_declaration_errors(
    env: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
    resolved_pint_units: Mapping[str, pint.Unit | dict[str | int, Any]],
    registry: pint.UnitRegistry,
) -> list[str]:
    """Declared-vs-derived errors for every group aggregation."""
    return [
        error
        for qname, obj in env.items()
        if isinstance(obj, AggByGroupFunction)
        and (
            error := _agg_declaration_inconsistency(
                qname=qname,
                obj=obj,
                resolved_pint_units=resolved_pint_units,
                registry=registry,
            )
        )
        is not None
    ]


def _agg_declaration_inconsistency(
    qname: str,
    obj: AggByGroupFunction,
    resolved_pint_units: Mapping[str, pint.Unit | dict[str | int, Any]],
    registry: pint.UnitRegistry,
) -> str | None:
    """Error message if an aggregation's declared unit ≠ what it derives."""
    if not obj.verify_units:
        # An opted-out aggregation keeps its declared unit as the contract for
        # consumers, but its declaration is not checked against the derivation —
        # for the rare group-property aggregation whose declared level the
        # derivation cannot express (a MEAN the author states `PER_KIN`).
        return None
    derived = resolved_pint_units.get(qname)
    declared_token = getattr(obj, "unit", UNSET_UNIT)
    if (
        derived is None
        or isinstance(derived, dict)
        or isinstance(declared_token, UnsetUnit)
    ):
        return None
    # An aggregation is a column: it declares an agnostic currency, and its
    # declared level is checked against the derived unit right below rather than
    # against a name suffix.
    declared_unit = pint_unit_from_ttsim_unit_for_column(
        unit=cast("CompositeUnit", declared_token),
        name=None,
        grouping_levels=(),
        where=f"Aggregation {qname!r}",
        registry=registry,
    )
    derived_unit = cast("pint.Unit", derived)
    if units_are_equivalent(left=declared_unit, right=derived_unit, registry=registry):
        return None
    return (
        f"{qname}: declares `{declared_token}` but its {obj.agg_type.name} "
        f"aggregation derives '{derived_unit}'. An aggregation's declared unit must "
        "match what it produces exactly — physical kind, flow period, and grouping "
        "level."
    )


def _rounding_spec_declaration_errors(
    env: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
) -> list[str]:
    """Spec-vs-function unit errors for every rounded column function."""
    return [
        error
        for qname, obj in env.items()
        if isinstance(obj, ColumnFunction)
        and (error := _rounding_spec_declaration_inconsistency(qname=qname, obj=obj))
        is not None
    ]


def _rounding_spec_declaration_inconsistency(
    qname: str,
    obj: ColumnFunction,
) -> str | None:
    """Error message if a rounding spec's unit disagrees with its function's.

    A rounding spec's magnitudes are statutory numbers in a concrete currency,
    exactly like a parameter's, so:

    - on a **currency-valued** function the spec pins down a registered currency
      and spells the full composite, which must equal the function's declared unit
      with the agnostic base swapped for the concrete one;
    - on a **non-currency** function the magnitudes are in the function's own unit
      and there is nothing to convert, so a declaration is rejected.

    A *missing* declaration on a currency-valued function — and a function without
    a unit — is the mandatory-units check's to report, not this one.
    """
    spec = getattr(obj, "rounding_spec", None)
    declared = getattr(obj, "unit", UNSET_UNIT)
    if spec is None or spec.unit is None or isinstance(declared, UnsetUnit):
        return None
    if not ttsim_unit_has_agnostic_currency(cast("CompositeUnit", declared)):
        return (
            f"{qname}: the rounding spec declares `{spec.unit}` but the function's "
            f"unit `{declared}` has no currency base, so there is nothing to "
            f"convert; drop the spec's `unit=`."
        )
    if ttsim_unit_has_agnostic_currency(spec.unit):
        return (
            f"{qname}: the rounding spec's magnitudes are written in a concrete "
            f"currency; declare it (e.g. `TTSIMUnit.DM.PER_YEAR`), not the agnostic "
            f"`{spec.unit}`."
        )
    if ttsim_unit_currency(spec.unit) is None:
        return (
            f"{qname}: the rounding spec's unit `{spec.unit}` does not pin down a "
            f"registered currency."
        )
    if ttsim_unit_with_agnostic_currency(spec.unit) != declared:
        return (
            f"{qname}: the rounding spec's unit `{spec.unit}` must equal the "
            f"function's declared `{declared}` with the agnostic base swapped for "
            f"the concrete currency — same flow period, same grouping level."
        )
    return None


def _body_verification_errors(
    env: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
    resolved_pint_units: Mapping[str, pint.Unit | dict[str | int, Any]],
    representative_values: Mapping[str, Any],
    boolean_nodes: set[str],
    unit_system: UnitSystem,
) -> list[str]:
    """Unit-check every human-written scalar body and collect its inference errors.

    Only ``@policy_function`` / ``@param_function`` bodies are unit-checked; everything
    else (aggregations, time-conversions, group ids) is unit-assigned by
    construction. A body is skipped when it is an unimplemented-period stub, has
    no resolved unit, opts out (``verify_units=False``), declares a structured
    unit, or has an unannotated producer.
    """
    inference = _unit_inference_module()
    registry = unit_system.registry
    # The helper stand-ins close over this run's registry, so they are built per run
    # rather than once at import.
    unit_check_helper_stand_ins, explorer_holder = (
        inference._unit_check_helper_stand_ins(  # noqa: SLF001
            unit_system=unit_system
        )
    )
    _anchor_schedules_on_body_explorer(
        representative_values=representative_values, explorer_holder=explorer_holder
    )
    errors: list[str] = []
    for qname, obj in env.items():
        if not isinstance(obj, PolicyFunction | ParamFunction):
            continue
        if getattr(obj, "fail_msg_if_included", None) is not None:
            continue
        if isinstance(obj, ParamFunction) and _returns_a_schedule(obj):
            # Its body builds a lookup/piecewise table, not a scalar; the declared
            # unit is the schedule's output contract (screened at the consumer's
            # `look_up`/`piecewise_polynomial`), so there is no scalar body to infer.
            continue
        if qname not in resolved_pint_units or not obj.verify_units:
            # Still UNSET (the mandatory-units check reports it), or the body
            # opted out — its declared unit stays the edge contract either way.
            continue
        declared = resolved_pint_units[qname]
        if isinstance(declared, dict):
            continue
        parameters = tuple(inspect.signature(obj.function).parameters)
        boolean_parameters = tuple(p for p in parameters if p in boolean_nodes)
        base_kwargs = inference._base_unit_check_kwargs(  # noqa: SLF001
            parameters=parameters,
            boolean_parameters=boolean_parameters,
            representative_values=representative_values,
        )
        if base_kwargs is None:
            continue
        # Feed each boolean parameter its resolved (possibly leveled) value, so a
        # leveled boolean carries its level into the body while only its truth value
        # is explorer-controlled; an unresolved producer falls back to level-less.
        boolean_values = {
            name: representative_values.get(name, registry.Quantity(1.0, ""))
            for name in boolean_parameters
        }
        error = inference._verify_one_body(  # noqa: SLF001
            qname=qname,
            function=recompile_with_logical_ops_as_calls(
                func=obj.function,
                module="xnp",
                module_obj=inference._NON_UNIT_ARGUMENT_VALUES["xnp"],  # noqa: SLF001
                extra_globals=unit_check_helper_stand_ins,
            ),
            declared=declared,
            boolean_values=boolean_values,
            base_kwargs=base_kwargs,
            unit_system=unit_system,
            explorer_holder=explorer_holder,
        )
        if error is not None:
            errors.append(error)
    return errors


def _anchor_schedules_on_body_explorer(
    representative_values: Mapping[str, Any],
    explorer_holder: list[Any],
) -> None:
    """Hand every top-level schedule the body's live explorer cell.

    A bare-literal ``look_up`` on such a schedule anchors on the current body's
    branch path via this shared, per-body-updated cell (see
    :meth:`_UnitCheckSchedule._produce`). A schedule plucked from a structured field
    is anchored where it is rebuilt per run instead (:func:`_wrap_for_unit_check`).
    """
    inference = _unit_inference_module()
    for value in representative_values.values():
        if isinstance(value, inference._UnitCheckSchedule):  # noqa: SLF001
            value.explorer_holder = explorer_holder
