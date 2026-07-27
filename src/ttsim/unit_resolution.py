"""Resolve every node's declared unit into a `pint` unit.

The declaration side of GEP 10: what a `unit=` / `unit:` / field annotation
*means*, with no reference to what any body computes. `unit_validation` checks
declarations against each other, `_unit_inference` checks them against bodies;
both start here.
"""

from __future__ import annotations

import dataclasses
import inspect
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

import dags.tree as dt
import pint
from dags import get_annotations

from ttsim.exceptions import UnitDefinitionError
from ttsim.interface_dag_elements.shared import (
    FRAMEWORK_PARTIAL_ARGUMENTS,
    get_re_pattern_for_all_time_units_and_groupings,
)
from ttsim.time_converters import TIME_UNIT_IDS_TO_LABELS
from ttsim.tt.aggregation import AggType
from ttsim.tt.column_objects_param_function import (
    AggByGroupFunction,
    ColumnObject,
    ParamFunction,
    PolicyInput,
    qname_is_person_pointer,
)
from ttsim.tt.param_objects import (
    ConsecutiveIntLookupTableParamValue,
    ParamMappingObject,
    ParamObject,
    PiecewisePolynomialParamValue,
    ScalarParam,
)
from ttsim.tt.type_resolution import (
    BOOL_KINDS,
    TypeResolutionError,
    resolve_kind_of_annotation,
    resolve_kind_of_column_function,
)
from ttsim.tt.units import (
    UNSET_UNIT,
    CompositeUnit,
    InputOutputUnits,
    UnitDeclaration,
    UnitSystem,
    UnsetUnit,
    head_count_from_boolean_sum,
    pint_unit_from_string,
    pint_unit_from_ttsim_unit_for_column,
    pint_unit_from_ttsim_unit_for_param,
    register_grouping_levels,
    resolved_unit_for_aggregation,
    ttsim_unit_has_currency,
)
from ttsim.typing import (
    OrderedQNames,
    SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
)


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


def _schedule_axis_errors(
    builds_lookup_table: bool,
    input_unit: CompositeUnit | tuple[CompositeUnit, ...],
    where: str,
) -> list[str]:
    """Type-specific `input_unit` errors for a schedule, whatever declares it.

    The same two rules bind a schedule-returning `@param_function`'s `unit=` and a
    schedule-typed parameter-dataclass field's `Annotated[...]` marker, so both
    sites ask here:

    - a piecewise polynomial takes a single domain argument, so a tuple
      `input_unit` (positional axes) belongs to a multi-dimensional lookup table
      only;
    - a lookup table is keyed by consecutive integers, so no input axis is ever a
      currency — the integer keys are never rescaled between currencies.

    The caller decides what to do with the messages: a declaration mismatch is
    collected into the environment-wide consistency report, a malformed field
    annotation is raised on the spot.
    """
    axes = (
        input_unit
        if isinstance(input_unit, tuple)
        else (cast("CompositeUnit", input_unit),)
    )
    errors: list[str] = []
    if not builds_lookup_table and isinstance(input_unit, tuple):
        errors.append(
            f"{where}: declares a tuple `input_unit` but is a piecewise polynomial, "
            f"which takes a single domain argument; a tuple of positional axes is "
            f"only for a multi-dimensional lookup table (GEP 10)."
        )
    if builds_lookup_table and any(
        ttsim_unit_has_currency(cast("CompositeUnit", axis)) for axis in axes
    ):
        errors.append(
            f"{where}: is a lookup table keyed by consecutive integers, so no "
            f"`input_unit` axis may be a currency (got `input_unit={input_unit}`); "
            f"the integer keys are never rescaled between currencies (GEP 10)."
        )
    return errors


FRAMEWORK_DATE_NODE_UNITS: Mapping[str, str] = {
    "policy_year": "calendar_year",
    "policy_month": "dimensionless",
    "policy_day": "dimensionless",
    "evaluation_year": "calendar_year",
    "evaluation_month": "dimensionless",
    "evaluation_day": "dimensionless",
}


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
    axis_errors = _schedule_axis_errors(
        builds_lookup_table=issubclass(base, ConsecutiveIntLookupTableParamValue),
        input_unit=io_token.input_unit,
        where=where,
    )
    if axis_errors:
        raise UnitDefinitionError(axis_errors[0])
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
    """The class's type hints, dropping only those that do not resolve."""
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


def _suffix_grouping_level(match: re.Match[str] | None) -> str | None:
    """The grouping level named by a name's aggregation suffix."""
    if match is None:
        return None
    return match.group("grouping") or None


def _composite_token_level(token: CompositeUnit) -> str | None:
    """The grouping level a TTSIM unit spells, or ``None`` if bare."""
    return token.level.lower() if token.level is not None else None


def _dimensionless_unit(registry: pint.UnitRegistry) -> pint.Unit:
    """The dimensionless unit, used when reporting a logical op's bare operand."""
    return registry.dimensionless


def _spell_ttsim_unit(ttsim_unit: Any) -> str:  # noqa: ANN401
    """Spell a declared TTSIM unit for an error message."""
    if isinstance(ttsim_unit, UnsetUnit):
        return "unset"
    return str(ttsim_unit)
