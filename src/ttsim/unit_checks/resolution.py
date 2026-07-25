"""Declared TTSIM units resolved to pint units, node kind by node kind.

Turns each node's declared :class:`~ttsim.tt.units.CompositeUnit` — plus what
its *name* says (a time suffix, an aggregation suffix) — into the single pint
unit the rest of the check compares against. The rules differ by node kind:
parameters (scalar, per-leaf mapping, schedule axes), group aggregations (which
mint, swap, or preserve a grouping level), schedule-returning param functions
(whose environment-level unit is the output axis), and plain columns.

Type-level questions come from :mod:`ttsim.unit_checks.contracts`. The resolved
units feed :mod:`ttsim.unit_checks.declarations`, which compares them against
each other, and :mod:`ttsim.unit_checks.execution`, which turns them into the
representative quantities a body is run on.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Mapping
from typing import (
    Any,
    NamedTuple,
    cast,
)

import dags.tree as dt
import pint

from ttsim.exceptions import (
    UnitDefinitionError,
)
from ttsim.interface_dag_elements.shared import (
    FRAMEWORK_PARTIAL_ARGUMENTS,
    get_re_pattern_for_all_time_units_and_groupings,
)
from ttsim.tt.aggregation import AggType
from ttsim.tt.column_objects_param_function import (
    AggByGroupFunction,
    ColumnObject,
    ParamFunction,
    PolicyInput,
    qname_is_person_pointer,
)
from ttsim.tt.currencies import UnitSystem
from ttsim.tt.grouping_levels import register_grouping_levels
from ttsim.tt.param_objects import (
    ParamMappingObject,
    ParamObject,
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
    _QNAME_TIME_SUFFIX_PATTERN,
    UNSET_UNIT,
    CompositeUnit,
    InputOutputUnit,
    _grouping_levels_with_exponent,
    _unit_level_denominator,
    _unit_without_grouping_levels,
    head_count_from_boolean_sum,
    parse_unit,
    resolve_agnostic_ttsim_unit,
    resolve_ttsim_unit_for_column,
    resolve_ttsim_unit_for_param,
    resolved_unit_for_aggregation,
    ttsim_unit_has_agnostic_currency,
)
from ttsim.typing import (
    OrderedQNames,
    SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
)
from ttsim.unit_checks.contracts import (
    _returns_a_schedule,
    _spell_ttsim_unit,
)
from ttsim.unit_converters import TIME_UNIT_IDS_TO_LABELS

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
        # `parse_unit` guides declarations to the DIMENSIONLESS token, so the
        # framework-internal ordinal spelling resolves directly.
        qname: (
            _dimensionless_unit(registry)
            if unit == "dimensionless"
            else parse_unit(unit_str=unit, registry=registry)
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
                registry=registry,
                name_time_unit_id=name_time_unit_id,
            )
            if param_unit is not None:
                resolved[qname] = param_unit
        elif isinstance(obj, AggByGroupFunction):
            agg_unit = _resolve_agg_by_group_unit(
                qname=qname, obj=obj, env=env, pattern=pattern, registry=registry
            )
            if agg_unit is not None:
                resolved[qname] = agg_unit
        elif isinstance(obj, ParamFunction) and _returns_a_schedule(obj):
            token = getattr(obj, "unit", UNSET_UNIT)
            if isinstance(token, InputOutputUnit):
                # A schedule-returning param function declares its schedule's two
                # axes with `unit=InputOutputUnit(...)`; the environment-level
                # resolved unit is the OUTPUT axis (what `look_up` /
                # `piecewise_polynomial` consumers receive), resolved agnostically
                # like a field annotation — concrete currencies rejected, no
                # name-suffix rules.
                resolved[qname] = resolve_agnostic_ttsim_unit(
                    unit=token.output_unit,
                    registry=registry,
                    where=f"Schedule param function {qname!r}",
                    what="the declaration",
                )
        else:  # ColumnObject | scalar ParamFunction
            token = getattr(obj, "unit", UNSET_UNIT)
            if isinstance(token, CompositeUnit) and token is not UNSET_UNIT:
                leaf_name = dt.tree_path_from_qname(qname)[-1]
                match = pattern.fullmatch(leaf_name)
                resolved[qname] = _resolve_leveled_column_unit(
                    token=token,
                    match=match,
                    registry=registry,
                )
    return resolved


def _resolve_leveled_column_unit(
    token: CompositeUnit,
    match: re.Match[str] | None,
    registry: pint.UnitRegistry,
) -> pint.Unit:
    """Resolve a column/function's full unit, including its grouping level."""
    time_unit_id = match.group("time_unit") if match else None
    grouping_level = _suffix_grouping_level(match)
    return resolve_ttsim_unit_for_column(
        unit=token,
        time_unit_id=time_unit_id,
        grouping_level=grouping_level,
        where="A column/function",
        registry=registry,
    )


def _resolve_agg_by_group_unit(
    qname: str,
    obj: AggByGroupFunction,
    env: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
    pattern: re.Pattern[str],
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
      per-head average belongs to the individual (GEP 10);
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
            if token is UNSET_UNIT
            else _resolve_leveled_column_unit(
                token=cast("CompositeUnit", token),
                match=match,
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
    if source_token is UNSET_UNIT:
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
    source_match = pattern.fullmatch(dt.tree_path_from_qname(source_qname)[-1])
    source_unit = _resolve_leveled_column_unit(
        token=cast("CompositeUnit", source_token),
        match=source_match,
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
            registry=registry,
        )
    if obj.unit is UNSET_UNIT:
        return None
    if isinstance(obj.unit, Mapping):
        return _resolve_unit_mapping(
            qname=qname,
            unit_mapping=cast("Mapping[str | int, Any]", obj.unit),
            registry=registry,
        )
    token = cast("CompositeUnit", obj.unit)
    _fail_if_param_ttsim_unit_is_agnostic_currency(
        ttsim_unit=token, where=f"Parameter {qname!r}"
    )
    # A scalar parameter takes its period from a time suffix on its name; a
    # dict/raw parameter has no single name to suffix.
    return resolve_ttsim_unit_for_param(
        unit=token,
        registry=registry,
        time_unit_id=name_time_unit_id if isinstance(obj, ScalarParam) else None,
        where=f"Parameter {qname!r}",
    )


def _resolve_param_mapping_object_units(
    qname: str,
    obj: ParamMappingObject,
    name_time_unit_id: str | None,
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
    tokens: dict[str, CompositeUnit] = {}
    for axis, raw in (("input_unit", obj.input_unit), ("output_unit", obj.output_unit)):
        if raw is UNSET_UNIT:
            tokens[axis] = UNSET_UNIT
            continue
        where = f"Parameter {qname!r}, {axis}"
        if isinstance(raw, Mapping):
            raise UnitDefinitionError(
                f"{where}: per-axis declarations are single units, not "
                f"mappings (GEP 10)."
            )
        token = cast("CompositeUnit", raw)
        _fail_if_param_ttsim_unit_is_agnostic_currency(ttsim_unit=token, where=where)
        tokens[axis] = token
    output_token = tokens["output_unit"]
    if name_time_unit_id is not None:
        _fail_if_name_suffix_disagrees_with_output_axis(
            qname=qname,
            output_token=output_token,
            name_time_unit_id=name_time_unit_id,
            registry=registry,
        )
    input_token = tokens["input_unit"]
    if input_token is not UNSET_UNIT:
        resolve_ttsim_unit_for_param(
            unit=input_token, registry=registry, where=f"Parameter {qname!r}"
        )
    if output_token is UNSET_UNIT:
        return None
    return resolve_ttsim_unit_for_param(
        unit=output_token, registry=registry, where=f"Parameter {qname!r}"
    )


def _fail_if_name_suffix_disagrees_with_output_axis(
    qname: str,
    output_token: Any,  # noqa: ANN401
    name_time_unit_id: str,
    registry: pint.UnitRegistry,
) -> None:
    """Check the name-suffix ⟺ flow-output coincidence rules.

    A name time suffix denotes a flow, so the ``output_unit`` must be a flow and
    its spelled period must agree with the suffix (validated by resolving the
    output unit against the suffix).
    """
    if output_token is UNSET_UNIT or not output_token.is_flow:
        raise UnitDefinitionError(
            f"Parameter {qname!r}: the name carries a time-unit suffix "
            f"(_{name_time_unit_id}), which denotes a flow, but "
            f"`output_unit:` is {_spell_ttsim_unit(output_token)} (GEP 10)."
        )
    resolve_ttsim_unit_for_param(
        unit=output_token,
        registry=registry,
        time_unit_id=name_time_unit_id,
        where=f"Parameter {qname!r}",
    )


def _resolve_unit_mapping(
    qname: str,
    unit_mapping: Mapping[str | int, Any],
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
                qname=qname, unit_mapping=token, registry=registry
            )
            continue
        where = f"Parameter {qname!r}, unit of leaf {key!r}"
        _fail_if_param_ttsim_unit_is_agnostic_currency(ttsim_unit=token, where=where)
        match = _QNAME_TIME_SUFFIX_PATTERN.search(str(key))
        suffix_id = match.group("time_unit") if match else None
        resolved[key] = resolve_ttsim_unit_for_param(
            unit=cast("CompositeUnit", token),
            registry=registry,
            time_unit_id=suffix_id,
            where=where,
        )
    return resolved


def _fail_if_param_ttsim_unit_is_agnostic_currency(
    ttsim_unit: CompositeUnit | None,
    where: str,
) -> None:
    """Reject a parameter whose declared TTSIM unit is the agnostic currency."""
    if ttsim_unit_has_agnostic_currency(ttsim_unit):
        suffixes = str(ttsim_unit).removeprefix("CURRENCY")
        raise UnitDefinitionError(
            f"{where}: parameters must pin down the concrete currency their "
            f"numbers are written in; the agnostic unit {ttsim_unit} is not "
            f"allowed here. Declare the statutory currency at the parameter's "
            f"dates, e.g. DM{suffixes} or EUR{suffixes} (GEP 10)."
        )


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
    from a leveled boolean here — both are the plain number over their group
    (GEP 10).

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
    is **bare** (``None``, GEP 10).

    Two fam-level indicators give ``"fam"``; the mixed
    ``wealth_fam >= threshold_fam or wealth_kin >= threshold_kin`` combines a fam-
    and a kin-level operand, so the result is bare (``None``).
    """
    return left if left == right else None


def _dimensionless_unit(registry: pint.UnitRegistry) -> pint.Unit:
    """The dimensionless unit, used when reporting a logical op's bare operand."""
    return registry.dimensionless
