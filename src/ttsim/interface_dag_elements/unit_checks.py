"""Whole-environment unit checks.

Builds on the per-function engine in :mod:`ttsim.tt.units` to verify a fully
assembled policy environment on two counts:

- **Every active node declares a TTSIM unit.** Author declarations (``unit=`` /
  ``unit:``) cover inputs, functions, parameters, and param functions; derived
  nodes (aggregations, time-conversion variants, group ids) and the framework
  date nodes get theirs assigned here.
- **Each function body agrees with its declaration.** Every ``@policy_function``
  / ``@param_function`` body is unit-checked on representative quantities built
  from its producers' resolved units. A body the check cannot evaluate must opt
  out with ``verify_units=False``.

The two layers of :mod:`ttsim.tt.units` both appear here: a node's declared
*TTSIM unit* (:class:`~ttsim.tt.units.CompositeUnit`) is resolved once against the
environment's registry, and every check downstream compares the resulting *pint
units*.

Checking a body is an abstract interpretation over pint units: the real body runs,
but on stand-in values carrying a pint unit and no meaningful magnitude. Each
``Quantity(1.0, unit)`` is wrapped in a :class:`_UnitCheckQuantity`, whose
arithmetic propagates units while a :class:`_PathExplorer` drives its branch
decisions — so a body is checked down every reachable path.

The unit check always runs in NumPy, regardless of the backend of the actual run.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import re
import sys
from collections.abc import Callable, Mapping
from typing import (
    Any,
    NamedTuple,
    NoReturn,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

import dags.tree as dt
import numpy
import pint
from dags import get_annotations

from ttsim.exceptions import (
    TTSIMError,
    UnitConsistencyError,
    UnitDefinitionError,
)
from ttsim.interface_dag_elements.interface_node_objects import interface_function
from ttsim.interface_dag_elements.shared import (
    FRAMEWORK_PARTIAL_ARGUMENTS,
    get_re_pattern_for_all_time_units_and_groupings,
)
from ttsim.tt._source_rewriting import recompile_with_logical_ops_as_calls
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
from ttsim.tt.currencies import UnitSystem
from ttsim.tt.grouping_levels import register_grouping_levels
from ttsim.tt.param_objects import (
    ConsecutiveIntLookupTableParamValue,
    DictParam,
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
    _QNAME_TIME_SUFFIX_PATTERN,
    TIME_UNIT_ID_TO_PINT_NAME,
    UNSET_UNIT,
    CompositeUnit,
    InputOutputUnit,
    UnitAnnotatedColumn,
    UserNestedUnitAnnotatedData,
    _grouping_levels_with_exponent,
    _unit_level_denominator,
    _unit_without_grouping_levels,
    fail_if_units_are_missing,
    head_count_from_boolean_sum,
    is_calendar_point_unit,
    parse_unit,
    pint_unit_has_currency,
    resolve_agnostic_ttsim_unit,
    resolve_ttsim_unit,
    resolve_ttsim_unit_for_column,
    resolve_ttsim_unit_for_param,
    resolved_unit_for_aggregation,
    ttsim_unit_currency,
    ttsim_unit_from_yaml_value,
    ttsim_unit_has_agnostic_currency,
    ttsim_unit_has_currency,
    ttsim_unit_with_agnostic_currency,
    unit_residual_excluding_currency_and_flow_period,
    units_are_equivalent,
)
from ttsim.typing import (
    OrderedQNames,
    SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
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


class _UnitCheckXnp:
    """The unit check's ``xnp``: NumPy with the unit-bearing ops routed through
    ``_UnitCheckQuantity``'s checks, so a vectorized (``not_required``) body is checked
    at full parity with a scalar one.

    An op not modelled here falls through to raw NumPy, raises, and is reported as
    needing ``verify_units=False`` — never silently passed through.
    """

    @staticmethod
    def logical_and(left: Any, right: Any) -> Any:  # noqa: ANN401
        return left & right

    @staticmethod
    def logical_or(left: Any, right: Any) -> Any:  # noqa: ANN401
        return left | right

    @staticmethod
    def logical_not(value: Any) -> Any:  # noqa: ANN401
        return ~value

    @staticmethod
    def maximum(left: Any, right: Any) -> Any:  # noqa: ANN401
        return _clamping_op(left=left, right=right, op="maximum")

    @staticmethod
    def minimum(left: Any, right: Any) -> Any:  # noqa: ANN401
        return _clamping_op(left=left, right=right, op="minimum")

    @staticmethod
    def where(condition: Any, x: Any, y: Any) -> Any:  # noqa: ANN401, ARG004
        return _where_op(x=x, y=y)

    @staticmethod
    def clip(value: Any, a_min: Any, a_max: Any) -> Any:  # noqa: ANN401
        return _clip_op(value=value, a_min=a_min, a_max=a_max)

    @staticmethod
    def sum(value: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401, ARG004
        return _unit_preserving_op(value)

    @staticmethod
    def amin(value: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401, ARG004
        return _unit_preserving_op(value)

    @staticmethod
    def amax(value: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401, ARG004
        return _unit_preserving_op(value)

    @staticmethod
    def floor(value: Any) -> Any:  # noqa: ANN401
        return _unit_preserving_op(value)

    @staticmethod
    def ceil(value: Any) -> Any:  # noqa: ANN401
        return _unit_preserving_op(value)

    @staticmethod
    def round(value: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401, ARG004
        return _unit_preserving_op(value)

    @staticmethod
    def abs(value: Any) -> Any:  # noqa: ANN401
        return _unit_preserving_op(value)

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        return getattr(numpy, name)


#: Representative values for the framework arguments in a unit check.
_NON_UNIT_ARGUMENT_VALUES: Mapping[str, Any] = {
    "xnp": _UnitCheckXnp(),
    "dnp": numpy,
    "backend": "numpy",
    "num_segments": 1,
    "len_p_id": 1,
}

#: The logical operators the unit check screens for boolean (dimensionless) operands.
_LOGICAL_OPS = frozenset({"&", "|", "^", "~"})


def _dimensionless_unit(registry: pint.UnitRegistry) -> pint.Unit:
    """The dimensionless unit, used when reporting a logical op's bare operand."""
    return registry.dimensionless


@interface_function()
def resolved_pint_units(
    specialized_environment__without_tree_logic_and_with_derived_functions: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,  # noqa: E501
    labels__grouping_levels: OrderedQNames,
    unit_system: UnitSystem,
) -> dict[str, pint.Unit | dict[str | int, Any]]:
    """The pint unit every annotated node's declared TTSIM unit resolves to."""
    return resolve_environment_units(
        env=specialized_environment__without_tree_logic_and_with_derived_functions,
        grouping_levels=labels__grouping_levels,
        unit_system=unit_system,
    )


@interface_function()
def declared_ttsim_units(
    specialized_environment__without_tree_logic_and_with_derived_functions: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,  # noqa: E501
) -> dict[str, CompositeUnit]:
    """Each node's declared TTSIM unit, by qname."""
    env = specialized_environment__without_tree_logic_and_with_derived_functions
    return {
        qname: token
        for qname, obj in env.items()
        if isinstance((token := getattr(obj, "unit", UNSET_UNIT)), CompositeUnit)
        and token is not UNSET_UNIT
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


def _returns_a_schedule(obj: ParamFunction) -> bool:
    """Whether a param function is annotated as returning a schedule/lookup value."""
    return _return_annotation_name(obj.function) in _SCHEDULE_RETURN_TYPE_NAMES


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


def _suffix_grouping_level(match: re.Match[str] | None) -> str | None:
    """The grouping level named by a name's aggregation suffix."""
    if match is None:
        return None
    return match.group("grouping") or None


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


def fail_if_environment_units_are_missing(
    env: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
    grouping_levels: OrderedQNames,  # noqa: ARG001  (kept for symmetry of the two checks)
) -> None:
    """Mandatory-units check over a fully assembled environment.

    Every active node must declare a unit:

    - a dict or require_converter parameter with per-leaf units must cover every
      leaf of the value active at the policy date;
    - a ``@param_function`` declaring ``unit=UNSET_UNIT`` (a structured value) or
      ``unit=InputOutputUnit(...)`` (a schedule builder) is exempt — its output is
      not a single quantity, and its units live in the field annotations or the
      declaration itself, so neither is an omission (GEP 10);
    - a rounding spec on a currency-valued function must declare its own unit (its
      magnitudes are statutory numbers in a concrete currency, like a parameter's
      — GEP 10); a missing one is reported as ``<qname> (rounding_spec)``.

    Raises:
        UnitDefinitionError: If any node (or per-leaf-mapping leaf) lacks a unit
            declaration.
    """
    units_by_qname: dict[str, CompositeUnit] = {}
    for qname, obj in env.items():
        if not isinstance(obj, ColumnObject | ParamFunction | ParamObject):
            continue
        if qname in FRAMEWORK_DATE_NODE_UNITS:
            continue
        declared_unit = getattr(obj, "unit", UNSET_UNIT)
        if isinstance(obj, ParamFunction) and (
            declared_unit is UNSET_UNIT or isinstance(declared_unit, InputOutputUnit)
        ):
            # A structured value (`UNSET_UNIT`) states its units in the return
            # type's field annotations; a schedule builder (`InputOutputUnit`)
            # states them in the declaration itself.
            continue
        if isinstance(obj, ParamMappingObject | RawParam) and (
            isinstance(obj, ParamMappingObject)
            or obj.input_unit is not UNSET_UNIT
            or obj.output_unit is not UNSET_UNIT
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
    if derived is None or isinstance(derived, dict) or declared_token is UNSET_UNIT:
        return None
    declared_unit = resolve_ttsim_unit_for_param(
        unit=cast("CompositeUnit", declared_token),
        registry=registry,
        where=f"Aggregation {qname!r}",
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


def _rounding_spec_declaration_inconsistency(
    qname: str,
    obj: ColumnFunction,
) -> str | None:
    """Error message if a rounding spec's unit disagrees with its function's.

    A rounding spec's magnitudes are statutory numbers in a concrete currency,
    exactly like a parameter's (GEP 10), so:

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
    if spec is None or spec.unit is None or declared is UNSET_UNIT:
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
            derives, or a schedule builder breaks the ``InputOutputUnit`` contract
            (declaration vs. return annotation vs. ``verify_units``). All offending
            nodes are reported together.
        UnitDefinitionError: If an ``InputOutputUnit`` axis or a
            parameter-dataclass field annotation is invalid.
    """
    registry = unit_system.registry
    _fail_if_structured_field_annotations_are_invalid(env=env, unit_system=unit_system)
    if resolved_pint_units is None:
        resolved_pint_units = resolve_environment_units(
            env=env, grouping_levels=grouping_levels, unit_system=unit_system
        )
    representative_values = _representative_values_by_qname(
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


def _anchor_schedules_on_body_explorer(
    representative_values: Mapping[str, Any],
    explorer_holder: list[_PathExplorer | None],
) -> None:
    """Hand every top-level schedule the body's live explorer cell.

    A bare-literal ``look_up`` on such a schedule anchors on the current body's
    branch path via this shared, per-body-updated cell (see
    :meth:`_UnitCheckSchedule._produce`). A schedule plucked from a structured field
    is anchored where it is rebuilt per run instead (:func:`_wrap_for_unit_check`).
    """
    for value in representative_values.values():
        if isinstance(value, _UnitCheckSchedule):
            value.explorer_holder = explorer_holder


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
    registry = unit_system.registry
    # The helper stand-ins close over this run's registry, so they are built per run
    # rather than once at import.
    unit_check_helper_stand_ins, explorer_holder = _unit_check_helper_stand_ins(
        unit_system=unit_system
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
        base_kwargs = _base_unit_check_kwargs(
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
        error = _verify_one_body(
            qname=qname,
            function=recompile_with_logical_ops_as_calls(
                func=obj.function,
                module="xnp",
                module_obj=_NON_UNIT_ARGUMENT_VALUES["xnp"],
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


def _composite_token_level(token: CompositeUnit) -> str | None:
    """The grouping level a TTSIM unit spells, or ``None`` if bare."""
    return token.level.lower() if token.level is not None else None


def fail_if_input_units_are_inconsistent(
    input_unit_tokens: Mapping[str, CompositeUnit],
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
      off the suffix — GEP 10);
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
    for qname, tag_token in input_unit_tokens.items():
        tag = resolve_ttsim_unit(unit=tag_token, registry=registry, with_level=True)
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
        tag_residual = unit_residual_excluding_currency_and_flow_period(
            units=tag, registry=registry
        )
        expected_residual = unit_residual_excluding_currency_and_flow_period(
            units=expected, registry=registry
        )
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


def _spell_ttsim_unit(ttsim_unit: Any) -> str:  # noqa: ANN401
    """Spell a declared TTSIM unit for an error message."""
    if ttsim_unit is UNSET_UNIT:
        return "unset"
    return str(ttsim_unit)


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


def _representative_value(
    resolved_unit: pint.Unit | dict[str | int, Any],
    registry: pint.UnitRegistry,
) -> Any:  # noqa: ANN401
    """A representative unit-check value: ``Quantity(1.0, unit)``, or a dict thereof."""
    if isinstance(resolved_unit, dict):
        return {
            key: _representative_value(
                resolved_unit=cast("pint.Unit | dict[str | int, Any]", unit),
                registry=registry,
            )
            for key, unit in resolved_unit.items()
        }
    return registry.Quantity(1.0, resolved_unit)


def _uniform_quantity_tree(
    value: Any,  # noqa: ANN401
    resolved_unit: pint.Unit,
    registry: pint.UnitRegistry,
) -> Any:  # noqa: ANN401
    """Mirror a dict param's value structure with uniform representative quantities."""
    if isinstance(value, Mapping):
        return {
            key: _uniform_quantity_tree(
                value=sub_value, resolved_unit=resolved_unit, registry=registry
            )
            for key, sub_value in value.items()
        }
    return registry.Quantity(1.0, resolved_unit)


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


def _dataclass_or_none(obj: Any) -> type | None:  # noqa: ANN401
    """``obj`` if it is a dataclass type, else ``None``."""
    return obj if isinstance(obj, type) and dataclasses.is_dataclass(obj) else None


def _resolve_dotted_dataclass(name: str, func: Any) -> type | None:  # noqa: ANN401
    """Resolve a (possibly dotted) dataclass name through the function's module."""
    obj: Any = sys.modules.get(getattr(func, "__module__", ""))
    for part in name.split("."):
        obj = getattr(obj, part, None)
    return _dataclass_or_none(obj)


def _mapping_value_dataclass(annotation: Any) -> type | None:  # noqa: ANN401
    """The value dataclass of a resolved ``Mapping[..., <dataclass>]``, or ``None``."""
    origin = get_origin(annotation)
    if not (isinstance(origin, type) and issubclass(origin, Mapping)):
        return None
    args = get_args(annotation)
    return _dataclass_or_none(args[-1]) if args else None


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


def _param_function_stand_in(
    qname: str,
    obj: ParamFunction,
    unit_system: UnitSystem,
) -> _UnitCheckStructuredValue:
    """The unit check's stand-in for a structured param-function output (GEP 10).

    A ``@param_function(unit=UNSET_UNIT)`` builds a dataclass of related
    parameters, so its stand-in is a :class:`_UnitCheckStructuredValue` typed with
    the return dataclass where one resolves — annotated plucks then carry their
    field units, and an unannotated pluck stays opaque and is cast at the site.
    A schedule builder (``unit=InputOutputUnit(...)``) is handled separately as a
    :class:`_UnitCheckSchedule` in :func:`_representative_values_by_qname`.
    """
    cls, item_cls = _resolved_return_structure(obj.function)
    return _UnitCheckStructuredValue(
        producer=qname,
        unit_system=unit_system,
        cls=cls,
        item_cls=item_cls,
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


def _representative_values_by_qname(
    env: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
    resolved_pint_units: Mapping[str, pint.Unit | dict[str | int, Any]],
    unit_system: UnitSystem,
) -> dict[str, Any]:
    """Representative unit-check values for every unit-resolved node.

    - A ``piecewise_*``/lookup-table parameter becomes a :class:`_UnitCheckSchedule`
      carrying its input/output axes, so a consumer's ``piecewise_polynomial`` /
      ``look_up`` call resolves to the output unit.
    - A schedule-building param function (``unit=InputOutputUnit(...)``) becomes a
      :class:`_UnitCheckSchedule` carrying the declared axes, so its consumers
      screen and resolve exactly as for a schedule parameter.
    - A dict parameter with a scalar ``unit:`` declaration becomes a dict of
      uniform representative quantities mirroring its value structure, so
      subscripting works inside a consumer's unit check.
    - A structured param function (``unit=UNSET_UNIT``) becomes its
      :func:`_param_function_stand_in` — a :class:`_UnitCheckStructuredValue`.
    """
    registry = unit_system.registry
    out: dict[str, Any] = {}
    for qname, unit in resolved_pint_units.items():
        obj = env.get(qname)
        if isinstance(obj, ParamMappingObject) and not isinstance(unit, dict):
            out[qname] = _UnitCheckSchedule(
                input_unit=_resolve_schedule_input_unit(obj=obj, registry=registry),
                output_unit=cast("pint.Unit", unit),
                unit_system=unit_system,
            )
        elif (
            isinstance(obj, ParamFunction)
            and _returns_a_schedule(obj)
            and isinstance(obj.unit, InputOutputUnit)
            and not isinstance(unit, dict)
        ):
            # A schedule builder declares its axes with `unit=InputOutputUnit(...)`:
            # `look_up`/`piecewise_polynomial` screens each domain argument against
            # the input axis (a single axis applied to every argument, or a tuple
            # screened positionally) and yields the output.
            out[qname] = _UnitCheckSchedule(
                input_unit=_resolve_input_axes(
                    input_unit=obj.unit.input_unit,
                    registry=registry,
                    where=f"Schedule param function {qname!r}",
                ),
                output_unit=cast("pint.Unit", unit),
                unit_system=unit_system,
            )
        elif isinstance(obj, DictParam | RawParam) and not isinstance(unit, dict):
            out[qname] = _uniform_quantity_tree(
                value=obj.value,
                resolved_unit=cast("pint.Unit", unit),
                registry=registry,
            )
        else:
            out[qname] = _representative_value(resolved_unit=unit, registry=registry)
    for qname, obj in env.items():
        if isinstance(obj, ParamFunction) and obj.unit is UNSET_UNIT:
            out[qname] = _param_function_stand_in(
                qname=qname, obj=obj, unit_system=unit_system
            )
    return out


def _base_unit_check_kwargs(
    parameters: tuple[str, ...],
    boolean_parameters: tuple[str, ...],
    representative_values: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Representative kwargs for a body's non-boolean parameters.

    Returns ``None`` if any parameter has no representative value (an
    unannotated producer): the body cannot be evaluated and its declared
    unit is the fallback.
    """
    out: dict[str, Any] = {}
    for parameter in parameters:
        if parameter in boolean_parameters:
            continue
        if parameter in FRAMEWORK_PARTIAL_ARGUMENTS:
            out[parameter] = _NON_UNIT_ARGUMENT_VALUES[parameter]
        elif parameter in representative_values:
            out[parameter] = representative_values[parameter]
        else:
            return None
    return out


# Caps on the path-exploring unit check (see ``_PathExplorer``): only a pathological
# body (deep independent branching, or a data-driven loop) hits them, so the build
# check can never blow up.
_MAX_PATHS = 1024
_MAX_DECISIONS_PER_RUN = 64

#: How many of a failing run's branch decisions an error message spells out.
_MAX_NAMED_DECISIONS = 4


class _PathBudgetExceededError(TTSIMError):
    """A single unit-check run made too many branch decisions (likely a loop)."""


class _UnitMixError(TTSIMError):
    """A body combined two non-equivalent unit-carrying operands.

    ``+``, ``-`` and the ordering comparisons are *unit-blind at run time*:
    once the policy environment is built there is no pint, so a body that does
    ``monthly_flow + yearly_flow`` (or compares them) just adds or compares the
    bare arrays — no conversion happens. The two operands must therefore already
    be in equivalent units (same dimension *and* period). pint would mask such a
    bug during the unit check — silently auto-converting a same-dimension mismatch
    (``CURRENCY / month + CURRENCY / year``) or raising a swallowed
    ``DimensionalityError`` on a cross-dimension one — so ``_UnitCheckQuantity`` checks
    before delegating instead.
    """

    def __init__(
        self,
        op: str,
        left: pint.Unit,
        right: pint.Unit,
        literal: Any = None,  # noqa: ANN401
    ) -> None:
        super().__init__()
        self.op = op
        self.left = left
        self.right = right
        # The bare numeric literal an ordering comparison was made against, when
        # that is the offence; ``None`` for the unit-vs-unit mismatch cases.
        self.literal = literal


class _PathExplorer:
    """Drives a body down every reachable branch path across re-runs.

    Concolic-style depth-first search: each run forces a fixed prefix of branch
    outcomes, takes the ``False`` arm at the frontier, and records the outcomes
    actually taken. After a run, the last ``False`` outcome is flipped to
    ``True`` and the suffix dropped, so successive runs walk the whole path
    tree. The number of runs equals the number of *reachable* paths — not
    ``2**(branches)`` — because only branches actually executed become decisions
    (an unreached branch never asks). A boolean input is just another decision,
    and numeric-driven branches (``if income > limit``) are reached too — a
    single representative value would silently fix them to one arm.
    """

    def __init__(self) -> None:
        self._prefix: list[bool] = []
        self._trail: list[bool] = []
        self._labels: list[str | None] = []
        self._index = 0

    def start_run(self) -> None:
        self._index = 0
        self._trail = []
        self._labels = []

    def decide(self, label: str | None = None) -> bool:
        """Resolve the next branch: replay the prefix, then explore ``False``.

        ``label`` names the condition consulted — an argument tested directly, a
        comparison of named operands — so a failure can report the branch in the
        body's own terms (:meth:`branch_detail`); ``None`` where the unit check has
        no name for it.
        """
        if self._index >= _MAX_DECISIONS_PER_RUN:
            raise _PathBudgetExceededError
        value = self._prefix[self._index] if self._index < len(self._prefix) else False
        self._index += 1
        self._trail.append(value)
        self._labels.append(label)
        return value

    def advance(self) -> bool:
        """Queue the next path; return ``False`` once the tree is exhausted."""
        for k in range(len(self._trail) - 1, -1, -1):
            if not self._trail[k]:
                self._prefix = [*self._trail[:k], True]
                return True
        return False

    def branch_detail(self) -> str:
        """The current run's branch combination, phrased for an error message.

        Empty when the run made no branch decision. Decisions are named where a
        label was recorded and fall back to their ordinal otherwise; long trails
        are truncated after :data:`_MAX_NAMED_DECISIONS` decisions.
        """
        if not self._trail:
            return ""
        parts = [
            f"`{label}` is {value}"
            if label is not None
            else f"branch decision {position} is {value}"
            for position, (label, value) in enumerate(
                zip(self._labels, self._trail, strict=True), start=1
            )
        ]
        if len(parts) > _MAX_NAMED_DECISIONS:
            dropped = len(parts) - _MAX_NAMED_DECISIONS
            parts = [*parts[:_MAX_NAMED_DECISIONS], f"{dropped} more decision(s)"]
        return " on the branch where " + " and ".join(parts)


def _unwrap(value: Any) -> Any:  # noqa: ANN401
    return value.q if isinstance(value, _UnitCheckQuantity) else value


class _UnitCheckQuantity:
    """A pint ``Quantity`` wrapped so branch decisions route to a ``_PathExplorer``.

    Arithmetic forwards to the wrapped quantity, so units propagate exactly as
    in a real run (the whole point of the check). Comparisons and truth tests
    instead return an explorer-controlled value, so the explorer — not the
    representative magnitude — decides which branch is taken; the magnitude is
    always ``1.0`` and never matters. Anything the wrapper cannot model raises,
    which the caller treats as "not evaluable on this path" and falls back to
    the declaration — so the wrapper can never produce a false positive.
    """

    __slots__ = ("_explorer", "_label", "_unit_system", "q")
    # Keep NumPy from broadcasting over us: defer binary ops with a NumPy operand
    # to our reflected dunders instead.
    __array_ufunc__ = None
    __array_priority__ = 1000
    __hash__ = object.__hash__

    def __init__(
        self,
        q: Any,  # noqa: ANN401
        explorer: _PathExplorer,
        unit_system: UnitSystem,
        label: str | None = None,
    ) -> None:
        self.q = q
        self._explorer = explorer
        # The system whose registry `q` lives in — every unit the wrapper mints
        # (a boolean's level, a dimensionless truth value) must land there too.
        self._unit_system = unit_system
        # How the body's author would name this value — the argument name for a
        # direct input, a composed description for a comparison or logical
        # combination, ``None`` once arithmetic has mixed it beyond naming. Used
        # to report the branch a failure sits on (`_PathExplorer.branch_detail`).
        self._label = label

    @property
    def _registry(self) -> pint.UnitRegistry:
        return self._unit_system.registry

    def _wrap(self, q: Any) -> _UnitCheckQuantity:  # noqa: ANN401
        return _UnitCheckQuantity(
            q=q, explorer=self._explorer, unit_system=self._unit_system
        )

    def _controlled_bool_at(
        self, level: str | None, label: str | None = None
    ) -> _UnitCheckQuantity:
        return _UnitCheckQuantity(
            q=_boolean_quantity(level=level, registry=self._registry),
            explorer=self._explorer,
            unit_system=self._unit_system,
            label=label,
        )

    def _composed_label(self, other: Any, op: str) -> str | None:  # noqa: ANN401
        """Describe ``self <op> other`` for branch naming, if either side has
        a name; a bare literal operand shows as itself."""
        if isinstance(other, _UnitCheckQuantity):
            right = other._label  # noqa: SLF001
        elif isinstance(other, int | float | numpy.number | numpy.bool_):
            right = repr(other)
        else:
            right = None
        if self._label is None and right is None:
            return None
        return f"{self._label or '…'} {op} {right or '…'}"

    def _comparison_level(self, other: Any) -> str | None:  # noqa: ANN401
        """The grouping level a comparison result carries.

        A comparison of a leveled quantity yields a boolean at that level
        (``einkommen_m_bg > schwelle`` is a bg-level indicator). The level is
        read off ``self``, falling back to the other operand — for an ordering
        comparison the two are equivalent, so they agree. A comparison of two
        bare quantities is evaluated per person and therefore yields a bare,
        individual boolean (``None``, GEP 10).
        """
        level = _unit_level_denominator(cast("pint.Unit", self.q.units))
        if level is not None:
            return level
        other_q = _unwrap(other)
        if isinstance(other_q, pint.Quantity):
            other_level = _unit_level_denominator(cast("pint.Unit", other_q.units))
            if other_level is not None:
                return other_level
        return None

    def _logical_result(self, other: Any, op: str) -> _UnitCheckQuantity:  # noqa: ANN401
        """Combine two booleans under a logical operator ``&``/``|``/``^``.

        Logical operators combine truth values. Each operand must be a (possibly
        leveled) boolean — a non-dimensionless operand carrying physical content
        (``wealth & is_adult``) or a head count is a mistake the run-time arrays
        would silently swallow. The result is a boolean whose level follows the
        combine rule (:func:`_combined_boolean_level`): equal levels are kept, a
        mismatch downcasts to the bare individual level. A bare literal carries no
        unit and stays a lenient, bare boolean.
        """
        self_is_boolean, self_level = _as_boolean_level(
            unit=cast("pint.Unit", self.q.units), registry=self._registry
        )
        other_q = _unwrap(other)
        if isinstance(other_q, _UnitCheckStructuredValue):
            other_q._raise_used_as_quantity(op)  # noqa: SLF001
        if isinstance(other_q, pint.Quantity):
            other_is_boolean, other_level = _as_boolean_level(
                unit=cast("pint.Unit", other_q.units), registry=self._registry
            )
        else:
            other_is_boolean, other_level = True, None
        if not self_is_boolean or not other_is_boolean:
            right = (
                cast("pint.Unit", other_q.units)
                if isinstance(other_q, pint.Quantity)
                else _dimensionless_unit(self._registry)
            )
            raise _UnitMixError(
                op=op, left=cast("pint.Unit", self.q.units), right=right
            )
        return self._controlled_bool_at(
            level=_combined_boolean_level(left=self_level, right=other_level),
            label=self._composed_label(other=other, op=op),
        )

    def _fail_if_additive_operand_is_invalid(self, other: Any, op: str) -> None:  # noqa: ANN401
        """Screen an operand of ``+``/``-``.

        The rules are those of :meth:`_fail_if_other_unit_is_not_equivalent`,
        with one dispensation: a calendar point (an affine offset unit). Its
        valid ``point +/- duration`` is *not* equivalence (a point and a duration
        differ), yet pint's offset algebra permits exactly it. Two *different*
        offset units of the same ``[time]`` dimension are the trap: pint
        subtracts ``calendar_year - calendar_month`` with a silent /12
        (``0.917 delta_calendar_year``) while the run-time subtraction is raw
        and unconverted, so a point - point across axes is rejected here rather
        than delegated. A same-axis point +/- duration (or point - point) is left
        to pint, which raises ``OffsetUnitCalculusError`` /
        ``DimensionalityError`` on the remaining misuses — caught in
        :func:`_verify_one_body` and reported as a calendar misuse. Only
        ``+``/``-`` get the dispensation: they alone run a forward pint
        operation afterwards, so nothing would catch a point mixed into an
        ordering or a ``where`` later.
        """
        other_q = _unwrap(other)
        if isinstance(other_q, _UnitCheckStructuredValue):
            other_q._raise_used_as_quantity(op)  # noqa: SLF001
        self_is_point = is_calendar_point_unit(
            unit=cast("pint.Unit", self.q.units), registry=self._registry
        )
        other_is_point = isinstance(other_q, pint.Quantity) and is_calendar_point_unit(
            unit=cast("pint.Unit", other_q.units), registry=self._registry
        )
        if (
            self_is_point
            and other_is_point
            and not units_are_equivalent(
                left=cast("pint.Unit", self.q.units),
                right=cast("pint.Unit", other_q.units),
                registry=self._registry,
            )
        ):
            raise _UnitMixError(
                op=op,
                left=cast("pint.Unit", self.q.units),
                right=cast("pint.Unit", other_q.units),
            )
        if self_is_point or other_is_point:
            return
        self._fail_if_other_unit_is_not_equivalent(other=other, op=op)

    def _fail_if_other_unit_is_not_equivalent(self, other: Any, op: str) -> None:  # noqa: ANN401
        """Reject an invalid operand of an ordering comparison or ``where``.

        At run time there is no pint, so these operations are unit-blind (raw
        arrays are added or compared without conversion); two unit-carrying
        operands must already be in equivalent units. Equivalence decides
        calendar points by *identity* (:func:`units_are_equivalent`): ordering
        two same-axis points (``geburtsjahr <= policy_year``) passes, while a
        point against a duration — or any other unit — is rejected. A non-zero
        *bare literal* next to a non-dimensionless quantity is rejected too: it
        silently carries the quantity's unit (``betrag_m + 100.0`` hides a
        monthly amount) — promote it to a parameter or tag it with
        ``cast_ttsim_unit``. Only ``0`` (the ``x + 0.0`` guard, the floor at zero) is
        allowed inline, and literals next to a dimensionless quantity stay
        lenient. Unlike ``+``/``-``, an ordering comparison runs no forward pint
        operation, so calendar points get no delegate-to-pint dispensation here
        (equivalence decides them by identity: only same-axis points order).
        """
        other_q = _unwrap(other)
        if isinstance(other_q, _UnitCheckStructuredValue):
            other_q._raise_used_as_quantity(op)  # noqa: SLF001
        if isinstance(other_q, pint.Quantity) and not units_are_equivalent(
            left=cast("pint.Unit", self.q.units),
            right=cast("pint.Unit", other_q.units),
            registry=self._registry,
        ):
            raise _UnitMixError(op=op, left=self.q.units, right=other_q.units)
        if (
            isinstance(other_q, int | float | numpy.number)
            and not self.q.dimensionless
            and other_q != 0
        ):
            raise _UnitMixError(
                op=op,
                left=cast("pint.Unit", self.q.units),
                right=_dimensionless_unit(self._registry),
                literal=other_q,
            )

    def __bool__(self) -> bool:
        return self._explorer.decide(self._label)

    # Ordering comparisons are unit-blind at run time, so a non-equivalent
    # unit-carrying operand is a bug; the explorer still forces which branch runs.
    def __lt__(self, other: Any) -> _UnitCheckQuantity:  # noqa: ANN401
        self._fail_if_other_unit_is_not_equivalent(other=other, op="<")
        return self._controlled_bool_at(
            level=self._comparison_level(other),
            label=self._composed_label(other=other, op="<"),
        )

    def __le__(self, other: Any) -> _UnitCheckQuantity:  # noqa: ANN401
        self._fail_if_other_unit_is_not_equivalent(other=other, op="<=")
        return self._controlled_bool_at(
            level=self._comparison_level(other),
            label=self._composed_label(other=other, op="<="),
        )

    def __gt__(self, other: Any) -> _UnitCheckQuantity:  # noqa: ANN401
        self._fail_if_other_unit_is_not_equivalent(other=other, op=">")
        return self._controlled_bool_at(
            level=self._comparison_level(other),
            label=self._composed_label(other=other, op=">"),
        )

    def __ge__(self, other: Any) -> _UnitCheckQuantity:  # noqa: ANN401
        self._fail_if_other_unit_is_not_equivalent(other=other, op=">=")
        return self._controlled_bool_at(
            level=self._comparison_level(other),
            label=self._composed_label(other=other, op=">="),
        )

    # ``==``/``!=`` are deliberately *not* unit-screened: they are routinely used
    # polymorphically (sentinels, ``x == 0``) and are not magnitude comparisons.
    def __eq__(self, other: object) -> _UnitCheckQuantity:  # ty: ignore[invalid-method-override]
        return self._controlled_bool_at(
            level=self._comparison_level(other),
            label=self._composed_label(other=other, op="=="),
        )

    def __ne__(self, other: object) -> _UnitCheckQuantity:  # ty: ignore[invalid-method-override]
        return self._controlled_bool_at(
            level=self._comparison_level(other),
            label=self._composed_label(other=other, op="!="),
        )

    def __and__(self, other: Any) -> _UnitCheckQuantity:  # noqa: ANN401
        return self._logical_result(other=other, op="&")

    def __rand__(self, other: Any) -> _UnitCheckQuantity:  # noqa: ANN401
        return self._logical_result(other=other, op="&")

    def __or__(self, other: Any) -> _UnitCheckQuantity:  # noqa: ANN401
        return self._logical_result(other=other, op="|")

    def __ror__(self, other: Any) -> _UnitCheckQuantity:  # noqa: ANN401
        return self._logical_result(other=other, op="|")

    def __xor__(self, other: Any) -> _UnitCheckQuantity:  # noqa: ANN401
        return self._logical_result(other=other, op="^")

    def __rxor__(self, other: Any) -> _UnitCheckQuantity:  # noqa: ANN401
        return self._logical_result(other=other, op="^")

    def __invert__(self) -> _UnitCheckQuantity:
        is_boolean, level = _as_boolean_level(
            unit=cast("pint.Unit", self.q.units), registry=self._registry
        )
        if not is_boolean:
            raise _UnitMixError(
                op="~",
                left=cast("pint.Unit", self.q.units),
                right=_dimensionless_unit(self._registry),
            )
        return self._controlled_bool_at(
            level=level,
            label=f"~{self._label}" if self._label is not None else None,
        )

    # Addition and subtraction require equivalent units (see ``_UnitMixError``);
    # multiplication, division, and powers legitimately combine different units.
    def __add__(self, other: Any) -> _UnitCheckQuantity:  # noqa: ANN401
        self._fail_if_additive_operand_is_invalid(other=other, op="+")
        return self._wrap(self.q + _unwrap(other))

    def __radd__(self, other: Any) -> _UnitCheckQuantity:  # noqa: ANN401
        self._fail_if_additive_operand_is_invalid(other=other, op="+")
        return self._wrap(_unwrap(other) + self.q)

    def __sub__(self, other: Any) -> _UnitCheckQuantity:  # noqa: ANN401
        self._fail_if_additive_operand_is_invalid(other=other, op="-")
        return self._wrap(self.q - _unwrap(other))

    def __rsub__(self, other: Any) -> _UnitCheckQuantity:  # noqa: ANN401
        self._fail_if_additive_operand_is_invalid(other=other, op="-")
        return self._wrap(_unwrap(other) - self.q)

    def __mul__(self, other: Any) -> _UnitCheckQuantity:  # noqa: ANN401
        return self._wrap(self.q * _unwrap(other))

    def __rmul__(self, other: Any) -> _UnitCheckQuantity:  # noqa: ANN401
        return self._wrap(_unwrap(other) * self.q)

    def _nonzero_like(self, value: Any) -> Any:  # noqa: ANN401
        """A magnitude-1.0 stand-in for a division operand.

        Representative magnitudes are all ``1.0``, so a difference of two equal
        quantities (``midijobgrenze_m - minijobgrenze_m``) is a zero-magnitude
        quantity whose *unit* is nonetheless well-defined. The unit check cares only
        about units, so when a division by such a value raises
        ``ZeroDivisionError`` the quotient's unit is recovered by re-dividing by
        a same-unit magnitude-1.0 stand-in (a bare literal becomes ``1.0``).
        """
        if isinstance(value, pint.Quantity):
            return self._registry.Quantity(1.0, value.units)
        return 1.0

    def __truediv__(self, other: Any) -> _UnitCheckQuantity:  # noqa: ANN401
        divisor = _unwrap(other)
        try:
            return self._wrap(self.q / divisor)
        except ZeroDivisionError:
            return self._wrap(self.q / self._nonzero_like(divisor))

    def __rtruediv__(self, other: Any) -> _UnitCheckQuantity:  # noqa: ANN401
        dividend = _unwrap(other)
        try:
            return self._wrap(dividend / self.q)
        except ZeroDivisionError:
            return self._wrap(dividend / self._nonzero_like(self.q))

    def __floordiv__(self, other: Any) -> _UnitCheckQuantity:  # noqa: ANN401
        divisor = _unwrap(other)
        try:
            return self._wrap(self.q // divisor)
        except ZeroDivisionError:
            return self._wrap(self.q // self._nonzero_like(divisor))

    def __rfloordiv__(self, other: Any) -> _UnitCheckQuantity:  # noqa: ANN401
        dividend = _unwrap(other)
        try:
            return self._wrap(dividend // self.q)
        except ZeroDivisionError:
            return self._wrap(dividend // self._nonzero_like(self.q))

    def __mod__(self, other: Any) -> _UnitCheckQuantity:  # noqa: ANN401
        return self._wrap(self.q % _unwrap(other))

    def __rmod__(self, other: Any) -> _UnitCheckQuantity:  # noqa: ANN401
        return self._wrap(_unwrap(other) % self.q)

    def __pow__(self, other: Any) -> _UnitCheckQuantity:  # noqa: ANN401
        return self._wrap(self.q ** _unwrap(other))

    def __rpow__(self, other: Any) -> _UnitCheckQuantity:  # noqa: ANN401
        return self._wrap(_unwrap(other) ** self.q)

    def __neg__(self) -> _UnitCheckQuantity:
        return self._wrap(-self.q)

    def __pos__(self) -> _UnitCheckQuantity:
        return self._wrap(+self.q)

    def __abs__(self) -> _UnitCheckQuantity:
        return self._wrap(abs(self.q))

    def __round__(self, ndigits: int | None = None) -> _UnitCheckQuantity:
        # `round` is unit-preserving (the vectorized `xnp.round` is handled the
        # same way), so a body using the builtin `round(x)` keeps its unit.
        return self._wrap(self.q)

    def astype(
        self,
        dtype: Any,  # noqa: ANN401
        *args: Any,  # noqa: ANN401, ARG002
        **kwargs: Any,  # noqa: ANN401, ARG002
    ) -> _UnitCheckQuantity:
        """The unit check's ``astype``, whose effect on the unit follows the dtype:

        - a real numeric dtype (integer or floating) re-types the magnitude only,
          so the unit is preserved
          (this is how a lookup table is indexed off a float column,
          ``xnp.floor(age).astype(int)``);
        - ``bool`` yields an indicator, so the physical dimension is dropped and
          only the grouping level survives — a per-``[fam]`` amount becomes a
          per-``[fam]`` truth value;
        - anything else (a datetime, a string, a complex number) has no unit
          reading here, so the body is left un-evaluable and must opt out.

        ``dtype`` is required but not positional-only, since both backends accept
        the keyword form; the remaining options only affect the magnitude.
        """
        kind = numpy.dtype(dtype).kind
        if kind == "b":
            return self._controlled_bool_at(level=_unit_level_denominator(self.q.units))
        if kind not in "iuf":
            raise _UnsupportedAstypeError(dtype)
        return self._wrap(self.q)


def _wrap_for_unit_check(
    value: Any,  # noqa: ANN401
    explorer: _PathExplorer,
    unit_system: UnitSystem,
    label: str | None = None,
) -> Any:  # noqa: ANN401
    """Wrap unit-carrying representative values; pass framework args through.

    Quantities (and the leaves of dict-param trees) become ``_UnitCheckQuantity`` so the
    explorer controls branches on them; ``xnp``/``num_segments``/… stay raw.
    ``label`` is the argument name the body sees, carried on the stand-in so a
    branch decision on it can be named in an error. A structured stand-in is
    re-anchored on the run's explorer, so its annotated plucks screen and
    branch like any other operand; the same explorer, held in a one-element
    cell, lets a schedule plucked from a schedule-typed field anchor a
    bare-literal ``look_up`` on the run's branch path.
    """
    if isinstance(value, pint.Quantity):
        return _UnitCheckQuantity(
            q=value, explorer=explorer, unit_system=unit_system, label=label
        )
    if isinstance(value, _UnitCheckStructuredValue):
        return _UnitCheckStructuredValue(
            producer=value._producer,  # noqa: SLF001
            unit_system=unit_system,
            cls=value._cls,  # noqa: SLF001
            explorer=explorer,
            label=label,
            item_cls=value._item_cls,  # noqa: SLF001
            explorer_holder=[explorer],
        )
    if isinstance(value, dict):
        return {
            key: _wrap_for_unit_check(
                value=leaf,
                explorer=explorer,
                unit_system=unit_system,
                label=f"{label}[{key!r}]" if label is not None else None,
            )
            for key, leaf in value.items()
        }
    return value


class _ScheduleNotEvaluableError(Exception):
    """A schedule/lookup/join call the unit check cannot resolve to a unit.

    Raised when a function-like parameter carries no axes (a converter-produced
    or unannotated schedule) or a gather has no unit-carrying target — caught by
    :func:`_verify_one_body`'s generic handler and reported as needing an explicit
    ``verify_units=False`` opt-out, exactly like any other un-evaluable op.
    """


class _LookupArityError(Exception):
    """A multi-dimensional ``look_up`` call supplies the wrong number of arguments.

    A lookup declaring a tuple ``input_unit`` screens each argument against its
    own axis positionally, so the call must supply exactly as many arguments as
    declared axes. Caught by :func:`_verify_one_body` and reported against the
    calling body, naming both counts.
    """

    def __init__(self, declared: int, supplied: int) -> None:
        super().__init__()
        self.declared = declared
        self.supplied = supplied


class _StructuredValueUsedAsQuantityError(Exception):
    """A value plucked off a structured parameter was used as a quantity —
    caught by :func:`_verify_one_body` and reported with the
    cast-at-the-pluck fix."""

    def __init__(self, producer: str, op: str) -> None:
        super().__init__()
        self.producer = producer
        self.op = op


class _UnitCheckStructuredValue:
    """The unit check's stand-in for a structured param-function output
    (``unit=UNSET_UNIT``, GEP 10). A pluck off an ``Annotated`` scalar field of
    the producer's return dataclass resolves to a quantity at the field's
    declared unit; a nested-dataclass pluck resolves recursively. Everything
    else stays opaque — attribute access, subscripting, method calls yield an
    opaque stand-in again — and using an opaque pluck as a quantity raises,
    demanding a ``cast_ttsim_unit`` at the pluck.
    """

    __slots__ = (
        "_cls",
        "_explorer",
        "_explorer_holder",
        "_item_cls",
        "_label",
        "_producer",
        "_unit_system",
    )
    # Defer binary NumPy ops to our (raising) reflected dunders.
    __array_ufunc__ = None
    __array_priority__ = 1000
    __hash__ = object.__hash__

    def __init__(
        self,
        producer: str,
        unit_system: UnitSystem,
        cls: type | None = None,
        explorer: _PathExplorer | None = None,
        label: str | None = None,
        item_cls: type | None = None,
        explorer_holder: list[_PathExplorer | None] | None = None,
    ) -> None:
        self._producer = producer
        # An annotated field's pluck resolves to a unit, so the stand-in needs
        # the system to resolve it in — an object graph, not a call tree, so it
        # carries the system rather than receiving it per call.
        self._unit_system = unit_system
        self._cls = cls
        self._explorer = explorer
        self._label = label
        # The value dataclass a mapping producer yields on subscript; `None`
        # unless the producer is typed `Mapping[..., <dataclass>]`.
        self._item_cls = item_cls
        # The body's live explorer cell, handed to a schedule plucked from a
        # schedule-typed field so a bare-literal `look_up` anchors on the body's
        # branch path (see `_UnitCheckSchedule._produce`). Carried through nested
        # plucks so a schedule any depth down still anchors.
        self._explorer_holder = explorer_holder

    def _raise_used_as_quantity(self, op: str) -> NoReturn:
        raise _StructuredValueUsedAsQuantityError(producer=self._producer, op=op)

    def _opaque(self) -> _UnitCheckStructuredValue:
        return _UnitCheckStructuredValue(
            producer=self._producer,
            unit_system=self._unit_system,
            explorer_holder=self._explorer_holder,
        )

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        # Refuse protocol probes (``__array__``, copy/pickle hooks, …).
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        kinds = (
            _structured_field_kinds(cls=self._cls, unit_system=self._unit_system)
            if self._cls is not None
            else None
        )
        resolved = (kinds or {}).get(name)
        label = f"{self._label}.{name}" if self._label is not None else None
        if isinstance(resolved, pint.Unit):
            # An annotated field's pluck is a known quantity; with the run's
            # explorer it screens and branches like any other operand.
            quantity = self._unit_system.registry.Quantity(1.0, resolved)
            if self._explorer is None:
                return quantity
            return _UnitCheckQuantity(
                q=quantity,
                explorer=self._explorer,
                unit_system=self._unit_system,
                label=label,
            )
        if isinstance(resolved, _ScheduleFieldKind):
            # A schedule-typed field declares both axes; the pluck yields a
            # schedule that screens each `look_up`/`piecewise_polynomial` argument
            # against `input_unit` and produces `output_unit`.
            return _UnitCheckSchedule(
                input_unit=resolved.input_unit,
                output_unit=resolved.output_unit,
                unit_system=self._unit_system,
                explorer_holder=self._explorer_holder,
            )
        if resolved is not None:
            return _UnitCheckStructuredValue(
                producer=self._producer,
                unit_system=self._unit_system,
                cls=resolved,
                explorer=self._explorer,
                label=label,
                explorer_holder=self._explorer_holder,
            )
        return self._opaque()

    def __getitem__(self, _key: Any) -> _UnitCheckStructuredValue:  # noqa: ANN401
        if self._item_cls is None:
            return self._opaque()
        # A mapping producer's value is the same dataclass at every key, so the
        # key itself is irrelevant — subscripting yields a stand-in of that class.
        label = f"{self._label}[…]" if self._label is not None else None
        return _UnitCheckStructuredValue(
            producer=self._producer,
            unit_system=self._unit_system,
            cls=self._item_cls,
            explorer=self._explorer,
            label=label,
            explorer_holder=self._explorer_holder,
        )

    def __call__(self, *_args: Any, **_kwargs: Any) -> _UnitCheckStructuredValue:  # noqa: ANN401
        return self._opaque()


#: Every arithmetic / ordering / logical / bool use of an opaque structured value is a
#: misuse (cast the pluck or opt out), so all of them raise. The dunders are generated
#: from this table rather than spelled out one near-identical method at a time.
_STRUCTURED_VALUE_FORBIDDEN_OPS: Mapping[str, str] = {
    "__bool__": "a branch decision",
    "__lt__": "<",
    "__le__": "<=",
    "__gt__": ">",
    "__ge__": ">=",
    "__eq__": "==",
    "__ne__": "!=",
    "__add__": "+",
    "__radd__": "+",
    "__sub__": "-",
    "__rsub__": "-",
    "__mul__": "*",
    "__rmul__": "*",
    "__truediv__": "/",
    "__rtruediv__": "/",
    "__floordiv__": "//",
    "__rfloordiv__": "//",
    "__mod__": "%",
    "__rmod__": "%",
    "__pow__": "**",
    "__rpow__": "**",
    "__and__": "&",
    "__rand__": "&",
    "__or__": "|",
    "__ror__": "|",
    "__xor__": "^",
    "__rxor__": "^",
    "__invert__": "~",
    "__neg__": "unary -",
    "__pos__": "unary +",
    "__abs__": "abs",
}


class _UnsupportedAstypeError(Exception):
    """A cast whose dtype has no unit reading (a datetime, a string)."""

    def __init__(self, dtype: Any) -> None:  # noqa: ANN401
        super().__init__(f"astype({dtype!r}) has no unit reading")


def _structured_value_forbidden_op(op: str) -> Callable[..., Any]:
    def method(self: _UnitCheckStructuredValue, *_a: Any, **_k: Any) -> Any:  # noqa: ANN401
        return self._raise_used_as_quantity(op)

    return method


def _install_structured_value_forbidden_ops() -> None:
    """Bind each dunder in `_STRUCTURED_VALUE_FORBIDDEN_OPS` to a raising stub.

    The table covers arithmetic, ordering, equality, logical and truth-value uses.
    A structured value is a container, not a quantity, so any of them is an
    authoring error the unit check reports against the operator it was used with —
    cast the pluck or opt out (GEP 10).
    """
    for dunder, op in _STRUCTURED_VALUE_FORBIDDEN_OPS.items():
        setattr(_UnitCheckStructuredValue, dunder, _structured_value_forbidden_op(op))


_install_structured_value_forbidden_ops()


class _UnitCheckSchedule:
    """The unit check's stand-in for a ``piecewise_*``/lookup-table parameter value.

    Such a parameter is a *function between quantities*: a body calls
    ``piecewise_polynomial(x, parameters=…)`` or ``….look_up(idx)`` on it and gets
    an array. The unit check needs only the unit that falls out. This stand-in
    carries the resolved ``input_unit``/``output_unit`` axes — a schedule
    parameter's own, a schedule builder's ``InputOutputUnit`` declaration, or a
    schedule-typed dataclass field's: it screens each domain argument against
    ``input_unit`` (as ``+`` screens an operand) and produces the ``output_unit``.

    ``input_unit`` takes three forms:

    - a single :class:`pint.Unit` — screened against every ``look_up`` argument
      (a one-dimensional look-up, or a piecewise ``x``);
    - a ``tuple`` of :class:`pint.Unit` — screened positionally (argument ``i``
      against axis ``i``); the call must supply exactly as many arguments as
      declared axes, else a :class:`_LookupArityError`;
    - ``None`` — unscreened; a schedule parameter that left its input axis unset.
    """

    __slots__ = ("explorer_holder", "input_unit", "output_unit", "unit_system")

    def __init__(
        self,
        input_unit: pint.Unit | tuple[pint.Unit, ...] | None,
        output_unit: pint.Unit,
        unit_system: UnitSystem,
        explorer_holder: list[_PathExplorer | None] | None = None,
    ) -> None:
        self.input_unit = input_unit
        self.output_unit = output_unit
        self.unit_system = unit_system
        self.explorer_holder = explorer_holder

    def _produce(self, domain_args: tuple[Any, ...]) -> _UnitCheckQuantity:
        if isinstance(self.input_unit, tuple) and len(domain_args) != len(
            self.input_unit
        ):
            raise _LookupArityError(
                declared=len(self.input_unit), supplied=len(domain_args)
            )
        explorer: _PathExplorer | None = None
        all_indices_are_scalar_literals = True
        for position, arg in enumerate(domain_args):
            # A single declared axis screens every argument; a tuple screens
            # positionally; `None` leaves the argument unscreened.
            axis = self._axis_at(position)
            if isinstance(arg, _UnitCheckQuantity):
                explorer = arg._explorer  # noqa: SLF001
                if axis is not None and not units_are_equivalent(
                    left=cast("pint.Unit", arg.q.units),
                    right=axis,
                    registry=self.unit_system.registry,
                ):
                    raise _UnitMixError(
                        op="look-up",
                        left=axis,
                        right=cast("pint.Unit", arg.q.units),
                    )
            elif not _is_scalar_literal(arg):
                # A non-quantity, non-literal index — an opaque structured pluck,
                # say — cannot be treated as a bare literal: it still owes a
                # `cast_ttsim_unit` or an annotation, so the anchoring fallback
                # below must not silently accept it.
                all_indices_are_scalar_literals = False
        if (
            explorer is None
            and all_indices_are_scalar_literals
            and self._literal_index_is_admissible()
        ):
            # No unit-carrying domain argument to anchor the result on (a bare or
            # computed-literal index), but every index is a bare dimensionless
            # literal and legitimate here. The output unit is fixed by the schedule
            # regardless of the index, so anchor the result on the body's own
            # branch path via `explorer_holder`. Only a schedule reached with no
            # body explorer at all (never during body verification) stays
            # un-anchorable and falls back to the opt-out.
            explorer = (
                self.explorer_holder[0] if self.explorer_holder is not None else None
            )
        if explorer is None:
            raise _ScheduleNotEvaluableError
        return _UnitCheckQuantity(
            q=self.unit_system.registry.Quantity(1.0, self.output_unit),
            explorer=explorer,
            unit_system=self.unit_system,
        )

    def _axis_at(self, position: int) -> pint.Unit | None:
        """The input axis screening the argument at ``position``.

        A single declared axis screens every argument; a tuple screens
        positionally (the arity is checked before this is reached); ``None`` leaves
        the argument unscreened.
        """
        if isinstance(self.input_unit, tuple):
            return self.input_unit[position]
        return self.input_unit

    def _literal_index_is_admissible(self) -> bool:
        """Whether bare literals are legitimate indices for this schedule.

        A bare Python literal is a dimensionless, person-level value. It matches
        an undeclared input axis (``None``, unscreened by design) or a declared
        dimensionless one. A dimensionful axis (a currency, an area, a calendar
        point) is never keyed by a bare literal, so such an index is not
        admissible and the body must opt out. Every declared axis — the one axis,
        or each element of a tuple — must be admissible.
        """
        if self.input_unit is None:
            return True
        axes = (
            self.input_unit
            if isinstance(self.input_unit, tuple)
            else (self.input_unit,)
        )
        return all(
            self.unit_system.registry.Quantity(
                1.0, cast("pint.Unit", axis)
            ).dimensionless
            for axis in axes
        )

    def look_up(self, *args: Any) -> _UnitCheckQuantity:  # noqa: ANN401
        return self._produce(args)


def _is_scalar_literal(value: Any) -> bool:  # noqa: ANN401
    """Whether ``value`` is a genuine numeric scalar (a bare or computed literal).

    A body's arithmetic on Python/NumPy number literals stays a plain number;
    every unit-check stand-in (a quantity, a structured value, a schedule) is
    something else. `bool` counts — it is an `int` subclass.
    """
    return isinstance(value, int | float | numpy.integer | numpy.floating)


def _piecewise_polynomial_for_unit_check(x: Any, parameters: Any, xnp: Any) -> Any:  # noqa: ANN401, ARG001
    """Unit-check stand-in for ``piecewise_polynomial``.

    Screen ``x`` against the schedule's ``input_unit`` and produce its
    ``output_unit``. Every schedule arrives as a :class:`_UnitCheckSchedule`
    carrying the axes its producer declared — a parameter's YAML axes or a
    builder's ``InputOutputUnit``. An opaque structured pluck (an unannotated
    field of a structured value) propagates unchanged for the caller to cast;
    anything else cannot be evaluated here.
    """
    if isinstance(parameters, _UnitCheckSchedule):
        return parameters._produce((x,))  # noqa: SLF001
    if isinstance(parameters, _UnitCheckStructuredValue):
        return parameters
    raise _ScheduleNotEvaluableError


def _join_for_unit_check(
    foreign_key: Any,  # noqa: ANN401, ARG001
    primary_key: Any,  # noqa: ANN401, ARG001
    target: Any,  # noqa: ANN401
    value_if_foreign_key_is_missing: Any,  # noqa: ANN401, ARG001
    xnp: Any,  # noqa: ANN401, ARG001
) -> Any:  # noqa: ANN401
    """Unit-check stand-in for ``join``.

    A person-to-person gather preserves the ``target`` column's unit and level
    (the keys are dimensionless ``p_id``s, the missing-value a sentinel literal).
    """
    if isinstance(target, _UnitCheckQuantity):
        return target._wrap(target.q)  # noqa: SLF001
    raise _ScheduleNotEvaluableError


def _cast_ttsim_unit_for_unit_check(
    value: Any,  # noqa: ANN401
    unit: str | CompositeUnit,
    unit_system: UnitSystem,
    explorer_holder: list[_PathExplorer | None],
) -> Any:  # noqa: ANN401
    """Unit-check stand-in for ``cast_ttsim_unit``.

    The cast is total: whatever flowed in — a quantity at another unit or
    level, a bare literal, an attribute plucked off a structured value — the
    stand-in flowing out carries the stated unit, resolved like a declaration
    (currency-agnostic; an omitted level is bare — both per-person and
    level-neutral). The result stays on the body's path: a
    ``_UnitCheckQuantity`` input keeps its explorer, and any other input
    (a bare literal) is wrapped with the body's explorer (``explorer_holder``),
    so a cast literal orders and combines like any quantity — ``max(x,
    cast_ttsim_unit(0, …))`` screens instead of reading as un-evaluable. A malformed
    token raises a :class:`UnitDefinitionError`, which :func:`_verify_one_body`
    re-raises rather than misreporting as an un-evaluable body.
    """
    token = ttsim_unit_from_yaml_value(value=unit, where="A `cast_ttsim_unit` call")
    resolved = resolve_agnostic_ttsim_unit(
        unit=token,
        registry=unit_system.registry,
        where="A `cast_ttsim_unit` call",
        what="a cast inside a body",
    )
    quantity = unit_system.registry.Quantity(1.0, resolved)
    if isinstance(value, _UnitCheckQuantity):
        return value._wrap(quantity)  # noqa: SLF001
    explorer = explorer_holder[0]
    if explorer is None:
        return quantity
    return _UnitCheckQuantity(q=quantity, explorer=explorer, unit_system=unit_system)


def _time_conversion_stand_in(
    from_pint: str, to_pint: str, registry: pint.UnitRegistry, *, is_flow: bool
) -> Callable[[Any], Any]:
    """Build the unit-check stand-in for one ``ttsim.unit_converters`` time converter.

    A converter restates a value on a different time period by multiplying by a
    whole number — a *dimensionless* factor pint cannot see — so the stand-in
    rebases the period token instead: a duration converter (``m_to_y``) swaps the
    numerator period (``MONTHS`` -> ``YEARS``), a flow converter
    (``per_m_to_per_y``) the denominator flow period (``CURRENCY/month`` ->
    ``CURRENCY/year``). A bare literal (no unit to rebase) flows through
    unchanged; a value whose period does not match the converter produces a
    mismatched unit that the declared-vs-inferred check then reports.
    """
    from_q = registry.Quantity(1.0, from_pint)
    to_q = registry.Quantity(1.0, to_pint)

    def stand_in(value: Any) -> Any:  # noqa: ANN401
        if not isinstance(value, _UnitCheckQuantity):
            return value
        rebased = value.q * from_q / to_q if is_flow else value.q / from_q * to_q
        return value._wrap(rebased)  # noqa: SLF001

    return stand_in


def _time_conversion_stand_ins(registry: pint.UnitRegistry) -> dict[str, Any]:
    """A stand-in for every ``<a>_to_<b>`` / ``per_<a>_to_per_<b>`` converter."""
    stand_ins: dict[str, Any] = {}
    for from_id, from_pint in TIME_UNIT_ID_TO_PINT_NAME.items():
        for to_id, to_pint in TIME_UNIT_ID_TO_PINT_NAME.items():
            if from_id == to_id:
                continue
            stand_ins[f"{from_id}_to_{to_id}"] = _time_conversion_stand_in(
                from_pint=from_pint,
                to_pint=to_pint,
                registry=registry,
                is_flow=False,
            )
            stand_ins[f"per_{from_id}_to_per_{to_id}"] = _time_conversion_stand_in(
                from_pint=from_pint,
                to_pint=to_pint,
                registry=registry,
                is_flow=True,
            )
    return stand_ins


def _unit_check_helper_stand_ins(
    unit_system: UnitSystem,
) -> tuple[Mapping[str, Any], list[_PathExplorer | None]]:
    """The module-level helpers swapped for unit-only stand-ins in a body's scope.

    Each stand-in mints quantities in ``unit_system``'s registry, so the set is
    built per run rather than shared. The returned ``explorer_holder`` lets the
    ``cast_ttsim_unit`` stand-in reach the explorer of the body currently under
    verification — :func:`_verify_one_body` sets it per body — so a cast literal
    becomes an explorer-carrying quantity rather than a bare pint one.
    """
    explorer_holder: list[_PathExplorer | None] = [None]
    stand_ins: Mapping[str, Any] = {
        "piecewise_polynomial": _piecewise_polynomial_for_unit_check,
        "join": _join_for_unit_check,
        "cast_ttsim_unit": functools.partial(
            _cast_ttsim_unit_for_unit_check,
            unit_system=unit_system,
            explorer_holder=explorer_holder,
        ),
        "max": _scalar_clamp_for_unit_check(op="maximum"),
        "min": _scalar_clamp_for_unit_check(op="minimum"),
        **_time_conversion_stand_ins(unit_system.registry),
    }
    return stand_ins, explorer_holder


def _scalar_clamp_for_unit_check(op: str) -> Any:  # noqa: ANN401
    """Stand in for scalar ``max``/``min``, screening like the vectorized ops.

    A scalar body's ``max(a, b)``/``min(a, b)`` runs the Python builtin, which
    returns one operand *whole* — so on the branch where a bare ``0`` floor wins
    the result is a unit-less ``0`` and a downstream ``/`` or ``*`` corrupts the
    unit. Routing through :func:`_clamping_op` (the vectorizer's own path for
    ``xnp.maximum``/``xnp.minimum``) makes the result carry the quantity's unit
    on every branch, so a zero floor needs no ``cast_ttsim_unit``. The two-argument and
    one-iterable spellings (GEP 1) both fold through the same screen.
    """

    def stand_in(*args: Any) -> Any:  # noqa: ANN401
        items = list(args[0]) if len(args) == 1 else list(args)
        result = items[0]
        for item in items[1:]:
            result = _clamping_op(left=result, right=item, op=op)
        return result

    return stand_in


def _clamping_op(left: Any, right: Any, op: str) -> Any:  # noqa: ANN401
    """``xnp.maximum``/``xnp.minimum``: an ordering-style screen, unit preserved.

    The vectorizer rewrites a scalar ``max(a, b)``/``min(a, b)`` to these, so the
    operands are screened exactly as an ordering comparison — two unit-carrying
    operands must be equivalent, a bare non-zero literal bound is rejected — and
    the result carries the quantity's unit.
    """
    quantity = left if isinstance(left, _UnitCheckQuantity) else right
    if not isinstance(quantity, _UnitCheckQuantity):
        return getattr(numpy, op)(left, right)
    other = right if quantity is left else left
    quantity._fail_if_other_unit_is_not_equivalent(other=other, op=op)  # noqa: SLF001
    return quantity._wrap(quantity.q)  # noqa: SLF001


def _where_op(x: Any, y: Any) -> Any:  # noqa: ANN401
    """``xnp.where``: the two branches become one column, so they must carry
    equivalent units (as for an ordering comparison — no forward pint op runs,
    so calendar points screen by identity); the result carries that unit."""
    quantity = x if isinstance(x, _UnitCheckQuantity) else y
    if not isinstance(quantity, _UnitCheckQuantity):
        return numpy.where(True, x, y)  # noqa: FBT003
    other = y if quantity is x else x
    quantity._fail_if_other_unit_is_not_equivalent(other=other, op="where")  # noqa: SLF001
    return quantity._wrap(quantity.q)  # noqa: SLF001


def _clip_op(value: Any, a_min: Any, a_max: Any) -> Any:  # noqa: ANN401
    """``xnp.clip``: each bound is screened against the value as an ordering
    operand (so a bare non-zero literal bound is rejected); the unit is preserved.
    """
    if not isinstance(value, _UnitCheckQuantity):
        return numpy.clip(value, a_min, a_max)
    for bound in (a_min, a_max):
        if bound is not None:
            value._fail_if_other_unit_is_not_equivalent(other=bound, op="clip")  # noqa: SLF001
    return value._wrap(value.q)  # noqa: SLF001


def _unit_preserving_op(value: Any) -> Any:  # noqa: ANN401
    """A unit-preserving reduction/unary op (``sum``/``floor``/``abs``/…)."""
    if isinstance(value, _UnitCheckQuantity):
        return value._wrap(value.q)  # noqa: SLF001
    return value


def _opt_out_required_error(qname: str, reason: str) -> str:
    """Message demanding an explicit opt-out for a body the unit check cannot evaluate.

    A body the unit check cannot evaluate is *not* waved through silently:
    the author must mark it ``verify_units=False`` so that every un-verified body
    is a visible, deliberate choice. The declared unit still
    stands and the body's edges are still checked — only its internal inference
    is skipped.
    """
    return (
        f"{qname}: its body cannot be unit-checked ({reason}). "
        f"Set `verify_units=False` on its decorator to opt out of body inference "
        f"— its declared unit and its edges stay checked (GEP 10)."
    )


def _structured_pluck_message(
    qname: str,
    error: _StructuredValueUsedAsQuantityError,
) -> str:
    """Message for a body computing with an un-cast pluck off a structured
    value."""
    return (
        f"{qname}: uses a value plucked off the structured parameter "
        f"'{error.producer}' as a quantity ('{error.op}'), but such a value "
        f"carries no unit. State its unit at the structure — annotate the "
        f"dataclass field (`Annotated[float, TTSIMUnit…]`) — or at the pluck with "
        f"`cast_ttsim_unit(<pluck>, <unit>)`, or opt out of body inference with "
        f"`verify_units=False` (GEP 10)."
    )


def _arithmetic_misuse_message(
    qname: str,
    error: _UnitMixError
    | _StructuredValueUsedAsQuantityError
    | pint.OffsetUnitCalculusError
    | pint.DimensionalityError,
    detail: str,
) -> str:
    """Message for a body that combines quantities unsoundly under ``+``/``-``/order.

    Dispatches the ways the unit check catches such a body: an explicit
    :class:`_UnitMixError` (non-equivalent units, a logical operator on a real
    quantity, a bare-literal threshold), a
    :class:`_StructuredValueUsedAsQuantityError` (an un-cast pluck off a
    structured value used as a quantity), or the pint error a calendar point
    raises when used outside its affine algebra — an
    :class:`pint.OffsetUnitCalculusError` (point + point, a scaled point) or a
    :class:`pint.DimensionalityError` (point + a foreign dimension such as a
    currency, ``geburtsjahr + income_m``). The latter two are genuine calendar
    bugs, not un-checkable bodies, so they report as misuse rather than
    demanding ``verify_units=False``.
    """
    if isinstance(error, _UnitMixError):
        return _unit_mix_error_message(qname=qname, mix=error, detail=detail)
    if isinstance(error, _StructuredValueUsedAsQuantityError):
        return _structured_pluck_message(qname=qname, error=error)
    return (
        f"{qname}: combines a calendar point unsoundly{detail} — a point cannot "
        f"be added to another point (subtract them to get a duration), scaled, or "
        f"combined with a non-duration quantity; shift a point only by a same-axis "
        f"duration (GEP 10)."
    )


def _unit_mix_error_message(qname: str, mix: _UnitMixError, detail: str) -> str:
    """Message for a body that combined two units unsoundly.

    A logical operator (``&``/``|``/``^``/``~``) carrying a real quantity is
    reported as a non-boolean operand; an ordering comparison against a bare
    non-zero literal is reported as an untagged threshold; ``+``/``-``/an ordering
    comparison of non-equivalent quantities is reported as a unit mix (no run-time
    conversion).
    """
    if mix.literal is not None:
        return (
            f"{qname}: combines '{mix.left}' {mix.op} the bare literal "
            f"{mix.literal}{detail} — a literal next to a quantity silently "
            f"carries that quantity's unit; promote it to a parameter, tag it "
            f"with `cast_ttsim_unit`, or use 0 (GEP 10)."
        )
    if mix.op not in _LOGICAL_OPS:
        return (
            f"{qname}: combines non-equivalent units '{mix.left}' "
            f"{mix.op} '{mix.right}'{detail} — at run time there is no unit "
            f"conversion, so both operands must already be in the same unit."
        )
    if mix.op == "~":
        return (
            f"{qname}: applies '~' to a non-boolean operand '{mix.left}'{detail} "
            f"— logical operators expect boolean (dimensionless) operands."
        )
    return (
        f"{qname}: applies '{mix.op}' to non-boolean operands '{mix.left}' and "
        f"'{mix.right}'{detail} — logical operators expect boolean "
        f"(dimensionless) operands."
    )


def _verify_one_body(
    qname: str,
    function: Any,  # noqa: ANN401  (a scalar body, possibly a dags wrapper)
    declared: pint.Unit,
    boolean_values: Mapping[str, Any],
    base_kwargs: dict[str, Any],
    unit_system: UnitSystem,
    explorer_holder: list[_PathExplorer | None],
) -> str | None:
    """Unit-check one body on every reachable branch path; return an error or ``None``.

    The body runs once per path through the branch tree (see ``_PathExplorer``);
    each run that infers a concrete unit must match the declaration. A body that
    adds, subtracts, or orders two non-equivalent unit-carrying operands is
    flagged directly (``_UnitMixError``): those operations are unit-blind at run
    time, where no pint conversion happens. A run that infers a dimensionless
    result (e.g. an early ``return 0.0`` guard) falls back to the declaration on
    that path. A run that *raises* — a body using a lookup table, a piecewise
    polynomial, ``join``, or a raw ``xnp`` op the unit check cannot evaluate — is
    reported as needing an explicit ``verify_units=False`` opt-out (callers reach
    this only for bodies that have not already opted out).

    A branch failure does not stop the exploration: the remaining paths still
    run, so the error can say whether the offence is confined to the reported
    branch combination (named via :meth:`_PathExplorer.branch_detail`) or other
    combinations fail as well.
    """
    explorer = _PathExplorer()
    # The `cast_ttsim_unit` stand-in reaches this body's explorer here, so a cast
    # literal (`max(x, cast_ttsim_unit(0, …))`) becomes an explorer-carrying quantity.
    explorer_holder[0] = explorer
    paths = 0
    branch_errors: list[str] = []
    clean_paths = 0
    while True:
        if paths >= _MAX_PATHS:
            # Truncating exploration must not pass silently: a wrong-unit branch
            # first reached past the cap would otherwise go unchecked.
            return _opt_out_required_error(
                qname=qname,
                reason=f"it explores more than {_MAX_PATHS} branch paths — too many to "
                "check exhaustively",
            )
        paths += 1
        explorer.start_run()
        kwargs = {
            name: _wrap_for_unit_check(
                value=value,
                explorer=explorer,
                unit_system=unit_system,
                label=name,
            )
            for name, value in {**base_kwargs, **boolean_values}.items()
        }
        error, terminal = _run_one_path(
            qname=qname,
            function=function,
            declared=declared,
            kwargs=kwargs,
            explorer=explorer,
            unit_system=unit_system,
        )
        if terminal:
            return error
        if error is None:
            clean_paths += 1
        elif error not in branch_errors:
            branch_errors.append(error)
        if not explorer.advance():
            break
    return _summarize_branch_errors(
        branch_errors=branch_errors, clean_paths=clean_paths
    )


def _run_one_path(
    qname: str,
    function: Any,  # noqa: ANN401  (a scalar body, possibly a dags wrapper)
    declared: pint.Unit,
    kwargs: dict[str, Any],
    explorer: _PathExplorer,
    unit_system: UnitSystem,
) -> tuple[str | None, bool]:
    """Run the body once along the explorer's current path.

    Returns ``(error, terminal)``. A *terminal* error aborts the exploration —
    an un-evaluable body or a blown decision budget is not branch-specific —
    while a non-terminal error is one branch combination's failure and the
    exploration continues.
    """
    try:
        result: Any = function(**kwargs)
    except _PathBudgetExceededError:
        return _opt_out_required_error(
            qname=qname,
            reason=f"it makes more than {_MAX_DECISIONS_PER_RUN} branch decisions "
            "in one run — a data-driven loop?",
        ), True
    except _LookupArityError as err:
        return (
            f"{qname}: calls a lookup declaring {err.declared} input "
            f"{'axis' if err.declared == 1 else 'axes'} with {err.supplied} "
            f"argument{'' if err.supplied == 1 else 's'} — a multi-dimensional "
            f"lookup screens each argument against the corresponding axis, so the "
            f"counts must match (GEP 10)."
        ), True
    except (
        _UnitMixError,
        _StructuredValueUsedAsQuantityError,
        pint.OffsetUnitCalculusError,
        pint.DimensionalityError,
    ) as err:
        return _arithmetic_misuse_message(
            qname=qname, error=err, detail=explorer.branch_detail()
        ), False
    except UnitDefinitionError:
        # A malformed `cast_ttsim_unit` token is a definition error, not an
        # un-evaluable body; report it as itself.
        raise
    except Exception as err:  # noqa: BLE001
        return _opt_out_required_error(
            qname=qname,
            reason=f"evaluating it raised {type(err).__name__}: {err} — either an "
            "operation pint cannot evaluate symbolically (a piecewise polynomial, a "
            "lookup table, `join`, a raw `xnp` op) or a defect in the body itself, "
            "which this message reproduces verbatim so the two can be told apart",
        ), True
    return _inferred_result_error(
        qname=qname,
        inferred=_unwrap(result),
        declared=declared,
        detail=explorer.branch_detail(),
        unit_system=unit_system,
    ), False


def _summarize_branch_errors(
    branch_errors: list[str],
    clean_paths: int,
) -> str | None:
    """Collapse an exploration's branch failures into one message.

    A single failing combination next to clean ones states that the others
    match; several distinct failures report the first and count the rest.
    """
    if not branch_errors:
        return None
    if len(branch_errors) > 1:
        return (
            f"{branch_errors[0]} ({len(branch_errors) - 1} further branch "
            f"combination(s) fail too.)"
        )
    if clean_paths:
        return (
            f"{branch_errors[0]} All other branch combinations match the declaration."
        )
    return branch_errors[0]


def _non_quantity_result_error(
    qname: str,
    inferred: Any,  # noqa: ANN401
) -> str:
    """The screen for a result that is neither a quantity nor a plain scalar:
    a bare structured pluck names its cast; any other opaque return — a
    dataclass, a tuple — must opt out."""
    if isinstance(inferred, _UnitCheckStructuredValue):
        return (
            f"{qname}: returns a value plucked off the structured parameter "
            f"'{inferred._producer}' without stating its unit; annotate the "  # noqa: SLF001
            f"dataclass field or tag it with `cast_ttsim_unit` at the pluck (GEP 10)."
        )
    return _opt_out_required_error(
        qname=qname,
        reason="it returns a value the unit check cannot handle — a dataclass, "
        "a tuple, or another non-scalar",
    )


def _bare_literal_result_error(
    qname: str,
    inferred: Any,  # noqa: ANN401
    declared: pint.Unit,
    detail: str,
    registry: pint.UnitRegistry,
) -> str | None:
    """The literal-return screen for a plain (non-``Quantity``) scalar result.

    Only ``0`` (the eligibility guard) and booleans fall through: a non-zero
    numeric literal returned under a non-dimensionless declaration is a hidden
    dimensioned constant, exactly as a bare bound in an ordering comparison.
    """
    if (
        isinstance(inferred, int | float | numpy.number)
        and not isinstance(inferred, bool | numpy.bool_)
        and inferred != 0
        and not registry.Quantity(1.0, declared).dimensionless
    ):
        return (
            f"{qname}: returns the bare literal {inferred}{detail} under the "
            f"declaration '{declared}' — a literal return silently carries "
            f"the declared unit; promote it to a parameter, tag it with "
            f"`cast_ttsim_unit`, or return 0 (GEP 10)."
        )
    return None


def _inferred_result_error(
    qname: str,
    inferred: Any,  # noqa: ANN401
    declared: pint.Unit,
    detail: str,
    unit_system: UnitSystem,
) -> str | None:
    """Check one inferred result against the declaration.

    An opaque return — a dataclass, a tuple, … — is neither a checkable quantity
    nor a plain scalar, so it must opt out. A concrete ``Quantity`` must match the
    declaration on two axes, checked separately:

    - its **physical content** (currency, period, area, …) — the unit with every
      grouping level divided out — must equal the declaration's; and
    - its **grouping levels** must equal the declared (resolved) unit's level
      signature *exactly* — every level with its exponent. A level-less inference
      under a declaration that spells a level fails, and so does the squared
      level of multiplying two group-owned quantities
      (``1/[fam] * CURRENCY/month/[fam]`` → ``…/[fam]**2``). The level is
      declared, not read off the suffix (GEP 10); a body whose arithmetic cannot
      produce the declared levels — an intensive group property computed from
      level-less material, a policy-mandated cross-level product — states the
      intended unit with ``cast_ttsim_unit`` at the site.

    A plain scalar result takes its own screen
    (:func:`_bare_literal_result_error`): ``return 0.0`` stays lenient, while a
    non-zero literal return is rejected. A dimensionless quantity is checked
    like every other inferred quantity and therefore cannot satisfy a
    dimensioned declaration.
    """
    registry = unit_system.registry
    if not isinstance(
        inferred, pint.Quantity | int | float | numpy.number | numpy.bool_
    ):
        return _non_quantity_result_error(qname=qname, inferred=inferred)
    if not isinstance(inferred, pint.Quantity):
        return _bare_literal_result_error(
            qname=qname,
            inferred=inferred,
            declared=declared,
            detail=detail,
            registry=registry,
        )
    inferred_unit = cast("pint.Unit", inferred.units)
    if not units_are_equivalent(
        left=_unit_without_grouping_levels(unit=inferred_unit, registry=registry),
        right=_unit_without_grouping_levels(unit=declared, registry=registry),
        registry=registry,
    ):
        return (
            f"{qname}: declares '{declared}' but its body infers "
            f"'{inferred_unit}'{detail}."
        )
    inferred_levels = dict(_grouping_levels_with_exponent(inferred_unit))
    declared_levels = dict(_grouping_levels_with_exponent(declared))
    if inferred_levels != declared_levels:
        return (
            f"{qname}: its body infers '{inferred_unit}'{detail}, whose grouping "
            f"levels do not match its declaration's resolution '{declared}'. A "
            f"quantity carries a group level iff it is a property of the group "
            f"as a whole; where the mismatch is deliberate, state the intended "
            f"unit at the site with `cast_ttsim_unit` (GEP 10)."
        )
    return None
