"""Whole-environment unit checks.

Builds on the per-function engine in :mod:`ttsim.tt.units` to verify a fully
assembled policy environment on two counts:

- **Every active node declares a unit.** Author declarations (``unit=`` /
  ``unit:``) cover inputs, functions, parameters, and param functions; derived
  nodes (aggregations, time-conversion variants, group ids) and the framework
  date nodes get theirs assigned here. Identifiers and booleans are
  ``DIMENSIONLESS``.
- **Each function body agrees with its declaration.** Every ``@policy_function``
  / ``@param_function`` body is dry-run on representative quantities built from
  its producers' resolved units. A body that cannot be evaluated symbolically
  must opt out with ``verify_units=False``, so every un-verified body is visible.
  A single deliberate irregularity — policy-mandated cross-level arithmetic, a
  dimensioned constant — re-tags itself with ``cast_unit`` instead, keeping the
  rest of the body checked.

The dry-run wraps each ``Quantity(1.0, unit)`` in a :class:`_DryRunQuantity`,
whose arithmetic propagates units while a :class:`_PathExplorer` drives its
branch decisions — so a body is checked down every reachable path. ``+``, ``-``
and the ordering comparisons additionally require equivalent operands, because
at run time there is no pint to convert between them. pint runs only at
build time; no live array is ever wrapped.
"""

from __future__ import annotations

import dataclasses
import inspect
import re
import sys
from collections.abc import Mapping
from typing import Any, NoReturn, cast, get_args, get_type_hints

import dags.tree as dt
import numpy
import pint
from dags import get_annotations

from ttsim.exceptions import TTSIMError, UnitConsistencyError, UnitDefinitionError
from ttsim.interface_dag_elements.interface_node_objects import interface_function
from ttsim.interface_dag_elements.shared import (
    FRAMEWORK_PARTIAL_ARGUMENTS,
    get_re_pattern_for_all_time_units_and_groupings,
)
from ttsim.tt.aggregation import AggType
from ttsim.tt.column_objects_param_function import (
    AggByGroupFunction,
    ColumnFunction,
    ColumnObject,
    ParamFunction,
    PolicyFunction,
    PolicyInput,
)
from ttsim.tt.param_objects import (
    DictParam,
    ParamMappingObject,
    ParamObject,
    RawParam,
    ScalarParam,
)
from ttsim.tt.type_resolution import (
    ResolvedKind,
    TypeResolutionError,
    resolve_kind_of_annotation,
    resolve_kind_of_column_function,
)
from ttsim.tt.units import (
    _GROUPING_LEVEL_PREFIX,
    _QNAME_TIME_SUFFIX_PATTERN,
    PERSON_LEVEL,
    UNIT_REGISTRY,
    UNSET_UNIT,
    CompositeUnit,
    UnitAnnotatedColumn,
    _grouping_levels_with_exponent,
    _unit_level_denominator,
    _unit_without_grouping_levels,
    coerce_unit_token,
    fail_if_units_are_missing,
    is_calendar_point_unit,
    parse_unit,
    register_grouping_levels,
    registered_base_currencies,
    resolve_compositional_cast_unit,
    resolve_compositional_column_unit,
    resolve_compositional_field_unit,
    resolve_compositional_param_unit,
    resolved_unit_for_aggregation,
    token_is_agnostic_currency,
    token_source_currency,
    unit_for_derived_node,
    unit_residual_excluding_currency_and_flow_period,
    units_are_equivalent,
)
from ttsim.tt.vectorization import recompile_with_logical_ops_as_calls
from ttsim.typing import (
    OrderedQNames,
    SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
)
from ttsim.unit_converters import TIME_UNIT_IDS_TO_LABELS

#: Units of the date nodes the framework injects into every policy environment
#: (see ``policy_environment.policy_environment``); their units live here rather
#: than in downstream annotations. The *year* nodes are calendar **points** —
#: they run without bound, so ``policy_year - geburtsjahr`` is a duration in
#: years and adding two points is rejected. The month/day nodes carry a
#: month-of-year (1-12) / day-of-month (1-31): **cyclic ordinals** that wrap and
#: pin nothing on a running calendar, hence ``DIMENSIONLESS`` (GEP 10) — exactly
#: like ``geburtsmonat``, so ``policy_month >= geburtsmonat`` screens as plain
#: dimensionless arithmetic.
FRAMEWORK_DATE_NODE_UNITS: Mapping[str, str] = {
    "policy_year": "calendar_year",
    "policy_month": "dimensionless",
    "policy_day": "dimensionless",
    "evaluation_year": "calendar_year",
    "evaluation_month": "dimensionless",
    "evaluation_day": "dimensionless",
}

#: Arguments of column/param functions that the framework partials in and that
#: never carry a unit, shared with `specialized_environment` so the two cannot
#: drift.
NON_UNIT_ARGUMENTS = FRAMEWORK_PARTIAL_ARGUMENTS


class _DryRunXnp:
    """The dry-run's ``xnp``: NumPy, but with the unit-bearing ops routed through
    ``_DryRunQuantity``'s checks so a vectorized (``not_required``) body is checked
    at full parity with a scalar one.

    ``_DryRunQuantity`` sets ``__array_ufunc__ = None`` to force ``+``/``-``/… onto
    its checking dunders, which also stops NumPy ufuncs (``numpy.maximum`` …) from
    running. So the array ops a body actually calls are intercepted here and routed
    to the *same* checking primitives the operators use — ``maximum``/``minimum``
    screen like an ordering comparison (the scalar ``max``/``min`` the vectorizer
    rewrote), ``where`` like ``+`` (its two branches become one column), reductions
    and unary shape ops preserve the unit. An op not modelled here falls through to
    raw NumPy, raises, and is reported as needing ``verify_units=False`` — never
    silently passed through.
    """

    @staticmethod
    def logical_and(left: Any, right: Any) -> Any:  # noqa: ANN401
        return left & right

    @staticmethod
    def logical_or(left: Any, right: Any) -> Any:  # noqa: ANN401
        return left | right

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


#: Representative values for the framework arguments in a dry-run. The
#: dry-run always executes in NumPy + pint (NEP 18), regardless of the
#: backend of the actual run.
_NON_UNIT_ARGUMENT_VALUES: Mapping[str, Any] = {
    "xnp": _DryRunXnp(),
    "dnp": numpy,
    "backend": "numpy",
    "num_segments": 1,
    "len_p_id": 1,
}

_BOOL_KINDS = frozenset({ResolvedKind.BOOL_SCALAR, ResolvedKind.BOOL_COLUMN})

#: The logical operators a dry-run screens for boolean (dimensionless) operands.
_LOGICAL_OPS = frozenset({"&", "|", "^", "~"})

#: The dimensionless unit, used when reporting a logical op's bare operand.
_DIMENSIONLESS_UNIT: pint.Unit = cast(
    "pint.Unit", UNIT_REGISTRY.Quantity(1.0, "").units
)


@interface_function()
def resolved_units(
    specialized_environment__without_tree_logic_and_with_derived_functions: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,  # noqa: E501
    labels__grouping_levels: OrderedQNames,
) -> dict[str, pint.Unit | dict[str | int, Any]]:
    """The resolved pint unit of every annotated node in the environment.

    A single interface-DAG node so the environment walk runs once per build: the
    unit checks (TT bodies, input tags) and the unit-annotated output tree all
    consume this rather than each recomputing it. Thin wrapper over
    :func:`resolve_environment_units`.
    """
    return resolve_environment_units(
        env=specialized_environment__without_tree_logic_and_with_derived_functions,
        grouping_levels=labels__grouping_levels,
    )


def resolve_environment_units(
    env: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
    grouping_levels: OrderedQNames,
) -> dict[str, pint.Unit | dict[str | int, Any]]:
    """Resolve the complete unit of every annotated node in the environment.

    Columns and param functions resolve their fully-spelled compositional
    ``unit`` against the time-unit and aggregation suffixes of their leaf name;
    parameters spell their period and level in the unit string.
    Dict parameters with per-leaf units resolve to nested dicts of pint units. The
    framework-injected date nodes resolve via
    :data:`FRAMEWORK_DATE_NODE_UNITS`. A node that declares no unit (still
    :data:`UNSET_UNIT`) is absent from the result; the mandatory-units check
    reports it.
    """
    register_grouping_levels(grouping_levels)
    pattern = get_re_pattern_for_all_time_units_and_groupings(
        time_units=tuple(TIME_UNIT_IDS_TO_LABELS),
        grouping_levels=grouping_levels,
    )
    resolved: dict[str, pint.Unit | dict[str | int, Any]] = {
        # `parse_unit` guides declarations to the DIMENSIONLESS token, so the
        # framework-internal ordinal spelling resolves directly.
        qname: (_DIMENSIONLESS_UNIT if unit == "dimensionless" else parse_unit(unit))
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
                qname=qname, obj=obj, name_time_unit_id=name_time_unit_id
            )
            if param_unit is not None:
                resolved[qname] = param_unit
        elif isinstance(obj, AggByGroupFunction):
            agg_unit = _resolve_agg_by_group_unit(
                qname=qname, obj=obj, env=env, pattern=pattern
            )
            if agg_unit is not None:
                resolved[qname] = agg_unit
        else:  # ColumnObject | ParamFunction
            token = getattr(obj, "unit", UNSET_UNIT)
            if token is not UNSET_UNIT:
                leaf_name = dt.tree_path_from_qname(qname)[-1]
                match = pattern.fullmatch(leaf_name)
                resolved[qname] = _resolve_leveled_column_unit(
                    token=cast("CompositeUnit", token),
                    match=match,
                    is_boolean=node_is_boolean(qname=qname, obj=obj),
                )
    return resolved


def _resolve_leveled_column_unit(
    token: CompositeUnit,
    match: re.Match[str] | None,
    *,
    is_boolean: bool = False,
) -> pint.Unit:
    """Resolve a column/function's full unit, including its grouping level.

    Both booleans and ordinary columns resolve via
    :func:`resolve_compositional_column_unit`, which validates the spelled
    period/level against the name suffix and resolves an omitted group level to
    the person grain — the implied person leaf for a level-carrying base or a
    boolean, bare for an intensive one. A **boolean** is leveled (``is_boolean``):
    it is ``DIMENSIONLESS`` per the level it is defined at — ``1 / [fam]`` for a
    fam-level indicator (spelled ``DIMENSIONLESS_PER_FAM``), ``1 / [person]`` for
    a person-level one (bare ``DIMENSIONLESS``, person implied).
    """
    time_unit_id = match.group("time_unit") if match else None
    grouping_level = _suffix_grouping_level(match)
    return resolve_compositional_column_unit(
        token,
        time_unit_id=time_unit_id,
        grouping_level=grouping_level,
        where="A column/function",
        is_boolean=is_boolean,
    )


def _argument_is_person_pointer(qname: str) -> bool:
    """Whether an aggregation argument is a ``p_id_*`` person pointer.

    An ``agg_by_p_id`` aggregation becomes an :class:`AggByGroupFunction` once tree
    logic is removed, carrying its foreign-key pointer (``p_id_recipient``, …) as a
    plain argument. The pointer is not a value source — it selects *where* the sum
    lands — so it must be excluded when finding the single source column, exactly
    as the ``@agg_by_p_id_function`` constructor does.
    """
    return any(e.startswith("p_id_") for e in dt.tree_path_from_qname(qname))


def _resolve_agg_by_group_unit(
    qname: str,
    obj: AggByGroupFunction,
    env: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
    pattern: re.Pattern[str],
) -> pint.Unit | None:
    """Resolve a group-aggregation node's unit, level-aware.

    The aggregation is where a grouping level is minted, swapped, or preserved.
    The *target* level is the node's own aggregation suffix (an ``_hh`` node
    aggregates to ``[hh]``). The rule depends on
    :attr:`AggByGroupFunction.agg_type`:

    - a **head count** — ``COUNT``, or a ``SUM`` over a *boolean* source (counting
      the persons the indicator is true for) — mints ``[person]/[target]``;
    - ``SUM`` / ``MIN`` / ``MAX`` resolve to the **target** level whatever the
      source (a level-less source acquires it); ``MEAN`` resolves to the
      **individual** level — a per-head average belongs to the person (GEP 10);
    - ``ANY`` / ``ALL`` yield a dimensionless boolean at the target level.

    The value source is the function's own summed/averaged argument — read off the
    signature, not by stripping the name suffix, so a hand-written aggregation
    (``number_of_adults_fam`` sums ``adult``, not ``number_of_adults``) resolves
    correctly. Returns ``None`` if a value source carries no resolvable unit — the
    mandatory-units check reports the source.
    """
    target_level = _suffix_grouping_level(
        pattern.fullmatch(dt.tree_path_from_qname(qname)[-1])
    )
    agg_type = obj.agg_type
    # COUNT and ANY/ALL are independent of the source's unit, so resolve them
    # before touching the source: well-defined even when the source declares none.
    if agg_type in (AggType.COUNT, AggType.ANY, AggType.ALL):
        return resolved_unit_for_aggregation(
            agg_type=agg_type, target_level=target_level
        )
    sources = {
        p
        for p in inspect.signature(obj.function).parameters
        if not p.endswith("_id")
        and not _argument_is_person_pointer(p)
        and p not in NON_UNIT_ARGUMENTS
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
    # same unit a COUNT mints, so resolve it as one.
    if agg_type is AggType.SUM and source_is_boolean:
        return resolved_unit_for_aggregation(
            agg_type=AggType.COUNT, target_level=target_level
        )
    source_match = pattern.fullmatch(dt.tree_path_from_qname(source_qname)[-1])
    source_unit = _resolve_leveled_column_unit(
        token=cast("CompositeUnit", source_token),
        match=source_match,
        is_boolean=source_is_boolean,
    )
    # Read the source's level off its *resolved* unit, not its declared token: the
    # token may leave a person leaf implicit that the resolved denominator spells.
    source_level = _unit_level_denominator(source_unit)
    return resolved_unit_for_aggregation(
        source_unit=source_unit,
        agg_type=agg_type,
        target_level=target_level,
        source_level=source_level,
    )


def _suffix_grouping_level(match: re.Match[str] | None) -> str:
    """The grouping level named by a name's aggregation suffix.

    The combined time+grouping regex captures the aggregation suffix in its
    ``grouping`` group (``betrag_m_hh`` → ``"hh"``); an unsuffixed name has no
    such group and is at the individual leaf level :data:`PERSON_LEVEL`.
    """
    if match is None:
        return PERSON_LEVEL
    return match.group("grouping") or PERSON_LEVEL


def _has_grouping_level_numerator(unit: pint.Unit) -> bool:
    """Whether a unit carries a grouping level as a *numerator* — a head count."""
    return any(exponent > 0 for _, exponent in _grouping_levels_with_exponent(unit))


def _boolean_level(unit: pint.Unit) -> tuple[bool, str | None]:
    """Classify a unit as a (possibly leveled) boolean and read its level.

    A boolean is a truth value: dimensionless apart from at most a single grouping
    level it is measured *per* — ``1 / [fam]`` for a fam-level indicator, plain
    dimensionless for a level-less share/flag. A unit with physical content
    (currency, area, a duration) or a grouping-level *numerator* (a head count
    ``[person] / [hh]``) is *not* a boolean. Returns ``(is_boolean, level)``; the
    level is ``None`` for a level-less boolean.

    ``1 / [fam]`` → ``(True, "fam")``; a plain ``1`` → ``(True, None)``;
    ``EUR_PER_MONTH`` or ``[person] / [hh]`` → ``(False, None)``.
    """
    if _has_grouping_level_numerator(unit):
        return (False, None)
    if not UNIT_REGISTRY.Quantity(
        1.0, _unit_without_grouping_levels(unit)
    ).dimensionless:
        return (False, None)
    return (True, _unit_level_denominator(unit))


def _boolean_quantity(level: str | None) -> pint.Quantity:
    """A representative boolean ``Quantity`` at ``level`` — ``1 / [level]``.

    ``_boolean_quantity("fam")`` is ``1 / [fam]`` (a fam-level indicator);
    ``_boolean_quantity(None)`` is a plain dimensionless ``1`` (a level-less flag).
    """
    truth = UNIT_REGISTRY.Quantity(1.0, "")
    if level is None:
        return truth
    return truth / UNIT_REGISTRY.Quantity(1.0, f"{_GROUPING_LEVEL_PREFIX}{level}")


def _combined_boolean_level(left: str | None, right: str | None) -> str | None:
    """Combine two boolean levels for a logical operator.

    Equal levels are kept; any mismatch downcasts to the individual
    :data:`PERSON_LEVEL`. The downcast is sound and conservative: grouping levels
    do not nest, and a cross-level logical combination is evaluated per person
    (each person sees its groups' indicators), so the result is person-level.

    Two fam-level indicators give ``"fam"``; the mixed
    ``wealth_fam >= threshold_fam or wealth_kin >= threshold_kin`` combines a fam-
    and a kin-level operand, so the result is :data:`PERSON_LEVEL`.
    """
    return left if left == right else PERSON_LEVEL


def fail_if_environment_units_are_missing(
    env: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
    grouping_levels: OrderedQNames,  # noqa: ARG001  (kept for symmetry of the two checks)
) -> None:
    """Mandatory-units check over a fully assembled environment.

    Every active node must declare a unit — where ``unit=Unit.DIMENSIONLESS``
    / ``unit: DIMENSIONLESS`` *is* a declaration (a dimensionless quantity).
    For a dict or require_converter parameter with per-leaf units, every leaf
    of the value active at the policy date must be covered. A ``@param_function``
    declaring ``unit=UNSET_UNIT`` is exempt: its output is a structured value,
    not a quantity, and the decorator requires the argument, so the sentinel is
    never an omission (GEP 10).

    A rounding spec on a currency-valued function must declare its own unit:
    its magnitudes are statutory numbers written in a concrete currency,
    exactly like a parameter's (GEP 10). A missing one is reported as
    ``<qname> (rounding_spec)``.

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
        if isinstance(obj, ParamFunction) and declared_unit is UNSET_UNIT:
            continue
        if isinstance(obj, ParamMappingObject):
            units_by_qname[f"{qname} (input_unit)"] = cast(
                "CompositeUnit", obj.input_unit
            )
            units_by_qname[f"{qname} (output_unit)"] = cast(
                "CompositeUnit", obj.output_unit
            )
            continue
        if isinstance(obj, RawParam) and (
            obj.input_unit is not UNSET_UNIT or obj.output_unit is not UNSET_UNIT
        ):
            # A require_converter declares per-axis units instead of a single `unit:`.
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
                leaf_path = dt.tree_path_from_qname(leaf_qname)
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
                and token_is_agnostic_currency(cast("CompositeUnit", declared_unit))
            ):
                units_by_qname[f"{qname} (rounding_spec)"] = UNSET_UNIT
    fail_if_units_are_missing(units_by_qname)


def _agg_declaration_inconsistency(
    qname: str,
    obj: AggByGroupFunction,
    resolved_units: Mapping[str, pint.Unit | dict[str | int, Any]],
) -> str | None:
    """Error message if an aggregation's declared unit ≠ what it derives.

    The resolved unit (:func:`_resolve_agg_by_group_unit`) is the *derived* one —
    minted / swapped / preserved from the source and agg_type. A hand-written
    aggregation's declared unit must be **precise and complete**: it must equal the
    derived unit in full — physical kind, flow period, *and* grouping level — with
    no implicit matching of time units or group levels. The author spells the group
    level (``CURRENCY_PER_YEAR_PER_HH``, ``PERSON_COUNT_PER_BG``, even
    ``MONTHS_PER_FG``); only the ``[person]`` leaf is implied, never spelled. So a
    ``SUM`` over a boolean declared ``DIMENSIONLESS`` rather than
    ``PERSON_COUNT_PER_BG`` is rejected, and a ``_hh`` sum declared
    ``CURRENCY_PER_YEAR`` (omitting the level) or ``CURRENCY_PER_YEAR_PER_BG``
    (wrong level) is rejected too.

    A *purely derived* aggregation (a ``COUNT`` / ``ANY`` with no hand-written
    declaration) carries the derived unit directly and has nothing to check.
    Returns ``None`` when there is nothing to check: the derivation could not
    resolve the source, or no unit is declared (the mandatory-units check reports
    either).
    """
    derived = resolved_units.get(qname)
    declared_token = getattr(obj, "unit", UNSET_UNIT)
    if derived is None or isinstance(derived, dict) or declared_token is UNSET_UNIT:
        return None
    declared_unit = resolve_compositional_param_unit(
        cast("CompositeUnit", declared_token), where=f"Aggregation {qname!r}"
    )
    derived_unit = cast("pint.Unit", derived)
    if units_are_equivalent(left=declared_unit, right=derived_unit):
        return None
    return (
        f"{qname}: declares `{declared_token}` but its {obj.agg_type.name} "
        f"aggregation derives '{derived_unit}'. An aggregation's declared unit must "
        f"match what it produces exactly — physical kind, flow period, and grouping "
        f"level; spell the group level (the ``[person]`` leaf is implied), e.g. "
        f"`PERSON_COUNT_PER_<level>` for a count, the source's currency and period "
        f"for a sum of money (GEP 10)."
    )


def _aggregation_declaration_errors(
    env: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
    resolved_units: Mapping[str, pint.Unit | dict[str | int, Any]],
) -> list[str]:
    """Declared-vs-derived errors for every group aggregation."""
    return [
        error
        for qname, obj in env.items()
        if isinstance(obj, AggByGroupFunction)
        for error in [
            _agg_declaration_inconsistency(
                qname=qname, obj=obj, resolved_units=resolved_units
            )
        ]
        if error is not None
    ]


def _rounding_spec_declaration_inconsistency(
    qname: str,
    obj: ColumnFunction,
) -> str | None:
    """Error message if a rounding spec's unit disagrees with its function's.

    A rounding spec's magnitudes are statutory numbers written in a concrete
    currency, exactly like a parameter's — so on a currency-valued function the
    spec pins down a registered currency and spells the full composite, which
    must equal the function's declared unit with the agnostic base swapped for
    the concrete one. On a non-currency function the magnitudes are in the
    function's own unit and there is nothing to convert, so a declaration is
    rejected (GEP 10). The *missing* declaration on a currency-valued function
    is the mandatory-units check's to report, as is a function without a unit.
    """
    spec = getattr(obj, "rounding_spec", None)
    declared = getattr(obj, "unit", UNSET_UNIT)
    if spec is None or spec.unit is None or declared is UNSET_UNIT:
        return None
    if not token_is_agnostic_currency(cast("CompositeUnit", declared)):
        return (
            f"{qname}: the rounding spec declares `{spec.unit}` but the function's "
            f"unit `{declared}` has no currency base, so there is nothing to "
            f"convert; drop the spec's `unit=` (GEP 10)."
        )
    if token_is_agnostic_currency(spec.unit):
        return (
            f"{qname}: the rounding spec's magnitudes are written in a concrete "
            f"currency; declare it (e.g. `Unit.DM.PER_YEAR`), not the agnostic "
            f"`{spec.unit}` (GEP 10)."
        )
    if token_source_currency(spec.unit) is None:
        return (
            f"{qname}: the rounding spec's unit `{spec.unit}` does not pin down a "
            f"registered currency (GEP 10)."
        )
    if unit_for_derived_node(spec.unit) != declared:
        return (
            f"{qname}: the rounding spec's unit `{spec.unit}` must equal the "
            f"function's declared `{declared}` with the agnostic base swapped for "
            f"the concrete currency — same flow period, same grouping level "
            f"(GEP 10)."
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
        for error in [_rounding_spec_declaration_inconsistency(qname=qname, obj=obj)]
        if error is not None
    ]


def fail_if_environment_units_are_inconsistent(
    env: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
    grouping_levels: OrderedQNames,
    resolved_units: dict[str, pint.Unit | dict[str | int, Any]] | None = None,
) -> None:
    """Conservative body/edge verification over an assembled environment.

    Each ``@policy_function`` / ``@param_function`` body is dry-run on
    representative values built from its producers' resolved units (the DAG
    edges) — see the module docstring for the conservative rules and the
    boolean-enumeration strategy. An aggregation has no scalar body, but it
    *derives* a unit from its source and agg_type; its declared token is checked
    against that derivation here, the same declared-vs-produced contract a body is
    held to. A rounding spec's declared unit is checked against its function's
    (:func:`_rounding_spec_declaration_inconsistency`), and a converter of an
    axes-declaring ``require_converter`` blob against the axes contract
    (:func:`_axes_converter_contract_errors`). Time-conversion variants
    and group-creation functions are unit-assigned by construction and need no
    check.

    In the interface DAG the resolved units are supplied by the
    :func:`resolved_units` node, so the environment walk runs once per build
    regardless of how many checks consume it. ``resolved_units`` defaults to
    ``None`` purely for direct callers (tests), where it is computed on demand.

    Raises:
        UnitConsistencyError: If any body infers a concrete unit that disagrees
            with its declaration, or an aggregation's declared unit disagrees
            with what it derives. All offending nodes are reported together.
    """
    if resolved_units is None:
        resolved_units = resolve_environment_units(
            env=env, grouping_levels=grouping_levels
        )
    representative_values = _representative_values_by_qname(
        env=env, resolved_units=resolved_units
    )
    boolean_nodes = {
        qname
        for qname, obj in env.items()
        if isinstance(obj, ColumnObject | ParamFunction)
        and node_is_boolean(qname=qname, obj=obj)
    }
    errors: list[str] = _aggregation_declaration_errors(
        env=env, resolved_units=resolved_units
    )
    errors.extend(_rounding_spec_declaration_errors(env=env))
    errors.extend(_axes_converter_contract_errors(env=env))
    errors.extend(
        _structured_annotation_drift_errors(env=env, resolved_units=resolved_units)
    )
    for qname, obj in env.items():
        # Only these two have a human-written scalar body; everything else
        # (aggregations validated above, time-conversions, group ids) is assigned
        # by construction.
        if not isinstance(obj, PolicyFunction | ParamFunction):
            continue
        if qname not in resolved_units:
            # Still UNSET — the mandatory-units check reports it.
            continue
        if not obj.verify_units:
            # Body opted out of unit inference; its declared unit still stands as
            # the edge contract, so consumers are still checked.
            continue
        declared = resolved_units[qname]
        if isinstance(declared, dict):
            continue
        parameters = tuple(inspect.signature(obj.function).parameters)
        boolean_parameters = tuple(p for p in parameters if p in boolean_nodes)
        base_kwargs = _base_dry_run_kwargs(
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
            name: representative_values.get(name, UNIT_REGISTRY.Quantity(1.0, ""))
            for name in boolean_parameters
        }
        error = _verify_one_body(
            qname=qname,
            function=recompile_with_logical_ops_as_calls(
                obj.function,
                module="xnp",
                module_obj=_NON_UNIT_ARGUMENT_VALUES["xnp"],
                extra_globals=_DRY_RUN_HELPER_SHIMS,
            ),
            declared=declared,
            boolean_values=boolean_values,
            base_kwargs=base_kwargs,
        )
        if error is not None:
            errors.append(error)
    if errors:
        raise UnitConsistencyError(
            "Environment unit-consistency check failed:\n  " + "\n  ".join(errors)
        )


def fail_if_not_all_leaves_are_unit_annotated_columns(
    flat: Mapping[tuple[str, ...], Any],
) -> None:
    """Reject a unit-annotated input tree with any bare (untagged) leaf.

    Every leaf of the unit-annotated input tree must be a
    :class:`UnitAnnotatedColumn`. The producers that strip the tags
    (``input_data__flat`` / ``input_data__units``) assume this, so the
    ``not_all_input_leaves_are_unit_annotated_columns`` fail node — which the
    ``fail_if`` namespace orders ahead of them (see ``entry_point.lexsort_key``)
    — calls this first, turning a bare leaf into a clean error rather than an
    ``AttributeError`` when a producer reaches for ``.unit``.

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
            "with `UnitAnnotatedColumn(values=arr, unit=Unit.DIMENSIONLESS)`, or "
            "pass untagged data via input_data__tree."
        )


def fail_if_input_units_are_inconsistent(
    input_units: Mapping[str, pint.Unit],
    resolved_units: Mapping[str, Any],
) -> None:
    """Fail if a tagged input column is not equivalent to its declared unit.

    ``input_units`` maps each tagged input column to its pint unit tag;
    ``resolved_units`` maps every declared node to its resolved (agnostic) DAG
    unit. The tag must be *equivalent* to the declared unit once two axes the
    boundary handles separately are factored out: the currency (converted to the
    run currency at the boundary — a DM tag on a euro-run column passes) and the
    flow period (screened against the name suffix by the dedicated period guard).
    What remains — the numerator scale — must match *exactly*, not merely share a
    dimension: a ``HECTARES`` column tagged ``m²`` (a 10,000-fold level error) or
    a ``YEARS`` age tagged ``month`` is rejected here rather than silently
    mis-stripped at the boundary, while a currency tag on a ``YEARS`` column
    (different residual dimension) is rejected as before.

    Raises:
        UnitConsistencyError: If any tagged column is not equivalent to its
            declared unit. All offending columns are reported together.
    """
    errors: list[str] = []
    for qname, tag in input_units.items():
        expected = resolved_units.get(qname)
        if not isinstance(expected, pint.Unit):
            # No scalar declared unit (absent, or a dict parameter); nothing to check.
            continue
        tag_residual = unit_residual_excluding_currency_and_flow_period(tag)
        expected_residual = unit_residual_excluding_currency_and_flow_period(expected)
        if not units_are_equivalent(left=tag_residual, right=expected_residual):
            errors.append(
                f"  {qname}: tagged '{tag}', which is not equivalent to the declared "
                f"unit '{expected}' (the boundary converts currency and screens the "
                f"flow period against the name suffix, but the remaining magnitude "
                f"must match exactly)."
            )
    if errors:
        raise UnitConsistencyError(
            "Input unit annotations are inconsistent with the DAG's declared "
            "units:\n" + "\n".join(sorted(errors))
        )


def node_is_boolean(qname: str, obj: Any) -> bool:  # noqa: ANN401
    """Whether a node's output is boolean (used to build the dry-run's symbolic
    values and drive its branch exploration; orthogonal to the declared unit)."""
    try:
        if isinstance(obj, PolicyInput):
            kind = resolve_kind_of_annotation(obj.data_type, node_name=qname)
        elif isinstance(obj, ColumnObject | ParamFunction):
            kind = resolve_kind_of_column_function(obj.function, node_name=qname)
        else:
            return False
    except TypeResolutionError:
        return False
    return kind in _BOOL_KINDS


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


def _spell_token(token: Any) -> str:  # noqa: ANN401
    """Spell a declaration token for an error message."""
    if token is UNSET_UNIT:
        return "unset"
    return str(token)


def _fail_if_param_token_is_agnostic_currency(
    token: CompositeUnit | None,
    where: str,
) -> None:
    """Reject an agnostic currency unit on a parameter.

    Once a concrete currency is registered, a parameter's numbers are
    written in *some* currency — the declaration must name it
    (``SILVER_PENNY``, ``DM_PER_YEAR``, …), so the build-time conversion
    to the run currency knows what to convert from. The agnostic ``CURRENCY``
    base stays legal — and required — on columns and functions, which are
    currency-agnostic by design.
    """
    bases = registered_base_currencies()
    if bases and token_is_agnostic_currency(token):
        concrete = f"{bases[0].upper()}{str(token).removeprefix('CURRENCY')}"
        raise UnitDefinitionError(
            f"{where}: parameters must pin down the concrete currency their "
            f"numbers are written in; the agnostic unit {token} is not "
            f"allowed here. Declare e.g. {concrete} (GEP 10)."
        )


def _resolve_param_mapping_object_units(
    qname: str,
    obj: ParamMappingObject,
    name_time_unit_id: str | None,
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
        _fail_if_param_token_is_agnostic_currency(token=token, where=where)
        tokens[axis] = token
    output_token = tokens["output_unit"]
    if name_time_unit_id is not None:
        _fail_if_name_suffix_disagrees_with_output_axis(
            qname=qname,
            output_token=output_token,
            name_time_unit_id=name_time_unit_id,
        )
    input_token = tokens["input_unit"]
    if input_token is not UNSET_UNIT:
        resolve_compositional_param_unit(input_token, where=f"Parameter {qname!r}")
    if output_token is UNSET_UNIT:
        return None
    return resolve_compositional_param_unit(output_token, where=f"Parameter {qname!r}")


def _fail_if_name_suffix_disagrees_with_output_axis(
    qname: str,
    output_token: Any,  # noqa: ANN401
    name_time_unit_id: str,
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
            f"`output_unit:` is {_spell_token(output_token)} (GEP 10)."
        )
    resolve_compositional_param_unit(
        output_token, time_unit_id=name_time_unit_id, where=f"Parameter {qname!r}"
    )


def _resolve_param_object_unit(
    qname: str,
    obj: ParamObject,
    name_time_unit_id: str | None = None,
) -> pint.Unit | dict[str | int, Any] | None:
    """Resolve a parameter's declared compositional unit.

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
            qname=qname, obj=obj, name_time_unit_id=name_time_unit_id
        )
    if obj.unit is UNSET_UNIT:
        return None
    if isinstance(obj.unit, Mapping):
        return _resolve_unit_mapping(
            qname=qname, unit_mapping=cast("Mapping[str | int, Any]", obj.unit)
        )
    token = cast("CompositeUnit", obj.unit)
    _fail_if_param_token_is_agnostic_currency(token=token, where=f"Parameter {qname!r}")
    # A scalar parameter takes its period from a time suffix on its name; a
    # dict/raw parameter has no single name to suffix.
    return resolve_compositional_param_unit(
        token,
        time_unit_id=name_time_unit_id if isinstance(obj, ScalarParam) else None,
        where=f"Parameter {qname!r}",
    )


def _resolve_unit_mapping(
    qname: str,
    unit_mapping: Mapping[str | int, Any],
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
            resolved[key] = _resolve_unit_mapping(qname=qname, unit_mapping=token)
            continue
        where = f"Parameter {qname!r}, unit of leaf {key!r}"
        _fail_if_param_token_is_agnostic_currency(token=token, where=where)
        match = _QNAME_TIME_SUFFIX_PATTERN.search(str(key))
        suffix_id = match.group("time_unit") if match else None
        resolved[key] = resolve_compositional_param_unit(
            cast("CompositeUnit", token), time_unit_id=suffix_id, where=where
        )
    return resolved


def _representative_value(
    resolved_unit: pint.Unit | dict[str | int, Any],
) -> Any:  # noqa: ANN401
    """A representative dry-run value: ``Quantity(1.0, unit)``, or a dict thereof."""
    if isinstance(resolved_unit, dict):
        return {
            key: _representative_value(cast("pint.Unit | dict[str | int, Any]", unit))
            for key, unit in resolved_unit.items()
        }
    return UNIT_REGISTRY.Quantity(1.0, resolved_unit)


def _uniform_quantity_tree(value: Any, resolved_unit: pint.Unit) -> Any:  # noqa: ANN401
    """Mirror a dict param's value structure with uniform representative quantities."""
    if isinstance(value, Mapping):
        return {
            key: _uniform_quantity_tree(value=sub_value, resolved_unit=resolved_unit)
            for key, sub_value in value.items()
        }
    return UNIT_REGISTRY.Quantity(1.0, resolved_unit)


#: The unqualified return-annotation names the axes contract accepts: the two
#: schedule types whose typed output the per-axis conversion can restate.
_SCHEDULE_RETURN_TYPE_NAMES = frozenset(
    {"PiecewisePolynomialParamValue", "ConsecutiveIntLookupTableParamValue"}
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


def _axes_declaring_raw_dependencies(
    obj: ParamFunction,
    env: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
) -> list[tuple[str, RawParam]]:
    """The axes-declaring ``require_converter`` blobs a param function consumes."""
    out: list[tuple[str, RawParam]] = []
    for dep in sorted(obj.dependencies):
        raw = env.get(dep)
        if isinstance(raw, RawParam) and (
            raw.input_unit is not UNSET_UNIT or raw.output_unit is not UNSET_UNIT
        ):
            out.append((dep, raw))
    return out


def _resolved_raw_param_axis(
    token: Any,  # noqa: ANN401
    qname: str,
) -> pint.Unit | None:
    """Resolve one declared axis of an axes-declaring ``require_converter`` blob."""
    if token is UNSET_UNIT:
        return None
    where = f"Parameter {qname!r}"
    unit_token = cast("CompositeUnit", token)
    _fail_if_param_token_is_agnostic_currency(token=unit_token, where=where)
    return resolve_compositional_param_unit(unit_token, where=where)


def _resolved_return_dataclass(func: Any) -> type | None:  # noqa: ANN401
    """The dataclass a param function's return annotation names, or ``None``.

    Resolved by walking the (possibly dotted) annotation string through the
    function's module namespace — annotations are strings under
    ``from __future__ import annotations``, and the class must be importable
    at runtime for its field annotations to matter. Anything unresolvable
    simply yields ``None``: the output stays fully opaque and plucks are cast
    at the site (GEP 10).
    """
    annotation = get_annotations(func, default="").get("return", "")
    if isinstance(annotation, type):
        return annotation if dataclasses.is_dataclass(annotation) else None
    if not isinstance(annotation, str) or not annotation:
        return None
    obj: Any = sys.modules.get(getattr(func, "__module__", ""))
    for part in annotation.split("."):
        obj = getattr(obj, part, None)
    return obj if isinstance(obj, type) and dataclasses.is_dataclass(obj) else None


#: Per-class memo of :func:`_structured_field_kinds`. ``None`` records a class
#: whose annotations do not resolve at runtime. Field annotations are
#: currency-agnostic by rule, so a cached resolution never goes stale when a
#: currency registration is rolled back.
_STRUCTURED_FIELD_KINDS: dict[type, dict[str, pint.Unit | type] | None] = {}


def _structured_field_kinds(cls: type) -> dict[str, pint.Unit | type] | None:
    """Resolve a parameter dataclass's field annotations for the dry-run.

    Maps each field to what its pluck yields: the resolved unit of an
    ``Annotated[<scalar>, Unit…]`` field, or the class of a nested-dataclass
    field (whose plucks resolve recursively). Fields that are neither — a bare
    scalar, a dict, an array, a schedule value — are absent: their plucks stay
    opaque and are cast at the site (GEP 10). ``None`` when the annotations do
    not resolve at runtime (a name imported only under ``TYPE_CHECKING``), in
    which case every pluck stays opaque.

    Raises:
        UnitDefinitionError: If a field annotates several units, annotates a
            non-scalar field, or pins a concrete currency.
    """
    if cls in _STRUCTURED_FIELD_KINDS:
        return _STRUCTURED_FIELD_KINDS[cls]
    try:
        hints = get_type_hints(cls, include_extras=True)
    except NameError:
        _STRUCTURED_FIELD_KINDS[cls] = None
        return None
    kinds: dict[str, pint.Unit | type] = {}
    for field in dataclasses.fields(cls):
        hint = hints.get(field.name, field.type)
        tokens = [
            token
            for token in getattr(hint, "__metadata__", ())
            if isinstance(token, CompositeUnit)
        ]
        base = get_args(hint)[0] if hasattr(hint, "__metadata__") else hint
        where = f"Field '{cls.__name__}.{field.name}'"
        if len(tokens) > 1:
            raise UnitDefinitionError(
                f"{where}: annotates {len(tokens)} units "
                f"({', '.join(str(t) for t in tokens)}); a field states exactly "
                f"one (GEP 10)."
            )
        if tokens:
            if base not in (int, float, bool):
                raise UnitDefinitionError(
                    f"{where}: a unit annotation sits on a scalar field "
                    f"(int/float/bool); a structured or container field has no "
                    f"single unit — cast at the pluck instead (GEP 10)."
                )
            kinds[field.name] = resolve_compositional_field_unit(tokens[0], where=where)
        elif isinstance(base, type) and dataclasses.is_dataclass(base):
            kinds[field.name] = base
    _STRUCTURED_FIELD_KINDS[cls] = kinds
    return kinds


def _annotated_field_units(cls: type) -> dict[tuple[str, ...], pint.Unit]:
    """Flatten a parameter dataclass's annotated field units to field paths."""
    out: dict[tuple[str, ...], pint.Unit] = {}
    for name, resolved in (_structured_field_kinds(cls) or {}).items():
        if isinstance(resolved, pint.Unit):
            out[(name,)] = resolved
        else:
            for path, unit in _annotated_field_units(resolved).items():
                out[(name, *path)] = unit
    return out


def _flattened_unit_mapping(
    units: Mapping[str | int, Any],
) -> dict[tuple[str, ...], pint.Unit]:
    """Flatten a resolved per-leaf ``unit:`` mapping to string leaf paths."""
    out: dict[tuple[str, ...], pint.Unit] = {}
    for key, value in units.items():
        if isinstance(value, dict):
            for path, unit in _flattened_unit_mapping(value).items():
                out[(str(key), *path)] = unit
        else:
            out[(str(key),)] = value
    return out


def _structured_annotation_drift_errors(
    env: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
    resolved_units: Mapping[str, pint.Unit | dict[str | int, Any]],
) -> list[str]:
    """YAML-vs-annotation drift check for dataclass-building converters.

    The per-leaf ``unit:`` mapping of a parameter drives the numeric
    conversion of its raw value; the field annotations of the dataclass a
    converter builds from it drive the checking of consumer bodies. Where a
    mapping leaf's path coincides with a field's path — the converter kept the
    name — the two independent declarations describe the same number and must
    agree; a drift would convert the number one way and check it another,
    silently. A renamed or derived field has no matching leaf and is not
    checked here (GEP 10).
    """
    errors: list[str] = []
    for qname, obj in env.items():
        if not isinstance(obj, ParamFunction) or obj.unit is not UNSET_UNIT:
            continue
        cls = _resolved_return_dataclass(obj.function)
        if cls is None:
            continue
        field_units = _annotated_field_units(cls)
        if not field_units:
            continue
        for dep in sorted(obj.dependencies):
            declared = resolved_units.get(dep)
            if not isinstance(env.get(dep), ParamObject) or not isinstance(
                declared, dict
            ):
                continue
            leaf_units = _flattened_unit_mapping(
                cast("Mapping[str | int, Any]", declared)
            )
            for path, field_unit in sorted(field_units.items()):
                leaf_unit = leaf_units.get(path)
                if leaf_unit is None or units_are_equivalent(
                    left=leaf_unit, right=field_unit
                ):
                    continue
                errors.append(
                    f"{qname}: the field '{cls.__name__}.{'.'.join(path)}' is "
                    f"annotated '{field_unit}' but parameter '{dep}' declares "
                    f"'{leaf_unit}' for the same leaf — the declaration "
                    f"converts the number, the annotation checks its uses, so "
                    f"the two must state the same unit (GEP 10)."
                )
    return errors


def _param_function_stand_in(
    qname: str,
    obj: ParamFunction,
    env: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
) -> _DryRunSchedule | _DryRunStructuredValue:
    """The dry-run stand-in for a structured param-function output (GEP 10).

    A converter of a single axes-declaring ``require_converter`` blob that is
    annotated as returning a schedule type gets a :class:`_DryRunSchedule`
    carrying the blob's axes: the per-axis currency conversion already assumes
    the declared axes describe the typed output, so consumers screen against
    them exactly as for a parameter-declared schedule — no cast at the call.
    Everything else stays a :class:`_DryRunStructuredValue`, typed with the
    return dataclass where one resolves so that annotated plucks carry their
    field units; a converter that breaks the axes contract is reported by
    :func:`_axes_converter_contract_errors`.
    """
    axes_deps = _axes_declaring_raw_dependencies(obj=obj, env=env)
    if (
        len(axes_deps) == 1
        and _return_annotation_name(obj.function) in _SCHEDULE_RETURN_TYPE_NAMES
    ):
        raw_qname, raw = axes_deps[0]
        output_unit = _resolved_raw_param_axis(raw.output_unit, qname=raw_qname)
        if output_unit is not None:
            return _DryRunSchedule(
                input_unit=_resolved_raw_param_axis(raw.input_unit, qname=raw_qname),
                output_unit=output_unit,
            )
    return _DryRunStructuredValue(
        producer=qname, cls=_resolved_return_dataclass(obj.function)
    )


def _axes_converter_contract_errors(
    env: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
) -> list[str]:
    """The axes contract: declared axes must reach a schedule-typed output.

    A ``require_converter`` declaring ``input_unit:``/``output_unit:`` axes
    promises that its converters build a schedule, which the framework
    converts per axis and screens call sites against. A converter that is not
    annotated as returning one of the two schedule types, that declares a
    quantity unit, or that mixes several axes-declaring blobs (the per-axis
    conversion would scale its output once per blob) breaks that promise — at
    build time, independent of whether a currency conversion is active in this
    run (GEP 10).
    """
    errors: list[str] = []
    for qname, obj in env.items():
        if not isinstance(obj, ParamFunction):
            continue
        axes_deps = _axes_declaring_raw_dependencies(obj=obj, env=env)
        if not axes_deps:
            continue
        names = ", ".join(f"'{dep}'" for dep, _ in axes_deps)
        if len(axes_deps) > 1:
            errors.append(
                f"{qname}: consumes {len(axes_deps)} axes-declaring "
                f"require_converter parameters ({names}); the per-axis "
                f"conversion of a converter's typed output is defined against "
                f"exactly one (GEP 10)."
            )
        elif obj.unit is not UNSET_UNIT:
            errors.append(
                f"{qname}: consumes the axes-declaring require_converter "
                f"parameter {names} but declares a quantity unit; a converter "
                f"of an axes-declaring blob builds a schedule, a structured "
                f"value — declare `unit=UNSET_UNIT` (GEP 10)."
            )
        elif _return_annotation_name(obj.function) not in _SCHEDULE_RETURN_TYPE_NAMES:
            errors.append(
                f"{qname}: consumes the axes-declaring require_converter "
                f"parameter {names}, so it must be annotated as returning a "
                f"PiecewisePolynomialParamValue or a "
                f"ConsecutiveIntLookupTableParamValue — the declared axes "
                f"describe that typed output, and its call sites are screened "
                f"against them (GEP 10)."
            )
    return errors


def _resolve_schedule_input_unit(obj: ParamMappingObject) -> pint.Unit | None:
    """The resolved ``input_unit`` of a schedule/lookup parameter, or ``None``.

    Resolved the same way as the ``output_unit`` the environment exposes, so a
    concrete-currency input axis and an agnostic ``CURRENCY`` consumer argument
    compare as equivalent. ``None`` when the parameter left ``input_unit`` unset.
    """
    if obj.input_unit is UNSET_UNIT:
        return None
    return resolve_compositional_param_unit(
        cast("CompositeUnit", obj.input_unit), where="A schedule input axis"
    )


def _representative_values_by_qname(
    env: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
    resolved_units: Mapping[str, pint.Unit | dict[str | int, Any]],
) -> dict[str, Any]:
    """Representative dry-run values for every unit-resolved node.

    A ``piecewise_*``/lookup-table parameter becomes a :class:`_DryRunSchedule`
    carrying its input/output axes, so a consumer's ``piecewise_polynomial`` /
    ``look_up`` call resolves to the output unit. A dict parameter with a scalar
    ``unit:`` declaration becomes a dict of uniform representative quantities
    mirroring its value structure, so that subscripting works inside a consumer's
    dry-run. A structured param function (``unit=UNSET_UNIT``) becomes its
    :func:`_param_function_stand_in` — a schedule where a ``require_converter``
    blob declares axes for it, a :class:`_DryRunStructuredValue` otherwise.
    """
    out: dict[str, Any] = {}
    for qname, unit in resolved_units.items():
        obj = env.get(qname)
        if isinstance(obj, ParamMappingObject) and not isinstance(unit, dict):
            out[qname] = _DryRunSchedule(
                input_unit=_resolve_schedule_input_unit(obj),
                output_unit=cast("pint.Unit", unit),
            )
        elif isinstance(obj, DictParam | RawParam) and not isinstance(unit, dict):
            out[qname] = _uniform_quantity_tree(
                value=obj.value, resolved_unit=cast("pint.Unit", unit)
            )
        else:
            out[qname] = _representative_value(unit)
    for qname, obj in env.items():
        if isinstance(obj, ParamFunction) and obj.unit is UNSET_UNIT:
            out[qname] = _param_function_stand_in(qname=qname, obj=obj, env=env)
    return out


def _base_dry_run_kwargs(
    parameters: tuple[str, ...],
    boolean_parameters: tuple[str, ...],
    representative_values: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Representative kwargs for a body's non-boolean parameters.

    Returns ``None`` if any parameter has no representative value (an
    unannotated producer): the body cannot be dry-run and its declared
    unit is the fallback.
    """
    out: dict[str, Any] = {}
    for parameter in parameters:
        if parameter in boolean_parameters:
            continue
        if parameter in NON_UNIT_ARGUMENTS:
            out[parameter] = _NON_UNIT_ARGUMENT_VALUES[parameter]
        elif parameter in representative_values:
            out[parameter] = representative_values[parameter]
        else:
            return None
    return out


# Caps on the path-exploring dry-run (see ``_PathExplorer``): only a pathological
# body (deep independent branching, or a data-driven loop) hits them, so the build
# check can never blow up.
_MAX_PATHS = 1024
_MAX_DECISIONS_PER_RUN = 64

#: How many of a failing run's branch decisions an error message spells out.
_MAX_NAMED_DECISIONS = 4


class _PathBudgetExceededError(TTSIMError):
    """A single dry-run made too many branch decisions (likely a loop)."""


class _UnitMixError(TTSIMError):
    """A body combined two non-equivalent unit-carrying operands.

    ``+``, ``-`` and the ordering comparisons are *unit-blind at run time*:
    once the policy environment is built there is no pint, so a body that does
    ``monthly_flow + yearly_flow`` (or compares them) just adds or compares the
    bare arrays — no conversion happens. The two operands must therefore already
    be in equivalent units (same dimension *and* period). pint would mask such a
    bug during the dry-run — silently auto-converting a same-dimension mismatch
    (``CURRENCY / month + CURRENCY / year``) or raising a swallowed
    ``DimensionalityError`` on a cross-dimension one — so ``_DryRunQuantity`` checks
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
    (an unreached branch never asks). This subsumes the former boolean
    enumeration (a boolean input is just another decision) and additionally
    reaches numeric-driven branches (``if income > limit``), which a single
    representative value would silently fix to one arm.
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
        body's own terms (:meth:`branch_detail`); ``None`` where the dry-run has
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
    return value.q if isinstance(value, _DryRunQuantity) else value


class _DryRunQuantity:
    """A pint ``Quantity`` wrapped so branch decisions route to a ``_PathExplorer``.

    Arithmetic forwards to the wrapped quantity, so units propagate exactly as
    in a real run (the whole point of the check). Comparisons and truth tests
    instead return an explorer-controlled value, so the explorer — not the
    representative magnitude — decides which branch is taken; the magnitude is
    always ``1.0`` and never matters. Anything the wrapper cannot model raises,
    which the caller treats as "not dry-runnable on this path" and falls back to
    the declaration — so the wrapper can never produce a false positive.
    """

    __slots__ = ("_explorer", "_label", "q")
    # Keep NumPy from broadcasting over us: defer binary ops with a NumPy operand
    # to our reflected dunders instead.
    __array_ufunc__ = None
    __array_priority__ = 1000
    __hash__ = object.__hash__

    def __init__(
        self,
        q: Any,  # noqa: ANN401
        explorer: _PathExplorer,
        label: str | None = None,
    ) -> None:
        self.q = q
        self._explorer = explorer
        # How the body's author would name this value — the argument name for a
        # direct input, a composed description for a comparison or logical
        # combination, ``None`` once arithmetic has mixed it beyond naming. Used
        # to report the branch a failure sits on (`_PathExplorer.branch_detail`).
        self._label = label

    def _wrap(self, q: Any) -> _DryRunQuantity:  # noqa: ANN401
        return _DryRunQuantity(q=q, explorer=self._explorer)

    def _controlled_bool_at(
        self, level: str | None, label: str | None = None
    ) -> _DryRunQuantity:
        return _DryRunQuantity(
            q=_boolean_quantity(level), explorer=self._explorer, label=label
        )

    def _composed_label(self, other: Any, op: str) -> str | None:  # noqa: ANN401
        """Describe ``self <op> other`` for branch naming, if either side has
        a name; a bare literal operand shows as itself."""
        if isinstance(other, _DryRunQuantity):
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
        comparison the two are equivalent, so they agree.
        """
        level = _unit_level_denominator(cast("pint.Unit", self.q.units))
        if level is not None:
            return level
        other_q = _unwrap(other)
        if isinstance(other_q, pint.Quantity):
            return _unit_level_denominator(cast("pint.Unit", other_q.units))
        return None

    def _logical_result(self, other: Any, op: str) -> _DryRunQuantity:  # noqa: ANN401
        """Combine two booleans under a logical operator ``&``/``|``/``^``.

        Logical operators combine truth values. Each operand must be a (possibly
        leveled) boolean — a non-dimensionless operand carrying physical content
        (``wealth & is_adult``) or a head count is a mistake the run-time arrays
        would silently swallow. The result is a boolean whose level follows the
        combine rule (:func:`_combined_boolean_level`): equal levels are kept, a
        mismatch downcasts to the per-person level. A bare literal carries no unit
        and stays a lenient, level-less boolean.
        """
        self_is_boolean, self_level = _boolean_level(cast("pint.Unit", self.q.units))
        other_q = _unwrap(other)
        if isinstance(other_q, _DryRunStructuredValue):
            other_q._raise_used_as_quantity(op)  # noqa: SLF001
        if isinstance(other_q, pint.Quantity):
            other_is_boolean, other_level = _boolean_level(
                cast("pint.Unit", other_q.units)
            )
        else:
            other_is_boolean, other_level = True, None
        if not self_is_boolean or not other_is_boolean:
            right = (
                cast("pint.Unit", other_q.units)
                if isinstance(other_q, pint.Quantity)
                else _DIMENSIONLESS_UNIT
            )
            raise _UnitMixError(
                op=op, left=cast("pint.Unit", self.q.units), right=right
            )
        return self._controlled_bool_at(
            _combined_boolean_level(self_level, other_level),
            label=self._composed_label(other, op),
        )

    def _fail_if_additive_operand_is_invalid(self, other: Any, op: str) -> None:  # noqa: ANN401
        """Screen an operand of ``+``/``-``.

        The rules are those of :meth:`_fail_if_other_unit_is_not_equivalent`,
        with one dispensation: a calendar point (an affine offset unit). Its
        valid ``point ± duration`` is *not* equivalence (a point and a duration
        differ), yet pint's offset algebra permits exactly it and forbids the
        genuine misuses (``point + point``, cross-axis mixes). So when either
        operand is a calendar point the magnitude pre-screen is skipped and the
        forward operation delegates to pint, which raises
        ``OffsetUnitCalculusError`` on a misuse — caught in
        :func:`_verify_one_body` and reported as a calendar misuse. Only
        ``+``/``-`` get the dispensation: they alone run a forward pint
        operation afterwards, so nothing would catch a point mixed into an
        ordering or a ``where`` later.
        """
        other_q = _unwrap(other)
        if isinstance(other_q, _DryRunStructuredValue):
            other_q._raise_used_as_quantity(op)  # noqa: SLF001
        if is_calendar_point_unit(cast("pint.Unit", self.q.units)) or (
            isinstance(other_q, pint.Quantity)
            and is_calendar_point_unit(cast("pint.Unit", other_q.units))
        ):
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
        ``cast_unit``. Only ``0`` (the ``x + 0.0`` guard, the floor at zero) is
        allowed inline, and literals next to a dimensionless quantity stay
        lenient.
        """
        other_q = _unwrap(other)
        if isinstance(other_q, _DryRunStructuredValue):
            other_q._raise_used_as_quantity(op)  # noqa: SLF001
        if isinstance(other_q, pint.Quantity) and not units_are_equivalent(
            left=cast("pint.Unit", self.q.units),
            right=cast("pint.Unit", other_q.units),
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
                right=_DIMENSIONLESS_UNIT,
                literal=other_q,
            )

    def _fail_if_ordering_operand_is_invalid(self, other: Any, op: str) -> None:  # noqa: ANN401
        """Screen an operand of an ordering comparison (``<``/``<=``/``>``/``>=``).

        Two unit-carrying operands must be equivalent (calendar points by
        identity, so only same-axis points order), and a non-zero bare literal
        next to a non-dimensionless quantity is rejected (``wealth >
        1_000_000`` reads the bound as currency) — see
        :meth:`_fail_if_other_unit_is_not_equivalent`. Unlike ``+``/``-``, an
        ordering runs no forward pint operation, so calendar points get no
        delegate-to-pint dispensation here.
        """
        self._fail_if_other_unit_is_not_equivalent(other=other, op=op)

    def __bool__(self) -> bool:
        return self._explorer.decide(self._label)

    # Ordering comparisons are unit-blind at run time, so a non-equivalent
    # unit-carrying operand is a bug; the explorer still forces which branch runs.
    def __lt__(self, other: Any) -> _DryRunQuantity:  # noqa: ANN401
        self._fail_if_ordering_operand_is_invalid(other=other, op="<")
        return self._controlled_bool_at(
            self._comparison_level(other), label=self._composed_label(other, "<")
        )

    def __le__(self, other: Any) -> _DryRunQuantity:  # noqa: ANN401
        self._fail_if_ordering_operand_is_invalid(other=other, op="<=")
        return self._controlled_bool_at(
            self._comparison_level(other), label=self._composed_label(other, "<=")
        )

    def __gt__(self, other: Any) -> _DryRunQuantity:  # noqa: ANN401
        self._fail_if_ordering_operand_is_invalid(other=other, op=">")
        return self._controlled_bool_at(
            self._comparison_level(other), label=self._composed_label(other, ">")
        )

    def __ge__(self, other: Any) -> _DryRunQuantity:  # noqa: ANN401
        self._fail_if_ordering_operand_is_invalid(other=other, op=">=")
        return self._controlled_bool_at(
            self._comparison_level(other), label=self._composed_label(other, ">=")
        )

    # ``==``/``!=`` are deliberately *not* unit-screened: they are routinely used
    # polymorphically (sentinels, ``x == 0``) and are not magnitude comparisons.
    def __eq__(self, other: object) -> _DryRunQuantity:  # ty: ignore[invalid-method-override]
        return self._controlled_bool_at(
            self._comparison_level(other), label=self._composed_label(other, "==")
        )

    def __ne__(self, other: object) -> _DryRunQuantity:  # ty: ignore[invalid-method-override]
        return self._controlled_bool_at(
            self._comparison_level(other), label=self._composed_label(other, "!=")
        )

    def __and__(self, other: Any) -> _DryRunQuantity:  # noqa: ANN401
        return self._logical_result(other=other, op="&")

    def __rand__(self, other: Any) -> _DryRunQuantity:  # noqa: ANN401
        return self._logical_result(other=other, op="&")

    def __or__(self, other: Any) -> _DryRunQuantity:  # noqa: ANN401
        return self._logical_result(other=other, op="|")

    def __ror__(self, other: Any) -> _DryRunQuantity:  # noqa: ANN401
        return self._logical_result(other=other, op="|")

    def __xor__(self, other: Any) -> _DryRunQuantity:  # noqa: ANN401
        return self._logical_result(other=other, op="^")

    def __rxor__(self, other: Any) -> _DryRunQuantity:  # noqa: ANN401
        return self._logical_result(other=other, op="^")

    def __invert__(self) -> _DryRunQuantity:
        is_boolean, level = _boolean_level(cast("pint.Unit", self.q.units))
        if not is_boolean:
            raise _UnitMixError(
                op="~", left=cast("pint.Unit", self.q.units), right=_DIMENSIONLESS_UNIT
            )
        return self._controlled_bool_at(
            level, label=f"~{self._label}" if self._label is not None else None
        )

    # Addition and subtraction require equivalent units (see ``_UnitMixError``);
    # multiplication, division, and powers legitimately combine different units.
    def __add__(self, other: Any) -> _DryRunQuantity:  # noqa: ANN401
        self._fail_if_additive_operand_is_invalid(other=other, op="+")
        return self._wrap(self.q + _unwrap(other))

    def __radd__(self, other: Any) -> _DryRunQuantity:  # noqa: ANN401
        self._fail_if_additive_operand_is_invalid(other=other, op="+")
        return self._wrap(_unwrap(other) + self.q)

    def __sub__(self, other: Any) -> _DryRunQuantity:  # noqa: ANN401
        self._fail_if_additive_operand_is_invalid(other=other, op="-")
        return self._wrap(self.q - _unwrap(other))

    def __rsub__(self, other: Any) -> _DryRunQuantity:  # noqa: ANN401
        self._fail_if_additive_operand_is_invalid(other=other, op="-")
        return self._wrap(_unwrap(other) - self.q)

    def __mul__(self, other: Any) -> _DryRunQuantity:  # noqa: ANN401
        return self._wrap(self.q * _unwrap(other))

    def __rmul__(self, other: Any) -> _DryRunQuantity:  # noqa: ANN401
        return self._wrap(_unwrap(other) * self.q)

    def __truediv__(self, other: Any) -> _DryRunQuantity:  # noqa: ANN401
        return self._wrap(self.q / _unwrap(other))

    def __rtruediv__(self, other: Any) -> _DryRunQuantity:  # noqa: ANN401
        return self._wrap(_unwrap(other) / self.q)

    def __floordiv__(self, other: Any) -> _DryRunQuantity:  # noqa: ANN401
        return self._wrap(self.q // _unwrap(other))

    def __rfloordiv__(self, other: Any) -> _DryRunQuantity:  # noqa: ANN401
        return self._wrap(_unwrap(other) // self.q)

    def __mod__(self, other: Any) -> _DryRunQuantity:  # noqa: ANN401
        return self._wrap(self.q % _unwrap(other))

    def __rmod__(self, other: Any) -> _DryRunQuantity:  # noqa: ANN401
        return self._wrap(_unwrap(other) % self.q)

    def __pow__(self, other: Any) -> _DryRunQuantity:  # noqa: ANN401
        return self._wrap(self.q ** _unwrap(other))

    def __rpow__(self, other: Any) -> _DryRunQuantity:  # noqa: ANN401
        return self._wrap(_unwrap(other) ** self.q)

    def __neg__(self) -> _DryRunQuantity:
        return self._wrap(-self.q)

    def __pos__(self) -> _DryRunQuantity:
        return self._wrap(+self.q)

    def __abs__(self) -> _DryRunQuantity:
        return self._wrap(abs(self.q))


def _wrap_for_dry_run(
    value: Any,  # noqa: ANN401
    explorer: _PathExplorer,
    label: str | None = None,
) -> Any:  # noqa: ANN401
    """Wrap unit-carrying representative values; pass framework args through.

    Quantities (and the leaves of dict-param trees) become ``_DryRunQuantity`` so the
    explorer controls branches on them; ``xnp``/``num_segments``/… stay raw.
    ``label`` is the argument name the body sees, carried on the stand-in so a
    branch decision on it can be named in an error. A structured stand-in is
    re-anchored on the run's explorer, so its annotated plucks screen and
    branch like any other operand.
    """
    if isinstance(value, pint.Quantity):
        return _DryRunQuantity(q=value, explorer=explorer, label=label)
    if isinstance(value, _DryRunStructuredValue):
        return _DryRunStructuredValue(
            producer=value._producer,  # noqa: SLF001
            cls=value._cls,  # noqa: SLF001
            explorer=explorer,
            label=label,
        )
    if isinstance(value, dict):
        return {
            key: _wrap_for_dry_run(
                value=leaf,
                explorer=explorer,
                label=f"{label}[{key!r}]" if label is not None else None,
            )
            for key, leaf in value.items()
        }
    return value


class _ScheduleNotDryRunnableError(Exception):
    """A schedule/lookup/join call the dry-run cannot resolve to a unit.

    Raised when a function-like parameter carries no axes (a converter-produced
    or unannotated schedule) or a gather has no unit-carrying target — caught by
    :func:`_verify_one_body`'s generic handler and reported as needing an explicit
    ``verify_units=False`` opt-out, exactly like any other un-evaluable op.
    """


class _StructuredValueUsedAsQuantityError(Exception):
    """A value plucked off a structured parameter was used as a quantity —
    caught by :func:`_verify_one_body` and reported with the
    cast-at-the-pluck fix."""

    def __init__(self, producer: str, op: str) -> None:
        super().__init__()
        self.producer = producer
        self.op = op


class _DryRunStructuredValue:
    """The dry-run stand-in for a structured param-function output
    (``unit=UNSET_UNIT``, GEP 10). A pluck off an ``Annotated`` scalar field of
    the producer's return dataclass resolves to a quantity at the field's
    declared unit; a nested-dataclass pluck resolves recursively. Everything
    else stays opaque — attribute access, subscripting, method calls yield an
    opaque stand-in again — and using an opaque pluck as a quantity raises,
    demanding a ``cast_unit`` at the pluck.
    """

    __slots__ = ("_cls", "_explorer", "_label", "_producer")
    # Defer binary NumPy ops to our (raising) reflected dunders.
    __array_ufunc__ = None
    __array_priority__ = 1000
    __hash__ = object.__hash__

    def __init__(
        self,
        producer: str,
        cls: type | None = None,
        explorer: _PathExplorer | None = None,
        label: str | None = None,
    ) -> None:
        self._producer = producer
        self._cls = cls
        self._explorer = explorer
        self._label = label

    def _raise_used_as_quantity(self, op: str) -> NoReturn:
        raise _StructuredValueUsedAsQuantityError(producer=self._producer, op=op)

    def _opaque(self) -> _DryRunStructuredValue:
        return _DryRunStructuredValue(producer=self._producer)

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        # Refuse protocol probes (``__array__``, copy/pickle hooks, …).
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        kinds = _structured_field_kinds(self._cls) if self._cls is not None else None
        resolved = (kinds or {}).get(name)
        label = f"{self._label}.{name}" if self._label is not None else None
        if isinstance(resolved, pint.Unit):
            # An annotated field's pluck is a known quantity; with the run's
            # explorer it screens and branches like any other operand.
            quantity = UNIT_REGISTRY.Quantity(1.0, resolved)
            if self._explorer is None:
                return quantity
            return _DryRunQuantity(q=quantity, explorer=self._explorer, label=label)
        if resolved is not None:
            return _DryRunStructuredValue(
                producer=self._producer,
                cls=resolved,
                explorer=self._explorer,
                label=label,
            )
        return self._opaque()

    def __getitem__(self, _key: Any) -> _DryRunStructuredValue:  # noqa: ANN401
        return self._opaque()

    def __call__(self, *_args: Any, **_kwargs: Any) -> _DryRunStructuredValue:  # noqa: ANN401
        return self._opaque()

    def __bool__(self) -> Any:  # noqa: ANN401
        return self._raise_used_as_quantity("a branch decision")

    def __lt__(self, _other: Any) -> Any:  # noqa: ANN401
        return self._raise_used_as_quantity("<")

    def __le__(self, _other: Any) -> Any:  # noqa: ANN401
        return self._raise_used_as_quantity("<=")

    def __gt__(self, _other: Any) -> Any:  # noqa: ANN401
        return self._raise_used_as_quantity(">")

    def __ge__(self, _other: Any) -> Any:  # noqa: ANN401
        return self._raise_used_as_quantity(">=")

    def __eq__(self, _other: object) -> Any:  # noqa: ANN401
        return self._raise_used_as_quantity("==")

    def __ne__(self, _other: object) -> Any:  # noqa: ANN401
        return self._raise_used_as_quantity("!=")

    def __add__(self, _other: Any) -> Any:  # noqa: ANN401
        return self._raise_used_as_quantity("+")

    __radd__ = __add__

    def __sub__(self, _other: Any) -> Any:  # noqa: ANN401
        return self._raise_used_as_quantity("-")

    __rsub__ = __sub__

    def __mul__(self, _other: Any) -> Any:  # noqa: ANN401
        return self._raise_used_as_quantity("*")

    __rmul__ = __mul__

    def __truediv__(self, _other: Any) -> Any:  # noqa: ANN401
        return self._raise_used_as_quantity("/")

    __rtruediv__ = __truediv__

    def __floordiv__(self, _other: Any) -> Any:  # noqa: ANN401
        return self._raise_used_as_quantity("//")

    __rfloordiv__ = __floordiv__

    def __mod__(self, _other: Any) -> Any:  # noqa: ANN401
        return self._raise_used_as_quantity("%")

    __rmod__ = __mod__

    def __pow__(self, _other: Any) -> Any:  # noqa: ANN401
        return self._raise_used_as_quantity("**")

    __rpow__ = __pow__

    def __and__(self, _other: Any) -> Any:  # noqa: ANN401
        return self._raise_used_as_quantity("&")

    __rand__ = __and__

    def __or__(self, _other: Any) -> Any:  # noqa: ANN401
        return self._raise_used_as_quantity("|")

    __ror__ = __or__

    def __xor__(self, _other: Any) -> Any:  # noqa: ANN401
        return self._raise_used_as_quantity("^")

    __rxor__ = __xor__

    def __invert__(self) -> Any:  # noqa: ANN401
        return self._raise_used_as_quantity("~")

    def __neg__(self) -> Any:  # noqa: ANN401
        return self._raise_used_as_quantity("unary -")

    def __pos__(self) -> Any:  # noqa: ANN401
        return self._raise_used_as_quantity("unary +")

    def __abs__(self) -> Any:  # noqa: ANN401
        return self._raise_used_as_quantity("abs")


class _DryRunSchedule:
    """A dry-run stand-in for a ``piecewise_*``/lookup-table parameter value.

    Such a parameter is a *function between quantities*: a body calls
    ``piecewise_polynomial(x, parameters=…)`` or ``….look_up(idx)`` on it and gets
    an array. The dry-run needs only the unit that falls out. This stand-in
    carries the resolved ``input_unit``/``output_unit`` axes — a schedule
    parameter's own, or those a ``require_converter`` blob declares for its
    converter's typed output: it screens each domain argument against
    ``input_unit`` (as ``+`` screens an operand) and produces the
    ``output_unit``. ``input_unit`` is ``None`` when the parameter left it
    unset, in which case the domain is not screened.
    """

    __slots__ = ("input_unit", "output_unit")

    def __init__(self, input_unit: pint.Unit | None, output_unit: pint.Unit) -> None:
        self.input_unit = input_unit
        self.output_unit = output_unit

    def _produce(self, domain_args: tuple[Any, ...]) -> _DryRunQuantity:
        explorer: _PathExplorer | None = None
        for arg in domain_args:
            if isinstance(arg, _DryRunQuantity):
                explorer = arg._explorer  # noqa: SLF001
                if self.input_unit is not None and not units_are_equivalent(
                    left=cast("pint.Unit", arg.q.units), right=self.input_unit
                ):
                    raise _UnitMixError(
                        op="look-up",
                        left=self.input_unit,
                        right=cast("pint.Unit", arg.q.units),
                    )
        if explorer is None:
            # No unit-carrying domain argument to anchor the result on (a bare
            # literal index): not dry-runnable, fall back to the opt-out.
            raise _ScheduleNotDryRunnableError
        return _DryRunQuantity(
            q=UNIT_REGISTRY.Quantity(1.0, self.output_unit), explorer=explorer
        )

    def look_up(self, *args: Any) -> _DryRunQuantity:  # noqa: ANN401
        return self._produce(args)


def _piecewise_polynomial_dry_run(x: Any, parameters: Any, xnp: Any) -> Any:  # noqa: ANN401, ARG001
    """Dry-run shim for ``piecewise_polynomial``.

    Screen ``x`` against the schedule's ``input_unit`` and produce its
    ``output_unit``. A schedule built from an axes-declaring
    ``require_converter`` blob arrives as a :class:`_DryRunSchedule` too and
    screens the same way; only an axis-less converter-built schedule stays
    opaque — the caller casts the result. Anything else is not dry-runnable
    here.
    """
    if isinstance(parameters, _DryRunSchedule):
        return parameters._produce((x,))  # noqa: SLF001
    if isinstance(parameters, _DryRunStructuredValue):
        return parameters
    raise _ScheduleNotDryRunnableError


def _join_dry_run(
    foreign_key: Any,  # noqa: ANN401, ARG001
    primary_key: Any,  # noqa: ANN401, ARG001
    target: Any,  # noqa: ANN401
    value_if_foreign_key_is_missing: Any,  # noqa: ANN401, ARG001
    xnp: Any,  # noqa: ANN401, ARG001
) -> Any:  # noqa: ANN401
    """Dry-run shim for ``join``.

    A person-to-person gather preserves the ``target`` column's unit and level
    (the keys are dimensionless ``p_id``s, the missing-value a sentinel literal).
    """
    if isinstance(target, _DryRunQuantity):
        return target._wrap(target.q)  # noqa: SLF001
    raise _ScheduleNotDryRunnableError


def _cast_unit_dry_run(value: Any, unit: str | CompositeUnit) -> Any:  # noqa: ANN401
    """Dry-run shim for ``cast_unit``.

    The cast is total: whatever flowed in — a quantity at another unit or
    level, a bare literal, an attribute plucked off a structured value — the
    stand-in flowing out carries the stated unit, resolved like a declaration
    (currency-agnostic, the person leaf implied). A ``_DryRunQuantity`` input
    keeps its explorer, so branch decisions stay on the run's path; any other
    input anchors a plain representative quantity, which a wrapped operand
    combines with like any parameter value. A malformed token raises a
    :class:`UnitDefinitionError`, which :func:`_verify_one_body` re-raises
    rather than misreporting as an un-evaluable body.
    """
    token = coerce_unit_token(unit, where="A `cast_unit` call")
    resolved = resolve_compositional_cast_unit(token, where="A `cast_unit` call")
    quantity = UNIT_REGISTRY.Quantity(1.0, resolved)
    if isinstance(value, _DryRunQuantity):
        return value._wrap(quantity)  # noqa: SLF001
    return quantity


#: Module-level helpers swapped for unit-only shims in a dry-run body's scope.
_DRY_RUN_HELPER_SHIMS: Mapping[str, Any] = {
    "piecewise_polynomial": _piecewise_polynomial_dry_run,
    "join": _join_dry_run,
    "cast_unit": _cast_unit_dry_run,
}


def _clamping_op(left: Any, right: Any, op: str) -> Any:  # noqa: ANN401
    """``xnp.maximum``/``xnp.minimum``: an ordering-style screen, unit preserved.

    The vectorizer rewrites a scalar ``max(a, b)``/``min(a, b)`` to these, so the
    operands are screened exactly as an ordering comparison — two unit-carrying
    operands must be equivalent, a bare non-zero literal bound is rejected — and
    the result carries the quantity's unit.
    """
    quantity = left if isinstance(left, _DryRunQuantity) else right
    if not isinstance(quantity, _DryRunQuantity):
        return getattr(numpy, op)(left, right)
    other = right if quantity is left else left
    quantity._fail_if_ordering_operand_is_invalid(other=other, op=op)  # noqa: SLF001
    return quantity._wrap(quantity.q)  # noqa: SLF001


def _where_op(x: Any, y: Any) -> Any:  # noqa: ANN401
    """``xnp.where``: the two branches become one column, so they must carry
    equivalent units (as for an ordering comparison — no forward pint op runs,
    so calendar points screen by identity); the result carries that unit."""
    quantity = x if isinstance(x, _DryRunQuantity) else y
    if not isinstance(quantity, _DryRunQuantity):
        return numpy.where(True, x, y)  # noqa: FBT003
    other = y if quantity is x else x
    quantity._fail_if_other_unit_is_not_equivalent(other=other, op="where")  # noqa: SLF001
    return quantity._wrap(quantity.q)  # noqa: SLF001


def _clip_op(value: Any, a_min: Any, a_max: Any) -> Any:  # noqa: ANN401
    """``xnp.clip``: each bound is screened against the value as an ordering
    operand (so a bare non-zero literal bound is rejected); the unit is preserved.
    """
    if not isinstance(value, _DryRunQuantity):
        return numpy.clip(value, a_min, a_max)
    for bound in (a_min, a_max):
        if bound is not None:
            value._fail_if_ordering_operand_is_invalid(other=bound, op="clip")  # noqa: SLF001
    return value._wrap(value.q)  # noqa: SLF001


def _unit_preserving_op(value: Any) -> Any:  # noqa: ANN401
    """A unit-preserving reduction/unary op (``sum``/``floor``/``abs``/…)."""
    if isinstance(value, _DryRunQuantity):
        return value._wrap(value.q)  # noqa: SLF001
    return value


def _opt_out_required_error(qname: str, reason: str) -> str:
    """Message demanding an explicit opt-out for a body the dry-run cannot check.

    A body that cannot be evaluated symbolically is *not* waved through silently:
    the author must mark it ``verify_units=False`` so that every un-verified body
    is a visible, deliberate choice. The declared unit still
    stands and the body's edges are still checked — only its internal inference
    is skipped.
    """
    return (
        f"{qname}: its body cannot be unit-checked by the dry-run ({reason}). "
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
        f"dataclass field (`Annotated[float, Unit…]`) — or at the pluck with "
        f"`cast_unit(<pluck>, <unit>)`, or opt out of body inference with "
        f"`verify_units=False` (GEP 10)."
    )


def _arithmetic_misuse_message(
    qname: str,
    error: _UnitMixError
    | _StructuredValueUsedAsQuantityError
    | pint.OffsetUnitCalculusError,
    detail: str,
) -> str:
    """Message for a body that combines quantities unsoundly under ``+``/``-``/order.

    Dispatches the three ways the dry-run catches such a body: an explicit
    :class:`_UnitMixError` (non-equivalent units, a logical operator on a real
    quantity, a bare-literal threshold), a
    :class:`_StructuredValueUsedAsQuantityError` (an un-cast pluck off a
    structured value used as a quantity), or a
    :class:`pint.OffsetUnitCalculusError` raised by pint when a calendar point
    is used outside its affine algebra.
    """
    if isinstance(error, _UnitMixError):
        return _unit_mix_error_message(qname=qname, mix=error, detail=detail)
    if isinstance(error, _StructuredValueUsedAsQuantityError):
        return _structured_pluck_message(qname=qname, error=error)
    return (
        f"{qname}: combines calendar points unsoundly{detail} — two calendar "
        f"points cannot be added (subtract them to get a duration) and a point "
        f"cannot be scaled or mixed across calendar axes; shift a point only by "
        f"a same-axis duration (GEP 10)."
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
            f"with `cast_unit`, or use 0 (GEP 10)."
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
) -> str | None:
    """Dry-run one body on every reachable branch path; return an error or ``None``.

    The body runs once per path through the branch tree (see ``_PathExplorer``);
    each run that infers a concrete unit must match the declaration. A body that
    adds, subtracts, or orders two non-equivalent unit-carrying operands is
    flagged directly (``_UnitMixError``): those operations are unit-blind at run
    time, where no pint conversion happens. A run that infers a dimensionless
    result (e.g. an early ``return 0.0`` guard) falls back to the declaration on
    that path. A run that *raises* — a body using a lookup table, a piecewise
    polynomial, ``join``, or a raw ``xnp`` op the dry-run cannot evaluate — is
    reported as needing an explicit ``verify_units=False`` opt-out (callers reach
    this only for bodies that have not already opted out).

    A branch failure does not stop the exploration: the remaining paths still
    run, so the error can say whether the offence is confined to the reported
    branch combination (named via :meth:`_PathExplorer.branch_detail`) or other
    combinations fail as well.
    """
    explorer = _PathExplorer()
    paths = 0
    branch_errors: list[str] = []
    clean_paths = 0
    while True:
        if paths >= _MAX_PATHS:
            # Truncating exploration must not pass silently: a wrong-unit branch
            # first reached past the cap would otherwise go unchecked.
            return _opt_out_required_error(
                qname,
                f"it explores more than {_MAX_PATHS} branch paths — too many to "
                "check exhaustively",
            )
        paths += 1
        explorer.start_run()
        kwargs = {
            name: _wrap_for_dry_run(value=value, explorer=explorer, label=name)
            for name, value in {**base_kwargs, **boolean_values}.items()
        }
        error, terminal = _run_one_path(
            qname=qname,
            function=function,
            declared=declared,
            kwargs=kwargs,
            explorer=explorer,
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
            qname,
            f"it makes more than {_MAX_DECISIONS_PER_RUN} branch decisions "
            "in one run — a data-driven loop?",
        ), True
    except (
        _UnitMixError,
        _StructuredValueUsedAsQuantityError,
        pint.OffsetUnitCalculusError,
    ) as err:
        return _arithmetic_misuse_message(
            qname=qname, error=err, detail=explorer.branch_detail()
        ), False
    except UnitDefinitionError:
        # A malformed `cast_unit` token is a definition error, not an
        # un-evaluable body; report it as itself.
        raise
    except Exception:  # noqa: BLE001
        return _opt_out_required_error(
            qname,
            "it uses an operation pint cannot evaluate symbolically — a "
            "piecewise polynomial, a lookup table, `join`, or a raw `xnp` op",
        ), True
    return _inferred_result_error(
        qname=qname,
        inferred=_unwrap(result),
        declared=declared,
        detail=explorer.branch_detail(),
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
    if isinstance(inferred, _DryRunStructuredValue):
        return (
            f"{qname}: returns a value plucked off the structured parameter "
            f"'{inferred._producer}' without stating its unit; annotate the "  # noqa: SLF001
            f"dataclass field or tag it with `cast_unit` at the pluck (GEP 10)."
        )
    return _opt_out_required_error(
        qname,
        "it returns a value the dry-run cannot unit-check — a dataclass, "
        "a tuple, or another non-scalar",
    )


def _bare_literal_result_error(
    qname: str,
    inferred: Any,  # noqa: ANN401
    declared: pint.Unit,
    detail: str,
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
        and not UNIT_REGISTRY.Quantity(1.0, declared).dimensionless
    ):
        return (
            f"{qname}: returns the bare literal {inferred}{detail} under the "
            f"declaration '{declared}' — a literal return silently carries "
            f"the declared unit; promote it to a parameter, tag it with "
            f"`cast_unit`, or return 0 (GEP 10)."
        )
    return None


def _dimensionless_claim_error(
    qname: str,
    declared: pint.Unit,
    detail: str,
) -> str | None:
    """The group-ownership screen for a plain dimensionless inference.

    A dimensionless result is unit-polymorphic — what an identifier, a share,
    or a count magnitude produces (``p_id * 2.0``) — so it may stand in for any
    *person-grain* declaration. It cannot claim a group-owned one: ownership is
    a statement that arithmetic on level-less material can never produce, so it
    is made explicitly, with ``cast_unit``.
    """
    declared_group_levels = {
        name
        for name, exponent in _grouping_levels_with_exponent(declared)
        if exponent < 0 and name != PERSON_LEVEL
    }
    if not declared_group_levels:
        return None
    return (
        f"{qname}: its body infers a plain dimensionless result{detail}, "
        f"which cannot claim the group-owned declaration '{declared}'; "
        f"state the intended unit at the site with `cast_unit` (GEP 10)."
    )


def _inferred_result_error(
    qname: str,
    inferred: Any,  # noqa: ANN401
    declared: pint.Unit,
    detail: str,
) -> str | None:
    """Check one dry-run result against the declaration.

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
      intended unit with ``cast_unit`` at the site.

    A plain scalar result and a plain dimensionless inference take their own
    screens (:func:`_bare_literal_result_error`,
    :func:`_dimensionless_claim_error`): ``return 0.0`` and dimensionless
    magnitudes standing in for a person-grain quantity (``p_id * 2.0`` under
    ``CURRENCY``) stay lenient; a non-zero literal return and a dimensionless
    claim on a group-owned declaration are rejected.
    """
    if not isinstance(
        inferred, pint.Quantity | int | float | numpy.number | numpy.bool_
    ):
        return _non_quantity_result_error(qname=qname, inferred=inferred)
    if not isinstance(inferred, pint.Quantity):
        return _bare_literal_result_error(
            qname=qname, inferred=inferred, declared=declared, detail=detail
        )
    if inferred.dimensionless:
        return _dimensionless_claim_error(qname=qname, declared=declared, detail=detail)
    inferred_unit = cast("pint.Unit", inferred.units)
    if not units_are_equivalent(
        left=_unit_without_grouping_levels(inferred_unit),
        right=_unit_without_grouping_levels(declared),
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
            f"unit at the site with `cast_unit` (GEP 10)."
        )
    return None
