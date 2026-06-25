"""Whole-environment unit checks (GEP 10).

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

The dry-run wraps each ``Quantity(1.0, unit)`` in a :class:`_DryRunQuantity`,
whose arithmetic propagates units while a :class:`_PathExplorer` drives its
branch decisions — so a body is checked down every reachable path. ``+``, ``-``
and the ordering comparisons additionally require equivalent operands, because
at run time there is no pint to convert between them. pint runs only at
build time; no live array is ever wrapped.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Mapping
from typing import Any, cast

import dags.tree as dt
import numpy
import pint

from ttsim.exceptions import TTSIMError, UnitConsistencyError, UnitDefinitionError
from ttsim.interface_dag_elements.shared import (
    get_re_pattern_for_all_time_units_and_groupings,
)
from ttsim.tt.aggregation import AggType
from ttsim.tt.column_objects_param_function import (
    AggByGroupFunction,
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
    PERSON_LEVEL,
    REFERENCE_PERIOD_TO_PINT_NAME,
    TIME_UNIT_ID_TO_PINT_NAME,
    UNIT_REGISTRY,
    UNSET_UNIT,
    CurrencyUnitToken,
    Unit,
    UnsetUnitType,
    base_currency,
    divide_by_grouping_level,
    fail_if_units_are_missing,
    is_calendar_point_unit,
    parse_unit,
    register_grouping_levels,
    resolve_column_unit,
    resolve_param_unit,
    resolve_scalar_param_unit,
    resolved_unit_for_aggregation,
    token_is_agnostic_currency,
    unit_token_carries_level,
    unit_token_is_flow,
    units_are_equivalent,
)
from ttsim.typing import (
    OrderedQNames,
    SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
)
from ttsim.unit_converters import TIME_UNIT_IDS_TO_LABELS

#: Units of the date nodes the framework itself injects into every policy
#: environment (see ``policy_environment.policy_environment``). They are
#: framework constructs, so their units live here rather than in downstream
#: annotations. Each is a *calendar point* (GEP 10), not a duration: a year on
#: the calendar is an affine point, so ``policy_year - geburtsjahr`` is a
#: duration in years (``Unit.YEARS``) and adding two of them is rejected. (The
#: month/day nodes are typed as points too; in practice they are read as cyclic
#: ordinals via equality/lookup, which the dry-run does not unit-screen.)
FRAMEWORK_DATE_NODE_UNITS: Mapping[str, str] = {
    "policy_year": "calendar_year",
    "policy_month": "calendar_month",
    "policy_day": "calendar_day",
    "evaluation_year": "calendar_year",
    "evaluation_month": "calendar_month",
    "evaluation_day": "calendar_day",
}

#: Arguments of column/param functions that the framework partials in and
#: that never carry a unit.
NON_UNIT_ARGUMENTS = frozenset({"xnp", "dnp", "backend", "num_segments", "len_p_id"})

#: Representative values for the framework arguments in a dry-run. The
#: dry-run always executes in NumPy + pint (NEP 18), regardless of the
#: backend of the actual run.
_NON_UNIT_ARGUMENT_VALUES: Mapping[str, Any] = {
    "xnp": numpy,
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


def resolve_environment_units(
    env: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
    grouping_levels: OrderedQNames,
) -> dict[str, pint.Unit | dict[str | int, Any]]:
    """Resolve the complete unit of every annotated node in the environment.

    Columns and param functions combine their declared non-time ``unit`` with
    the time-unit suffix of their leaf name (GEP 1 / GEP 10); parameters
    combine theirs with the functional ``reference_period``. Dict parameters
    with per-leaf units resolve to nested dicts of pint units. The
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
        qname: parse_unit(unit)
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
            agg_unit = _resolve_agg_by_group_unit(qname=qname, env=env, pattern=pattern)
            if agg_unit is not None:
                resolved[qname] = agg_unit
        else:  # ColumnObject | ParamFunction
            token = getattr(obj, "unit", UNSET_UNIT)
            if token is not UNSET_UNIT:
                leaf_name = dt.tree_path_from_qname(qname)[-1]
                match = pattern.fullmatch(leaf_name)
                resolved[qname] = _resolve_leveled_column_unit(
                    token=cast("Unit", token), match=match
                )
    return resolved


def _resolve_leveled_column_unit(
    token: Unit,
    match: re.Match[str] | None,
) -> pint.Unit:
    """Resolve a column/function's full unit, including its grouping level (GEP 10).

    Combines the declared token with the name's time-unit suffix
    (:func:`resolve_column_unit`) and then, when the token carries a level by
    default (currency, area, the ``[person]`` count), divides by the name's
    aggregation-suffix level — an unsuffixed name is at :data:`PERSON_LEVEL`. A
    level-less token stays level-less regardless of any suffix: the suffix is then
    a pure *index* level (a ``MIN``-of-age at ``_fg``), not a unit level.
    """
    time_unit_id = match.group("time_unit") if match else None
    column_unit = resolve_column_unit(token=token, time_unit_id=time_unit_id)
    if unit_token_carries_level(token):
        return divide_by_grouping_level(
            unit=column_unit, level=_suffix_grouping_level(match)
        )
    return column_unit


def _resolve_agg_by_group_unit(
    qname: str,
    env: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
    pattern: re.Pattern[str],
) -> pint.Unit | None:
    """Resolve an auto-aggregation node's unit, level-aware (GEP 10, #119).

    Auto-aggregations are the orchestration site where a grouping level is minted,
    swapped, or preserved (GEP 10): this routes through
    :func:`resolved_unit_for_aggregation`, the level-aware resolver, rather than the
    plain token+suffix path. The aggregation's *target* level is its own
    aggregation suffix (an ``_hh`` node aggregates to ``[hh]``); the *source* level
    is the source column's own level (:data:`PERSON_LEVEL` for an extensive
    person-level source, ``None`` for a level-less source such as an age).

    The stored ``unit`` token already encodes the aggregation rule's effect on the
    physical token (``SUM`` preserves, ``COUNT`` → ``DIMENSIONLESS``; see
    :func:`unit_for_aggregation`); here it is combined with the levels. Returns
    ``None`` if the source carries no resolvable unit — the mandatory-units check
    reports the source.
    """
    leaf_name = dt.tree_path_from_qname(qname)[-1]
    match = pattern.fullmatch(leaf_name)
    target_level = _suffix_grouping_level(match)
    source_qname, source_match = _agg_source(qname=qname, pattern=pattern)
    source_obj = env.get(source_qname)
    source_token = getattr(source_obj, "unit", UNSET_UNIT)
    if source_token is UNSET_UNIT:
        return None
    source_unit = _resolve_leveled_column_unit(
        token=cast("Unit", source_token), match=source_match
    )
    source_level = (
        _suffix_grouping_level(source_match)
        if unit_token_carries_level(cast("Unit", source_token))
        else None
    )
    return resolved_unit_for_aggregation(
        source_unit=source_unit,
        agg_type=AggType.SUM,
        target_level=target_level,
        source_level=source_level,
    )


def _agg_source(
    qname: str,
    pattern: re.Pattern[str],
) -> tuple[str, re.Match[str] | None]:
    """The source qname of an auto-aggregation and its parsed name (GEP 10).

    An auto-aggregation ``…_hh`` sums the same-named individual-level source with
    the grouping suffix stripped (``betrag_m_hh`` → ``betrag_m``); the source keeps
    the time-unit suffix, since a flow is summed period-for-period.
    """
    leaf_match = pattern.fullmatch(qname)
    if leaf_match is None or not leaf_match.group("grouping"):
        return qname, pattern.fullmatch(qname)
    base = leaf_match.group("base_name")
    time_unit = leaf_match.group("time_unit")
    source_qname = f"{base}_{time_unit}" if time_unit else base
    return source_qname, pattern.fullmatch(source_qname)


def _suffix_grouping_level(match: re.Match[str] | None) -> str:
    """The grouping level named by a name's aggregation suffix (GEP 10).

    The combined time+grouping regex captures the aggregation suffix in its
    ``grouping`` group (``betrag_m_hh`` → ``"hh"``); an unsuffixed name has no
    such group and is at the individual leaf level :data:`PERSON_LEVEL`.
    """
    if match is None:
        return PERSON_LEVEL
    return match.group("grouping") or PERSON_LEVEL


#: The dimensionality-key prefix of a grouping-level dimension: the internal pint
#: unit name :data:`_GROUPING_LEVEL_PREFIX` wrapped in pint's ``[…]`` dimension
#: brackets (e.g. ``[grouping_level_hh]``).
_GROUPING_LEVEL_DIM_PREFIX = f"[{_GROUPING_LEVEL_PREFIX}"


def _unit_level_denominator(unit: pint.Unit) -> str | None:
    """The grouping level a resolved unit carries as a denominator (GEP 10).

    A leveled quantity carries its level as a ``/[level]`` denominator (negative
    exponent in the dimensionality), exactly as a flow carries its period. Returns
    the level name (``"hh"``, ``"person"``, …) found in the denominator, or
    ``None`` for a level-less unit. A head count's ``[person]`` *numerator*
    (positive exponent) is not a denominator level and is ignored, so a
    ``[person]/[hh]`` count reports ``"hh"`` — its index level (GEP 10).
    """
    for dimension, exponent in UNIT_REGISTRY.Quantity(1.0, unit).dimensionality.items():
        if isinstance(exponent, complex):  # pint exponents are real; narrow for ty
            continue
        if exponent < 0 and dimension.startswith(_GROUPING_LEVEL_DIM_PREFIX):
            return dimension[len(_GROUPING_LEVEL_DIM_PREFIX) : -1]
    return None


def fail_if_environment_units_are_missing(
    env: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
    grouping_levels: OrderedQNames,  # noqa: ARG001  (kept for symmetry of the two checks)
) -> None:
    """Mandatory-units check over a fully assembled environment (GEP 10).

    Every active node must declare a unit — where ``unit=Unit.DIMENSIONLESS``
    / ``unit: DIMENSIONLESS`` *is* a declaration (a dimensionless quantity,
    GEP 10). For a dict parameter with per-leaf units, every leaf of the value
    active at the policy date must be covered.

    Raises:
        UnitDefinitionError: If any node (or dict-param leaf) lacks a unit
            declaration.
    """
    units_by_qname: dict[str, Unit | CurrencyUnitToken | UnsetUnitType] = {}
    for qname, obj in env.items():
        if not isinstance(obj, ColumnObject | ParamFunction | ParamObject):
            continue
        if qname in FRAMEWORK_DATE_NODE_UNITS:
            continue
        declared_unit = getattr(obj, "unit", UNSET_UNIT)
        if isinstance(obj, ParamMappingObject):
            # A function between quantities declares one token per axis
            # (coerced to vocabulary tokens at construction).
            units_by_qname[f"{qname} (input_unit)"] = cast(
                "Unit | CurrencyUnitToken | UnsetUnitType", obj.input_unit
            )
            units_by_qname[f"{qname} (output_unit)"] = cast(
                "Unit | CurrencyUnitToken | UnsetUnitType", obj.output_unit
            )
            continue
        if isinstance(obj, ParamObject) and isinstance(declared_unit, Mapping):
            value = getattr(obj, "value", None)
            value_tree = value if isinstance(value, Mapping) else {}
            units_by_leaf = dt.flatten_to_qnames(declared_unit)
            for leaf_qname in dt.flatten_to_qnames(value_tree):
                token = units_by_leaf.get(leaf_qname, UNSET_UNIT)
                leaf_path = dt.tree_path_from_qname(leaf_qname)
                display = f"{qname}[{']['.join(leaf_path)}]"
                units_by_qname[display] = (
                    token if isinstance(token, str) else UNSET_UNIT
                )
        else:
            units_by_qname[qname] = declared_unit
    fail_if_units_are_missing(units_by_qname)


def fail_if_environment_units_are_inconsistent(
    env: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
    grouping_levels: OrderedQNames,
) -> None:
    """Conservative body/edge verification over an assembled environment.

    Each ``@policy_function`` / ``@param_function`` body is dry-run on
    representative values built from its producers' resolved units (the DAG
    edges) — see the module docstring for the conservative rules and the
    boolean-enumeration strategy. Aggregations, time-conversion variants, and
    group-creation functions have no scalar body to dry-run; their units are
    auto-assigned or checked via their consumers.

    Raises:
        UnitConsistencyError: If any body infers a concrete unit that
            disagrees with its declaration. All offending nodes are reported
            together.
    """
    resolved_units = resolve_environment_units(env=env, grouping_levels=grouping_levels)
    representative_values = _representative_values_by_qname(
        env=env, resolved_units=resolved_units
    )
    suffix_pattern = get_re_pattern_for_all_time_units_and_groupings(
        time_units=tuple(TIME_UNIT_IDS_TO_LABELS),
        grouping_levels=grouping_levels,
    )
    boolean_nodes = {
        qname
        for qname, obj in env.items()
        if isinstance(obj, ColumnObject | ParamFunction)
        and node_is_boolean(qname=qname, obj=obj)
    }
    errors: list[str] = []
    for qname, obj in env.items():
        # Only these two have a scalar body written by a human; everything
        # else is generated and unit-assigned by construction.
        if not isinstance(obj, PolicyFunction | ParamFunction):
            continue
        if qname not in resolved_units:
            # No resolved unit (still UNSET) — the mandatory-units check reports
            # it; there is nothing to dry-run against here. A boolean-returning
            # body is *not* skipped: it declares DIMENSIONLESS like any other
            # dimensionless node, so it is dry-run too — the comparisons and
            # logical combinations inside it are checked, and its truth-value
            # result is dimensionless, matching the declaration (GEP 10).
            continue
        if not obj.verify_units:
            # Body opted out of unit inference (GEP 10): a genuine code-level
            # literal (or an opaque/standardized body) would otherwise trip the
            # dry-run. The declared unit still stands (resolved_units[qname]) as
            # the edge contract, so this node's *consumers* are still checked, and
            # the inputs it consumes are themselves verified producer outputs. Not
            # checked is anything *internal* to the body: e.g. a schedule's domain
            # (input_unit) is never bound to the argument it is evaluated at, since
            # the body is never dry-run. See GEP 10 (known limitation).
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
        leaf_name = dt.tree_path_from_qname(qname)[-1]
        suffix_match = suffix_pattern.fullmatch(leaf_name)
        error = _verify_one_body(
            qname=qname,
            function=obj.function,
            declared=declared,
            suffix_level=_suffix_grouping_level(suffix_match),
            boolean_parameters=boolean_parameters,
            base_kwargs=base_kwargs,
        )
        if error is not None:
            errors.append(error)
    if errors:
        raise UnitConsistencyError(
            "Environment unit-consistency check failed:\n  " + "\n  ".join(errors)
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


#: Matches a GEP-1 time-unit suffix at the end of a dict-param leaf key.
_LEAF_TIME_SUFFIX_PATTERN = re.compile(
    rf"_(?P<time_unit>[{''.join(TIME_UNIT_ID_TO_PINT_NAME)}])$"
)


def _spell_token(token: Any) -> str:  # noqa: ANN401
    """Spell a declaration token for an error message."""
    if token is UNSET_UNIT:
        return "unset"
    return str(token)


def _fail_if_period_sources_disagree(
    where: str,
    suffix_id: str,
    reference_period: str | None,
) -> None:
    """Reject disagreeing period sources — coincidence, never precedence (GEP 10)."""
    if (
        reference_period is not None
        and REFERENCE_PERIOD_TO_PINT_NAME[reference_period]
        != TIME_UNIT_ID_TO_PINT_NAME[suffix_id]
    ):
        raise UnitDefinitionError(
            f"{where}: the time-unit suffix (_{suffix_id}) and "
            f"`reference_period: {reference_period}` disagree; wherever two "
            f"period sources apply they must coincide (GEP 10)."
        )


def _fail_if_param_token_is_agnostic_currency(
    token: Unit | Any | None,  # noqa: ANN401
    where: str,
) -> None:
    """Reject an agnostic currency token on a parameter (GEP 10).

    Once a concrete currency is registered, a parameter's numbers are
    written in *some* currency — the declaration must name it
    (``SILVER_PENNY``, ``DM_FLOW``, …), so the build-time conversion
    to the run currency knows what to convert from. The agnostic tokens stay
    legal — and required — on columns and functions, which are
    currency-agnostic by design.
    """
    base = base_currency()
    if base is not None and token_is_agnostic_currency(token):
        concrete = f"{base.upper()}{str(token).removeprefix('CURRENCY')}"
        raise UnitDefinitionError(
            f"{where}: parameters must pin down the concrete currency their "
            f"numbers are written in; the agnostic token {token} is not "
            f"allowed here. Declare e.g. {concrete} (GEP 10)."
        )


def _resolve_token_unit(
    token: Unit | CurrencyUnitToken,
    reference_period: str | None,
    reference_level: str | None = None,
) -> pint.Unit:
    """Resolve one axis token; ``reference_period`` only feeds flow tokens.

    ``reference_level`` is the grouping-level counterpart of ``reference_period``
    (GEP 10): a per-person or per-group axis carries its level as a denominator. It
    is divided onto any token (it has no flow constraint); ``None`` is
    level-agnostic.
    """
    if unit_token_is_flow(token):
        return resolve_param_unit(
            token=token,
            reference_period=reference_period,
            reference_level=reference_level,
        )
    return resolve_param_unit(
        token=token, reference_period=None, reference_level=reference_level
    )


def _resolve_param_mapping_object_units(
    qname: str,
    obj: ParamMappingObject,
    name_time_unit_id: str | None,
) -> pint.Unit | None:
    """Resolve a mapping parameter's per-axis unit declarations.

    A schedule or lookup table is a function between quantities: it declares
    ``input_unit:`` and ``output_unit:`` instead of ``unit:`` (GEP 10). Both
    tokens follow the kind rules; a ``…_FLOW`` token on either axis takes its
    period from the (single) ``reference_period``, and a dangling
    ``reference_period`` (no flow axis consuming it) is an error. A time
    suffix on the parameter's *name* describes what it yields, so it must
    coincide with a flow ``output_unit``.

    The environment-level resolved unit is the *output* unit (what consumers
    receive); the input unit is validated here and consumed by the build-time
    currency conversion. Returns ``None`` if the output unit is unset — the
    mandatory-units check reports it.
    """
    tokens = {}
    for axis, raw in (("input_unit", obj.input_unit), ("output_unit", obj.output_unit)):
        if raw is UNSET_UNIT:
            tokens[axis] = UNSET_UNIT
            continue
        where = f"Parameter {qname!r}, {axis}"
        if isinstance(raw, Mapping):
            raise UnitDefinitionError(
                f"{where}: per-axis declarations are single tokens, not "
                f"mappings (GEP 10)."
            )
        token = cast("Unit | CurrencyUnitToken", raw)
        _fail_if_param_token_is_agnostic_currency(token=token, where=where)
        tokens[axis] = token
    output_token = tokens["output_unit"]
    if name_time_unit_id is not None:
        _fail_if_name_suffix_disagrees_with_output_axis(
            qname=qname,
            obj=obj,
            output_token=output_token,
            name_time_unit_id=name_time_unit_id,
        )
    any_flow = any(
        token is not UNSET_UNIT and unit_token_is_flow(token)
        for token in tokens.values()
    )
    if obj.reference_period is not None and not any_flow:
        raise UnitDefinitionError(
            f"Parameter {qname!r} declares `reference_period: "
            f"{obj.reference_period}` but neither axis token is a `…_FLOW`; "
            f"a dangling reference_period is an error (GEP 10)."
        )
    reference_level = getattr(obj, "reference_level", None)
    input_token = tokens["input_unit"]
    if input_token is not UNSET_UNIT:
        _resolve_token_unit(
            token=input_token,
            reference_period=obj.reference_period,
            reference_level=reference_level,
        )
    if output_token is UNSET_UNIT:
        return None
    return _resolve_token_unit(
        token=output_token,
        reference_period=obj.reference_period,
        reference_level=reference_level,
    )


def _fail_if_name_suffix_disagrees_with_output_axis(
    qname: str,
    obj: ParamMappingObject,
    output_token: Any,  # noqa: ANN401
    name_time_unit_id: str,
) -> None:
    """Check the name-suffix ⟺ flow-output coincidence rules (GEP 10)."""
    if output_token is UNSET_UNIT or not unit_token_is_flow(output_token):
        raise UnitDefinitionError(
            f"Parameter {qname!r}: the name carries a time-unit suffix "
            f"(_{name_time_unit_id}), which denotes a flow, but "
            f"`output_unit:` is {_spell_token(output_token)} (GEP 10)."
        )
    _fail_if_period_sources_disagree(
        where=f"Parameter {qname!r}",
        suffix_id=name_time_unit_id,
        reference_period=obj.reference_period,
    )


def _resolve_param_object_unit(
    qname: str,
    obj: ParamObject,
    name_time_unit_id: str | None = None,
) -> pint.Unit | dict[str | int, Any] | None:
    """Resolve a parameter's declared unit.

    A **scalar** parameter takes its period from a time suffix on its *name*,
    just like a column (``lump_sum_deduction_y`` → ``CURRENCY / year``);
    ``reference_period`` is forbidden on it (GEP 10). ``reference_period`` is
    reserved for the period sources that have no name to suffix: a uniformly
    typed **dict** parameter, a **raw** parameter, integer-keyed leaves of a
    per-leaf ``unit:`` mapping, and the axes of a **mapping** parameter.
    A per-leaf ``unit:`` mapping (dict params) holds one token (``DIMENSIONLESS``
    for a dimensionless leaf) per leaf; a flow leaf gets its period from the leaf
    key's time suffix or from the dict-level ``reference_period`` (see
    :func:`_resolve_unit_mapping`). Mapping parameters (schedules, lookup
    tables) declare per-axis tokens instead — see
    :func:`_resolve_param_mapping_object_units`. Resolves to a nested dict of
    pint units mirroring the value structure. Returns ``None`` for an
    unannotated parameter — the mandatory-units check reports it.
    """
    if isinstance(obj, ParamMappingObject):
        return _resolve_param_mapping_object_units(
            qname=qname, obj=obj, name_time_unit_id=name_time_unit_id
        )
    if obj.unit is UNSET_UNIT:
        return None
    # The grouping level the parameter is denominated per (GEP 10): the level
    # counterpart of ``reference_period``, declared on the parameter and divided
    # onto its resolved unit. ``None`` (the default, and the value until the field
    # is added downstream) is level-agnostic. Unlike ``reference_period`` it is
    # allowed on scalar parameters — they have no aggregation suffix to read it off.
    reference_level = getattr(obj, "reference_level", None)
    if isinstance(obj.unit, Mapping):
        reference_period = obj.reference_period
        if (
            reference_period is not None
            and reference_period not in REFERENCE_PERIOD_TO_PINT_NAME
        ):
            raise UnitDefinitionError(
                f"Parameter {qname!r}: unknown reference_period {reference_period!r}."
            )
        resolved, has_flow_leaf = _resolve_unit_mapping(
            qname=qname,
            unit_mapping=cast("Mapping[str | int, Any]", obj.unit),
            reference_period=reference_period,
            reference_level=reference_level,
        )
        if reference_period is not None and not has_flow_leaf:
            raise UnitDefinitionError(
                f"Parameter {qname!r} declares `reference_period: "
                f"{reference_period}` but no `…_FLOW` leaf consumes it; a "
                f"dangling reference_period is an error (GEP 10)."
            )
        return resolved
    token = cast("Unit | CurrencyUnitToken", obj.unit)
    _fail_if_param_token_is_agnostic_currency(token=token, where=f"Parameter {qname!r}")
    if isinstance(obj, ScalarParam):
        # A scalar parameter takes its period from a time suffix on its name
        # (GEP 10); reference_period is reserved for the suffix-impossible cases.
        if obj.reference_period is not None:
            raise UnitDefinitionError(
                f"Parameter {qname!r}: a scalar parameter takes its period from a "
                f"time suffix on its name, not `reference_period` (GEP 10). Drop "
                f"`reference_period: {obj.reference_period}`; name a flow parameter "
                f"with a `_y`/`_m`/… suffix."
            )
        return resolve_scalar_param_unit(
            token=token,
            time_unit_id=name_time_unit_id,
            reference_level=reference_level,
        )
    # DictParam with a uniform token, or RawParam: no single name to suffix, so
    # the period (if any) comes from the dict-level reference_period.
    return resolve_param_unit(
        token=token,
        reference_period=obj.reference_period,
        reference_level=reference_level,
    )


def _resolve_unit_mapping(
    qname: str,
    unit_mapping: Mapping[str | int, Any],
    reference_period: str | None,
    reference_level: str | None = None,
) -> tuple[dict[str | int, Any], bool]:
    """Resolve a per-leaf ``unit:`` mapping to pint units (GEP 10).

    Period sources for a flow leaf are checked under **strict coincidence** —
    there is no precedence order:

    - a leaf key with a time suffix must *agree* with a non-null dict-level
      ``reference_period``; disagreement is an error, the suffix does not win;
    - a suffix-less flow leaf takes ``reference_period``; if that is null,
      the leaf has no period source and fails;
    - a complete or ``DIMENSIONLESS`` leaf must not carry a suffixed key (a
      suffixed name denotes a flow).

    ``reference_level`` is the dict-level grouping level (GEP 10): a per-person or
    per-group dict parameter denominates every leaf per that level, so it is
    divided onto each leaf's resolved unit. ``None`` is level-agnostic.

    Returns the resolved mapping and whether any flow leaf was seen (the
    caller rejects a dangling ``reference_period``).
    """
    resolved: dict[str | int, Any] = {}
    any_flow = False
    for key, token in unit_mapping.items():
        if isinstance(token, Mapping):
            resolved[key], sub_flow = _resolve_unit_mapping(
                qname=qname,
                unit_mapping=token,
                reference_period=reference_period,
                reference_level=reference_level,
            )
            any_flow = any_flow or sub_flow
            continue
        where = f"Parameter {qname!r}, unit of leaf {key!r}"
        _fail_if_param_token_is_agnostic_currency(token=token, where=where)
        match = _LEAF_TIME_SUFFIX_PATTERN.search(str(key))
        suffix_id = match.group("time_unit") if match else None
        if unit_token_is_flow(token):
            any_flow = True
            if suffix_id is not None:
                _fail_if_period_sources_disagree(
                    where=where, suffix_id=suffix_id, reference_period=reference_period
                )
            leaf_reference_period = (
                TIME_UNIT_IDS_TO_LABELS[suffix_id] if suffix_id else reference_period
            )
            if leaf_reference_period is None:
                raise UnitDefinitionError(
                    f"{where}: token {token} denotes a flow but has no period "
                    f"source — give the leaf key a time suffix or declare a "
                    f"dict-level `reference_period` (GEP 10)."
                )
            resolved[key] = resolve_param_unit(
                token=token,
                reference_period=leaf_reference_period,
                reference_level=reference_level,
            )
        else:
            if suffix_id is not None:
                raise UnitDefinitionError(
                    f"{where}: the leaf key carries a time suffix "
                    f"(_{suffix_id}), which denotes a flow, but the declared "
                    f"token is {_spell_token(token)} (GEP 10)."
                )
            resolved[key] = resolve_param_unit(
                token=token, reference_period=None, reference_level=reference_level
            )
    return resolved, any_flow


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


def _representative_values_by_qname(
    env: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
    resolved_units: Mapping[str, pint.Unit | dict[str | int, Any]],
) -> dict[str, Any]:
    """Representative dry-run values for every unit-resolved node.

    A dict parameter with a scalar ``unit:`` declaration becomes a dict of
    uniform representative quantities mirroring its value structure, so that
    subscripting works inside a consumer's dry-run.
    """
    out: dict[str, Any] = {}
    for qname, unit in resolved_units.items():
        obj = env.get(qname)
        if isinstance(obj, DictParam | RawParam) and not isinstance(unit, dict):
            out[qname] = _uniform_quantity_tree(
                value=obj.value, resolved_unit=cast("pint.Unit", unit)
            )
        else:
            out[qname] = _representative_value(unit)
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


# Caps on the path-exploring dry-run (see ``_PathExplorer``). A real policy
# body has a handful of branches; these only guard a pathological body (deep
# independent branching, or a data-driven loop) so the whole-environment build
# check can never blow up. On either cap we warn and stop exploring (the paths seen
# so far are still checked), so coverage is never *silently* truncated.
_MAX_PATHS = 1024
_MAX_DECISIONS_PER_RUN = 64


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
    before delegating instead (GEP 10).
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
        # that is the offence (a unit-carrying operand compared to a non-zero
        # number, GEP 10); ``None`` for the unit-vs-unit mismatch cases.
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
        self._index = 0

    def start_run(self) -> None:
        self._index = 0
        self._trail = []

    def decide(self) -> bool:
        """Resolve the next branch: replay the prefix, then explore ``False``."""
        if self._index >= _MAX_DECISIONS_PER_RUN:
            raise _PathBudgetExceededError
        value = self._prefix[self._index] if self._index < len(self._prefix) else False
        self._index += 1
        self._trail.append(value)
        return value

    def advance(self) -> bool:
        """Queue the next path; return ``False`` once the tree is exhausted."""
        for k in range(len(self._trail) - 1, -1, -1):
            if not self._trail[k]:
                self._prefix = [*self._trail[:k], True]
                return True
        return False

    @property
    def on_a_branch(self) -> bool:
        return bool(self._trail)


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

    __slots__ = ("_explorer", "q")
    # Keep NumPy from trying to broadcast over us: defer binary ops involving a
    # NumPy operand to our reflected dunders instead (raw ``xnp`` ops then raise
    # and the path falls back, as before).
    __array_ufunc__ = None
    __array_priority__ = 1000
    __hash__ = object.__hash__

    def __init__(self, q: Any, explorer: _PathExplorer) -> None:  # noqa: ANN401
        self.q = q
        self._explorer = explorer

    def _wrap(self, q: Any) -> _DryRunQuantity:  # noqa: ANN401
        return _DryRunQuantity(q=q, explorer=self._explorer)

    def _controlled_bool(self) -> _DryRunQuantity:
        # A dimensionless, explorer-controlled stand-in for a comparison result.
        return self._wrap(UNIT_REGISTRY.Quantity(1.0, ""))

    def _fail_if_other_unit_is_not_equivalent(self, other: Any, op: str) -> None:  # noqa: ANN401
        """Reject a non-equivalent *unit-carrying* operand of ``+``/``-``/comparison.

        At run time there is no pint, so these operations are unit-blind (raw
        arrays are added or compared without conversion); two unit-carrying
        operands must already be in equivalent units. A bare literal carries no
        unit to compare against, so it stays lenient — the literal ambiguity we
        deliberately do not resolve (an ``x + 0.0`` guard must not be flagged).

        A calendar point (an affine offset unit) is the exception: its valid
        ``point + duration`` is *not* equivalence (a point and a duration differ),
        yet pint's offset algebra permits it and forbids the genuine misuses
        (``point + point``, cross-axis mixes). So when either operand is a
        calendar point we skip the magnitude pre-screen and let the forward
        operation delegate to pint, which raises ``OffsetUnitCalculusError`` on a
        misuse — caught in :func:`_verify_one_body` and reported as a calendar
        misuse (GEP 10).
        """
        other_q = _unwrap(other)
        if is_calendar_point_unit(cast("pint.Unit", self.q.units)) or (
            isinstance(other_q, pint.Quantity)
            and is_calendar_point_unit(cast("pint.Unit", other_q.units))
        ):
            return
        if isinstance(other_q, pint.Quantity) and not units_are_equivalent(
            left=cast("pint.Unit", self.q.units),
            right=cast("pint.Unit", other_q.units),
        ):
            raise _UnitMixError(op=op, left=self.q.units, right=other_q.units)

    def _fail_if_logical_operand_carries_unit(self, other: Any, op: str) -> None:  # noqa: ANN401
        """Reject a unit-carrying operand of a logical operator ``&``/``|``/``^``.

        Logical operators combine truth values, which are dimensionless. A
        non-dimensionless operand — ``wealth & is_adult`` where ``wealth`` is a
        currency stock — is a mistake, and unit-blind at run time where the bare
        arrays are combined with no check. Both operands are screened (the
        wrapper may sit on either side via the reflected dunders); a bare literal
        carries no unit and stays lenient.
        """
        other_q = _unwrap(other)
        other_has_unit = (
            isinstance(other_q, pint.Quantity) and not other_q.dimensionless
        )
        if not self.q.dimensionless or other_has_unit:
            right = (
                cast("pint.Unit", other_q.units)
                if isinstance(other_q, pint.Quantity)
                else _DIMENSIONLESS_UNIT
            )
            raise _UnitMixError(
                op=op, left=cast("pint.Unit", self.q.units), right=right
            )

    def _fail_if_ordering_operand_is_invalid(self, other: Any, op: str) -> None:  # noqa: ANN401
        """Screen an operand of an ordering comparison (``<``/``<=``/``>``/``>=``).

        Two unit-carrying operands must be equivalent, exactly as for ``+``/``-``.
        In addition a *bare numeric literal* is rejected here: an ordering
        comparison is unit-blind at run time, so comparing a non-dimensionless
        quantity against a non-zero number silently lends the number that
        quantity's unit — ``wealth > 1_000_000`` reads the bound as currency. Only
        ``0`` (the sign test / floor-at-zero) is allowed inline; a real threshold
        belongs in a parameter, which carries its own unit. When the quantity is
        itself dimensionless (a share, a count, working hours), bare literals are
        fine and stay lenient (GEP 10).
        """
        self._fail_if_other_unit_is_not_equivalent(other=other, op=op)
        other_q = _unwrap(other)
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

    def __bool__(self) -> bool:
        return self._explorer.decide()

    # Ordering comparisons are magnitude comparisons, so they are unit-blind at
    # run time: a non-equivalent unit-carrying operand is a bug (a monthly vs a
    # yearly flow). The explorer still forces the *outcome* (which branch runs);
    # the unit check only screens the operands first.
    def __lt__(self, other: Any) -> _DryRunQuantity:  # noqa: ANN401
        self._fail_if_ordering_operand_is_invalid(other=other, op="<")
        return self._controlled_bool()

    def __le__(self, other: Any) -> _DryRunQuantity:  # noqa: ANN401
        self._fail_if_ordering_operand_is_invalid(other=other, op="<=")
        return self._controlled_bool()

    def __gt__(self, other: Any) -> _DryRunQuantity:  # noqa: ANN401
        self._fail_if_ordering_operand_is_invalid(other=other, op=">")
        return self._controlled_bool()

    def __ge__(self, other: Any) -> _DryRunQuantity:  # noqa: ANN401
        self._fail_if_ordering_operand_is_invalid(other=other, op=">=")
        return self._controlled_bool()

    # ``==``/``!=`` are deliberately *not* unit-screened: they are routinely used
    # polymorphically (sentinels, ``x == 0``) and are not magnitude comparisons.
    # Returning a (non-bool) controlled stand-in is the standard proxy pattern
    # (cf. NumPy arrays); the explorer forces the branch.
    def __eq__(self, other: object) -> _DryRunQuantity:  # ty: ignore[invalid-method-override]
        return self._controlled_bool()

    def __ne__(self, other: object) -> _DryRunQuantity:  # ty: ignore[invalid-method-override]
        return self._controlled_bool()

    # Logical operators (``&`` ``|`` ``^`` ``~``) combine truth values, which are
    # dimensionless. They yield an explorer-controlled boolean stand-in, screening
    # out a non-dimensionless operand first — a boolean-returning body that mixes
    # a real quantity into a logical combination is a bug the run-time arrays would
    # silently swallow (GEP 10).
    def __and__(self, other: Any) -> _DryRunQuantity:  # noqa: ANN401
        self._fail_if_logical_operand_carries_unit(other=other, op="&")
        return self._controlled_bool()

    def __rand__(self, other: Any) -> _DryRunQuantity:  # noqa: ANN401
        self._fail_if_logical_operand_carries_unit(other=other, op="&")
        return self._controlled_bool()

    def __or__(self, other: Any) -> _DryRunQuantity:  # noqa: ANN401
        self._fail_if_logical_operand_carries_unit(other=other, op="|")
        return self._controlled_bool()

    def __ror__(self, other: Any) -> _DryRunQuantity:  # noqa: ANN401
        self._fail_if_logical_operand_carries_unit(other=other, op="|")
        return self._controlled_bool()

    def __xor__(self, other: Any) -> _DryRunQuantity:  # noqa: ANN401
        self._fail_if_logical_operand_carries_unit(other=other, op="^")
        return self._controlled_bool()

    def __rxor__(self, other: Any) -> _DryRunQuantity:  # noqa: ANN401
        self._fail_if_logical_operand_carries_unit(other=other, op="^")
        return self._controlled_bool()

    def __invert__(self) -> _DryRunQuantity:
        if not self.q.dimensionless:
            raise _UnitMixError(
                op="~", left=cast("pint.Unit", self.q.units), right=_DIMENSIONLESS_UNIT
            )
        return self._controlled_bool()

    # Arithmetic propagates real units through the wrapped quantity. Addition and
    # subtraction additionally require equivalent units (see ``_UnitMixError``);
    # multiplication, division, and powers legitimately combine different units,
    # so they are not screened.
    def __add__(self, other: Any) -> _DryRunQuantity:  # noqa: ANN401
        self._fail_if_other_unit_is_not_equivalent(other=other, op="+")
        return self._wrap(self.q + _unwrap(other))

    def __radd__(self, other: Any) -> _DryRunQuantity:  # noqa: ANN401
        self._fail_if_other_unit_is_not_equivalent(other=other, op="+")
        return self._wrap(_unwrap(other) + self.q)

    def __sub__(self, other: Any) -> _DryRunQuantity:  # noqa: ANN401
        self._fail_if_other_unit_is_not_equivalent(other=other, op="-")
        return self._wrap(self.q - _unwrap(other))

    def __rsub__(self, other: Any) -> _DryRunQuantity:  # noqa: ANN401
        self._fail_if_other_unit_is_not_equivalent(other=other, op="-")
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


def _wrap_for_dry_run(value: Any, explorer: _PathExplorer) -> Any:  # noqa: ANN401
    """Wrap unit-carrying representative values; pass framework args through.

    Quantities (and the leaves of dict-param trees) become ``_DryRunQuantity`` so the
    explorer controls branches on them; ``xnp``/``num_segments``/… stay raw.
    """
    if isinstance(value, pint.Quantity):
        return _DryRunQuantity(q=value, explorer=explorer)
    if isinstance(value, dict):
        return {
            key: _wrap_for_dry_run(value=leaf, explorer=explorer)
            for key, leaf in value.items()
        }
    return value


def _opt_out_required_error(qname: str, reason: str) -> str:
    """Message demanding an explicit opt-out for a body the dry-run cannot check.

    A body that cannot be evaluated symbolically is *not* waved through silently
    (GEP 10): the author must mark it ``verify_units=False`` so that every
    un-verified body is a visible, deliberate choice. The declared unit still
    stands and the body's edges are still checked — only its internal inference
    is skipped.
    """
    return (
        f"{qname}: its body cannot be unit-checked by the dry-run ({reason}). "
        f"Set `verify_units=False` on its decorator to opt out of body inference "
        f"— its declared unit and its edges stay checked (GEP 10)."
    )


def _arithmetic_misuse_message(
    qname: str,
    error: _UnitMixError | pint.OffsetUnitCalculusError,
    detail: str,
) -> str:
    """Message for a body that combines quantities unsoundly under ``+``/``-``/order.

    Dispatches the two ways the dry-run catches such a body (GEP 10): an explicit
    :class:`_UnitMixError` (non-equivalent units, a logical operator on a real
    quantity, a bare-literal threshold) or a :class:`pint.OffsetUnitCalculusError`
    raised by pint when a calendar point is used outside its affine algebra.
    """
    if isinstance(error, _UnitMixError):
        return _unit_mix_error_message(qname=qname, mix=error, detail=detail)
    # A calendar point was added to another point, scaled, or mixed across axes.
    return (
        f"{qname}: combines calendar points unsoundly{detail} — two calendar "
        f"points cannot be added (subtract them to get a duration) and a point "
        f"cannot be scaled or mixed across calendar axes; shift a point only by "
        f"a same-axis duration (GEP 10)."
    )


def _unit_mix_error_message(qname: str, mix: _UnitMixError, detail: str) -> str:
    """Message for a body that combined two units unsoundly (GEP 10).

    A logical operator (``&``/``|``/``^``/``~``) carrying a real quantity is
    reported as a non-boolean operand; an ordering comparison against a bare
    non-zero literal is reported as an untagged threshold; ``+``/``-``/an ordering
    comparison of non-equivalent quantities is reported as a unit mix (no run-time
    conversion).
    """
    if mix.literal is not None:
        return (
            f"{qname}: compares '{mix.left}' against the bare literal "
            f"{mix.literal}{detail} — a literal in an ordering comparison silently "
            f"carries the other operand's unit; promote it to a parameter, or "
            f"compare against 0 (GEP 10)."
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
    suffix_level: str,
    boolean_parameters: tuple[str, ...],
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
    """
    explorer = _PathExplorer()
    paths = 0
    while True:
        if paths >= _MAX_PATHS:
            # Truncating exploration must not pass silently (GEP 10): a wrong-unit
            # branch first reached past the cap would otherwise go unchecked. Demand
            # an explicit opt-out, as for the per-run decision cap above.
            return _opt_out_required_error(
                qname,
                f"it explores more than {_MAX_PATHS} branch paths — too many to "
                "check exhaustively",
            )
        paths += 1
        explorer.start_run()
        kwargs = {
            name: _wrap_for_dry_run(value=value, explorer=explorer)
            for name, value in base_kwargs.items()
        }
        for name in boolean_parameters:
            kwargs[name] = _DryRunQuantity(
                q=UNIT_REGISTRY.Quantity(1.0, ""), explorer=explorer
            )
        try:
            result: Any = function(**kwargs)
        except _PathBudgetExceededError:
            return _opt_out_required_error(
                qname,
                f"it makes more than {_MAX_DECISIONS_PER_RUN} branch decisions "
                "in one run — a data-driven loop?",
            )
        except (_UnitMixError, pint.OffsetUnitCalculusError) as err:
            # A unit-blind arithmetic mix (``_UnitMixError``) or a calendar point
            # used outside its affine algebra (``OffsetUnitCalculusError`` from
            # pint: two points added, a point scaled, axes mixed) — GEP 10.
            detail = " on a conditional branch" if explorer.on_a_branch else ""
            return _arithmetic_misuse_message(qname=qname, error=err, detail=detail)
        except Exception:  # noqa: BLE001
            return _opt_out_required_error(
                qname,
                "it uses an operation pint cannot evaluate symbolically — a "
                "piecewise polynomial, a lookup table, `join`, or a raw `xnp` op",
            )
        detail = " on a conditional branch" if explorer.on_a_branch else ""
        error = _inferred_result_error(
            qname=qname,
            inferred=_unwrap(result),
            declared=declared,
            suffix_level=suffix_level,
            detail=detail,
        )
        if error is not None:
            return error
        if not explorer.advance():
            break
    return None


def _unit_without_grouping_levels(unit: pint.Unit) -> pint.Unit:
    """A unit with every grouping-level component (numerator or denominator) removed.

    The level-free *residual* used to compare an inferred unit against the
    declaration on its physical content alone (currency, period, area, …) — the
    grouping level is screened separately under the index-vs-unit rule (GEP 10).
    Both a denominator level (a ``…/[hh]`` total) and a numerator level (a head
    count's ``[person]/…``) are divided out.
    """
    quantity = UNIT_REGISTRY.Quantity(1.0, unit)
    for dimension, exponent in dict(quantity.dimensionality).items():
        if isinstance(exponent, complex):  # pint exponents are real; narrow for ty
            continue
        if dimension.startswith(_GROUPING_LEVEL_DIM_PREFIX):
            level = dimension[len(_GROUPING_LEVEL_DIM_PREFIX) : -1]
            level_unit = UNIT_REGISTRY.Quantity(1.0, f"{_GROUPING_LEVEL_PREFIX}{level}")
            # Cancel the level: multiply by its unit raised to the *negated*
            # exponent (a `…/[hh]` denominator, exponent -1, is multiplied by
            # `[hh] ** 1`).
            quantity = quantity * level_unit ** (-exponent)
    return cast("pint.Unit", quantity.units)


def _inferred_result_error(
    qname: str,
    inferred: Any,  # noqa: ANN401
    declared: pint.Unit,
    suffix_level: str,
    detail: str,
) -> str | None:
    """Check one dry-run result against the declaration (GEP 10).

    An opaque return — a dataclass, a tuple, … — is neither a checkable quantity
    nor a plain scalar, so it must opt out. A concrete ``Quantity`` must match the
    declaration on two axes, checked separately:

    - its **physical content** (currency, period, area, …) — the unit with every
      grouping level divided out — must equal the declaration's; and
    - its **grouping level**, under the index-vs-unit rule: *when the inferred unit
      carries a level denominator it must equal the name's aggregation-suffix
      level*; a level-less inferred unit is exempt (its index level is the
      structural system's concern, not the unit check's). So a per-person rent
      share mis-named ``…_hh`` (inferred ``…/[person]``, suffix ``[hh]``) is caught,
      while a level-less ``MIN``-of-age at ``_fg`` passes.

    A *dimensionless* inference is deliberately not flagged against a concrete
    declaration: it is unit-polymorphic — exactly what an identifier, a head count,
    or a share produces (``p_id * 2.0``), legitimately standing in for the
    magnitude of a concrete quantity — and separating a genuine cancellation
    (``wealth / income`` declared ``CURRENCY``) from it would need operand
    provenance the dry-run does not track. A bare-literal ``return 0.0`` is not a
    ``Quantity``, so it is a plain scalar and falls through cleanly.
    """
    if not isinstance(
        inferred, pint.Quantity | int | float | numpy.number | numpy.bool_
    ):
        return _opt_out_required_error(
            qname,
            "it returns a value the dry-run cannot unit-check — a dataclass, "
            "a tuple, or another non-scalar",
        )
    if not isinstance(inferred, pint.Quantity) or inferred.dimensionless:
        return None
    inferred_unit = cast("pint.Unit", inferred.units)
    if not units_are_equivalent(
        left=_unit_without_grouping_levels(inferred_unit),
        right=_unit_without_grouping_levels(declared),
    ):
        return (
            f"{qname}: declares '{declared}' but its body infers "
            f"'{inferred_unit}'{detail}."
        )
    inferred_level = _unit_level_denominator(inferred_unit)
    if inferred_level is not None and inferred_level != suffix_level:
        return (
            f"{qname}: its body infers a '[{inferred_level}]'-level result "
            f"('{inferred_unit}'){detail}, but its name's aggregation suffix is "
            f"'[{suffix_level}]' (it resolves to '{declared}'). A unit's grouping "
            f"level, when present, must match the name's suffix level (GEP 10)."
        )
    return None
