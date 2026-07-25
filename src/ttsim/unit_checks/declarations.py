"""Environment-wide checks of declarations against each other and against names.

The entry points a caller reaches for: every active node must declare a unit
(:func:`fail_if_environment_units_are_missing`), every declaration must agree
with what its node produces (:func:`fail_if_environment_units_are_inconsistent`),
and a user's input-column tags must agree with the units the environment
declares for those columns (:func:`fail_if_input_units_are_inconsistent`).

The comparisons are made against the pint units
:mod:`ttsim.unit_checks.resolution` produces and the type-level contracts
:mod:`ttsim.unit_checks.contracts` resolves. The one check that needs to look
inside a function is body verification, which is delegated to
:mod:`ttsim.unit_checks.execution` and whose errors are collected here alongside
the declaration-only ones.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import (
    Any,
    cast,
)

import dags.tree as dt
import pint

from ttsim.exceptions import (
    UnitConsistencyError,
)
from ttsim.tt._source_rewriting import recompile_with_logical_ops_as_calls
from ttsim.tt.column_objects_param_function import (
    AggByGroupFunction,
    ColumnFunction,
    ColumnObject,
    ParamFunction,
    PolicyFunction,
)
from ttsim.tt.currencies import UnitSystem
from ttsim.tt.param_objects import (
    ParamMappingObject,
    ParamObject,
    RawParam,
)
from ttsim.tt.units import (
    UNSET_UNIT,
    CompositeUnit,
    InputOutputUnit,
    UnitAnnotatedColumn,
    UserNestedUnitAnnotatedData,
    _unit_level_denominator,
    fail_if_units_are_missing,
    pint_unit_has_currency,
    resolve_ttsim_unit,
    resolve_ttsim_unit_for_param,
    ttsim_unit_currency,
    ttsim_unit_has_agnostic_currency,
    ttsim_unit_with_agnostic_currency,
    unit_residual_excluding_currency_and_flow_period,
    units_are_equivalent,
)
from ttsim.typing import (
    OrderedQNames,
    SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
)
from ttsim.unit_checks.contracts import (
    _fail_if_structured_field_annotations_are_invalid,
    _returns_a_schedule,
    _schedule_param_function_contract_errors,
)
from ttsim.unit_checks.execution import (
    _NON_UNIT_ARGUMENT_VALUES,
    _base_unit_check_kwargs,
    _PathExplorer,
    _representative_values_by_qname,
    _unit_check_helper_stand_ins,
    _UnitCheckSchedule,
    _verify_one_body,
)
from ttsim.unit_checks.resolution import (
    FRAMEWORK_DATE_NODE_UNITS,
    _composite_token_level,
    node_is_boolean,
    resolve_environment_units,
)


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
