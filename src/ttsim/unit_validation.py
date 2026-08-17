"""Check unit declarations against each other across an assembled environment."""

from __future__ import annotations

import dataclasses
import datetime
import inspect
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import (
    Any,
    cast,
    get_args,
)

import dags.tree as dt
import pint

from ttsim._quantity_kinds import (
    documented_quantity_kind,
    quantity_kind,
    quantity_kind_for_leaf,
    quantity_kind_for_scalar_type,
)
from ttsim._unit_inference import (
    body_error_is_unsupported,
    body_verification_errors,
    function_uses_local_cast,
)
from ttsim.exceptions import UnitConsistencyError
from ttsim.interface_dag_elements.shared import FRAMEWORK_PARTIAL_ARGUMENTS
from ttsim.tt.column_objects_param_function import (
    AggByGroupFunction,
    AggByPIDFunction,
    ColumnFunction,
    ColumnObject,
    GroupCreationFunction,
    ParamFunction,
    PolicyFunction,
    TimeConversionFunction,
)
from ttsim.tt.param_objects import (
    ParamMappingObject,
    ParamObject,
    RawParam,
)
from ttsim.tt.units import (
    UNSET_UNIT,
    CompositeUnit,
    InputOutputUnits,
    QuantityKind,
    UnitAnnotatedColumn,
    UnitDeclaration,
    UnitSystem,
    UnsetUnit,
    _flow_period_of,
    _pint_unit_currency,
    _unit_level_denominator,
    _unit_without_grouping_levels,
    fail_if_units_are_missing,
    pint_unit_from_ttsim_unit,
    pint_unit_from_ttsim_unit_for_column,
    pint_unit_has_currency,
    ttsim_unit_currency,
    ttsim_unit_has_agnostic_currency,
    ttsim_unit_with_agnostic_currency,
    units_are_equivalent,
)
from ttsim.typing import (
    OrderedQNames,
    SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
    UserNestedUnitAnnotatedData,
)
from ttsim.unit_resolution import (
    _SCHEDULE_RETURN_TYPE_NAMES,
    FRAMEWORK_DATE_NODE_UNITS,
    _composite_token_level,
    _fail_if_structured_field_annotations_are_invalid,
    _resolvable_type_hints,
    _resolved_return_structure,
    _return_annotation_name,
    _returns_a_schedule,
    _schedule_axis_errors,
    _spell_ttsim_unit,
    resolve_environment_units,
)


def fail_if_environment_units_are_missing(
    env: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
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
                # A flat int-keyed dict
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
    if resolved_pint_units is None:
        resolved_pint_units = resolve_environment_units(
            env=env, grouping_levels=grouping_levels, unit_system=unit_system
        )
    _fail_if_structured_field_annotations_are_invalid(env=env, unit_system=unit_system)
    errors = _non_body_consistency_errors(
        env=env,
        resolved_pint_units=resolved_pint_units,
        registry=unit_system.registry,
    )
    errors.extend(
        body_verification_errors(
            env=env,
            resolved_pint_units=resolved_pint_units,
            unit_system=unit_system,
        )
    )
    if errors:
        raise UnitConsistencyError(
            "Environment unit-consistency check failed:\n  " + "\n  ".join(errors)
        )


@dataclass(frozen=True)
class UncheckedBody:
    """A function body omitted from inference, together with the exact reason."""

    qname: str
    reason: str


@dataclass(frozen=True)
class UnitValidationReport:
    """Auditable coverage of one or more policy-environment unit checks."""

    resolved_declarations: tuple[str, ...]
    checked_function_bodies: tuple[str, ...]
    checked_aggregations: tuple[str, ...]
    generated_rules: tuple[str, ...]
    local_casts: tuple[str, ...]
    body_opt_outs: tuple[str, ...]
    unsupported_bodies: tuple[UncheckedBody, ...]
    other_unchecked_bodies: tuple[UncheckedBody, ...]
    policy_date_regimes: tuple[datetime.date, ...]

    def summary(self) -> str:
        """Return a compact CI-oriented coverage summary with named exceptions."""
        lines = [
            f"Declarations resolved: {len(self.resolved_declarations)}",
            f"Bodies checked: {len(self.checked_function_bodies)}",
            f"Aggregations checked: {len(self.checked_aggregations)}",
            f"Generated rules: {len(self.generated_rules)}",
            f"Casts: {len(self.local_casts)}",
            f"Body opt-outs: {len(self.body_opt_outs)}",
            f"Unsupported bodies: {len(self.unsupported_bodies)}",
            f"Other unchecked bodies: {len(self.other_unchecked_bodies)}",
            f"Date regimes: {len(self.policy_date_regimes)}",
        ]
        if self.local_casts:
            lines.append("Cast functions: " + ", ".join(self.local_casts))
        if self.body_opt_outs:
            lines.append("Body opt-outs: " + ", ".join(self.body_opt_outs))
        lines.extend(
            f"Unsupported body {item.qname}: {item.reason}"
            for item in self.unsupported_bodies
        )
        lines.extend(
            f"Unchecked body {item.qname}: {item.reason}"
            for item in self.other_unchecked_bodies
        )
        return "\n".join(lines)


def create_unit_validation_report(
    env: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
    grouping_levels: OrderedQNames,
    unit_system: UnitSystem,
    policy_dates: tuple[datetime.date, ...] = (),
) -> UnitValidationReport:
    """Validate an environment and report what provided each kind of evidence.

    The report deliberately keeps declarations, inferred bodies, generated rules,
    local assertions, and whole-body opt-outs separate. Invalid declarations still
    raise immediately. A body rejected by inference is returned under
    ``unsupported_bodies`` so diagnostic callers can inspect the report; ordinary
    environment assembly and :func:`check_policy_environment_units` remain fail closed.
    """
    fail_if_environment_units_are_missing(env)
    resolved = resolve_environment_units(
        env=env, grouping_levels=grouping_levels, unit_system=unit_system
    )
    _fail_if_structured_field_annotations_are_invalid(env=env, unit_system=unit_system)
    declaration_errors = _non_body_consistency_errors(
        env=env,
        resolved_pint_units=resolved,
        registry=unit_system.registry,
    )
    if declaration_errors:
        raise UnitConsistencyError(
            "Environment unit-consistency check failed:\n  "
            + "\n  ".join(declaration_errors)
        )
    body_errors = body_verification_errors(
        env=env,
        resolved_pint_units=resolved,
        unit_system=unit_system,
    )
    invalid_body_errors = [
        error for error in body_errors if not body_error_is_unsupported(error)
    ]
    if invalid_body_errors:
        raise UnitConsistencyError(
            "Environment unit-consistency check failed:\n  "
            + "\n  ".join(invalid_body_errors)
        )
    unsupported = tuple(
        UncheckedBody(*_split_body_error(error))
        for error in body_errors
        if body_error_is_unsupported(error)
    )
    unsupported_qnames = {item.qname for item in unsupported}

    generated_rules = tuple(
        sorted(
            qname
            for qname, obj in env.items()
            if isinstance(obj, GroupCreationFunction | TimeConversionFunction)
            or (
                isinstance(obj, AggByGroupFunction | AggByPIDFunction)
                and obj.orig_location == "automatically generated"
            )
        )
    )
    casts = tuple(
        sorted(
            qname
            for qname, obj in env.items()
            if isinstance(obj, PolicyFunction | ParamFunction)
            and function_uses_local_cast(obj.function)
        )
    )

    checked, checked_aggregations, opt_outs, other = _classify_body_coverage(
        env=env,
        resolved=resolved,
        unsupported_qnames=unsupported_qnames,
    )

    return UnitValidationReport(
        resolved_declarations=tuple(sorted(resolved)),
        checked_function_bodies=tuple(sorted(checked)),
        checked_aggregations=tuple(sorted(checked_aggregations)),
        generated_rules=generated_rules,
        local_casts=casts,
        body_opt_outs=tuple(sorted(opt_outs)),
        unsupported_bodies=unsupported,
        other_unchecked_bodies=tuple(sorted(other, key=lambda item: item.qname)),
        policy_date_regimes=tuple(sorted(set(policy_dates))),
    )


def _classify_body_coverage(
    env: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
    resolved: Mapping[str, pint.Unit | dict[str | int, Any]],
    unsupported_qnames: set[str],
) -> tuple[list[str], list[str], list[str], list[UncheckedBody]]:
    """Classify every human-written function body in the report."""
    checked: list[str] = []
    checked_aggregations: list[str] = []
    opt_outs: list[str] = []
    other: list[UncheckedBody] = []
    for qname, obj in env.items():
        if isinstance(obj, AggByGroupFunction | AggByPIDFunction):
            if obj.orig_location != "automatically generated":
                if obj.verify_units:
                    checked_aggregations.append(qname)
                else:
                    opt_outs.append(qname)
            continue
        if not isinstance(obj, PolicyFunction | ParamFunction):
            continue
        category, unchecked = _classify_policy_body(
            qname=qname,
            obj=obj,
            resolved=resolved,
            unsupported_qnames=unsupported_qnames,
            env=env,
        )
        if category == "checked":
            checked.append(qname)
        elif category == "opt-out":
            opt_outs.append(qname)
        elif unchecked is not None:
            other.append(unchecked)
    return checked, checked_aggregations, opt_outs, other


def _classify_policy_body(
    qname: str,
    obj: PolicyFunction | ParamFunction,
    resolved: Mapping[str, pint.Unit | dict[str | int, Any]],
    unsupported_qnames: set[str],
    env: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
) -> tuple[str, UncheckedBody | None]:
    """Return the report category for one ordinary policy or parameter body."""
    if getattr(obj, "fail_msg_if_included", None) is not None:
        category = "other"
        unchecked = UncheckedBody(qname, "unimplemented-period failure stub")
    elif not obj.verify_units:
        category, unchecked = "opt-out", None
    elif isinstance(obj, ParamFunction) and _returns_a_schedule(obj):
        category = "other"
        unchecked = UncheckedBody(
            qname,
            "schedule-construction body; its declared input/output axes are checked "
            "at consumers",
        )
    elif qname not in resolved:
        category = "other"
        unchecked = UncheckedBody(qname, "no resolved scalar declaration")
    elif isinstance(resolved[qname], dict):
        category = "other"
        unchecked = UncheckedBody(qname, "structured return value")
    elif qname in unsupported_qnames:
        category, unchecked = "unsupported", None
    elif _body_has_unresolved_producer(obj=obj, resolved=resolved, env=env):
        category = "other"
        unchecked = UncheckedBody(qname, "at least one producer has no resolved unit")
    else:
        category, unchecked = "checked", None
    return category, unchecked


def merge_unit_validation_reports(
    reports: Iterable[UnitValidationReport],
) -> UnitValidationReport:
    """Combine per-regime reports into one policy-history coverage report."""
    materialized = tuple(reports)

    def merged_strings(attribute: str) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    item
                    for report in materialized
                    for item in cast("tuple[str, ...]", getattr(report, attribute))
                }
            )
        )

    def merged_unchecked(attribute: str) -> tuple[UncheckedBody, ...]:
        return tuple(
            sorted(
                {
                    item
                    for report in materialized
                    for item in cast(
                        "tuple[UncheckedBody, ...]", getattr(report, attribute)
                    )
                },
                key=lambda item: (item.qname, item.reason),
            )
        )

    return UnitValidationReport(
        resolved_declarations=merged_strings("resolved_declarations"),
        checked_function_bodies=merged_strings("checked_function_bodies"),
        checked_aggregations=merged_strings("checked_aggregations"),
        generated_rules=merged_strings("generated_rules"),
        local_casts=merged_strings("local_casts"),
        body_opt_outs=merged_strings("body_opt_outs"),
        unsupported_bodies=merged_unchecked("unsupported_bodies"),
        other_unchecked_bodies=merged_unchecked("other_unchecked_bodies"),
        policy_date_regimes=tuple(
            sorted(
                {date for report in materialized for date in report.policy_date_regimes}
            )
        ),
    )


def _split_body_error(error: str) -> tuple[str, str]:
    """Split the checker's stable ``<qname>: <reason>`` diagnostic shape."""
    qname, separator, reason = error.partition(": ")
    if not separator:
        return "<unknown>", error
    return qname, reason


def _non_body_consistency_errors(
    env: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
    resolved_pint_units: dict[str, pint.Unit | dict[str | int, Any]],
    registry: pint.UnitRegistry,
) -> list[str]:
    """Return declaration, aggregation, rounding, and schedule-contract errors."""
    errors: list[str] = _dimensionless_group_declaration_errors(env=env)
    errors.extend(
        _aggregation_declaration_errors(
            env=env,
            resolved_pint_units=resolved_pint_units,
            registry=registry,
        )
    )
    errors.extend(_rounding_spec_declaration_errors(env=env))
    errors.extend(_schedule_param_function_contract_errors(env=env))
    return errors


def _body_has_unresolved_producer(
    obj: PolicyFunction | ParamFunction,
    resolved: Mapping[str, pint.Unit | dict[str | int, Any]],
    env: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
) -> bool:
    return any(
        name not in FRAMEWORK_PARTIAL_ARGUMENTS
        and name not in resolved
        and not (
            isinstance((producer := env.get(name)), ParamFunction)
            and isinstance(producer.unit, UnsetUnit)
        )
        for name in inspect.signature(obj.function).parameters
    )


def _dimensionless_group_declaration_errors(
    env: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
) -> list[str]:
    """Reject group markers on dimensionless shares, rates, ids, and categories.

    A direct ``DIMENSIONLESS.PER_<LEVEL>`` declaration needs evidence outside
    the unit itself that the value is a count or yes/no indicator. Generated
    counts and indicators provide that evidence by rule; a direct integer must
    document its count interpretation.
    """
    errors: list[str] = []
    for qname, obj in env.items():
        # Aggregations derive their grouping level from their operation and source.
        # Their declarations are checked independently below against that derivation,
        # so they are not direct assertions about a dimensionless quantity's kind.
        if isinstance(obj, AggByGroupFunction | TimeConversionFunction):
            continue
        errors.extend(
            f"{declaration.where}: declares `{declaration.token}`, but a group marker "
            "on a "
            "dimensionless value is reserved for a known count or yes/no "
            "indicator. Shares, rates, identifiers, and categories stay "
            "bare `DIMENSIONLESS`; for an integer count, document that "
            "interpretation explicitly (GEP 10)."
            for declaration in _declared_quantities(qname=qname, obj=obj, env=env)
            if (
                declaration.token.base == "DIMENSIONLESS"
                and declaration.token.level is not None
                and declaration.kind not in (QuantityKind.COUNT, QuantityKind.INDICATOR)
            )
        )
    return errors


@dataclass(frozen=True)
class _DeclaredQuantity:
    """One declaration token and the evidence belonging to exactly that token."""

    where: str
    token: CompositeUnit
    kind: QuantityKind


def _declared_quantities(
    qname: str,
    obj: Any,  # noqa: ANN401
    env: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
) -> list[_DeclaredQuantity]:
    """Return scalar, mapping-leaf, structured-field, and schedule-axis evidence."""
    declaration = getattr(obj, "unit", UNSET_UNIT)
    if isinstance(declaration, InputOutputUnits):
        return _schedule_axis_declarations(where=qname, declaration=declaration)
    if isinstance(declaration, CompositeUnit):
        return [
            _DeclaredQuantity(
                where=qname,
                token=declaration,
                kind=quantity_kind(qname=qname, obj=obj, env=env),
            )
        ]
    if isinstance(declaration, Mapping):
        values = dt.flatten_to_qnames(
            cast("Mapping[str, Any]", getattr(obj, "value", {}))
        )
        return [
            _DeclaredQuantity(
                where=f"{qname}[{leaf}]",
                token=token,
                kind=quantity_kind_for_leaf(
                    qname=f"{qname}__{leaf}", value=values.get(leaf)
                ),
            )
            for leaf, token in dt.flatten_to_qnames(
                cast("Mapping[str, Any]", declaration)
            ).items()
            if isinstance(token, CompositeUnit)
        ]

    out: list[_DeclaredQuantity] = []
    if isinstance(obj, ParamMappingObject | RawParam):
        axis_kind = documented_quantity_kind(qname=qname, obj=obj)
        for axis_name in ("input_unit", "output_unit"):
            token = getattr(obj, axis_name, UNSET_UNIT)
            if isinstance(token, CompositeUnit):
                out.append(
                    _DeclaredQuantity(
                        where=f"{qname} ({axis_name})",
                        token=token,
                        kind=axis_kind,
                    )
                )
    if isinstance(obj, ParamFunction) and isinstance(declaration, UnsetUnit):
        cls, item_cls = _resolved_return_structure(obj.function)
        for structured_cls in (cls, item_cls):
            if structured_cls is not None:
                out.extend(
                    _structured_declarations(
                        qname=qname, cls=structured_cls, visited=set()
                    )
                )
    return out


def _schedule_axis_declarations(
    where: str, declaration: InputOutputUnits
) -> list[_DeclaredQuantity]:
    """Return one evidence record per schedule input/output axis."""
    input_units: tuple[CompositeUnit, ...] = (
        cast("tuple[CompositeUnit, ...]", declaration.input_unit)
        if isinstance(declaration.input_unit, tuple)
        else (declaration.input_unit,)
    )
    input_kinds: tuple[QuantityKind, ...] = (
        declaration.input_kind
        if isinstance(declaration.input_kind, tuple)
        else (declaration.input_kind,) * len(input_units)
    )
    if len(input_units) != len(input_kinds):
        return [
            _DeclaredQuantity(
                where=f"{where} (input_kind arity mismatch)",
                token=unit,
                kind=QuantityKind.GENERIC,
            )
            for unit in input_units
        ]
    out = [
        _DeclaredQuantity(
            where=f"{where} (input axis {position})",
            token=unit,
            kind=kind,
        )
        for position, (unit, kind) in enumerate(
            zip(input_units, input_kinds, strict=True), start=1
        )
    ]
    out.append(
        _DeclaredQuantity(
            where=f"{where} (output axis)",
            token=declaration.output_unit,
            kind=declaration.output_kind,
        )
    )
    return out


def _structured_declarations(
    qname: str,
    cls: type,
    visited: set[type],
) -> list[_DeclaredQuantity]:
    """Return exact evidence for every annotated scalar/schedule field."""
    if cls in visited:
        return []
    visited.add(cls)
    hints = _resolvable_type_hints(cls=cls)
    out: list[_DeclaredQuantity] = []
    for field in dataclasses.fields(cast("Any", cls)):
        hint = hints.get(field.name, field.type)
        metadata = getattr(hint, "__metadata__", ())
        base = get_args(hint)[0] if hasattr(hint, "__metadata__") else hint
        field_qname = f"{qname}__{field.name}"
        out.extend(
            _DeclaredQuantity(
                where=f"Field '{cls.__name__}.{field.name}'",
                token=token,
                kind=quantity_kind_for_scalar_type(qname=field_qname, scalar_type=base),
            )
            for token in metadata
            if isinstance(token, CompositeUnit)
        )
        for io_token in (
            token for token in metadata if isinstance(token, InputOutputUnits)
        ):
            out.extend(
                _schedule_axis_declarations(
                    where=f"Field '{cls.__name__}.{field.name}'",
                    declaration=io_token,
                )
            )
        if isinstance(base, type) and dataclasses.is_dataclass(base):
            out.extend(
                _structured_declarations(qname=field_qname, cls=base, visited=visited)
            )
    return out


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
        errors.extend(
            _schedule_axis_errors(
                builds_lookup_table=(
                    return_type_name == "ConsecutiveIntLookupTableParamValue"
                ),
                input_unit=obj.unit.input_unit,
                where=qname,
            )
        )
        if obj.verify_units:
            errors.append(
                f"{qname}: declares `unit=InputOutputUnits(...)` but leaves "
                f"`verify_units=True`; a schedule builder's body builds a table, not "
                f"a scalar, so it cannot be unit-verified — state "
                f"`verify_units=False` explicitly (GEP 10)."
            )
    return errors


def _measurement_unit(units: pint.Unit, registry: pint.UnitRegistry) -> pint.Unit:
    """Return a unit without currency, flow-period, or grouping-level components."""
    currency = _pint_unit_currency(units=units, registry=registry)
    residual = units / currency if currency is not None else units
    period = _flow_period_of(units=residual, registry=registry)
    residual = residual * period if period is not None else residual
    return _unit_without_grouping_levels(unit=residual, registry=registry)
