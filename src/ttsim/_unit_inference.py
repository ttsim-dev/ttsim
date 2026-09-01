"""Infer the unit of every DAG node's body and check it against the declaration.

GEP 10 specifies the declaration rules the inferred units are checked against.
"""

from __future__ import annotations

import ast
import functools
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import (
    Any,
    NoReturn,
    cast,
)

import numpy
import pint

from ttsim._quantity_kinds import (
    QuantityKind,
    QuantityKindTree,
    quantity_kinds_by_qname,
)
from ttsim.exceptions import (
    TTSIMError,
    UnitDefinitionError,
)
from ttsim.interface_dag_elements.shared import (
    FRAMEWORK_PARTIAL_ARGUMENTS,
)
from ttsim.tt._function_rewriting import (
    func_to_ast,
    recompile_with_logical_ops_as_calls,
)
from ttsim.tt.column_objects_param_function import (
    ColumnObject,
    ParamFunction,
    PolicyFunction,
)
from ttsim.tt.param_objects import (
    DictParam,
    ParamMappingObject,
    RawParam,
)
from ttsim.tt.units import (
    _GROUPING_LEVEL_PREFIX,
    TIME_UNIT_ID_TO_PINT_NAME,
    CompositeUnit,
    InputOutputUnits,
    UnitSystem,
    UnsetUnit,
    _grouping_levels_with_exponent,
    _unit_level_denominator,
    _unit_without_grouping_levels,
    cast_ttsim_unit,
    is_calendar_ordinal_unit,
    is_calendar_point_unit,
    pint_unit_from_ttsim_unit_for_column,
    ttsim_unit_from_yaml_value,
    units_are_equivalent,
)
from ttsim.typing import (
    SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
)
from ttsim.unit_resolution import (
    _dimensionless_unit,
    _resolve_input_axes,
    _resolve_schedule_input_unit,
    _resolved_return_structure,
    _returns_a_schedule,
    _ScalarFieldKind,
    _ScheduleFieldKind,
    _structured_field_kinds,
    node_is_boolean,
)


def body_verification_errors(
    env: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
    resolved_pint_units: Mapping[str, pint.Unit | dict[str | int, Any]],
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
    representative_values = _representative_values_by_qname(
        env=env, resolved_pint_units=resolved_pint_units, unit_system=unit_system
    )
    quantity_kinds = quantity_kinds_by_qname(env)
    boolean_nodes = {
        qname
        for qname, obj in env.items()
        if isinstance(obj, ColumnObject | ParamFunction)
        and node_is_boolean(qname=qname, obj=obj)
    }
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
                extra_globals=_unit_check_scope_bindings(
                    function=obj.function,
                    stand_ins=unit_check_helper_stand_ins,
                ),
            ),
            declared=declared,
            boolean_values=boolean_values,
            base_kwargs=base_kwargs,
            unit_system=unit_system,
            explorer_holder=explorer_holder,
            quantity_kinds=quantity_kinds,
        )
        if error is not None:
            errors.append(error)
    return errors


def body_error_is_unsupported(error: str) -> bool:
    """Whether ``error`` means symbolic evaluation is unsupported, not invalid."""
    return isinstance(error, _UnsupportedBodyError)


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
    for value in representative_values.values():
        if isinstance(value, _UnitCheckSchedule):
            value.explorer_holder = explorer_holder


def _has_grouping_level_numerator(unit: pint.Unit) -> bool:
    """Whether a unit carries a grouping level as a *numerator*."""
    return any(exponent > 0 for _, exponent in _grouping_levels_with_exponent(unit))


def _has_grouping_component(unit: pint.Unit) -> bool:
    """Whether a unit carries any grouping level, in numerator or denominator."""
    return bool(_grouping_levels_with_exponent(unit))


# A dataclass, not a `NamedTuple`: under `from __future__ import annotations` the
# `__new__` a `NamedTuple` synthesizes carries stringified annotations that the
# package claw cannot resolve, so beartype skips it with a warning.
@dataclass(frozen=True)
class BooleanLevel:
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
    if not _is_dimensionless_up_to_grouping_level(unit=unit, registry=registry):
        return BooleanLevel(is_boolean=False, level=None)
    return BooleanLevel(is_boolean=True, level=_unit_level_denominator(unit))


def _is_dimensionless_up_to_grouping_level(
    unit: pint.Unit, registry: pint.UnitRegistry
) -> bool:
    """Whether ``unit`` is a plain number, apart from a grouping-level denominator.

    ``1``, ``1 / [fam]`` → ``True``; ``EUR_PER_MONTH``, ``[hh]``, a calendar point
    → ``False``. This is what the unit model can say about the values that carry no
    physical content — truth values, identifiers, shares, counts. It cannot tell
    them apart from one another (GEP 10 leaves semantic kinds to a later stage), so
    a screen built on it rejects physical content rather than certifying a kind.
    """
    if _has_grouping_level_numerator(unit):
        return False
    return registry.Quantity(
        1.0, _unit_without_grouping_levels(unit=unit, registry=registry)
    ).dimensionless


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


def _verify_one_body(
    qname: str,
    function: Any,  # noqa: ANN401  (a scalar body, possibly a dags wrapper)
    declared: pint.Unit,
    boolean_values: Mapping[str, Any],
    base_kwargs: dict[str, Any],
    unit_system: UnitSystem,
    explorer_holder: list[_PathExplorer | None],
    quantity_kinds: Mapping[str, QuantityKindTree],
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
                kind=quantity_kinds.get(name, QuantityKind.GENERIC),
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
    except _ReductionNotEvaluableError as err:
        return _opt_out_required_error(
            qname=qname,
            reason=f"it reduces an array in the body (`xnp.{err.op}`) — a reduction "
            "changes which rows the result belongs to, and the unit check has no "
            "array-axis information to derive the result's grouping level from; an "
            "aggregation node states that level instead",
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
        _CalendarOrdinalArithmeticError,
        _UnsupportedGroupArithmeticError,
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
    error = _inferred_result_error(
        qname=qname,
        inferred=_unwrap(result),
        declared=declared,
        detail=explorer.branch_detail(),
        unit_system=unit_system,
    )
    return error, isinstance(error, _UnsupportedBodyError)


def _representative_values_by_qname(
    env: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
    resolved_pint_units: Mapping[str, pint.Unit | dict[str | int, Any]],
    unit_system: UnitSystem,
) -> dict[str, Any]:
    """Representative unit-check values for every unit-resolved node.

    - A ``piecewise_*``/lookup-table parameter becomes a :class:`_UnitCheckSchedule`
      carrying its input/output axes, so a consumer's ``piecewise_polynomial`` /
      ``look_up`` call resolves to the output unit.
    - A schedule-building param function (``unit=InputOutputUnits(...)``) becomes a
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
                output_kind=cast("CompositeUnit", obj.output_unit).kind,
                unit_system=unit_system,
            )
        elif (
            isinstance(obj, ParamFunction)
            and _returns_a_schedule(obj)
            and isinstance(obj.unit, InputOutputUnits)
            and not isinstance(unit, dict)
        ):
            # A schedule builder declares its axes with `unit=InputOutputUnits(...)`:
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
                output_kind=obj.unit.output_unit.kind,
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
        if isinstance(obj, ParamFunction) and isinstance(obj.unit, UnsetUnit):
            out[qname] = _param_function_stand_in(
                qname=qname, obj=obj, unit_system=unit_system
            )
    return out


def _param_function_stand_in(
    qname: str,
    obj: ParamFunction,
    unit_system: UnitSystem,
) -> _UnitCheckStructuredValue:
    """The unit check's stand-in for a structured param-function output.

    A ``@param_function(unit=UNSET_UNIT)`` builds a dataclass of related
    parameters, so its stand-in is a :class:`_UnitCheckStructuredValue` typed with
    the return dataclass where one resolves — annotated plucks then carry their
    field units, and an unannotated pluck stays opaque and is cast at the site.
    A schedule builder (``unit=InputOutputUnits(...)``) is handled separately as a
    :class:`_UnitCheckSchedule` in :func:`_representative_values_by_qname`.
    """
    cls, item_cls = _resolved_return_structure(obj.function)
    return _UnitCheckStructuredValue(
        producer=qname,
        unit_system=unit_system,
        cls=cls,
        item_cls=item_cls,
    )


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


class _ObjectWithCastStandIn:
    """Delegate every attribute except the canonical unit cast to an object."""

    def __init__(self, wrapped: Any, cast_stand_in: Any) -> None:  # noqa: ANN401
        self._wrapped = wrapped
        self.cast_ttsim_unit = cast_stand_in

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        return getattr(self._wrapped, name)


def _function_scope(function: Any) -> dict[str, Any]:  # noqa: ANN401
    """Return the globals and dereferenced closure cells visible to a function."""
    scope = dict(getattr(function, "__globals__", {}))
    closure = getattr(function, "__closure__", None)
    code = getattr(function, "__code__", None)
    if closure and code is not None:
        scope.update(
            dict(
                zip(
                    code.co_freevars,
                    (cell.cell_contents for cell in closure),
                    strict=True,
                )
            )
        )
    return scope


def _unit_check_scope_bindings(
    function: Any,  # noqa: ANN401
    stand_ins: Mapping[str, Any],
) -> dict[str, Any]:
    """Add every true alias of ``cast_ttsim_unit`` to checker stand-ins.

    Identity, not spelling, is authoritative. Import aliases and closure aliases
    are rebound directly. A module alias is replaced by a delegating proxy whose
    cast attribute is the stand-in, so its other helpers remain untouched.
    """
    cast_stand_in = stand_ins["cast_ttsim_unit"]
    bindings = {
        name: value for name, value in stand_ins.items() if name != "cast_ttsim_unit"
    }
    for name, value in _function_scope(function).items():
        if value is cast_ttsim_unit:
            bindings[name] = cast_stand_in
        else:
            try:
                attribute = inspect.getattr_static(value, "cast_ttsim_unit")
            except AttributeError:
                continue
            if attribute is cast_ttsim_unit:
                bindings[name] = _ObjectWithCastStandIn(value, cast_stand_in)
    return bindings


def _cast_aliases_in_scope(scope: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    """Return direct and module aliases that resolve to the canonical cast."""
    direct_aliases = {name for name, value in scope.items() if value is cast_ttsim_unit}
    module_aliases: set[str] = set()
    for name, value in scope.items():
        try:
            attribute = inspect.getattr_static(value, "cast_ttsim_unit")
        except AttributeError:
            continue
        if attribute is cast_ttsim_unit:
            module_aliases.add(name)
    return direct_aliases, module_aliases


def _assignment_rhs_is_cast(
    node: ast.Assign,
    aliases: set[str],
    module_aliases: set[str],
) -> bool:
    """Whether an assignment copies a known canonical-cast alias."""
    return (isinstance(node.value, ast.Name) and node.value.id in aliases) or (
        isinstance(node.value, ast.Attribute)
        and node.value.attr == "cast_ttsim_unit"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id in module_aliases
    )


def _local_cast_aliases(
    tree: ast.Module,
    direct_aliases: set[str],
    module_aliases: set[str],
) -> set[str]:
    """Expand aliases through local ``alias = existing_alias`` assignments."""
    aliases = set(direct_aliases)
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not _assignment_rhs_is_cast(
                node=node, aliases=aliases, module_aliases=module_aliases
            ):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id not in aliases:
                    aliases.add(target.id)
                    changed = True
    return aliases


def function_uses_local_cast(function: Any) -> bool:  # noqa: ANN401
    """Whether a body calls the canonical cast, directly or through a true alias."""
    direct_aliases, module_aliases = _cast_aliases_in_scope(_function_scope(function))
    try:
        tree = func_to_ast(function)
    except (OSError, TypeError, IndentationError, SyntaxError):
        code = getattr(function, "__code__", None)
        return code is not None and bool(
            set(code.co_names).intersection(direct_aliases | module_aliases)
            or set(code.co_freevars).intersection(direct_aliases)
        )

    aliases = _local_cast_aliases(
        tree=tree,
        direct_aliases=direct_aliases,
        module_aliases=module_aliases,
    )
    return any(
        isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id in aliases)
            or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "cast_ttsim_unit"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in module_aliases
            )
        )
        for node in ast.walk(tree)
    )


# Caps on the path-exploring unit check (see ``_PathExplorer``): only a pathological
# body (deep independent branching, or a data-driven loop) hits them, so the build
# check can never blow up.
_MAX_PATHS = 1024
_MAX_DECISIONS_PER_RUN = 64

#: How many of a failing run's branch decisions an error message spells out.
_MAX_NAMED_DECISIONS = 4


class _UnitCheckError(TTSIMError):
    """Base class of the signals the unit check throws and catches internally.

    Every subclass is raised inside a body run by :func:`_run_one_path` and caught
    there or by :func:`_verify_one_body`, then translated into a
    :class:`~ttsim.exceptions.UnitConsistencyError` naming the body. None ever
    reaches a user, but each subclasses ``TTSIMError`` so that a signal escaping
    its handler through a defect is still caught by ``except TTSIMError``.
    """


class _PathBudgetExceededError(_UnitCheckError):
    """A single unit-check run made too many branch decisions (likely a loop)."""


class _UnitMixError(_UnitCheckError):
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


class _CalendarOrdinalArithmeticError(_UnitCheckError):
    """A quarter-, month-, or day-within-period value was used arithmetically."""

    def __init__(self, op: str) -> None:
        super().__init__()
        self.op = op


class _UnsupportedGroupArithmeticError(_UnitCheckError):
    """A product or ratio lies outside GEP 10's restricted group rules."""

    def __init__(self, op: str, left: pint.Unit, right: pint.Unit) -> None:
        super().__init__()
        self.op = op
        self.left = left
        self.right = right


class _ScheduleNotEvaluableError(_UnitCheckError):
    """A schedule/lookup/join call the unit check cannot resolve to a unit.

    Raised when a function-like parameter carries no axes (a converter-produced
    or unannotated schedule) or a gather has no unit-carrying target — caught by
    :func:`_verify_one_body`'s generic handler and reported as needing an explicit
    ``verify_units=False`` opt-out, exactly like any other un-evaluable op.
    """


class _ReductionNotEvaluableError(_UnitCheckError):
    """An in-body array reduction the unit check cannot type.

    A reduction over rows changes which entity the result belongs to — per-person
    amounts summed over a group are the group's total — and the unit check sees no
    array axes to read that from, so it can neither preserve nor derive the
    result's grouping level. Caught by :func:`_run_one_path` and reported as
    needing an explicit ``verify_units=False`` opt-out; an aggregation node, which
    states its target level, is the checked alternative.
    """

    def __init__(self, op: str) -> None:
        super().__init__()
        self.op = op


class _LookupArityError(_UnitCheckError):
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


class _StructuredValueUsedAsQuantityError(_UnitCheckError):
    """A value plucked off a structured parameter was used as a quantity —
    caught by :func:`_verify_one_body` and reported with the
    cast-at-the-pluck fix."""

    def __init__(self, producer: str, op: str) -> None:
        super().__init__()
        self.producer = producer
        self.op = op


class _UnsupportedAstypeError(_UnitCheckError):
    """A cast whose dtype has no unit reading (a datetime, a string)."""

    def __init__(self, dtype: Any) -> None:  # noqa: ANN401
        super().__init__(f"astype({dtype!r}) has no unit reading")


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


class _UnitCheckQuantity:
    """A pint ``Quantity`` wrapped so branch decisions route to a ``_PathExplorer``.

    Arithmetic forwards to the wrapped quantity, so units propagate exactly as
    in a real run (the whole point of the check). Comparisons and truth tests
    instead return an explorer-controlled value, so the explorer — not the
    representative magnitude — decides which branch is taken; the magnitude is
    always ``1.0`` and never matters. A truth test is screened before it is
    handed on: only a boolean may control a branch. Anything the wrapper cannot
    model raises, which the caller treats as "not evaluable on this path" and
    falls back to the declaration — so the wrapper can never produce a false
    positive.
    """

    __slots__ = ("_explorer", "_kind", "_label", "_unit_system", "q")
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
        kind: QuantityKind = QuantityKind.GENERIC,
    ) -> None:
        self.q = q
        self._explorer = explorer
        # The system whose registry `q` lives in — every unit the wrapper mints
        # (a boolean's level, a dimensionless truth value) must land there too.
        self._unit_system = unit_system
        self._kind = kind
        # How the body's author would name this value — the argument name for a
        # direct input, a composed description for a comparison or logical
        # combination, ``None`` once arithmetic has mixed it beyond naming. Used
        # to report the branch a failure sits on (`_PathExplorer.branch_detail`).
        self._label = label

    @property
    def _registry(self) -> pint.UnitRegistry:
        return self._unit_system.registry

    def _wrap(
        self,
        q: Any,  # noqa: ANN401
        kind: QuantityKind = QuantityKind.GENERIC,
    ) -> _UnitCheckQuantity:
        return _UnitCheckQuantity(
            q=q,
            explorer=self._explorer,
            unit_system=self._unit_system,
            kind=kind,
        )

    def _controlled_bool_at(
        self, level: str | None, label: str | None = None
    ) -> _UnitCheckQuantity:
        return _UnitCheckQuantity(
            q=_boolean_quantity(level=level, registry=self._registry),
            explorer=self._explorer,
            unit_system=self._unit_system,
            label=label,
            kind=QuantityKind.INDICATOR,
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
        individual boolean (``None``).
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
        self_boolean = _as_boolean_level(
            unit=cast("pint.Unit", self.q.units), registry=self._registry
        )
        other_q = _unwrap(other)
        if isinstance(other_q, _UnitCheckStructuredValue):
            other_q._raise_used_as_quantity(op)  # noqa: SLF001
        if isinstance(other_q, pint.Quantity):
            other_boolean = _as_boolean_level(
                unit=cast("pint.Unit", other_q.units), registry=self._registry
            )
        else:
            other_boolean = BooleanLevel(is_boolean=True, level=None)
        if not self_boolean.is_boolean or not other_boolean.is_boolean:
            right = (
                cast("pint.Unit", other_q.units)
                if isinstance(other_q, pint.Quantity)
                else _dimensionless_unit(self._registry)
            )
            raise _UnitMixError(
                op=op, left=cast("pint.Unit", self.q.units), right=right
            )
        return self._controlled_bool_at(
            level=_combined_boolean_level(
                left=self_boolean.level, right=other_boolean.level
            ),
            label=self._composed_label(other=other, op=op),
        )

    def _fail_if_additive_operand_is_invalid(self, other: Any, op: str) -> None:  # noqa: ANN401
        """Screen an operand of ``+``/``-``.

        The rules are those of :meth:`_fail_if_other_unit_is_not_equivalent`, with one
        dispensation: a calendar point (an affine offset unit). Its valid ``point +/-
        duration`` is *not* equivalence (a point and a duration differ), yet pint's
        offset algebra permits exactly it. Two *different* offset units of the same
        ``[time]`` dimension are the trap: pint subtracts ``calendar_year -
        calendar_month`` with a silent /12 (``0.917 delta_calendar_year``) while the
        run-time subtraction is raw and unconverted, so a point - point across axes is
        rejected here rather than delegated. A same-axis point +/- duration (or point -
        point) is left to pint, which raises ``OffsetUnitCalculusError`` /
        ``DimensionalityError`` on the remaining misuses — caught in
        :func:`_verify_one_body` and reported as a calendar misuse. Only ``+``/``-`` get
        the dispensation: they alone run a forward pint operation afterwards, so nothing
        would catch a point mixed into an ordering or a ``where`` later.
        """
        other_q = _unwrap(other)
        if isinstance(other_q, _UnitCheckStructuredValue):
            other_q._raise_used_as_quantity(op)  # noqa: SLF001
        self._fail_if_calendar_ordinal_arithmetic(other=other, op=op)
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

    def _fail_if_calendar_ordinal_arithmetic(self, other: Any, op: str) -> None:  # noqa: ANN401
        """Reject arithmetic on month/day/quarter ordinals.

        These values can be ordered on the same scale, but neither a difference
        nor a shift has a calendar-independent meaning.
        """
        other_q = _unwrap(other)
        if is_calendar_ordinal_unit(cast("pint.Unit", self.q.units)) or (
            isinstance(other_q, pint.Quantity)
            and is_calendar_ordinal_unit(cast("pint.Unit", other_q.units))
        ):
            raise _CalendarOrdinalArithmeticError(op=op)

    def _group_arithmetic(
        self,
        other: Any,  # noqa: ANN401
        *,
        op: str,
        reflected: bool = False,
    ) -> tuple[Any, QuantityKind]:
        """Apply multiplication/division under the restricted group rules."""
        other_q = _unwrap(other)
        if isinstance(other_q, _UnitCheckStructuredValue):
            other_q._raise_used_as_quantity(op)  # noqa: SLF001
        if not isinstance(other_q, pint.Quantity):
            left_q, right_q = (other_q, self.q) if reflected else (self.q, other_q)
            if (
                op == "/"
                and reflected
                and _has_grouping_component(cast("pint.Unit", self.q.units))
            ):
                # ``bare / group_quantity`` creates a grouping numerator. Reject it
                # at the reciprocal rather than allowing a later multiplication to
                # cancel the marker and hide the illegal route.
                raise _UnsupportedGroupArithmeticError(
                    op=op,
                    left=_dimensionless_unit(self._registry),
                    right=cast("pint.Unit", self.q.units),
                )
            if op == "*":
                result = left_q * right_q
            else:
                try:
                    result = left_q / right_q
                except ZeroDivisionError:
                    result = left_q / self._nonzero_like(right_q)
            # Arithmetic no longer carries independent evidence that a value is
            # exactly a head count or a yes/no indicator.
            return result, QuantityKind.GENERIC

        other_kind = (
            other._kind  # noqa: SLF001
            if isinstance(other, _UnitCheckQuantity)
            else QuantityKind.GENERIC
        )
        left_q, right_q = (other_q, self.q) if reflected else (self.q, other_q)
        left_kind, right_kind = (
            (other_kind, self._kind) if reflected else (self._kind, other_kind)
        )
        left_level = _unit_level_denominator(cast("pint.Unit", left_q.units))
        right_level = _unit_level_denominator(cast("pint.Unit", right_q.units))

        if left_level is not None and right_level is not None:
            return self._group_arithmetic_with_two_levels(
                op=op,
                left_q=left_q,
                right_q=right_q,
                left_kind=left_kind,
                right_kind=right_kind,
                levels_match=left_level == right_level,
            )

        if op == "/" and left_level is None and right_level is not None:
            # This would create a grouping level in the numerator. GEP 10 only
            # derives the reverse, group-total / matching-head-count bridge.
            raise _UnsupportedGroupArithmeticError(
                op=op,
                left=cast("pint.Unit", left_q.units),
                right=cast("pint.Unit", right_q.units),
            )

        if op == "*":
            result = left_q * right_q
        else:
            try:
                result = left_q / right_q
            except ZeroDivisionError:
                result = left_q / self._nonzero_like(right_q)
        return result, QuantityKind.GENERIC

    @staticmethod
    def _group_arithmetic_with_two_levels(
        *,
        op: str,
        left_q: pint.Quantity,
        right_q: pint.Quantity,
        left_kind: QuantityKind,
        right_kind: QuantityKind,
        levels_match: bool,
    ) -> tuple[pint.Quantity, QuantityKind]:
        """Apply the matching-count and same-level-indicator exceptions."""
        if (
            levels_match
            and op == "*"
            and QuantityKind.INDICATOR
            in (
                left_kind,
                right_kind,
            )
        ):
            # A known yes/no value masks rather than multiplies the group marker.
            if left_kind is QuantityKind.INDICATOR:
                return right_q * left_q.magnitude, right_kind
            return left_q * right_q.magnitude, left_kind
        if levels_match and op == "/" and right_kind is QuantityKind.COUNT:
            return left_q / right_q, QuantityKind.GENERIC
        raise _UnsupportedGroupArithmeticError(
            op=op,
            left=cast("pint.Unit", left_q.units),
            right=cast("pint.Unit", right_q.units),
        )

    def _fail_if_grouping_operator_is_unsupported(
        self,
        other: object,
        op: str,
    ) -> None:
        """Reject operators for which GEP 10 defines no group-level bridge."""
        other_q = _unwrap(other)
        self_has_group = _has_grouping_component(cast("pint.Unit", self.q.units))
        other_has_group = isinstance(
            other_q, pint.Quantity
        ) and _has_grouping_component(cast("pint.Unit", other_q.units))
        if self_has_group or other_has_group:
            raise _UnsupportedGroupArithmeticError(
                op=op,
                left=cast("pint.Unit", self.q.units),
                right=(
                    cast("pint.Unit", other_q.units)
                    if isinstance(other_q, pint.Quantity)
                    else _dimensionless_unit(self._registry)
                ),
            )

    def _fail_if_other_unit_is_not_equivalent(self, other: Any, op: str) -> None:  # noqa: ANN401
        """Reject an invalid operand of an ordering comparison or ``where``.

        At run time there is no pint, so these operations are unit-blind (raw arrays are
        added or compared without conversion); two unit-carrying operands must already
        be in equivalent units. Equivalence decides calendar points by *identity*
        (:func:`units_are_equivalent`): ordering two same-axis points (``geburtsjahr <=
        policy_year``) passes, while a point against a duration — or any other unit — is
        rejected. A non-zero *bare literal* next to a non-dimensionless quantity is
        rejected too: it silently carries the quantity's unit (``betrag_m + 100.0``
        hides a monthly amount) — promote it to a parameter or tag it with
        ``cast_ttsim_unit``. Only ``0`` (the ``x + 0.0`` guard, the floor at zero) is
        allowed inline, and literals next to a dimensionless quantity stay lenient.
        Unlike ``+``/``-``, an ordering comparison runs no forward pint operation, so
        calendar points get no delegate-to-pint dispensation here (equivalence decides
        them by identity: only same-axis points order).
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
        """Resolve a truth context — an ``if``, a conditional expression, ``and``,
        ``or`` — on the explorer, once the value is established to be a truth value.

        A branch condition is a demand that the operand have boolean semantics, not
        merely that Python accept its numerical truthiness: a stock, a flow, a
        calendar point or a duration controlling a branch is a bug (GEP 10). The
        screen is the one the logical operators use, so ``if x`` and ``~x`` agree on
        what a boolean is.
        """
        self._fail_if_not_a_truth_value(op=_TRUTH_VALUE_OP)
        return self._explorer.decide(self._label)

    def _fail_if_not_a_truth_value(self, op: str) -> None:
        """Reject a value with physical content where a boolean is required."""
        if not _as_boolean_level(
            unit=cast("pint.Unit", self.q.units), registry=self._registry
        ).is_boolean:
            raise _UnitMixError(
                op=op,
                left=cast("pint.Unit", self.q.units),
                right=_dimensionless_unit(self._registry),
            )

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
        boolean = _as_boolean_level(
            unit=cast("pint.Unit", self.q.units), registry=self._registry
        )
        if not boolean.is_boolean:
            raise _UnitMixError(
                op="~",
                left=cast("pint.Unit", self.q.units),
                right=_dimensionless_unit(self._registry),
            )
        return self._controlled_bool_at(
            level=boolean.level,
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
        self._fail_if_calendar_ordinal_arithmetic(other=other, op="*")
        result, kind = self._group_arithmetic(other, op="*")
        return self._wrap(result, kind=kind)

    def __rmul__(self, other: Any) -> _UnitCheckQuantity:  # noqa: ANN401
        self._fail_if_calendar_ordinal_arithmetic(other=other, op="*")
        result, kind = self._group_arithmetic(other, op="*", reflected=True)
        return self._wrap(result, kind=kind)

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
        self._fail_if_calendar_ordinal_arithmetic(other=other, op="/")
        result, kind = self._group_arithmetic(other, op="/")
        return self._wrap(result, kind=kind)

    def __rtruediv__(self, other: Any) -> _UnitCheckQuantity:  # noqa: ANN401
        self._fail_if_calendar_ordinal_arithmetic(other=other, op="/")
        result, kind = self._group_arithmetic(other, op="/", reflected=True)
        return self._wrap(result, kind=kind)

    def __floordiv__(self, other: Any) -> _UnitCheckQuantity:  # noqa: ANN401
        self._fail_if_calendar_ordinal_arithmetic(other=other, op="//")
        self._fail_if_grouping_operator_is_unsupported(other=other, op="//")
        divisor = _unwrap(other)
        try:
            return self._wrap(self.q // divisor)
        except ZeroDivisionError:
            return self._wrap(self.q // self._nonzero_like(divisor))

    def __rfloordiv__(self, other: Any) -> _UnitCheckQuantity:  # noqa: ANN401
        self._fail_if_calendar_ordinal_arithmetic(other=other, op="//")
        self._fail_if_grouping_operator_is_unsupported(other=other, op="//")
        dividend = _unwrap(other)
        try:
            return self._wrap(dividend // self.q)
        except ZeroDivisionError:
            return self._wrap(dividend // self._nonzero_like(self.q))

    def __mod__(self, other: Any) -> _UnitCheckQuantity:  # noqa: ANN401
        self._fail_if_calendar_ordinal_arithmetic(other=other, op="%")
        return self._wrap(self.q % _unwrap(other))

    def __rmod__(self, other: Any) -> _UnitCheckQuantity:  # noqa: ANN401
        self._fail_if_calendar_ordinal_arithmetic(other=other, op="%")
        return self._wrap(_unwrap(other) % self.q)

    def __pow__(self, other: Any) -> _UnitCheckQuantity:  # noqa: ANN401
        self._fail_if_calendar_ordinal_arithmetic(other=other, op="**")
        self._fail_if_grouping_operator_is_unsupported(other=other, op="**")
        return self._wrap(self.q ** _unwrap(other))

    def __rpow__(self, other: Any) -> _UnitCheckQuantity:  # noqa: ANN401
        self._fail_if_calendar_ordinal_arithmetic(other=other, op="**")
        self._fail_if_grouping_operator_is_unsupported(other=other, op="**")
        return self._wrap(_unwrap(other) ** self.q)

    def __neg__(self) -> _UnitCheckQuantity:
        self._fail_if_calendar_ordinal_arithmetic(other=1, op="unary -")
        return self._wrap(-self.q)

    def __pos__(self) -> _UnitCheckQuantity:
        self._fail_if_calendar_ordinal_arithmetic(other=1, op="unary +")
        return self._wrap(+self.q)

    def __abs__(self) -> _UnitCheckQuantity:
        self._fail_if_calendar_ordinal_arithmetic(other=1, op="abs")
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


def _unwrap(value: Any) -> Any:  # noqa: ANN401
    return value.q if isinstance(value, _UnitCheckQuantity) else value


def _wrap_for_unit_check(
    value: Any,  # noqa: ANN401
    explorer: _PathExplorer,
    unit_system: UnitSystem,
    label: str | None = None,
    kind: QuantityKindTree = QuantityKind.GENERIC,
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
            q=value,
            explorer=explorer,
            unit_system=unit_system,
            label=label,
            kind=kind if isinstance(kind, QuantityKind) else QuantityKind.GENERIC,
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
                kind=(
                    kind.get(key, QuantityKind.GENERIC)
                    if isinstance(kind, Mapping)
                    else kind
                ),
            )
            for key, leaf in value.items()
        }
    return value


class _UnitCheckStructuredValue:
    """The unit check's stand-in for a structured param-function output
    (``unit=UNSET_UNIT``). A pluck off an ``Annotated`` scalar field of
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
        if isinstance(resolved, pint.Unit | _ScalarFieldKind):
            # An annotated field's pluck is a known quantity; with the run's
            # explorer it screens and branches like any other operand.
            unit = resolved.unit if isinstance(resolved, _ScalarFieldKind) else resolved
            kind = (
                resolved.kind
                if isinstance(resolved, _ScalarFieldKind)
                else QuantityKind.GENERIC
            )
            quantity = self._unit_system.registry.Quantity(1.0, unit)
            if self._explorer is None:
                return quantity
            return _UnitCheckQuantity(
                q=quantity,
                explorer=self._explorer,
                unit_system=self._unit_system,
                label=label,
                kind=kind,
            )
        if isinstance(resolved, _ScheduleFieldKind):
            # A schedule-typed field declares both axes; the pluck yields a
            # schedule that screens each `look_up`/`piecewise_polynomial` argument
            # against `input_unit` and produces `output_unit`.
            return _UnitCheckSchedule(
                input_unit=resolved.input_unit,
                output_unit=resolved.output_unit,
                output_kind=resolved.output_kind,
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


def _structured_value_forbidden_op(op: str) -> Callable[..., Any]:
    def method(self: _UnitCheckStructuredValue, *_a: Any, **_k: Any) -> Any:  # noqa: ANN401
        return self._raise_used_as_quantity(op)

    return method


def _install_structured_value_forbidden_ops() -> None:
    """Bind each dunder in `_STRUCTURED_VALUE_FORBIDDEN_OPS` to a raising stub.

    The table covers arithmetic, ordering, equality, logical and truth-value uses.
    A structured value is a container, not a quantity, so any of them is an
    authoring error the unit check reports against the operator it was used with —
    cast the pluck or opt out.
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
    parameter's own, a schedule builder's ``InputOutputUnits`` declaration, or a
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

    __slots__ = (
        "explorer_holder",
        "input_unit",
        "output_kind",
        "output_unit",
        "unit_system",
    )

    def __init__(
        self,
        input_unit: pint.Unit | tuple[pint.Unit, ...] | None,
        output_unit: pint.Unit,
        unit_system: UnitSystem,
        output_kind: QuantityKind = QuantityKind.GENERIC,
        explorer_holder: list[_PathExplorer | None] | None = None,
    ) -> None:
        self.input_unit = input_unit
        self.output_unit = output_unit
        self.output_kind = output_kind
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
            kind=self.output_kind,
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


class _UnitCheckXnp:
    """The unit check's ``xnp``: NumPy with the unit-bearing ops routed through
    ``_UnitCheckQuantity``'s checks, so a vectorized (``not_required``) body is checked
    at full parity with a scalar one.

    An op not modelled here falls through to raw NumPy, raises, and is reported as
    needing ``verify_units=False`` — never silently passed through. The array
    reductions are modelled only to refuse them: their result belongs to rows the
    unit check cannot identify without array-axis metadata (see
    :class:`_ReductionNotEvaluableError`).
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
    def where(condition: Any, x: Any, y: Any) -> Any:  # noqa: ANN401
        _fail_if_condition_is_not_a_truth_value(condition=condition)
        return _where_op(x=x, y=y)

    @staticmethod
    def clip(value: Any, a_min: Any, a_max: Any) -> Any:  # noqa: ANN401
        return _clip_op(value=value, a_min=a_min, a_max=a_max)

    @staticmethod
    def sum(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401, ARG004
        raise _ReductionNotEvaluableError(op="sum")

    @staticmethod
    def amin(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401, ARG004
        raise _ReductionNotEvaluableError(op="amin")

    @staticmethod
    def amax(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401, ARG004
        raise _ReductionNotEvaluableError(op="amax")

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


def _piecewise_polynomial_for_unit_check(x: Any, parameters: Any, xnp: Any) -> Any:  # noqa: ANN401, ARG001
    """Unit-check stand-in for ``piecewise_polynomial``.

    Screen ``x`` against the schedule's ``input_unit`` and produce its
    ``output_unit``. Every schedule arrives as a :class:`_UnitCheckSchedule`
    carrying the axes its producer declared — a parameter's YAML axes or a
    builder's ``InputOutputUnits``. An opaque structured pluck (an unannotated
    field of a structured value) propagates unchanged for the caller to cast;
    anything else cannot be evaluated here.
    """
    if isinstance(parameters, _UnitCheckSchedule):
        return parameters._produce((x,))  # noqa: SLF001
    if isinstance(parameters, _UnitCheckStructuredValue):
        return parameters
    raise _ScheduleNotEvaluableError


def _join_for_unit_check(
    foreign_key: Any,  # noqa: ANN401
    primary_key: Any,  # noqa: ANN401
    target: Any,  # noqa: ANN401
    value_if_foreign_key_is_missing: Any,  # noqa: ANN401
    xnp: Any,  # noqa: ANN401, ARG001
) -> Any:  # noqa: ANN401
    """Unit-check stand-in for ``join``, screening every operand it can read.

    A gather hands on the ``target`` column's unit and grouping level, but only if
    the rest of the call is sound (GEP 10):

    - both keys must be identifiers — values with no physical content. A currency,
      a flow, or a calendar point never identifies a row, and the run-time gather
      would take it as an index without complaint. (That two identifiers belong to
      the *same* key domain is beyond what the unit model can say.)
    - the missing-key fallback becomes part of the gathered column, so it is
      screened against the target exactly as an arm of ``where`` is: a fallback in
      another unit is a bug that only unmatched keys would ever expose, and a bare
      sentinel next to a dimensionless target (``-1`` for a group id) stays
      admissible.
    """
    _fail_if_join_key_is_not_an_identifier(key=foreign_key, name="foreign_key")
    _fail_if_join_key_is_not_an_identifier(key=primary_key, name="primary_key")
    if not isinstance(target, _UnitCheckQuantity):
        raise _ScheduleNotEvaluableError
    target._fail_if_other_unit_is_not_equivalent(  # noqa: SLF001
        other=value_if_foreign_key_is_missing, op=_JOIN_FALLBACK_OP
    )
    return target._wrap(target.q)  # noqa: SLF001


def _fail_if_join_key_is_not_an_identifier(key: Any, name: str) -> None:  # noqa: ANN401
    """Reject a ``join`` key carrying physical content.

    A raw key (a literal, an array the check does not model) carries no unit to
    screen and passes; a structured pluck is reported at the pluck, as anywhere.
    """
    if isinstance(key, _UnitCheckStructuredValue):
        key._raise_used_as_quantity(f"join {name}")  # noqa: SLF001
    if not isinstance(key, _UnitCheckQuantity):
        return
    if not _is_dimensionless_up_to_grouping_level(
        unit=cast("pint.Unit", key.q.units),
        registry=key._registry,  # noqa: SLF001
    ):
        raise _UnitMixError(
            op=_JOIN_KEY_OP,
            left=cast("pint.Unit", key.q.units),
            right=_dimensionless_unit(key._registry),  # noqa: SLF001
        )


def _fail_if_condition_is_not_a_truth_value(condition: Any) -> None:  # noqa: ANN401
    """Screen a selection primitive's condition as a truth value.

    The vectorized selectors run the same screen as a scalar branch, so
    ``xnp.where(wealth, …)`` is rejected exactly as ``… if wealth else …`` is
    (GEP 10). A raw condition — a bare literal, an array the check does not model
    — carries no unit to screen and passes.
    """
    if isinstance(condition, _UnitCheckStructuredValue):
        condition._raise_used_as_quantity(_TRUTH_VALUE_OP)  # noqa: SLF001
    if isinstance(condition, _UnitCheckQuantity):
        condition._fail_if_not_a_truth_value(op=_TRUTH_VALUE_OP)  # noqa: SLF001


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
    resolved = pint_unit_from_ttsim_unit_for_column(
        unit=token,
        name=None,
        grouping_levels=(),
        where="A `cast_ttsim_unit` call",
        registry=unit_system.registry,
    )
    quantity = unit_system.registry.Quantity(1.0, resolved)
    if isinstance(value, _UnitCheckQuantity):
        return value._wrap(quantity, kind=token.kind)  # noqa: SLF001
    explorer = explorer_holder[0]
    if explorer is None:
        return quantity
    return _UnitCheckQuantity(
        q=quantity, explorer=explorer, unit_system=unit_system, kind=token.kind
    )


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


def _time_conversion_stand_in(
    from_pint: str, to_pint: str, registry: pint.UnitRegistry, *, is_flow: bool
) -> Callable[[Any], Any]:
    """Build the unit-check stand-in for one ``ttsim.time_converters`` time converter.

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
    """An elementwise unit-preserving op (``floor``/``ceil``/``round``/``abs``)."""
    if isinstance(value, _UnitCheckQuantity):
        return value._wrap(value.q)  # noqa: SLF001
    return value


def _is_scalar_literal(value: Any) -> bool:  # noqa: ANN401
    """Whether ``value`` is a genuine numeric scalar (a bare or computed literal).

    A body's arithmetic on Python/NumPy number literals stays a plain number;
    every unit-check stand-in (a quantity, a structured value, a schedule) is
    something else. `bool` counts — it is an `int` subclass.
    """
    return isinstance(value, int | float | numpy.integer | numpy.floating)


#: The logical operators the unit check screens for boolean (dimensionless) operands.
_LOGICAL_OPS = frozenset({"&", "|", "^", "~"})

#: Names the screens above report themselves under, where the offending construct is
#: not an operator: a truth context (a branch, a selector's condition) and the two
#: `join` operands beyond the target.
_TRUTH_VALUE_OP = "truth value"
_JOIN_KEY_OP = "join key"
_JOIN_FALLBACK_OP = "missing-key fallback"


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
      declared, not read off the suffix; a body whose arithmetic cannot
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


def _arithmetic_misuse_message(
    qname: str,
    error: _CalendarOrdinalArithmeticError
    | _UnsupportedGroupArithmeticError
    | _UnitMixError
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
    if isinstance(error, _CalendarOrdinalArithmeticError):
        return (
            f"{qname}: uses a calendar ordinal in `{error.op}`{detail} — a quarter "
            "of year, month of year, or day of month may only be compared with the "
            "same calendar scale; use an explicit local unit assertion for a "
            "policy-specific conversion (GEP 10)."
        )
    if isinstance(error, _UnsupportedGroupArithmeticError):
        return (
            f"{qname}: uses an unsupported group calculation "
            f"'{error.left}' {error.op} '{error.right}'{detail}. TTSIM only "
            "derives ordinary scalar scaling, a group total divided by its "
            "matching head count, the reverse head-count multiplication, and a "
            "same-level yes/no mask; use an aggregation or a local unit assertion "
            "for a deliberate exception (GEP 10)."
        )
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
    reported as a non-boolean operand; a dimensioned branch condition or selector
    as a non-boolean truth value; a ``join`` key or missing-key fallback against
    the contract it broke; an ordering comparison against a bare non-zero literal
    as an untagged threshold; ``+``/``-``/an ordering comparison of non-equivalent
    quantities as a unit mix (no run-time conversion).
    """
    named_screen = _named_screen_message(qname=qname, mix=mix, detail=detail)
    if named_screen is not None:
        return named_screen
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


def _named_screen_message(qname: str, mix: _UnitMixError, detail: str) -> str | None:
    """Message for the screens that name themselves rather than an operator.

    A truth context, a ``join`` key, and a ``join`` fallback each broke a contract
    of their own, so each is reported in its own terms; ``None`` where the offence
    is an operator's and :func:`_unit_mix_error_message` phrases it.
    """
    if mix.op == _TRUTH_VALUE_OP:
        return (
            f"{qname}: uses '{mix.left}' as a truth value{detail} — only a boolean "
            f"may control a branch or select between two arms; compare it against "
            f"a quantity in the same unit instead (GEP 10)."
        )
    if mix.op == _JOIN_KEY_OP:
        return (
            f"{qname}: gathers with '{mix.left}' as a `join` key{detail} — a key is "
            f"an identifier and carries no physical content, so a dimensioned "
            f"column cannot identify a row (GEP 10)."
        )
    if mix.op == _JOIN_FALLBACK_OP:
        return _join_fallback_message(qname=qname, mix=mix, detail=detail)
    return None


def _join_fallback_message(qname: str, mix: _UnitMixError, detail: str) -> str:
    """Message for a ``join`` whose missing-key fallback is not in the target's unit.

    The fallback fills the rows whose key found no match, so it lands in the
    gathered column next to the target's own values — a bare non-zero literal
    carries the target's unit silently, another unit is not converted at run time.
    """
    if mix.literal is not None:
        return (
            f"{qname}: gathers with the bare literal {mix.literal} as the "
            f"missing-key fallback for a '{mix.left}' target{detail} — the fallback "
            f"lands in the gathered column and silently carries the target's unit; "
            f"promote it to a parameter, tag it with `cast_ttsim_unit`, or use 0 "
            f"(GEP 10)."
        )
    return (
        f"{qname}: gathers a '{mix.left}' target with a '{mix.right}' missing-key "
        f"fallback{detail} — the fallback fills the unmatched rows of the same "
        f"column, and there is no unit conversion at run time (GEP 10)."
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


class _UnsupportedBodyError(str):
    """Diagnostic marker for a body the symbolic checker cannot evaluate."""

    __slots__ = ()


def _opt_out_required_error(qname: str, reason: str) -> _UnsupportedBodyError:
    """Message demanding an explicit opt-out for a body the unit check cannot evaluate.

    A body the unit check cannot evaluate is *not* waved through silently:
    the author must mark it ``verify_units=False`` so that every un-verified body
    is a visible, deliberate choice. The declared unit still
    stands and the body's edges are still checked — only its internal inference
    is skipped.
    """
    return _UnsupportedBodyError(
        f"{qname}: its body cannot be unit-checked ({reason}). "
        f"Set `verify_units=False` on its decorator to opt out of body inference "
        f"— its declared unit and its edges stay checked (GEP 10)."
    )
