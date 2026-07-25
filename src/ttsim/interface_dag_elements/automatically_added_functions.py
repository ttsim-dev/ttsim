from __future__ import annotations

import datetime
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, cast, overload

import dags.tree as dt
from dags import get_annotations, get_free_arguments, rename_arguments

from ttsim.interface_dag_elements.shared import (
    get_base_name_and_grouping_suffix,
    get_re_pattern_for_all_time_units_and_groupings,
    get_re_pattern_for_specific_time_units_and_groupings,
    group_pattern,
)
from ttsim.tt.aggregation import AggType, grouped_sum
from ttsim.tt.column_objects_param_function import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    AggByGroupFunction,
    ColumnFunction,
    ColumnObject,
    ParamFunction,
    PolicyInput,
    TimeConversionFunction,
)
from ttsim.tt.param_objects import ScalarParam
from ttsim.tt.type_resolution import (
    BOOL_KINDS,
    ResolvedKind,
    TypeResolutionError,
    resolve_agg_output_kind,
    resolve_kind_of_annotation,
    resolve_kind_of_column_function,
    synthesize_typed_aggregation_wrapper,
    vectorized_column_kind,
)
from ttsim.tt.units import (
    UNSET_UNIT,
    CompositeUnit,
    ttsim_unit_with_agnostic_currency,
    unit_for_aggregation,
    unit_with_rebased_period,
)
from ttsim.typing import (
    OrderedQNames,
    PolicyEnvironment,
    QNameStrings,
    UnorderedQNames,
)
from ttsim.unit_converters import (
    TIME_UNIT_IDS_TO_LABELS,
    per_d_to_per_m,
    per_d_to_per_q,
    per_d_to_per_w,
    per_d_to_per_y,
    per_m_to_per_d,
    per_m_to_per_q,
    per_m_to_per_w,
    per_m_to_per_y,
    per_q_to_per_d,
    per_q_to_per_m,
    per_q_to_per_w,
    per_q_to_per_y,
    per_w_to_per_d,
    per_w_to_per_m,
    per_w_to_per_q,
    per_w_to_per_y,
    per_y_to_per_d,
    per_y_to_per_m,
    per_y_to_per_q,
    per_y_to_per_w,
)

if TYPE_CHECKING:
    import re
    from collections.abc import Callable

    from ttsim.typing import (
        BoolColumn,
        FloatColumn,
        IntColumn,
        OrderedQNames,
        PolicyEnvironment,
        UnorderedQNames,
    )


@dataclass(frozen=True)
class DerivedColumnObjects:
    """What one derivation rule contributes to the policy environment.

    A derivation rule (time conversion, aggregation by group) turns a declared
    column into further columns at derived names. Where the input data supplies
    such a name, no function is created — the data would override it — and the
    name gets a `PolicyInput` stub carrying the unit and dtype the rule implies
    instead, so that every input column has a unit declaration (GEP 10).

    The two are kept apart because they are not interchangeable as resolution
    sources: a stub is data, not a node the DAG computes, and its dtype is the
    one a supplied column carries rather than the one the declaration states.
    """

    functions: MappingProxyType[str, ColumnObject]
    """The column objects created at derived names, by qualified name."""
    input_stubs: MappingProxyType[str, PolicyInput]
    """Stubs for derived names the input data supplies, by qualified name."""

    @property
    def all_objects(self) -> dict[str, ColumnObject]:
        """Functions and stubs together, as they enter the environment."""
        return {**self.functions, **self.input_stubs}


def fail_if_multiple_time_units_for_same_base_name_and_group(
    base_names_and_groups_to_variations: dict[tuple[str, str], list[str]],
) -> None:
    invalid = {
        b: q for b, q in base_names_and_groups_to_variations.items() if len(q) > 1
    }
    if invalid:
        raise ValueError(f"Multiple time units for base names: {invalid}")


_automatic_time_converters = {
    "per_y_to_per_m": per_y_to_per_m,
    "per_y_to_per_q": per_y_to_per_q,
    "per_y_to_per_w": per_y_to_per_w,
    "per_y_to_per_d": per_y_to_per_d,
    "per_q_to_per_y": per_q_to_per_y,
    "per_q_to_per_m": per_q_to_per_m,
    "per_q_to_per_w": per_q_to_per_w,
    "per_q_to_per_d": per_q_to_per_d,
    "per_m_to_per_y": per_m_to_per_y,
    "per_m_to_per_q": per_m_to_per_q,
    "per_m_to_per_w": per_m_to_per_w,
    "per_m_to_per_d": per_m_to_per_d,
    "per_w_to_per_y": per_w_to_per_y,
    "per_w_to_per_q": per_w_to_per_q,
    "per_w_to_per_m": per_w_to_per_m,
    "per_w_to_per_d": per_w_to_per_d,
    "per_d_to_per_y": per_d_to_per_y,
    "per_d_to_per_m": per_d_to_per_m,
    "per_d_to_per_q": per_d_to_per_q,
    "per_d_to_per_w": per_d_to_per_w,
}


def _convertibles(
    qname_policy_environment: PolicyEnvironment,
) -> dict[str, ColumnObject | ParamFunction | ScalarParam]:
    return {
        qn: e
        for qn, e in qname_policy_environment.items()
        if isinstance(e, (ColumnObject, ScalarParam))
        or (
            isinstance(e, ParamFunction)
            and get_annotations(e.function)["return"] in {"float", "int"}
        )
    }


def create_time_conversion_functions(
    qname_policy_environment: PolicyEnvironment,
    input_columns: UnorderedQNames,
    grouping_levels: OrderedQNames,
) -> DerivedColumnObjects:
    """
    Create functions converting elements of the policy environment to other time units.

    Convertible elements are column objects, scalar parameters and param functions
    returning a scalar (see function *_convertibles*)

    The time unit of a function is determined by a naming convention:

    * Functions referring to yearly values end with "_y", or "_y_x" where "x" is a
      grouping level.
    * Functions referring to monthly values end with "_m", or "_m_x" where "x" is a
      grouping level.
    * Functions referring to weekly values end with "_w", or "_w_x" where "x" is a
      grouping level.
    * Functions referring to daily values end with "_d", or "_d_x" where "x" is a
      grouping level.

    Unless the corresponding function already exists, the following will be created:

    * For functions referring to yearly values, create monthly, weekly and daily
      functions.
    * For functions referring to monthly values, create yearly, weekly and daily
      functions.
    * For functions referring to weekly values, create yearly, monthly and daily
      functions.
    * For functions referring to daily values, create yearly, monthly and weekly
      functions.

    Args:
        qname_policy_environment: The policy environment with qualified names as keys.
        input_columns: The names of the input columns, represented by qualified names.
        grouping_levels: The grouping levels.

    Returns:
        The time conversion functions, plus a `PolicyInput` stub for each time
        variant the input data supplies (no function is created there).

    """
    time_units = tuple(TIME_UNIT_IDS_TO_LABELS)
    pattern_all = get_re_pattern_for_all_time_units_and_groupings(
        grouping_levels=grouping_levels,
        time_units=time_units,
    )
    # Map base name and grouping suffix to time conversion inputs.
    bngs_to_time_conversion_inputs = {}
    bngs_to_variations = {}
    for qname, element in _convertibles(qname_policy_environment).items():
        match = cast("re.Match[str]", pattern_all.fullmatch(qname))
        # We must not find multiple time units for the same base name and group.
        bngs = get_base_name_and_grouping_suffix(match)
        if match.group("time_unit"):
            if bngs not in bngs_to_variations:
                bngs_to_variations[bngs] = [qname]
            else:
                bngs_to_variations[bngs].append(qname)
            bngs_to_time_conversion_inputs[bngs] = {
                "base_name": bngs[0],
                "qname_source": qname,
                "element": element,
                "time_unit": match.group("time_unit"),
                "grouping_suffix": bngs[1],
                "time_units": time_units,
            }

    fail_if_multiple_time_units_for_same_base_name_and_group(bngs_to_variations)

    converted_elements: dict[str, ColumnObject] = {}
    input_stubs: dict[str, PolicyInput] = {}
    for bngs, inputs in bngs_to_time_conversion_inputs.items():
        for col_name in input_columns:
            # If a time variant of this base name and grouping level is provided
            # in the data, base the time conversions on that. The input column
            # must sit at the same grouping level as the declaration: a
            # household total is not the source of a per-person quantity, nor
            # the other way round.
            pattern_specific = get_re_pattern_for_specific_time_units_and_groupings(
                base_name=bngs[0],
                all_time_units=time_units,
                grouping_levels=grouping_levels,
            ).fullmatch(col_name)
            if (
                pattern_specific
                and pattern_specific.group("time_unit")
                and get_base_name_and_grouping_suffix(pattern_specific)[1] == bngs[1]
            ):
                inputs["qname_source"] = col_name
                inputs["time_unit"] = pattern_specific.group("time_unit")
                break

        element = cast("ColumnObject | ParamFunction | ScalarParam", inputs["element"])
        grouping_suffix = cast("str", inputs["grouping_suffix"])
        variations = _create_one_set_of_time_conversion_functions(
            base_name=cast("str", inputs["base_name"]),
            qname_source=cast("str", inputs["qname_source"]),
            element=element,
            time_unit=cast("str", inputs["time_unit"]),
            grouping_suffix=grouping_suffix,
            time_units=cast("OrderedQNames", inputs["time_units"]),
        )
        converted_elements = {**converted_elements, **variations}

        for time_unit_id in time_units:
            qname = f"{bngs[0]}_{time_unit_id}{grouping_suffix}"
            if (
                qname in input_columns
                and qname not in variations
                and qname not in qname_policy_environment
            ):
                stub = _time_converted_input_stub(
                    qname=qname, element=element, time_unit_id=time_unit_id
                )
                if stub is not None:
                    input_stubs[qname] = stub

    return DerivedColumnObjects(
        functions=MappingProxyType(converted_elements),
        input_stubs=MappingProxyType(input_stubs),
    )


def _time_converted_input_stub(
    qname: str,
    element: ColumnObject | ParamFunction | ScalarParam,
    time_unit_id: str,
) -> PolicyInput | None:
    """A `PolicyInput` stub for data supplied at a time-converted name.

    Data at such a name is the *source* of the conversions, so no function is
    created for it and it would carry no unit declaration — yet the input-side
    currency conversion and the input-tag checks read units off the environment.
    The stub carries the declared element's unit re-based to the supplied name's
    period, exactly as a conversion *away* from that element would (GEP 10).
    """
    declared = getattr(element, "unit", UNSET_UNIT)
    if declared is UNSET_UNIT:
        return None
    return _input_column_stub(
        qname=qname,
        unit=unit_with_rebased_period(
            unit=ttsim_unit_with_agnostic_currency(declared),
            time_unit_id=time_unit_id,
        ),
        # The synthesized time converters return floats, so a column supplied at
        # a converted name is a float column.
        data_type="FloatColumn",
    )


def _create_one_set_of_time_conversion_functions(
    base_name: str,
    qname_source: str,
    element: ColumnObject | ParamFunction | ScalarParam,
    time_unit: str,
    grouping_suffix: str,
    time_units: OrderedQNames,
) -> dict[str, TimeConversionFunction]:
    result: dict[str, TimeConversionFunction] = {}
    dependencies = (
        set(inspect.signature(element).parameters)
        if isinstance(element, ColumnFunction)
        else set()
    )
    # `ScalarParam.start_date` / `end_date` are typed `date | None`, but
    # every convertible element here carries concrete dates (column objects
    # and param functions always do; scalar params are built with explicit
    # `start_date` / `end_date`). Cast to `date` for the
    # `TimeConversionFunction` constructor which requires non-optional values.
    start_date = cast("datetime.date", element.start_date)
    end_date = cast("datetime.date", element.end_date)

    for target_time_unit in [tu for tu in time_units if tu != time_unit]:
        new_name = f"{base_name}_{target_time_unit}{grouping_suffix}"

        # Without the following check, we could create cycles in the DAG: Consider a
        # hard-coded function `var_y` that takes `var_m` as an input, assuming it
        # to be provided in the input data. If we create a function `var_m`, which
        # would take `var_y` as input, we create a cycle. If `var_m` is actually
        # provided as an input, `var_m` would be overwritten, removing the cycle.
        # However, if `var_m` is not provided as an input, an error message would
        # be shown that a cycle between `var_y` and `var_m` was detected. This
        # hides the actual problem, which is that `var_m` is not provided as an
        # input.
        if new_name in dependencies:
            continue

        result[new_name] = TimeConversionFunction(
            leaf_name=dt.tree_path_from_qname(new_name)[-1],
            function=_create_function_for_time_unit(
                source=qname_source,
                converter=_automatic_time_converters[
                    f"per_{time_unit}_to_per_{target_time_unit}"
                ],
            ),
            source=qname_source,
            start_date=start_date,
            end_date=end_date,
            description=(
                f"Time conversion of {dt.tree_path_from_qname(qname_source)} "
                f"from per {time_unit} to per {target_time_unit}"
            ),
            unit=unit_with_rebased_period(
                unit=ttsim_unit_with_agnostic_currency(
                    getattr(element, "unit", UNSET_UNIT)
                ),
                time_unit_id=target_time_unit,
            ),
        )

    return result


def _create_function_for_time_unit(
    source: str,
    converter: Callable[[BoolColumn | FloatColumn | IntColumn], FloatColumn],
) -> Callable[[BoolColumn | FloatColumn | IntColumn], FloatColumn]:
    @overload
    @rename_arguments(mapper={"x": source})
    def func(x: FloatColumn) -> FloatColumn: ...

    @overload
    @rename_arguments(mapper={"x": source})
    def func(x: IntColumn) -> FloatColumn: ...

    @overload
    @rename_arguments(mapper={"x": source})
    def func(x: BoolColumn) -> FloatColumn: ...

    @rename_arguments(mapper={"x": source})
    def func(x: FloatColumn | IntColumn | BoolColumn) -> FloatColumn:
        return converter(x)

    return func


def create_agg_by_group_functions(
    column_functions: dict[str, ColumnFunction],
    qname_policy_environment: PolicyEnvironment,
    time_converted_input_stubs: Mapping[str, PolicyInput],
    input_columns: UnorderedQNames,
    tt_targets: QNameStrings,
    grouping_levels: OrderedQNames,
) -> DerivedColumnObjects:
    """Create auto-aggregation functions, each with a concrete return annotation.

    Auto-aggregations are sum aggregations of an individual-level source
    column. The source column's kind (float / int / bool) determines the
    aggregation's output kind via the hand-written rule table in
    `ttsim.tt.type_resolution` (SUM: float -> float, int -> int,
    bool -> int). The synthesized `grouped_sum` wrapper is stamped with that
    concrete return annotation so the DAG's annotation-consistency check
    never sees an imprecise `FloatColumn | IntColumn` union for the node.

    Args:
        column_functions: Qualified-name to column function mapping.
        qname_policy_environment: The flat policy environment, used to look
            up `PolicyInput` declarations of pure input-column sources so
            their dtype can be resolved.
        time_converted_input_stubs: Stubs for time variants the input data
            supplies.
        input_columns: The qualified names of the input data columns.
        tt_targets: The requested targets.
        grouping_levels: The grouping levels.

    Returns:
        The `AggByGroupFunction`s, plus a `PolicyInput` stub for each group
        aggregate the input data supplies (no function is created there).
    """
    gp = group_pattern(grouping_levels)
    all_functions_and_data = {
        **column_functions,
        **dict.fromkeys(input_columns),
    }
    potential_agg_by_group_function_names = {
        # Targets that end with a grouping suffix are potential aggregation targets.
        *[t for t in tt_targets if gp.match(t)],
        *_get_potential_agg_by_group_function_names_from_function_arguments(
            functions=column_functions,
            group_pattern=gp,
        ),
    }
    # We will only aggregate from individual-level objects.
    potential_agg_by_group_sources = {
        qn: o for qn, o in all_functions_and_data.items() if not gp.match(qn)
    }
    # Exclude objects that have been explicitly provided.
    agg_by_group_function_names = {
        t
        for t in potential_agg_by_group_function_names
        if t not in all_functions_and_data
    }
    agg_by_group_input_names = {
        c
        for c in input_columns
        if gp.match(c)
        and c not in column_functions
        and c not in qname_policy_environment
        and c not in time_converted_input_stubs
    }
    functions: dict[str, ColumnObject] = {}
    input_stubs: dict[str, PolicyInput] = {}
    for abgfn in sorted(agg_by_group_function_names | agg_by_group_input_names):
        supplied_by_data = abgfn in agg_by_group_input_names
        # A stub's source may itself be data supplied at a time-converted name,
        # which has a stub rather than a node of its own.
        source_environment = (
            {**qname_policy_environment, **time_converted_input_stubs}
            if supplied_by_data
            else qname_policy_environment
        )
        match = cast("re.Match[str]", gp.match(abgfn))
        base_name_with_time_unit = match.group("base_name_with_time_unit")

        if base_name_with_time_unit in potential_agg_by_group_sources or (
            supplied_by_data and not gp.match(base_name_with_time_unit)
        ):
            # Check if the aggregation target is already a dependency of the source
            # function to avoid creating cycles in the DAG. Consider a function `x` that
            # takes `x_hh` as an input, assuming it to be provided in the input data. If
            # we create a function `x_hh`, which would aggregate `x` by household, we
            # create a cycle. If `x_hh` is actually provided as an input, `x_hh` would
            # be overwritten, removing the cycle. However, if `x_hh` is not provided as
            # an input, an error message would be shown that a cycle between `x` and
            # `x_hh` was detected. This hides the actual problem, which is that `x_hh`
            # is not provided as an input.
            source_function = column_functions.get(base_name_with_time_unit)
            if (
                not supplied_by_data
                and source_function
                and abgfn in get_free_arguments(source_function)
            ):
                continue

            group_id = match.group("group")
            source_unit = _resolve_source_unit(
                source_name=base_name_with_time_unit,
                column_functions=column_functions,
                qname_policy_environment=source_environment,
            )
            if supplied_by_data and source_unit is UNSET_UNIT:
                # Nothing to derive the stub's unit from; the mandatory-units
                # check reports the input.
                continue
            source_kind = _resolve_source_column_kind(
                source_name=base_name_with_time_unit,
                column_functions=column_functions,
                qname_policy_environment=source_environment,
                input_stub_name=abgfn if supplied_by_data else None,
            )
            unit = unit_for_aggregation(
                source_unit=source_unit,
                agg_type=AggType.SUM,
                target_level=group_id,
                source_is_boolean=source_kind in BOOL_KINDS,
            )
            output_kind = resolve_agg_output_kind(
                agg_type=AggType.SUM, input_kind=source_kind, node_name=abgfn
            )
            if supplied_by_data:
                # Add policy input stubs for already aggregated input data to ensure
                # fully specified DAG unit annotations (GEP 10).
                input_stubs[abgfn] = _input_column_stub(
                    qname=abgfn,
                    unit=unit,
                    data_type=output_kind,
                )
                continue
            group_id_name = f"{group_id}_id"
            mapper = {"group_id": group_id_name, "column": base_name_with_time_unit}
            # Stamp concrete column-type annotations onto the `grouped_sum`
            # wrapper. Its runtime implementation signature widens to
            # `FloatColumn | IntColumn`; left untouched, that union becomes
            # the node's producer type and the DAG's annotation-consistency
            # check rejects it against a concretely typed consumer.
            agg_func = synthesize_typed_aggregation_wrapper(
                rename_arguments(func=grouped_sum, mapper=mapper),
                agg_type=AggType.SUM,
                source_column_kind=source_kind,
                column_param_name=base_name_with_time_unit,
                node_name=abgfn,
            )
            functions[abgfn] = AggByGroupFunction(
                leaf_name=dt.tree_path_from_qname(abgfn)[-1],
                function=agg_func,
                start_date=DEFAULT_START_DATE,
                end_date=DEFAULT_END_DATE,
                description=(
                    f"Automatic sum aggregation of "
                    f"{dt.tree_path_from_qname(base_name_with_time_unit)} by "
                    f"{group_id} ID."
                ),
                unit=unit,
            )
    return DerivedColumnObjects(
        functions=MappingProxyType(functions),
        input_stubs=MappingProxyType(input_stubs),
    )


def _resolve_source_unit(
    source_name: str,
    column_functions: dict[str, ColumnFunction],
    qname_policy_environment: PolicyEnvironment,
) -> CompositeUnit:
    """Resolve the unit of an auto-aggregation source column.

    The source is a column function, a `PolicyInput` declared at `source_name`, or a
    user-supplied input at a different time unit than its declared `PolicyInput` sibling
    (e.g. caller passes `bonus_y` against a `bonus_m` declaration; the sibling's unit is
    re-based to the source's period).
    """
    source = column_functions.get(source_name) or qname_policy_environment.get(
        source_name
    )
    if source is not None:
        declared = getattr(source, "unit", UNSET_UNIT)
        # A derived function computes on already-converted values, so a
        # concrete currency token passes its agnostic counterpart on.
        return ttsim_unit_with_agnostic_currency(declared)
    sibling = _find_sibling_policy_input_at_other_time_unit(
        source_name=source_name,
        qname_policy_environment=qname_policy_environment,
    )
    if sibling is None or sibling.unit is UNSET_UNIT:
        return UNSET_UNIT
    return unit_with_rebased_period(
        unit=ttsim_unit_with_agnostic_currency(sibling.unit),
        time_unit_id=source_name.rpartition("_")[2],
    )


def _resolve_source_column_kind(
    source_name: str,
    column_functions: dict[str, ColumnFunction],
    qname_policy_environment: PolicyEnvironment,
    *,
    input_stub_name: str | None = None,
) -> ResolvedKind:
    """Resolve the column kind of an auto-aggregation source column.

    The source of an auto-aggregation is an individual-level column. It is one of:

    - a column function in the DAG (resolve from its return annotation);
    - a `PolicyInput` declared at `source_name` (resolve from its `data_type`);
    - a user-supplied input at a different time unit than the declared
      `PolicyInput` (e.g. caller passes `bonus_y` against a `bonus_m`
      declaration). The synthesised time-conversion function widens its
      return to `FloatColumn`, so resolve from the declared sibling
      `PolicyInput` to recover the precise dtype.

    Either annotation may be scalar-typed (a not-yet-vectorized scalar
    policy function, a `PolicyInput` declared `-> int`);
    `vectorized_column_kind` promotes a scalar kind to the column kind the
    node carries once it operates on data.

    Args:
        source_name: The qualified name of the source column.
        column_functions: Qualified-name to column function mapping.
        qname_policy_environment: The flat policy environment.
        input_stub_name: The derived name whose `PolicyInput` stub the kind is
            resolved for, or `None` when resolving for an aggregation function.
            A stub whose kind does not resolve would be dropped and its input
            would silently escape currency conversion and unit validation, so
            the failure is reported against the input's name.

    Returns:
        The source column's `ResolvedKind` (always a column kind).

    Raises:
        TypeResolutionError: If no column function, declared `PolicyInput`
            at `source_name`, or declared `PolicyInput` at a sibling time
            unit carries a kind.
    """
    if input_stub_name is None:
        msg = (
            f"Cannot resolve the dtype of auto-aggregation source column "
            f"{source_name!r}: it is neither a column function in the DAG, a "
            f"`PolicyInput` with a declared `data_type`, nor a sibling of any "
            f"declared `PolicyInput` at another time unit. A concrete source "
            f"dtype is required to synthesize a typed aggregation wrapper."
        )
    else:
        msg = (
            f"Cannot derive the unit of input {input_stub_name!r} from its "
            f"aggregation source {source_name!r}: the source declares a unit but "
            f"its column kind does not resolve, and a SUM over a boolean carries "
            f"a different unit than one over a number. Declare {source_name!r} "
            f"with a resolvable data type."
        )
    try:
        source_function = column_functions.get(source_name)
        if source_function is not None:
            kind = resolve_kind_of_column_function(
                source_function,
                node_name=source_name,
            )
            return vectorized_column_kind(kind, node_name=source_name)

        policy_input = qname_policy_environment.get(source_name)
        if isinstance(policy_input, PolicyInput):
            kind = resolve_kind_of_annotation(
                policy_input.data_type,
                node_name=source_name,
            )
            return vectorized_column_kind(kind, node_name=source_name)

        sibling = _find_sibling_policy_input_at_other_time_unit(
            source_name=source_name,
            qname_policy_environment=qname_policy_environment,
        )
        if sibling is not None:
            kind = resolve_kind_of_annotation(
                sibling.data_type,
                node_name=source_name,
            )
            return vectorized_column_kind(kind, node_name=source_name)
    except TypeResolutionError as error:
        if input_stub_name is None:
            raise
        raise TypeResolutionError(msg) from error

    raise TypeResolutionError(msg)


def _input_column_stub(
    qname: str,
    unit: CompositeUnit,
    data_type: str | ResolvedKind,
) -> PolicyInput:
    """A `PolicyInput` standing in for input data supplied at a derived name.

    `data_type` is either a column-type alias name or an already-resolved
    `ResolvedKind`; `resolve_kind_of_annotation` accepts both.
    """
    return PolicyInput(
        leaf_name=dt.tree_path_from_qname(qname)[-1],
        data_type=data_type,
        start_date=DEFAULT_START_DATE,
        end_date=DEFAULT_END_DATE,
        description=f"Input data supplied at the derived name {qname!r}.",
        unit=unit,
    )


def _find_sibling_policy_input_at_other_time_unit(
    source_name: str,
    qname_policy_environment: PolicyEnvironment,
) -> PolicyInput | None:
    """Find a `PolicyInput` declared at a sibling time unit of `source_name`.

    Strips the trailing time-unit suffix from `source_name` (if any) and
    looks up each other time unit at the same base name in
    `qname_policy_environment`. Returns the first `PolicyInput` found, or
    `None` if `source_name` has no time-unit suffix or no sibling is
    declared.
    """
    base, sep, time_unit = source_name.rpartition("_")
    if not sep or time_unit not in TIME_UNIT_IDS_TO_LABELS:
        return None
    for other_time_unit in TIME_UNIT_IDS_TO_LABELS:
        if other_time_unit == time_unit:
            continue
        candidate = qname_policy_environment.get(f"{base}_{other_time_unit}")
        if isinstance(candidate, PolicyInput):
            return candidate
    return None


def _get_potential_agg_by_group_function_names_from_function_arguments(
    functions: dict[str, ColumnFunction],
    group_pattern: re.Pattern[str],
) -> UnorderedQNames:
    """Get potential aggregation function names from function arguments.

    Args:
        functions: Dictionary containing functions to build the DAG.
        group_pattern: Compiled regex pattern for matching grouping suffixes.

    Returns:
        Set of potential aggregation targets.

    """
    all_names = {
        name for func in functions.values() for name in get_free_arguments(func)
    }
    return {n for n in all_names if group_pattern.match(n)}
