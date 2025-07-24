from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, overload

import dags.tree as dt
from dags import get_free_arguments, rename_arguments

from ttsim.interface_dag_elements.shared import (
    get_base_name_and_grouping_suffix,
    get_re_pattern_for_all_time_units_and_groupings,
    get_re_pattern_for_specific_time_units_and_groupings,
    group_pattern,
)
from ttsim.tt.aggregation import grouped_sum
from ttsim.tt.column_objects_param_function import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    AggByGroupFunction,
    ColumnFunction,
    ColumnObject,
    ParamFunction,
    TimeConversionFunction,
)
from ttsim.tt.param_objects import ScalarParam

if TYPE_CHECKING:
    import re
    from collections.abc import Callable

    from ttsim.typing import (
        BoolColumn,
        FloatColumn,
        IntColumn,
        OrderedQNames,
        QNamePolicyEnvironment,
        UnorderedQNames,
    )


TIME_UNIT_LABELS = {
    "y": "Year",
    "q": "Quarter",
    "m": "Month",
    "w": "Week",
    "d": "Day",
}

_Q_PER_Y = 4
_M_PER_Y = 12
_W_PER_Y = 365.25 / 7
_D_PER_Y = 365.25


def fail_if_multiple_time_units_for_same_base_name_and_group(
    base_names_and_groups_to_variations: dict[tuple[str, str], list[str]],
) -> None:
    invalid = {
        b: q for b, q in base_names_and_groups_to_variations.items() if len(q) > 1
    }
    if invalid:
        raise ValueError(f"Multiple time units for base names: {invalid}")


def y_to_q(value: float) -> float:
    """
    Converts yearly to quarterly values.

    Parameters
    ----------
    value
        Yearly value to be converted to quarterly value.

    Returns
    -------
    Quarterly value.
    """
    return value / _Q_PER_Y


def y_to_m(value: float) -> float:
    """
    Converts yearly to monthly values.

    Parameters
    ----------
    value
        Yearly value to be converted to monthly value.

    Returns
    -------
    Monthly value.
    """
    return value / _M_PER_Y


def y_to_w(value: float) -> float:
    """
    Converts yearly to weekly values.

    Parameters
    ----------
    value
        Yearly value to be converted to weekly value.

    Returns
    -------
    Weekly value.
    """
    return value / _W_PER_Y


def y_to_d(value: float) -> float:
    """
    Converts yearly to daily values.

    Parameters
    ----------
    value
        Yearly value to be converted to daily value.

    Returns
    -------
    Daily value.
    """
    return value / _D_PER_Y


def q_to_y(value: float) -> float:
    """
    Converts quarterly to yearly values.

    Parameters
    ----------
    value
        Quarterly value to be converted to yearly value.

    Returns
    -------
    Yearly value.
    """
    return value * _Q_PER_Y


def q_to_m(value: float) -> float:
    """
    Converts quarterly to monthly values.

    Parameters
    ----------
    value
        Quarterly value to be converted to monthly value.

    Returns
    -------
    Monthly value.
    """
    return value * _M_PER_Y / _Q_PER_Y


def q_to_w(value: float) -> float:
    """
    Converts quarterly to weekly values.

    Parameters
    ----------
    value
        Quarterly value to be converted to weekly value.

    Returns
    -------
    Weekly value.
    """
    return value * _Q_PER_Y / _W_PER_Y


def q_to_d(value: float) -> float:
    """
    Converts quarterly to daily values.

    Parameters
    ----------
    value
        Quarterly value to be converted to daily value.

    Returns
    -------
    Daily value.
    """
    return value * _Q_PER_Y / _D_PER_Y


def m_to_y(value: float) -> float:
    """
    Converts monthly to yearly values.

    Parameters
    ----------
    value
        Monthly value to be converted to yearly value.

    Returns
    -------
    Yearly value.
    """
    return value * _M_PER_Y


def m_to_q(value: float) -> float:
    """
    Converts monthly to quarterly values.

    Parameters
    ----------
    value
        Monthly value to be converted to quarterly value.

    Returns
    -------
    Quarterly value.
    """
    return value * _M_PER_Y / _Q_PER_Y


def m_to_w(value: float) -> float:
    """
    Converts monthly to weekly values.

    Parameters
    ----------
    value
        Monthly value to be converted to weekly value.

    Returns
    -------
    Weekly value.
    """
    return value * _M_PER_Y / _W_PER_Y


def m_to_d(value: float) -> float:
    """
    Converts monthly to daily values.

    Parameters
    ----------
    value
        Monthly value to be converted to daily value.

    Returns
    -------
    Daily value.
    """
    return value * _M_PER_Y / _D_PER_Y


def w_to_y(value: float) -> float:
    """
    Converts weekly to yearly values.

    Parameters
    ----------
    value
        Weekly value to be converted to yearly value.

    Returns
    -------
    Yearly value.
    """
    return value * _W_PER_Y


def w_to_q(value: float) -> float:
    """
    Converts weekly to quarterly values.

    Parameters
    ----------
    value
        Weekly value to be converted to quarterly value.

    Returns
    -------
    Quarterly value.
    """
    return value * _W_PER_Y / _Q_PER_Y


def w_to_m(value: float) -> float:
    """
    Converts weekly to monthly values.

    Parameters
    ----------
    value
        Weekly value to be converted to monthly value.

    Returns
    -------
    Monthly value.
    """
    return value * _W_PER_Y / _M_PER_Y


def w_to_d(value: float) -> float:
    """
    Converts weekly to daily values.

    Parameters
    ----------
    value
        Weekly value to be converted to daily value.

    Returns
    -------
    Daily value.
    """
    return value * _W_PER_Y / _D_PER_Y


def d_to_y(value: float) -> float:
    """
    Converts daily to yearly values.

    Parameters
    ----------
    value
        Daily value to be converted to yearly value.

    Returns
    -------
    Yearly value.
    """
    return value * _D_PER_Y


def d_to_m(value: float) -> float:
    """
    Converts daily to monthly values.

    Parameters
    ----------
    value
        Daily value to be converted to monthly value.

    Returns
    -------
    Monthly value.
    """
    return value * _D_PER_Y / _M_PER_Y


def d_to_q(value: float) -> float:
    """
    Converts daily to quarterly values.

    Parameters
    ----------
    value
        Daily value to be converted to quarterly value.

    Returns
    -------
    Quarterly value.
    """
    return value * _D_PER_Y / _Q_PER_Y


def d_to_w(value: float) -> float:
    """
    Converts daily to weekly values.

    Parameters
    ----------
    value
        Daily value to be converted to weekly value.

    Returns
    -------
    Weekly value.
    """
    return value * _D_PER_Y / _W_PER_Y


_time_conversion_functions = {
    "y_to_m": y_to_m,
    "y_to_q": y_to_q,
    "y_to_w": y_to_w,
    "y_to_d": y_to_d,
    "q_to_y": q_to_y,
    "q_to_m": q_to_m,
    "q_to_w": q_to_w,
    "q_to_d": q_to_d,
    "m_to_y": m_to_y,
    "m_to_q": m_to_q,
    "m_to_w": m_to_w,
    "m_to_d": m_to_d,
    "w_to_y": w_to_y,
    "w_to_q": w_to_q,
    "w_to_m": w_to_m,
    "w_to_d": w_to_d,
    "d_to_y": d_to_y,
    "d_to_m": d_to_m,
    "d_to_q": d_to_q,
    "d_to_w": d_to_w,
}


def _convertibles(
    qname_policy_environment: QNamePolicyEnvironment,
) -> dict[str, ColumnObject | ParamFunction | ScalarParam]:
    return {
        qn: e
        for qn, e in qname_policy_environment.items()
        if isinstance(e, (ColumnObject, ScalarParam))
        or (
            isinstance(e, ParamFunction)
            and e.function.__annotations__["return"] in {"float", "int"}
        )
    }


def create_time_conversion_functions(
    qname_policy_environment: QNamePolicyEnvironment,
    input_columns: UnorderedQNames,
    grouping_levels: OrderedQNames,
) -> UnorderedQNames:
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

    Parameters
    ----------
    functions
        The functions dict with qualified function names as keys and functions as
        values.
    input_columns
        The names of the input columns, represented by qualified names.
    grouping_levels
        The grouping levels.

    Returns
    -------
    The functions dict with the new time conversion functions.
    """
    time_units = tuple(TIME_UNIT_LABELS)
    pattern_all = get_re_pattern_for_all_time_units_and_groupings(
        grouping_levels=grouping_levels,
        time_units=time_units,
    )
    # Map base name and grouping suffix to time conversion inputs.
    bngs_to_time_conversion_inputs = {}
    bngs_to_variations = {}
    for qname, element in _convertibles(qname_policy_environment).items():
        match = pattern_all.fullmatch(qname)
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
    for bngs, inputs in bngs_to_time_conversion_inputs.items():
        for col_name in input_columns:
            # If base_name is in provided data, base time conversions on that.
            if pattern_specific := get_re_pattern_for_specific_time_units_and_groupings(
                base_name=bngs[0],
                all_time_units=time_units,
                grouping_levels=grouping_levels,
            ).fullmatch(col_name):
                inputs["qname_source"] = col_name
                inputs["time_unit"] = pattern_specific.group("time_unit")
                break

        variations = _create_one_set_of_time_conversion_functions(**inputs)
        converted_elements = {**converted_elements, **variations}

    return converted_elements


def _create_one_set_of_time_conversion_functions(
    base_name: str,
    qname_source: str,
    element: ColumnObject,
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
                converter=_time_conversion_functions[
                    f"{time_unit}_to_{target_time_unit}"
                ],
            ),
            source=qname_source,
            start_date=element.start_date,
            end_date=element.end_date,
            description=(
                f"Time conversion of {dt.tree_path_from_qname(qname_source)} "
                f"from {time_unit} to {target_time_unit}"
            ),
        )

    return result


def _create_function_for_time_unit(
    source: str,
    converter: Callable[[float], float],
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
    input_columns: UnorderedQNames,
    tt_targets: OrderedQNames,
    grouping_levels: OrderedQNames,
    # backend: Literal["numpy", "jax"],
) -> UnorderedQNames:
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
    out = {}
    for abgfn in agg_by_group_function_names:
        match = gp.match(abgfn)
        base_name_with_time_unit = match.group("base_name_with_time_unit")
        if base_name_with_time_unit in potential_agg_by_group_sources:
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
            if source_function and abgfn in get_free_arguments(source_function):
                continue

            group_id = f"{match.group('group')}_id"
            mapper = {"group_id": group_id, "column": base_name_with_time_unit}
            agg_func = rename_arguments(
                func=grouped_sum,
                mapper=mapper,
            )
            out[abgfn] = AggByGroupFunction(
                leaf_name=dt.tree_path_from_qname(abgfn)[-1],
                function=agg_func,
                start_date=DEFAULT_START_DATE,
                end_date=DEFAULT_END_DATE,
                description=(
                    f"Automatic sum aggregation of "
                    f"{dt.tree_path_from_qname(base_name_with_time_unit)} by "
                    f"{match.group('group')} ID."
                ),
            )
    return out


def _get_potential_agg_by_group_function_names_from_function_arguments(
    functions: UnorderedQNames,
    group_pattern: re.Pattern[str],
) -> UnorderedQNames:
    """Get potential aggregation function names from function arguments.

    Parameters
    ----------
    functions
        Dictionary containing functions to build the DAG.

    Returns
    -------
    Set of potential aggregation targets.
    """
    all_names = {
        name for func in functions.values() for name in get_free_arguments(func)
    }
    return {n for n in all_names if group_pattern.match(n)}
