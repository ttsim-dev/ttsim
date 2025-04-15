from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import dags.tree as dt
from dags import rename_arguments

from ttsim.shared import (
    get_re_pattern_for_all_time_units_and_groupings,
    get_re_pattern_for_specific_time_units_and_groupings,
)
from ttsim.ttsim_objects import (
    DerivedTimeConversionFunction,
    TTSIMFunction,
    TTSIMObject,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ttsim.typing import (
        QualNameDataDict,
        QualNameTTSIMFunctionDict,
        QualNameTTSIMObjectDict,
    )


TIME_UNITS = {
    "y": "year",
    "q": "quarter",
    "m": "month",
    "w": "week",
    "d": "day",
}

_Q_PER_Y = 4
_M_PER_Y = 12
_W_PER_Y = 365.25 / 7
_D_PER_Y = 365.25


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


def create_time_conversion_functions(
    ttsim_objects: QualNameTTSIMObjectDict,
    data: QualNameDataDict,
    groupings: tuple[str, ...],
) -> QualNameTTSIMFunctionDict:
    """
     Create functions that convert variables to different time units.

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
    data
        The data dict with qualified data names as keys and pandas Series as values.

    Returns
    -------
    The functions dict with the new time conversion functions.
    """

    all_time_units = tuple(TIME_UNITS)
    pattern_all = get_re_pattern_for_all_time_units_and_groupings(
        groupings=groupings,
        supported_time_units=all_time_units,
    )

    base_names_to_time_conversion_inputs = {}
    base_names_to_variations = {}
    for qual_name, ttsim_object in ttsim_objects.items():
        match = pattern_all.fullmatch(qual_name)
        base_name = match.group("base_name")
        if match.group("time_unit"):
            if base_name in base_names_to_variations:
                (base_names_to_variations[base_name].append(qual_name),)
            else:
                base_names_to_variations[base_name] = [qual_name]
            base_names_to_time_conversion_inputs[base_name] = {
                "base_name": base_name,
                "qual_name_source": qual_name,
                "ttsim_object": ttsim_object,
                "time_unit": match.group("time_unit"),
                "aggregation_suffix": f"_{match.group('aggregation')}"
                if match.group("aggregation")
                else "",
                "all_time_units": all_time_units,
            }

    _fail_if_multiple_time_units_for_same_base_name(base_names_to_variations)

    converted_ttsim_objects = {}
    for base_name, inputs in base_names_to_time_conversion_inputs.items():
        for qual_name_data in data:
            # If base_name is in provided data, base time conversions on that.
            if pattern_specific := get_re_pattern_for_specific_time_units_and_groupings(
                base_name=base_name,
                all_time_units=all_time_units,
                groupings=groupings,
            ).fullmatch(qual_name_data):
                inputs["qual_name_source"] = qual_name_data
                inputs["time_unit"] = pattern_specific.group("time_unit")
                break

        variations = _create_one_set_of_time_conversion_functions(**inputs)
        for der_name in variations:
            if der_name in converted_ttsim_objects or der_name in data:
                raise ValueError("Fixme, I should not be here -- left for debugging")
        converted_ttsim_objects = {**converted_ttsim_objects, **variations}

    return converted_ttsim_objects


def _fail_if_multiple_time_units_for_same_base_name(
    base_names_to_variations: dict[str, list[str]],
) -> None:
    invalid = {b: q for b, q in base_names_to_variations.items() if len(q) > 1}
    if invalid:
        raise ValueError(f"Multiple time units for base names: {invalid}")


def _create_one_set_of_time_conversion_functions(
    base_name: str,
    qual_name_source: str,
    ttsim_object: TTSIMObject,
    time_unit: str,
    aggregation_suffix: str,
    all_time_units: tuple[str, ...],
) -> dict[str, DerivedTimeConversionFunction]:
    result: dict[str, DerivedTimeConversionFunction] = {}
    dependencies = (
        set(inspect.signature(ttsim_object).parameters)
        if isinstance(ttsim_object, TTSIMFunction)
        else set()
    )

    for target_time_unit in [tu for tu in all_time_units if tu != time_unit]:
        new_name = f"{base_name}_{target_time_unit}{aggregation_suffix}"

        # Without this check, we could create cycles in the DAG: Consider a
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

        result[new_name] = DerivedTimeConversionFunction(
            leaf_name=dt.tree_path_from_qual_name(new_name)[-1],
            function=_create_function_for_time_unit(
                source=qual_name_source,
                converter=_time_conversion_functions[
                    f"{time_unit}_to_{target_time_unit}"
                ],
            ),
            source=qual_name_source,
            start_date=ttsim_object.start_date,
            end_date=ttsim_object.end_date,
            vectorization_strategy="not_required",
        )

    return result


def _create_function_for_time_unit(
    source: str, converter: Callable[[float], float]
) -> Callable[[float], float]:
    @rename_arguments(mapper={"x": source})
    def func(x: float) -> float:
        return converter(x)

    return func
