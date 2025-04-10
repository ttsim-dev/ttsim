from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import dags.tree as dt
from dags import rename_arguments

from _gettsim.config import SUPPORTED_GROUPINGS
from ttsim.function_types import DerivedTimeConversionFunction, TTSIMObject
from ttsim.shared import (
    get_re_pattern_for_all_time_units_and_groupings,
    get_re_pattern_for_specific_time_units_and_groupings,
)

if TYPE_CHECKING:
    import re
    from collections.abc import Callable

    from ttsim.typing import QualNameDataDict, QualNameTTSIMObjectDict

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
) -> QualNameTTSIMObjectDict:
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

    Unless the corresponding function already exists, the following functions are
    created:
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

    converted_ttsim_objects = {}

    for source_name, ttsim_object in ttsim_objects.items():
        all_time_units = tuple(TIME_UNITS)
        pattern_all = get_re_pattern_for_all_time_units_and_groupings(
            supported_groupings=SUPPORTED_GROUPINGS,
            supported_time_units=all_time_units,
        )
        pattern_specific = pattern_all.fullmatch(source_name)
        base_name = pattern_specific.group("base_name")

        # If base_name is in data, base time conversions on this.
        for data_name in data:
            if pattern_specific := get_re_pattern_for_specific_time_units_and_groupings(
                base_name=base_name,
                supported_time_units=all_time_units,
                supported_groupings=SUPPORTED_GROUPINGS,
            ):
                source_name = pattern_specific.fullmatch(data_name)  # noqa: PLW2901
                break

        all_time_conversions_for_this_function = _create_time_conversion_functions(
            source_name=source_name,
            ttsim_object=ttsim_object,
            time_unit_pattern=pattern_all,
            all_time_units=all_time_units,
        )
        for der_name, der_func in all_time_conversions_for_this_function.items():
            if der_name in converted_ttsim_objects or der_name in data:
                continue
            else:
                converted_ttsim_objects[der_name] = der_func

    return converted_ttsim_objects


def _create_time_conversion_functions(
    source_name: str,
    ttsim_object: TTSIMObject,
    time_unit_pattern: re.Pattern,
    all_time_units: tuple[str, ...],
) -> dict[str, DerivedTimeConversionFunction]:
    result: dict[str, DerivedTimeConversionFunction] = {}
    match = time_unit_pattern.fullmatch(source_name)
    base_name = match.group("base_name")
    time_unit = match.group("time_unit") or ""
    aggregation = match.group("aggregation") or ""
    dependencies = (
        set(inspect.signature(ttsim_object).parameters) if ttsim_object else set()
    )

    if match and time_unit:
        missing_time_units = [unit for unit in all_time_units if unit != time_unit]
        for missing_time_unit in missing_time_units:
            new_name = (
                f"{base_name}_{missing_time_unit}_{aggregation}"
                if aggregation
                else f"{base_name}_{missing_time_unit}"
            )

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
                    source=source_name,
                    converter=_time_conversion_functions[
                        f"{time_unit}_to_{missing_time_unit}"
                    ],
                ),
                source=source_name,
                start_date=ttsim_object.start_date,
                end_date=ttsim_object.end_date,
            )

    return result


def _create_function_for_time_unit(
    source: str, converter: Callable[[float], float]
) -> Callable[[float], float]:
    @rename_arguments(mapper={"x": source})
    def func(x: float) -> float:
        return converter(x)

    return func
