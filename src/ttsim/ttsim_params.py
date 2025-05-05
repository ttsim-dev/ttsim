from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_float_dtype,
    is_integer_dtype,
)

from ttsim.shared import to_datetime, validate_date_range

if TYPE_CHECKING:
    import pandas as pd

    from ttsim.config import numpy_or_jax as np
    from ttsim.typing import DashedISOString

DEFAULT_START_DATE = datetime.date(1900, 1, 1)
DEFAULT_END_DATE = datetime.date(2100, 12, 31)


@dataclass(frozen=True)
class TTSIMParam:
    """
    Abstract base class for all TTSIM Parameters.
    """

    leaf_name: str
    start_date: datetime.date
    end_date: datetime.date
    unit: (
        None
        | Literal[
            "Euro",
            "DM",
            "Share",
            "Percent",
            "Factor",
            "Year",
            "Month",
            "Hour",
            "Square Meter",
            "Euro / Square Meter",
        ]
    )
    reference_period: None | Literal["Year", "Quarter", "Month", "Week", "Day"]
    name: dict[Literal["de", "en"], str]
    description: dict[Literal["de", "en"], str]
    note: str | None
    reference: str | None


@dataclass(frozen=True)
class ScalarTTSIMParam(TTSIMParam):
    """
    A scalar TTSIM parameter directly read from a YAML file.
    """

    value: bool | int | float


@dataclass(frozen=True)
class DictTTSIMParam(TTSIMParam):
    """
    A TTSIM parameter directly read from a YAML file that is a flat dictionary.
    """

    value: dict[str | int, float | int | bool]


@dataclass(frozen=True)
class ListTTSIMParam(TTSIMParam):
    """
    A TTSIM parameter directly read from a YAML file that is a list.
    """

    value: list[float | int | bool]


def _convert_and_validate_dates(
    start_date: datetime.date | DashedISOString,
    end_date: datetime.date | DashedISOString,
) -> tuple[datetime.date, datetime.date]:
    """Convert and validate date strings to datetime.date objects.

    Parameters
    ----------
    start_date
        The start date (inclusive) in the format YYYY-MM-DD (part of ISO 8601).
    end_date
        The end date (inclusive) in the format YYYY-MM-DD (part of ISO 8601).

    Returns
    -------
    tuple[datetime.date, datetime.date]
        The converted and validated start and end dates.
    """
    start_date = to_datetime(start_date)
    end_date = to_datetime(end_date)

    validate_date_range(start_date, end_date)

    return start_date, end_date


def check_series_has_expected_type(series: pd.Series, internal_type: np.dtype) -> bool:
    """Checks whether used series has already expected internal type.

    Parameters
    ----------
    series : pandas.Series or pandas.DataFrame or dict of pandas.Series
        Data provided by the user.
    internal_type : TypeVar
        One of the internal gettsim types.

    Returns
    -------
    Bool

    """
    if (
        (internal_type == float) & (is_float_dtype(series))
        or (internal_type == int) & (is_integer_dtype(series))
        or (internal_type == bool) & (is_bool_dtype(series))
        or (internal_type == numpy.datetime64) & (is_datetime64_any_dtype(series))
    ):
        out = True
    else:
        out = False

    return out
