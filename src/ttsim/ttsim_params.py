from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import datetime

    from ttsim.piecewise_polynomial import PiecewisePolynomialParameters


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

    value: (
        dict[str, int]
        | dict[str, float]
        | dict[str, bool]
        | dict[int, int]
        | dict[int, float]
        | dict[int, bool]
    )


@dataclass(frozen=True)
class ListTTSIMParam(TTSIMParam):
    """
    A TTSIM parameter directly read from a YAML file that is a list.
    """

    value: list[float] | list[int] | list[bool]


@dataclass(frozen=True)
class PiecewisePolynomialTTSIMParam(TTSIMParam):
    """A TTSIM parameter with its contents read and converted from a YAML file.

    Its value is a PiecewisePolynomialParameters object, i.e., it contains the
    parameters for calling `piecewise_polynomial`.
    """

    value: PiecewisePolynomialParameters
