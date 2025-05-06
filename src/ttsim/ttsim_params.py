from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import datetime


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
class PiecewiseLinearTTSIMParam(TTSIMParam):
    """
    A TTSIM parameter directly read from a YAML file that specifies a piecewise linear
    function.
    """

    value: dict[
        int,
        dict[
            Literal[
                "lower_threshold",
                "upper_threshold",
                "intercept_at_lower_threshold",
                "rate",
            ],
            float,
        ],
    ]


@dataclass(frozen=True)
class PiecewiseQuadraticTTSIMParam(TTSIMParam):
    """
    A TTSIM parameter directly read from a YAML file that specifies a piecewise
    quadratic function.
    """

    value: dict[
        int,
        dict[
            Literal[
                "lower_threshold",
                "upper_threshold",
                "intercept_at_lower_threshold",
                "rate_linear",
                "rate_quadratic",
            ],
            float,
        ],
    ]
