from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, ParamSpec, TypeVar

if TYPE_CHECKING:
    import datetime

    from ttsim.config import numpy_or_jax as np

FunArgTypes = ParamSpec("FunArgTypes")
ReturnType = TypeVar("ReturnType")


@dataclass(frozen=True)
class ParamObject:
    """
    Abstract base class for all types of parameters.
    """

    leaf_name: str
    start_date: datetime.date
    end_date: datetime.date
    unit: (
        None
        | Literal[
            "Euros",
            "DM",
            "Share",
            "Percent",
            "Years",
            "Months",
            "Hours",
            "Square Meters",
            "Euros / Square Meter",
        ]
    )
    reference_period: None | Literal["Year", "Quarter", "Month", "Week", "Day"]
    name: dict[Literal["de", "en"], str]
    description: dict[Literal["de", "en"], str]


@dataclass(frozen=True)
class ScalarParam(ParamObject):
    """
    A scalar parameter directly read from a YAML file.
    """

    value: bool | int | float
    note: str | None = None
    reference: str | None = None


@dataclass(frozen=True)
class DictParam(ParamObject):
    """
    A parameter directly read from a YAML file that is a flat dictionary.
    """

    value: (
        dict[str, int]
        | dict[str, float]
        | dict[str, bool]
        | dict[int, int]
        | dict[int, float]
        | dict[int, bool]
    )
    note: str | None = None
    reference: str | None = None

    def __post_init__(self) -> None:
        assert all(x not in self.value for x in ["note", "reference"])


@dataclass(frozen=True)
class PiecewisePolynomialParam(ParamObject):
    """A parameter with its contents read and converted from a YAML file.

    Its value is a PiecewisePolynomialParamValue object, i.e., it contains the
    parameters for calling `piecewise_polynomial`.
    """

    value: PiecewisePolynomialParamValue
    note: str | None = None
    reference: str | None = None


@dataclass(frozen=True)
class ConsecutiveIntLookupTableParam(ParamObject):
    """A parameter with its contents read and converted from a YAML file.

    Its value is a ConsecutiveIntLookupTableParamValue object, i.e., it contains the
    parameters for calling `lookup_table`.
    """

    value: ConsecutiveIntLookupTableParamValue
    note: str | None = None
    reference: str | None = None


@dataclass(frozen=True)
class RawParam(ParamObject):
    """
    A parameter directly read from a YAML file that is an arbitrarily nested
    dictionary.
    """

    value: dict[str | int, Any]
    note: str | None = None
    reference: str | None = None

    def __post_init__(self) -> None:
        assert all(x not in self.value for x in ["note", "reference"])


@dataclass(frozen=True)
class PiecewisePolynomialParamValue:
    """The parameters expected by piecewise_polynomial"""

    thresholds: np.ndarray
    intercepts: np.ndarray
    rates: np.ndarray


@dataclass(frozen=True)
class ConsecutiveIntLookupTableParamValue:
    """The parameters expected by lookup_table"""

    base_value_to_subtract: int
    values_to_look_up: np.ndarray
