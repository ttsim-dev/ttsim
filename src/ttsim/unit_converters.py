from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from jaxtyping import Array, Float, Int

TIME_UNIT_IDS_TO_LABELS = {
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


# --- Year conversions (stocks) ---


@overload
def y_to_q(value: int) -> int: ...
@overload
def y_to_q(value: float) -> float: ...
@overload
def y_to_q(value: np.integer[Any]) -> np.integer[Any]: ...
@overload
def y_to_q(value: np.floating[Any]) -> np.floating[Any]: ...
@overload
def y_to_q(value: Int[Array, ...]) -> Int[Array, ...]: ...
@overload
def y_to_q(value: Float[Array, ...]) -> Float[Array, ...]: ...
@overload
def y_to_q(value: pd.Series) -> pd.Series: ...
@overload
def y_to_q(value: pd.DataFrame) -> pd.DataFrame: ...


def y_to_q(
    value: (
        float
        | np.integer[Any]
        | np.floating[Any]
        | Int[Array, ...]
        | Float[Array, ...]
        | pd.Series
        | pd.DataFrame
    ),
) -> (
    int
    | float
    | np.integer[Any]
    | np.floating[Any]
    | Int[Array, ...]
    | Float[Array, ...]
    | pd.Series
    | pd.DataFrame
):
    """
    Convert values (stocks) measured on the yearly level to values on the
    quarterly level.
    """
    return value * _Q_PER_Y


@overload
def per_y_to_per_q(value: int) -> float: ...
@overload
def per_y_to_per_q(value: float) -> float: ...
@overload
def per_y_to_per_q(value: np.integer[Any]) -> np.floating[Any]: ...
@overload
def per_y_to_per_q(value: np.floating[Any]) -> np.floating[Any]: ...
@overload
def per_y_to_per_q(
    value: Int[Array, ...],
) -> Float[Array, ...]: ...
@overload
def per_y_to_per_q(
    value: Float[Array, ...],
) -> Float[Array, ...]: ...
@overload
def per_y_to_per_q(value: pd.Series) -> pd.Series: ...
@overload
def per_y_to_per_q(value: pd.DataFrame) -> pd.DataFrame: ...


def per_y_to_per_q(
    value: (
        float
        | np.integer[Any]
        | np.floating[Any]
        | Int[Array, ...]
        | Float[Array, ...]
        | pd.Series
        | pd.DataFrame
    ),
) -> float | np.floating[Any] | Float[Array, ...] | pd.Series | pd.DataFrame:
    """Convert flows: values per year to values per quarter."""
    return q_to_y(value)


@overload
def y_to_m(value: int) -> int: ...
@overload
def y_to_m(value: float) -> float: ...
@overload
def y_to_m(value: np.integer[Any]) -> np.integer[Any]: ...
@overload
def y_to_m(value: np.floating[Any]) -> np.floating[Any]: ...
@overload
def y_to_m(value: Int[Array, ...]) -> Int[Array, ...]: ...
@overload
def y_to_m(value: Float[Array, ...]) -> Float[Array, ...]: ...
@overload
def y_to_m(value: pd.Series) -> pd.Series: ...
@overload
def y_to_m(value: pd.DataFrame) -> pd.DataFrame: ...


def y_to_m(
    value: (
        float
        | np.integer[Any]
        | np.floating[Any]
        | Int[Array, ...]
        | Float[Array, ...]
        | pd.Series
        | pd.DataFrame
    ),
) -> (
    int
    | float
    | np.integer[Any]
    | np.floating[Any]
    | Int[Array, ...]
    | Float[Array, ...]
    | pd.Series
    | pd.DataFrame
):
    """
    Convert values (stocks) measured on the yearly level to values on the
    monthly level.
    """
    return value * _M_PER_Y


@overload
def per_y_to_per_m(value: int) -> float: ...
@overload
def per_y_to_per_m(value: float) -> float: ...
@overload
def per_y_to_per_m(value: np.integer[Any]) -> np.floating[Any]: ...
@overload
def per_y_to_per_m(value: np.floating[Any]) -> np.floating[Any]: ...
@overload
def per_y_to_per_m(
    value: Int[Array, ...],
) -> Float[Array, ...]: ...
@overload
def per_y_to_per_m(
    value: Float[Array, ...],
) -> Float[Array, ...]: ...
@overload
def per_y_to_per_m(value: pd.Series) -> pd.Series: ...
@overload
def per_y_to_per_m(value: pd.DataFrame) -> pd.DataFrame: ...


def per_y_to_per_m(
    value: (
        float
        | np.integer[Any]
        | np.floating[Any]
        | Int[Array, ...]
        | Float[Array, ...]
        | pd.Series
        | pd.DataFrame
    ),
) -> float | np.floating[Any] | Float[Array, ...] | pd.Series | pd.DataFrame:
    """Convert flows: values per year to values per month."""
    return m_to_y(value)


@overload
def y_to_w(value: int) -> float: ...
@overload
def y_to_w(value: float) -> float: ...
@overload
def y_to_w(value: np.integer[Any]) -> np.floating[Any]: ...
@overload
def y_to_w(value: np.floating[Any]) -> np.floating[Any]: ...
@overload
def y_to_w(
    value: Int[Array, ...],
) -> Float[Array, ...]: ...
@overload
def y_to_w(
    value: Float[Array, ...],
) -> Float[Array, ...]: ...
@overload
def y_to_w(value: pd.Series) -> pd.Series: ...
@overload
def y_to_w(value: pd.DataFrame) -> pd.DataFrame: ...


def y_to_w(
    value: (
        float
        | np.integer[Any]
        | np.floating[Any]
        | Int[Array, ...]
        | Float[Array, ...]
        | pd.Series
        | pd.DataFrame
    ),
) -> float | np.floating[Any] | Float[Array, ...] | pd.Series | pd.DataFrame:
    """
    Convert values (stocks) measured on the yearly level to values on the
    weekly level.
    """
    return value * _W_PER_Y


@overload
def per_y_to_per_w(value: int) -> float: ...
@overload
def per_y_to_per_w(value: float) -> float: ...
@overload
def per_y_to_per_w(value: np.integer[Any]) -> np.floating[Any]: ...
@overload
def per_y_to_per_w(
    value: np.floating[Any],
) -> np.floating[Any]: ...
@overload
def per_y_to_per_w(
    value: Int[Array, ...],
) -> Float[Array, ...]: ...
@overload
def per_y_to_per_w(
    value: Float[Array, ...],
) -> Float[Array, ...]: ...
@overload
def per_y_to_per_w(value: pd.Series) -> pd.Series: ...
@overload
def per_y_to_per_w(
    value: pd.DataFrame,
) -> pd.DataFrame: ...


def per_y_to_per_w(
    value: (
        float
        | np.integer[Any]
        | np.floating[Any]
        | Int[Array, ...]
        | Float[Array, ...]
        | pd.Series
        | pd.DataFrame
    ),
) -> float | np.floating[Any] | Float[Array, ...] | pd.Series | pd.DataFrame:
    """Convert flows: values per year to values per week."""
    return w_to_y(value)


@overload
def y_to_d(value: int) -> float: ...
@overload
def y_to_d(value: float) -> float: ...
@overload
def y_to_d(value: np.integer[Any]) -> np.floating[Any]: ...
@overload
def y_to_d(value: np.floating[Any]) -> np.floating[Any]: ...
@overload
def y_to_d(
    value: Int[Array, ...],
) -> Float[Array, ...]: ...
@overload
def y_to_d(
    value: Float[Array, ...],
) -> Float[Array, ...]: ...
@overload
def y_to_d(value: pd.Series) -> pd.Series: ...
@overload
def y_to_d(value: pd.DataFrame) -> pd.DataFrame: ...


def y_to_d(
    value: (
        float
        | np.integer[Any]
        | np.floating[Any]
        | Int[Array, ...]
        | Float[Array, ...]
        | pd.Series
        | pd.DataFrame
    ),
) -> float | np.floating[Any] | Float[Array, ...] | pd.Series | pd.DataFrame:
    """
    Convert values (stocks) measured on the yearly level to values on the
    daily level.
    """
    return value * _D_PER_Y


@overload
def per_y_to_per_d(value: int) -> float: ...
@overload
def per_y_to_per_d(value: float) -> float: ...
@overload
def per_y_to_per_d(value: np.integer[Any]) -> np.floating[Any]: ...
@overload
def per_y_to_per_d(
    value: np.floating[Any],
) -> np.floating[Any]: ...
@overload
def per_y_to_per_d(
    value: Int[Array, ...],
) -> Float[Array, ...]: ...
@overload
def per_y_to_per_d(
    value: Float[Array, ...],
) -> Float[Array, ...]: ...
@overload
def per_y_to_per_d(value: pd.Series) -> pd.Series: ...
@overload
def per_y_to_per_d(
    value: pd.DataFrame,
) -> pd.DataFrame: ...


def per_y_to_per_d(
    value: (
        float
        | np.integer[Any]
        | np.floating[Any]
        | Int[Array, ...]
        | Float[Array, ...]
        | pd.Series
        | pd.DataFrame
    ),
) -> float | np.floating[Any] | Float[Array, ...] | pd.Series | pd.DataFrame:
    """Convert flows: values per year to values per day."""
    return d_to_y(value)


# --- Quarter conversions (stocks) ---


@overload
def q_to_y(value: int) -> float: ...
@overload
def q_to_y(value: float) -> float: ...
@overload
def q_to_y(value: np.integer[Any]) -> np.floating[Any]: ...
@overload
def q_to_y(value: np.floating[Any]) -> np.floating[Any]: ...
@overload
def q_to_y(
    value: Int[Array, ...],
) -> Float[Array, ...]: ...
@overload
def q_to_y(
    value: Float[Array, ...],
) -> Float[Array, ...]: ...
@overload
def q_to_y(value: pd.Series) -> pd.Series: ...
@overload
def q_to_y(value: pd.DataFrame) -> pd.DataFrame: ...


def q_to_y(
    value: (
        float
        | np.integer[Any]
        | np.floating[Any]
        | Int[Array, ...]
        | Float[Array, ...]
        | pd.Series
        | pd.DataFrame
    ),
) -> float | np.floating[Any] | Float[Array, ...] | pd.Series | pd.DataFrame:
    """
    Convert values (stocks) measured on the quarterly level to values on
    the yearly level.
    """
    return value / _Q_PER_Y


@overload
def per_q_to_per_y(value: int) -> int: ...
@overload
def per_q_to_per_y(value: float) -> float: ...
@overload
def per_q_to_per_y(value: np.integer[Any]) -> np.integer[Any]: ...
@overload
def per_q_to_per_y(
    value: np.floating[Any],
) -> np.floating[Any]: ...
@overload
def per_q_to_per_y(
    value: Int[Array, ...],
) -> Int[Array, ...]: ...
@overload
def per_q_to_per_y(
    value: Float[Array, ...],
) -> Float[Array, ...]: ...
@overload
def per_q_to_per_y(value: pd.Series) -> pd.Series: ...
@overload
def per_q_to_per_y(
    value: pd.DataFrame,
) -> pd.DataFrame: ...


def per_q_to_per_y(
    value: (
        float
        | np.integer[Any]
        | np.floating[Any]
        | Int[Array, ...]
        | Float[Array, ...]
        | pd.Series
        | pd.DataFrame
    ),
) -> (
    int
    | float
    | np.integer[Any]
    | np.floating[Any]
    | Int[Array, ...]
    | Float[Array, ...]
    | pd.Series
    | pd.DataFrame
):
    """Convert flows: values per quarter to values per year."""
    return y_to_q(value)


@overload
def q_to_m(value: int) -> float: ...
@overload
def q_to_m(value: float) -> float: ...
@overload
def q_to_m(value: np.integer[Any]) -> np.floating[Any]: ...
@overload
def q_to_m(value: np.floating[Any]) -> np.floating[Any]: ...
@overload
def q_to_m(
    value: Int[Array, ...],
) -> Float[Array, ...]: ...
@overload
def q_to_m(
    value: Float[Array, ...],
) -> Float[Array, ...]: ...
@overload
def q_to_m(value: pd.Series) -> pd.Series: ...
@overload
def q_to_m(value: pd.DataFrame) -> pd.DataFrame: ...


def q_to_m(
    value: (
        float
        | np.integer[Any]
        | np.floating[Any]
        | Int[Array, ...]
        | Float[Array, ...]
        | pd.Series
        | pd.DataFrame
    ),
) -> float | np.floating[Any] | Float[Array, ...] | pd.Series | pd.DataFrame:
    """
    Convert values (stocks) measured on the quarterly level to values on
    the monthly level.
    """
    return value * _M_PER_Y / _Q_PER_Y


@overload
def per_q_to_per_m(value: int) -> float: ...
@overload
def per_q_to_per_m(value: float) -> float: ...
@overload
def per_q_to_per_m(value: np.integer[Any]) -> np.floating[Any]: ...
@overload
def per_q_to_per_m(
    value: np.floating[Any],
) -> np.floating[Any]: ...
@overload
def per_q_to_per_m(
    value: Int[Array, ...],
) -> Float[Array, ...]: ...
@overload
def per_q_to_per_m(
    value: Float[Array, ...],
) -> Float[Array, ...]: ...
@overload
def per_q_to_per_m(value: pd.Series) -> pd.Series: ...
@overload
def per_q_to_per_m(
    value: pd.DataFrame,
) -> pd.DataFrame: ...


def per_q_to_per_m(
    value: (
        float
        | np.integer[Any]
        | np.floating[Any]
        | Int[Array, ...]
        | Float[Array, ...]
        | pd.Series
        | pd.DataFrame
    ),
) -> float | np.floating[Any] | Float[Array, ...] | pd.Series | pd.DataFrame:
    """Convert flows: values per quarter to values per month."""
    return m_to_q(value)


@overload
def q_to_w(value: int) -> float: ...
@overload
def q_to_w(value: float) -> float: ...
@overload
def q_to_w(value: np.integer[Any]) -> np.floating[Any]: ...
@overload
def q_to_w(value: np.floating[Any]) -> np.floating[Any]: ...
@overload
def q_to_w(
    value: Int[Array, ...],
) -> Float[Array, ...]: ...
@overload
def q_to_w(
    value: Float[Array, ...],
) -> Float[Array, ...]: ...
@overload
def q_to_w(value: pd.Series) -> pd.Series: ...
@overload
def q_to_w(value: pd.DataFrame) -> pd.DataFrame: ...


def q_to_w(
    value: (
        float
        | np.integer[Any]
        | np.floating[Any]
        | Int[Array, ...]
        | Float[Array, ...]
        | pd.Series
        | pd.DataFrame
    ),
) -> float | np.floating[Any] | Float[Array, ...] | pd.Series | pd.DataFrame:
    """
    Convert values (stocks) measured on the quarterly level to values on
    the weekly level.
    """
    return value * _W_PER_Y / _Q_PER_Y


@overload
def per_q_to_per_w(value: int) -> float: ...
@overload
def per_q_to_per_w(value: float) -> float: ...
@overload
def per_q_to_per_w(value: np.integer[Any]) -> np.floating[Any]: ...
@overload
def per_q_to_per_w(
    value: np.floating[Any],
) -> np.floating[Any]: ...
@overload
def per_q_to_per_w(
    value: Int[Array, ...],
) -> Float[Array, ...]: ...
@overload
def per_q_to_per_w(
    value: Float[Array, ...],
) -> Float[Array, ...]: ...
@overload
def per_q_to_per_w(value: pd.Series) -> pd.Series: ...
@overload
def per_q_to_per_w(
    value: pd.DataFrame,
) -> pd.DataFrame: ...


def per_q_to_per_w(
    value: (
        float
        | np.integer[Any]
        | np.floating[Any]
        | Int[Array, ...]
        | Float[Array, ...]
        | pd.Series
        | pd.DataFrame
    ),
) -> float | np.floating[Any] | Float[Array, ...] | pd.Series | pd.DataFrame:
    """Convert flows: values per quarter to values per week."""
    return w_to_q(value)


@overload
def q_to_d(value: int) -> float: ...
@overload
def q_to_d(value: float) -> float: ...
@overload
def q_to_d(value: np.integer[Any]) -> np.floating[Any]: ...
@overload
def q_to_d(value: np.floating[Any]) -> np.floating[Any]: ...
@overload
def q_to_d(
    value: Int[Array, ...],
) -> Float[Array, ...]: ...
@overload
def q_to_d(
    value: Float[Array, ...],
) -> Float[Array, ...]: ...
@overload
def q_to_d(value: pd.Series) -> pd.Series: ...
@overload
def q_to_d(value: pd.DataFrame) -> pd.DataFrame: ...


def q_to_d(
    value: (
        float
        | np.integer[Any]
        | np.floating[Any]
        | Int[Array, ...]
        | Float[Array, ...]
        | pd.Series
        | pd.DataFrame
    ),
) -> float | np.floating[Any] | Float[Array, ...] | pd.Series | pd.DataFrame:
    """
    Convert values (stocks) measured on the quarterly level to values on
    the daily level.
    """
    return value * _D_PER_Y / _Q_PER_Y


@overload
def per_q_to_per_d(value: int) -> float: ...
@overload
def per_q_to_per_d(value: float) -> float: ...
@overload
def per_q_to_per_d(value: np.integer[Any]) -> np.floating[Any]: ...
@overload
def per_q_to_per_d(
    value: np.floating[Any],
) -> np.floating[Any]: ...
@overload
def per_q_to_per_d(
    value: Int[Array, ...],
) -> Float[Array, ...]: ...
@overload
def per_q_to_per_d(
    value: Float[Array, ...],
) -> Float[Array, ...]: ...
@overload
def per_q_to_per_d(value: pd.Series) -> pd.Series: ...
@overload
def per_q_to_per_d(
    value: pd.DataFrame,
) -> pd.DataFrame: ...


def per_q_to_per_d(
    value: (
        float
        | np.integer[Any]
        | np.floating[Any]
        | Int[Array, ...]
        | Float[Array, ...]
        | pd.Series
        | pd.DataFrame
    ),
) -> float | np.floating[Any] | Float[Array, ...] | pd.Series | pd.DataFrame:
    """Convert flows: values per quarter to values per day."""
    return d_to_q(value)


# --- Month conversions (stocks) ---


@overload
def m_to_y(value: int) -> float: ...
@overload
def m_to_y(value: float) -> float: ...
@overload
def m_to_y(value: np.integer[Any]) -> np.floating[Any]: ...
@overload
def m_to_y(value: np.floating[Any]) -> np.floating[Any]: ...
@overload
def m_to_y(
    value: Int[Array, ...],
) -> Float[Array, ...]: ...
@overload
def m_to_y(
    value: Float[Array, ...],
) -> Float[Array, ...]: ...
@overload
def m_to_y(value: pd.Series) -> pd.Series: ...
@overload
def m_to_y(value: pd.DataFrame) -> pd.DataFrame: ...


def m_to_y(
    value: (
        float
        | np.integer[Any]
        | np.floating[Any]
        | Int[Array, ...]
        | Float[Array, ...]
        | pd.Series
        | pd.DataFrame
    ),
) -> float | np.floating[Any] | Float[Array, ...] | pd.Series | pd.DataFrame:
    """
    Convert values (stocks) measured on the monthly level to values on
    the yearly level.
    """
    return value / _M_PER_Y


@overload
def per_m_to_per_y(value: int) -> int: ...
@overload
def per_m_to_per_y(value: float) -> float: ...
@overload
def per_m_to_per_y(value: np.integer[Any]) -> np.integer[Any]: ...
@overload
def per_m_to_per_y(
    value: np.floating[Any],
) -> np.floating[Any]: ...
@overload
def per_m_to_per_y(
    value: Int[Array, ...],
) -> Int[Array, ...]: ...
@overload
def per_m_to_per_y(
    value: Float[Array, ...],
) -> Float[Array, ...]: ...
@overload
def per_m_to_per_y(value: pd.Series) -> pd.Series: ...
@overload
def per_m_to_per_y(
    value: pd.DataFrame,
) -> pd.DataFrame: ...


def per_m_to_per_y(
    value: (
        float
        | np.integer[Any]
        | np.floating[Any]
        | Int[Array, ...]
        | Float[Array, ...]
        | pd.Series
        | pd.DataFrame
    ),
) -> (
    int
    | float
    | np.integer[Any]
    | np.floating[Any]
    | Int[Array, ...]
    | Float[Array, ...]
    | pd.Series
    | pd.DataFrame
):
    """Convert flows: values per month to values per year."""
    return y_to_m(value)


@overload
def m_to_q(value: int) -> float: ...
@overload
def m_to_q(value: float) -> float: ...
@overload
def m_to_q(value: np.integer[Any]) -> np.floating[Any]: ...
@overload
def m_to_q(value: np.floating[Any]) -> np.floating[Any]: ...
@overload
def m_to_q(
    value: Int[Array, ...],
) -> Float[Array, ...]: ...
@overload
def m_to_q(
    value: Float[Array, ...],
) -> Float[Array, ...]: ...
@overload
def m_to_q(value: pd.Series) -> pd.Series: ...
@overload
def m_to_q(value: pd.DataFrame) -> pd.DataFrame: ...


def m_to_q(
    value: (
        float
        | np.integer[Any]
        | np.floating[Any]
        | Int[Array, ...]
        | Float[Array, ...]
        | pd.Series
        | pd.DataFrame
    ),
) -> float | np.floating[Any] | Float[Array, ...] | pd.Series | pd.DataFrame:
    """
    Convert values (stocks) measured on the monthly level to values on
    the quarterly level.
    """
    return value * _Q_PER_Y / _M_PER_Y


@overload
def per_m_to_per_q(value: int) -> float: ...
@overload
def per_m_to_per_q(value: float) -> float: ...
@overload
def per_m_to_per_q(value: np.integer[Any]) -> np.floating[Any]: ...
@overload
def per_m_to_per_q(
    value: np.floating[Any],
) -> np.floating[Any]: ...
@overload
def per_m_to_per_q(
    value: Int[Array, ...],
) -> Float[Array, ...]: ...
@overload
def per_m_to_per_q(
    value: Float[Array, ...],
) -> Float[Array, ...]: ...
@overload
def per_m_to_per_q(value: pd.Series) -> pd.Series: ...
@overload
def per_m_to_per_q(
    value: pd.DataFrame,
) -> pd.DataFrame: ...


def per_m_to_per_q(
    value: (
        float
        | np.integer[Any]
        | np.floating[Any]
        | Int[Array, ...]
        | Float[Array, ...]
        | pd.Series
        | pd.DataFrame
    ),
) -> float | np.floating[Any] | Float[Array, ...] | pd.Series | pd.DataFrame:
    """Convert flows: values per month to values per quarter."""
    return q_to_m(value)


@overload
def m_to_w(value: int) -> float: ...
@overload
def m_to_w(value: float) -> float: ...
@overload
def m_to_w(value: np.integer[Any]) -> np.floating[Any]: ...
@overload
def m_to_w(value: np.floating[Any]) -> np.floating[Any]: ...
@overload
def m_to_w(
    value: Int[Array, ...],
) -> Float[Array, ...]: ...
@overload
def m_to_w(
    value: Float[Array, ...],
) -> Float[Array, ...]: ...
@overload
def m_to_w(value: pd.Series) -> pd.Series: ...
@overload
def m_to_w(value: pd.DataFrame) -> pd.DataFrame: ...


def m_to_w(
    value: (
        float
        | np.integer[Any]
        | np.floating[Any]
        | Int[Array, ...]
        | Float[Array, ...]
        | pd.Series
        | pd.DataFrame
    ),
) -> float | np.floating[Any] | Float[Array, ...] | pd.Series | pd.DataFrame:
    """
    Convert values (stocks) measured on the monthly level to values on
    the weekly level.
    """
    return value * _W_PER_Y / _M_PER_Y


@overload
def per_m_to_per_w(value: int) -> float: ...
@overload
def per_m_to_per_w(value: float) -> float: ...
@overload
def per_m_to_per_w(value: np.integer[Any]) -> np.floating[Any]: ...
@overload
def per_m_to_per_w(
    value: np.floating[Any],
) -> np.floating[Any]: ...
@overload
def per_m_to_per_w(
    value: Int[Array, ...],
) -> Float[Array, ...]: ...
@overload
def per_m_to_per_w(
    value: Float[Array, ...],
) -> Float[Array, ...]: ...
@overload
def per_m_to_per_w(value: pd.Series) -> pd.Series: ...
@overload
def per_m_to_per_w(
    value: pd.DataFrame,
) -> pd.DataFrame: ...


def per_m_to_per_w(
    value: (
        float
        | np.integer[Any]
        | np.floating[Any]
        | Int[Array, ...]
        | Float[Array, ...]
        | pd.Series
        | pd.DataFrame
    ),
) -> float | np.floating[Any] | Float[Array, ...] | pd.Series | pd.DataFrame:
    """Convert flows: values per month to values per week."""
    return w_to_m(value)


@overload
def m_to_d(value: int) -> float: ...
@overload
def m_to_d(value: float) -> float: ...
@overload
def m_to_d(value: np.integer[Any]) -> np.floating[Any]: ...
@overload
def m_to_d(value: np.floating[Any]) -> np.floating[Any]: ...
@overload
def m_to_d(
    value: Int[Array, ...],
) -> Float[Array, ...]: ...
@overload
def m_to_d(
    value: Float[Array, ...],
) -> Float[Array, ...]: ...
@overload
def m_to_d(value: pd.Series) -> pd.Series: ...
@overload
def m_to_d(value: pd.DataFrame) -> pd.DataFrame: ...


def m_to_d(
    value: (
        float
        | np.integer[Any]
        | np.floating[Any]
        | Int[Array, ...]
        | Float[Array, ...]
        | pd.Series
        | pd.DataFrame
    ),
) -> float | np.floating[Any] | Float[Array, ...] | pd.Series | pd.DataFrame:
    """
    Convert values (stocks) measured on the monthly level to values on
    the daily level.
    """
    return value * _D_PER_Y / _M_PER_Y


@overload
def per_m_to_per_d(value: int) -> float: ...
@overload
def per_m_to_per_d(value: float) -> float: ...
@overload
def per_m_to_per_d(value: np.integer[Any]) -> np.floating[Any]: ...
@overload
def per_m_to_per_d(
    value: np.floating[Any],
) -> np.floating[Any]: ...
@overload
def per_m_to_per_d(
    value: Int[Array, ...],
) -> Float[Array, ...]: ...
@overload
def per_m_to_per_d(
    value: Float[Array, ...],
) -> Float[Array, ...]: ...
@overload
def per_m_to_per_d(value: pd.Series) -> pd.Series: ...
@overload
def per_m_to_per_d(
    value: pd.DataFrame,
) -> pd.DataFrame: ...


def per_m_to_per_d(
    value: (
        float
        | np.integer[Any]
        | np.floating[Any]
        | Int[Array, ...]
        | Float[Array, ...]
        | pd.Series
        | pd.DataFrame
    ),
) -> float | np.floating[Any] | Float[Array, ...] | pd.Series | pd.DataFrame:
    """Convert flows: values per month to values per day."""
    return d_to_m(value)


# --- Week conversions (stocks) ---


@overload
def w_to_y(value: int) -> float: ...
@overload
def w_to_y(value: float) -> float: ...
@overload
def w_to_y(value: np.integer[Any]) -> np.floating[Any]: ...
@overload
def w_to_y(value: np.floating[Any]) -> np.floating[Any]: ...
@overload
def w_to_y(
    value: Int[Array, ...],
) -> Float[Array, ...]: ...
@overload
def w_to_y(
    value: Float[Array, ...],
) -> Float[Array, ...]: ...
@overload
def w_to_y(value: pd.Series) -> pd.Series: ...
@overload
def w_to_y(value: pd.DataFrame) -> pd.DataFrame: ...


def w_to_y(
    value: (
        float
        | np.integer[Any]
        | np.floating[Any]
        | Int[Array, ...]
        | Float[Array, ...]
        | pd.Series
        | pd.DataFrame
    ),
) -> float | np.floating[Any] | Float[Array, ...] | pd.Series | pd.DataFrame:
    """
    Convert values (stocks) measured on the weekly level to values on
    the yearly level.
    """
    return value / _W_PER_Y


@overload
def per_w_to_per_y(value: int) -> float: ...
@overload
def per_w_to_per_y(value: float) -> float: ...
@overload
def per_w_to_per_y(value: np.integer[Any]) -> np.floating[Any]: ...
@overload
def per_w_to_per_y(
    value: np.floating[Any],
) -> np.floating[Any]: ...
@overload
def per_w_to_per_y(
    value: Int[Array, ...],
) -> Float[Array, ...]: ...
@overload
def per_w_to_per_y(
    value: Float[Array, ...],
) -> Float[Array, ...]: ...
@overload
def per_w_to_per_y(value: pd.Series) -> pd.Series: ...
@overload
def per_w_to_per_y(
    value: pd.DataFrame,
) -> pd.DataFrame: ...


def per_w_to_per_y(
    value: (
        float
        | np.integer[Any]
        | np.floating[Any]
        | Int[Array, ...]
        | Float[Array, ...]
        | pd.Series
        | pd.DataFrame
    ),
) -> float | np.floating[Any] | Float[Array, ...] | pd.Series | pd.DataFrame:
    """Convert flows: values per week to values per year."""
    return y_to_w(value)


@overload
def w_to_q(value: int) -> float: ...
@overload
def w_to_q(value: float) -> float: ...
@overload
def w_to_q(value: np.integer[Any]) -> np.floating[Any]: ...
@overload
def w_to_q(value: np.floating[Any]) -> np.floating[Any]: ...
@overload
def w_to_q(
    value: Int[Array, ...],
) -> Float[Array, ...]: ...
@overload
def w_to_q(
    value: Float[Array, ...],
) -> Float[Array, ...]: ...
@overload
def w_to_q(value: pd.Series) -> pd.Series: ...
@overload
def w_to_q(value: pd.DataFrame) -> pd.DataFrame: ...


def w_to_q(
    value: (
        float
        | np.integer[Any]
        | np.floating[Any]
        | Int[Array, ...]
        | Float[Array, ...]
        | pd.Series
        | pd.DataFrame
    ),
) -> float | np.floating[Any] | Float[Array, ...] | pd.Series | pd.DataFrame:
    """
    Convert values (stocks) measured on the weekly level to values on
    the quarterly level.
    """
    return value * _Q_PER_Y / _W_PER_Y


@overload
def per_w_to_per_q(value: int) -> float: ...
@overload
def per_w_to_per_q(value: float) -> float: ...
@overload
def per_w_to_per_q(value: np.integer[Any]) -> np.floating[Any]: ...
@overload
def per_w_to_per_q(
    value: np.floating[Any],
) -> np.floating[Any]: ...
@overload
def per_w_to_per_q(
    value: Int[Array, ...],
) -> Float[Array, ...]: ...
@overload
def per_w_to_per_q(
    value: Float[Array, ...],
) -> Float[Array, ...]: ...
@overload
def per_w_to_per_q(value: pd.Series) -> pd.Series: ...
@overload
def per_w_to_per_q(
    value: pd.DataFrame,
) -> pd.DataFrame: ...


def per_w_to_per_q(
    value: (
        float
        | np.integer[Any]
        | np.floating[Any]
        | Int[Array, ...]
        | Float[Array, ...]
        | pd.Series
        | pd.DataFrame
    ),
) -> float | np.floating[Any] | Float[Array, ...] | pd.Series | pd.DataFrame:
    """Convert flows: values per week to values per quarter."""
    return q_to_w(value)


@overload
def w_to_m(value: int) -> float: ...
@overload
def w_to_m(value: float) -> float: ...
@overload
def w_to_m(value: np.integer[Any]) -> np.floating[Any]: ...
@overload
def w_to_m(value: np.floating[Any]) -> np.floating[Any]: ...
@overload
def w_to_m(
    value: Int[Array, ...],
) -> Float[Array, ...]: ...
@overload
def w_to_m(
    value: Float[Array, ...],
) -> Float[Array, ...]: ...
@overload
def w_to_m(value: pd.Series) -> pd.Series: ...
@overload
def w_to_m(value: pd.DataFrame) -> pd.DataFrame: ...


def w_to_m(
    value: (
        float
        | np.integer[Any]
        | np.floating[Any]
        | Int[Array, ...]
        | Float[Array, ...]
        | pd.Series
        | pd.DataFrame
    ),
) -> float | np.floating[Any] | Float[Array, ...] | pd.Series | pd.DataFrame:
    """
    Convert values (stocks) measured on the weekly level to values on
    the monthly level.
    """
    return value * _M_PER_Y / _W_PER_Y


@overload
def per_w_to_per_m(value: int) -> float: ...
@overload
def per_w_to_per_m(value: float) -> float: ...
@overload
def per_w_to_per_m(value: np.integer[Any]) -> np.floating[Any]: ...
@overload
def per_w_to_per_m(
    value: np.floating[Any],
) -> np.floating[Any]: ...
@overload
def per_w_to_per_m(
    value: Int[Array, ...],
) -> Float[Array, ...]: ...
@overload
def per_w_to_per_m(
    value: Float[Array, ...],
) -> Float[Array, ...]: ...
@overload
def per_w_to_per_m(value: pd.Series) -> pd.Series: ...
@overload
def per_w_to_per_m(
    value: pd.DataFrame,
) -> pd.DataFrame: ...


def per_w_to_per_m(
    value: (
        float
        | np.integer[Any]
        | np.floating[Any]
        | Int[Array, ...]
        | Float[Array, ...]
        | pd.Series
        | pd.DataFrame
    ),
) -> float | np.floating[Any] | Float[Array, ...] | pd.Series | pd.DataFrame:
    """Convert flows: values per week to values per month."""
    return m_to_w(value)


@overload
def w_to_d(value: int) -> float: ...
@overload
def w_to_d(value: float) -> float: ...
@overload
def w_to_d(value: np.integer[Any]) -> np.floating[Any]: ...
@overload
def w_to_d(value: np.floating[Any]) -> np.floating[Any]: ...
@overload
def w_to_d(
    value: Int[Array, ...],
) -> Float[Array, ...]: ...
@overload
def w_to_d(
    value: Float[Array, ...],
) -> Float[Array, ...]: ...
@overload
def w_to_d(value: pd.Series) -> pd.Series: ...
@overload
def w_to_d(value: pd.DataFrame) -> pd.DataFrame: ...


def w_to_d(
    value: (
        float
        | np.integer[Any]
        | np.floating[Any]
        | Int[Array, ...]
        | Float[Array, ...]
        | pd.Series
        | pd.DataFrame
    ),
) -> float | np.floating[Any] | Float[Array, ...] | pd.Series | pd.DataFrame:
    """
    Convert values (stocks) measured on the weekly level to values on
    the daily level.
    """
    return value * _D_PER_Y / _W_PER_Y


@overload
def per_w_to_per_d(value: int) -> float: ...
@overload
def per_w_to_per_d(value: float) -> float: ...
@overload
def per_w_to_per_d(value: np.integer[Any]) -> np.floating[Any]: ...
@overload
def per_w_to_per_d(
    value: np.floating[Any],
) -> np.floating[Any]: ...
@overload
def per_w_to_per_d(
    value: Int[Array, ...],
) -> Float[Array, ...]: ...
@overload
def per_w_to_per_d(
    value: Float[Array, ...],
) -> Float[Array, ...]: ...
@overload
def per_w_to_per_d(value: pd.Series) -> pd.Series: ...
@overload
def per_w_to_per_d(
    value: pd.DataFrame,
) -> pd.DataFrame: ...


def per_w_to_per_d(
    value: (
        float
        | np.integer[Any]
        | np.floating[Any]
        | Int[Array, ...]
        | Float[Array, ...]
        | pd.Series
        | pd.DataFrame
    ),
) -> float | np.floating[Any] | Float[Array, ...] | pd.Series | pd.DataFrame:
    """Convert flows: values per week to values per day."""
    return d_to_w(value)


# --- Day conversions (stocks) ---


@overload
def d_to_y(value: int) -> float: ...
@overload
def d_to_y(value: float) -> float: ...
@overload
def d_to_y(value: np.integer[Any]) -> np.floating[Any]: ...
@overload
def d_to_y(value: np.floating[Any]) -> np.floating[Any]: ...
@overload
def d_to_y(
    value: Int[Array, ...],
) -> Float[Array, ...]: ...
@overload
def d_to_y(
    value: Float[Array, ...],
) -> Float[Array, ...]: ...
@overload
def d_to_y(value: pd.Series) -> pd.Series: ...
@overload
def d_to_y(value: pd.DataFrame) -> pd.DataFrame: ...


def d_to_y(
    value: (
        float
        | np.integer[Any]
        | np.floating[Any]
        | Int[Array, ...]
        | Float[Array, ...]
        | pd.Series
        | pd.DataFrame
    ),
) -> float | np.floating[Any] | Float[Array, ...] | pd.Series | pd.DataFrame:
    """
    Convert values (stocks) measured on the daily level to values on
    the yearly level.
    """
    return value / _D_PER_Y


@overload
def per_d_to_per_y(value: int) -> float: ...
@overload
def per_d_to_per_y(value: float) -> float: ...
@overload
def per_d_to_per_y(value: np.integer[Any]) -> np.floating[Any]: ...
@overload
def per_d_to_per_y(
    value: np.floating[Any],
) -> np.floating[Any]: ...
@overload
def per_d_to_per_y(
    value: Int[Array, ...],
) -> Float[Array, ...]: ...
@overload
def per_d_to_per_y(
    value: Float[Array, ...],
) -> Float[Array, ...]: ...
@overload
def per_d_to_per_y(value: pd.Series) -> pd.Series: ...
@overload
def per_d_to_per_y(
    value: pd.DataFrame,
) -> pd.DataFrame: ...


def per_d_to_per_y(
    value: (
        float
        | np.integer[Any]
        | np.floating[Any]
        | Int[Array, ...]
        | Float[Array, ...]
        | pd.Series
        | pd.DataFrame
    ),
) -> float | np.floating[Any] | Float[Array, ...] | pd.Series | pd.DataFrame:
    """Convert flows: values per day to values per year."""
    return y_to_d(value)


@overload
def d_to_m(value: int) -> float: ...
@overload
def d_to_m(value: float) -> float: ...
@overload
def d_to_m(value: np.integer[Any]) -> np.floating[Any]: ...
@overload
def d_to_m(value: np.floating[Any]) -> np.floating[Any]: ...
@overload
def d_to_m(
    value: Int[Array, ...],
) -> Float[Array, ...]: ...
@overload
def d_to_m(
    value: Float[Array, ...],
) -> Float[Array, ...]: ...
@overload
def d_to_m(value: pd.Series) -> pd.Series: ...
@overload
def d_to_m(value: pd.DataFrame) -> pd.DataFrame: ...


def d_to_m(
    value: (
        float
        | np.integer[Any]
        | np.floating[Any]
        | Int[Array, ...]
        | Float[Array, ...]
        | pd.Series
        | pd.DataFrame
    ),
) -> float | np.floating[Any] | Float[Array, ...] | pd.Series | pd.DataFrame:
    """
    Convert values (stocks) measured on the daily level to values on
    the monthly level.
    """
    return value * _M_PER_Y / _D_PER_Y


@overload
def per_d_to_per_m(value: int) -> float: ...
@overload
def per_d_to_per_m(value: float) -> float: ...
@overload
def per_d_to_per_m(value: np.integer[Any]) -> np.floating[Any]: ...
@overload
def per_d_to_per_m(
    value: np.floating[Any],
) -> np.floating[Any]: ...
@overload
def per_d_to_per_m(
    value: Int[Array, ...],
) -> Float[Array, ...]: ...
@overload
def per_d_to_per_m(
    value: Float[Array, ...],
) -> Float[Array, ...]: ...
@overload
def per_d_to_per_m(value: pd.Series) -> pd.Series: ...
@overload
def per_d_to_per_m(
    value: pd.DataFrame,
) -> pd.DataFrame: ...


def per_d_to_per_m(
    value: (
        float
        | np.integer[Any]
        | np.floating[Any]
        | Int[Array, ...]
        | Float[Array, ...]
        | pd.Series
        | pd.DataFrame
    ),
) -> float | np.floating[Any] | Float[Array, ...] | pd.Series | pd.DataFrame:
    """Convert flows: values per day to values per month."""
    return m_to_d(value)


@overload
def d_to_q(value: int) -> float: ...
@overload
def d_to_q(value: float) -> float: ...
@overload
def d_to_q(value: np.integer[Any]) -> np.floating[Any]: ...
@overload
def d_to_q(value: np.floating[Any]) -> np.floating[Any]: ...
@overload
def d_to_q(
    value: Int[Array, ...],
) -> Float[Array, ...]: ...
@overload
def d_to_q(
    value: Float[Array, ...],
) -> Float[Array, ...]: ...
@overload
def d_to_q(value: pd.Series) -> pd.Series: ...
@overload
def d_to_q(value: pd.DataFrame) -> pd.DataFrame: ...


def d_to_q(
    value: (
        float
        | np.integer[Any]
        | np.floating[Any]
        | Int[Array, ...]
        | Float[Array, ...]
        | pd.Series
        | pd.DataFrame
    ),
) -> float | np.floating[Any] | Float[Array, ...] | pd.Series | pd.DataFrame:
    """
    Convert values (stocks) measured on the daily level to values on
    the quarterly level.
    """
    return value * _Q_PER_Y / _D_PER_Y


@overload
def per_d_to_per_q(value: int) -> float: ...
@overload
def per_d_to_per_q(value: float) -> float: ...
@overload
def per_d_to_per_q(value: np.integer[Any]) -> np.floating[Any]: ...
@overload
def per_d_to_per_q(
    value: np.floating[Any],
) -> np.floating[Any]: ...
@overload
def per_d_to_per_q(
    value: Int[Array, ...],
) -> Float[Array, ...]: ...
@overload
def per_d_to_per_q(
    value: Float[Array, ...],
) -> Float[Array, ...]: ...
@overload
def per_d_to_per_q(value: pd.Series) -> pd.Series: ...
@overload
def per_d_to_per_q(
    value: pd.DataFrame,
) -> pd.DataFrame: ...


def per_d_to_per_q(
    value: (
        float
        | np.integer[Any]
        | np.floating[Any]
        | Int[Array, ...]
        | Float[Array, ...]
        | pd.Series
        | pd.DataFrame
    ),
) -> float | np.floating[Any] | Float[Array, ...] | pd.Series | pd.DataFrame:
    """Convert flows: values per day to values per quarter."""
    return q_to_d(value)


@overload
def d_to_w(value: int) -> float: ...
@overload
def d_to_w(value: float) -> float: ...
@overload
def d_to_w(value: np.integer[Any]) -> np.floating[Any]: ...
@overload
def d_to_w(value: np.floating[Any]) -> np.floating[Any]: ...
@overload
def d_to_w(
    value: Int[Array, ...],
) -> Float[Array, ...]: ...
@overload
def d_to_w(
    value: Float[Array, ...],
) -> Float[Array, ...]: ...
@overload
def d_to_w(value: pd.Series) -> pd.Series: ...
@overload
def d_to_w(value: pd.DataFrame) -> pd.DataFrame: ...


def d_to_w(
    value: (
        float
        | np.integer[Any]
        | np.floating[Any]
        | Int[Array, ...]
        | Float[Array, ...]
        | pd.Series
        | pd.DataFrame
    ),
) -> float | np.floating[Any] | Float[Array, ...] | pd.Series | pd.DataFrame:
    """
    Convert values (stocks) measured on the daily level to values on
    the weekly level.
    """
    return value * _W_PER_Y / _D_PER_Y


@overload
def per_d_to_per_w(value: int) -> float: ...
@overload
def per_d_to_per_w(value: float) -> float: ...
@overload
def per_d_to_per_w(value: np.integer[Any]) -> np.floating[Any]: ...
@overload
def per_d_to_per_w(
    value: np.floating[Any],
) -> np.floating[Any]: ...
@overload
def per_d_to_per_w(
    value: Int[Array, ...],
) -> Float[Array, ...]: ...
@overload
def per_d_to_per_w(
    value: Float[Array, ...],
) -> Float[Array, ...]: ...
@overload
def per_d_to_per_w(value: pd.Series) -> pd.Series: ...
@overload
def per_d_to_per_w(
    value: pd.DataFrame,
) -> pd.DataFrame: ...


def per_d_to_per_w(
    value: (
        float
        | np.integer[Any]
        | np.floating[Any]
        | Int[Array, ...]
        | Float[Array, ...]
        | pd.Series
        | pd.DataFrame
    ),
) -> float | np.floating[Any] | Float[Array, ...] | pd.Series | pd.DataFrame:
    """Convert flows: values per day to values per week."""
    return w_to_d(value)
