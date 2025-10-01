from __future__ import annotations

from typing import overload

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


@overload
def y_to_q(value: int) -> int: ...


@overload
def y_to_q(value: float) -> float: ...


def y_to_q(value: float) -> int | float:
    """
    Converts values on year level to values on quarter level.

    Parameters
    ----------
    value
        Value on year level to be converted to value on quarter level.

    Returns
    -------
    Value on quarter level.
    """
    return value * _Q_PER_Y


@overload
def per_y_to_per_q(value: int) -> float: ...


@overload
def per_y_to_per_q(value: float) -> float: ...


def per_y_to_per_q(value: float) -> float:
    """
    Converts values per year to values per quarter.

    Parameters
    ----------
    value
        Value per year to be converted to value per quarter.

    Returns
    -------
    Value per quarter.
    """
    return q_to_y(value)


@overload
def y_to_m(value: int) -> int: ...


@overload
def y_to_m(value: float) -> float: ...


def y_to_m(value: float) -> int | float:
    """
    Converts values on year level to values on month level.

    Parameters
    ----------
    value
        Value on year level to be converted to value on month level.

    Returns
    -------
    Value on month level.
    """
    return value * _M_PER_Y


@overload
def per_y_to_per_m(value: int) -> float: ...


@overload
def per_y_to_per_m(value: float) -> float: ...


def per_y_to_per_m(value: float) -> float:
    """
    Converts values per year to values per month.

    Parameters
    ----------
    value
        Value per year to be converted to value per month.

    Returns
    -------
    Value per month.
    """
    return m_to_y(value)


@overload
def y_to_w(value: int) -> float: ...


@overload
def y_to_w(value: float) -> float: ...


def y_to_w(value: float) -> float:
    """
    Converts values on year level to values on week level.

    Parameters
    ----------
    value
        Value on year level to be converted to value on week level.

    Returns
    -------
    Value on week level.
    """
    return value * _W_PER_Y


@overload
def per_y_to_per_w(value: int) -> float: ...


@overload
def per_y_to_per_w(value: float) -> float: ...


def per_y_to_per_w(value: float) -> float:
    """
    Converts values per year to values per week.

    Parameters
    ----------
    value
        Value per year to be converted to value per week.

    Returns
    -------
    Value per week.
    """
    return w_to_y(value)


@overload
def y_to_d(value: int) -> float: ...


@overload
def y_to_d(value: float) -> float: ...


def y_to_d(value: float) -> float:
    """
    Converts values on year level to values on day level.

    Parameters
    ----------
    value
        Value on year level to be converted to value on day level.

    Returns
    -------
    Value on day level.
    """
    return value * _D_PER_Y


@overload
def per_y_to_per_d(value: int) -> float: ...


@overload
def per_y_to_per_d(value: float) -> float: ...


def per_y_to_per_d(value: float) -> float:
    """
    Converts values per year to values per day.

    Parameters
    ----------
    value
        Value per year to be converted to value per day.

    Returns
    -------
    Value per day.
    """
    return d_to_y(value)


@overload
def q_to_y(value: int) -> float: ...


@overload
def q_to_y(value: float) -> float: ...


def q_to_y(value: float) -> float:
    """
    Converts values on quarter level to values on year level.

    Parameters
    ----------
    value
        Value on quarter level to be converted to value on year level.

    Returns
    -------
    Value on year level.
    """
    return value / _Q_PER_Y


@overload
def per_q_to_per_y(value: int) -> int: ...


@overload
def per_q_to_per_y(value: float) -> float: ...


def per_q_to_per_y(value: float) -> int | float:
    """
    Converts values per quarter to values per year.

    Parameters
    ----------
    value
        Value per quarter to be converted to value per year.

    Returns
    -------
    Value per year.
    """
    return y_to_q(value)


@overload
def q_to_m(value: int) -> float: ...


@overload
def q_to_m(value: float) -> float: ...


def q_to_m(value: float) -> float:
    """
    Converts values on quarter level to values on month level.

    Parameters
    ----------
    value
        Value on quarter level to be converted to value on month level.

    Returns
    -------
    Value on month level.
    """
    return value * _M_PER_Y / _Q_PER_Y


@overload
def per_q_to_per_m(value: int) -> float: ...


@overload
def per_q_to_per_m(value: float) -> float: ...


def per_q_to_per_m(value: float) -> float:
    """
    Converts values per quarter to values per month.

    Parameters
    ----------
    value
        Value per quarter to be converted to value per month.

    Returns
    -------
    Value per month.
    """
    return m_to_q(value)


@overload
def q_to_w(value: int) -> float: ...


@overload
def q_to_w(value: float) -> float: ...


def q_to_w(value: float) -> float:
    """
    Converts values on quarter level to values on week level.

    Parameters
    ----------
    value
        Value on quarter level to be converted to value on week level.

    Returns
    -------
    Value on week level.
    """
    return value * _W_PER_Y / _Q_PER_Y


@overload
def per_q_to_per_w(value: int) -> float: ...


@overload
def per_q_to_per_w(value: float) -> float: ...


def per_q_to_per_w(value: float) -> float:
    """
    Converts values per quarter to values per week.

    Parameters
    ----------
    value
        Value per quarter to be converted to value per week.

    Returns
    -------
    Value per week.
    """
    return w_to_q(value)


@overload
def q_to_d(value: int) -> float: ...


@overload
def q_to_d(value: float) -> float: ...


def q_to_d(value: float) -> float:
    """
    Converts values on quarter level to values on day level.

    Parameters
    ----------
    value
        Value on quarter level to be converted to value on day level.

    Returns
    -------
    Value on day level.
    """
    return value * _D_PER_Y / _Q_PER_Y


@overload
def per_q_to_per_d(value: int) -> float: ...


@overload
def per_q_to_per_d(value: float) -> float: ...


def per_q_to_per_d(value: float) -> float:
    """
    Converts values per quarter to values per day.

    Parameters
    ----------
    value
        Value per quarter to be converted to value per day.

    Returns
    -------
    Value per day.
    """
    return d_to_q(value)


@overload
def m_to_y(value: int) -> float: ...


@overload
def m_to_y(value: float) -> float: ...


def m_to_y(value: float) -> float:
    """
    Converts values on month level to values on year level.

    Parameters
    ----------
    value
        Value on month level to be converted to value on year level.

    Returns
    -------
    Value on year level.
    """
    return value / _M_PER_Y


@overload
def per_m_to_per_y(value: int) -> int: ...


@overload
def per_m_to_per_y(value: float) -> float: ...


def per_m_to_per_y(value: float) -> int | float:
    """
    Converts values per month to values per year.

    Parameters
    ----------
    value
        Value per month to be converted to value per year.

    Returns
    -------
    Value per year.
    """
    return y_to_m(value)


@overload
def m_to_q(value: int) -> float: ...


@overload
def m_to_q(value: float) -> float: ...


def m_to_q(value: float) -> float:
    """
    Converts values on month level to values on quarter level.

    Parameters
    ----------
    value
        Value on month level to be converted to value on quarter level.

    Returns
    -------
    Value on quarter level.
    """
    return value * _Q_PER_Y / _M_PER_Y


@overload
def per_m_to_per_q(value: int) -> float: ...


@overload
def per_m_to_per_q(value: float) -> float: ...


def per_m_to_per_q(value: float) -> float:
    """
    Converts values per month to values per quarter.

    Parameters
    ----------
    value
        Value per month to be converted to value per quarter.

    Returns
    -------
    Value per quarter.
    """
    return q_to_m(value)


@overload
def m_to_w(value: int) -> float: ...


@overload
def m_to_w(value: float) -> float: ...


def m_to_w(value: float) -> float:
    """
    Converts values on month level to values on week level.

    Parameters
    ----------
    value
        Value on month level to be converted to value on week level.

    Returns
    -------
    Value on week level.
    """
    return value * _W_PER_Y / _M_PER_Y


@overload
def per_m_to_per_w(value: int) -> float: ...


@overload
def per_m_to_per_w(value: float) -> float: ...


def per_m_to_per_w(value: float) -> float:
    """
    Converts values per month to values per week.

    Parameters
    ----------
    value
        Value per month to be converted to value per week.

    Returns
    -------
    Value per week.
    """
    return w_to_m(value)


@overload
def m_to_d(value: int) -> float: ...


@overload
def m_to_d(value: float) -> float: ...


def m_to_d(value: float) -> float:
    """
    Converts values on month level to values on day level.

    Parameters
    ----------
    value
        Value on month level to be converted to value on day level.

    Returns
    -------
    Value on day level.
    """
    return value * _D_PER_Y / _M_PER_Y


@overload
def per_m_to_per_d(value: int) -> float: ...


@overload
def per_m_to_per_d(value: float) -> float: ...


def per_m_to_per_d(value: float) -> float:
    """
    Converts values per month to values per day.

    Parameters
    ----------
    value
        Value per month to be converted to value per day.

    Returns
    -------
    Value per day.
    """
    return d_to_m(value)


@overload
def w_to_y(value: int) -> float: ...


@overload
def w_to_y(value: float) -> float: ...


def w_to_y(value: float) -> float:
    """
    Converts values on week level to values on year level.

    Parameters
    ----------
    value
        Value on week level to be converted to value on year level.

    Returns
    -------
    Value on year level.
    """
    return value / _W_PER_Y


@overload
def per_w_to_per_y(value: int) -> float: ...


@overload
def per_w_to_per_y(value: float) -> float: ...


def per_w_to_per_y(value: float) -> float:
    """
    Converts values per week to values per year.

    Parameters
    ----------
    value
        Value per week to be converted to value per year.

    Returns
    -------
    Value per year.
    """
    return y_to_w(value)


@overload
def w_to_q(value: int) -> float: ...


@overload
def w_to_q(value: float) -> float: ...


def w_to_q(value: float) -> float:
    """
    Converts values on week level to values on quarter level.

    Parameters
    ----------
    value
        Value on week level to be converted to value on quarter level.

    Returns
    -------
    Value on quarter level.
    """
    return value * _Q_PER_Y / _W_PER_Y


@overload
def per_w_to_per_q(value: int) -> float: ...


@overload
def per_w_to_per_q(value: float) -> float: ...


def per_w_to_per_q(value: float) -> float:
    """
    Converts values per week to values per quarter.

    Parameters
    ----------
    value
        Value per week to be converted to value per quarter.

    Returns
    -------
    Value per quarter.
    """
    return q_to_w(value)


@overload
def w_to_m(value: int) -> float: ...


@overload
def w_to_m(value: float) -> float: ...


def w_to_m(value: float) -> float:
    """
    Converts values on week level to values on month level.

    Parameters
    ----------
    value
        Value on week level to be converted to value on month level.

    Returns
    -------
    Value on month level.
    """
    return value * _M_PER_Y / _W_PER_Y


@overload
def per_w_to_per_m(value: int) -> float: ...


@overload
def per_w_to_per_m(value: float) -> float: ...


def per_w_to_per_m(value: float) -> float:
    """
    Converts values per week to values per month.

    Parameters
    ----------
    value
        Value per week to be converted to value per month.

    Returns
    -------
    Value per month.
    """
    return m_to_w(value)


@overload
def w_to_d(value: int) -> float: ...


@overload
def w_to_d(value: float) -> float: ...


def w_to_d(value: float) -> float:
    """
    Converts values on week level to values on day level.

    Parameters
    ----------
    value
        Value on week level to be converted to value on day level.

    Returns
    -------
    Value on day level.
    """
    return value * _D_PER_Y / _W_PER_Y


@overload
def per_w_to_per_d(value: int) -> float: ...


@overload
def per_w_to_per_d(value: float) -> float: ...


def per_w_to_per_d(value: float) -> float:
    """
    Converts values per week to values per day.

    Parameters
    ----------
    value
        Value per week to be converted to value per day.

    Returns
    -------
    Value per day.
    """
    return d_to_w(value)


@overload
def d_to_y(value: int) -> float: ...


@overload
def d_to_y(value: float) -> float: ...


def d_to_y(value: float) -> float:
    """
    Converts values on day level to values on year level.

    Parameters
    ----------
    value
        Value on day level to be converted to value on year level.

    Returns
    -------
    Value on year level.
    """
    return value / _D_PER_Y


@overload
def per_d_to_per_y(value: int) -> float: ...


@overload
def per_d_to_per_y(value: float) -> float: ...


def per_d_to_per_y(value: float) -> float:
    """
    Converts values per day to values per year.

    Parameters
    ----------
    value
        Value per day to be converted to value per year.

    Returns
    -------
    Value per year.
    """
    return y_to_d(value)


@overload
def d_to_m(value: int) -> float: ...


@overload
def d_to_m(value: float) -> float: ...


def d_to_m(value: float) -> float:
    """
    Converts values on day level to values on month level.

    Parameters
    ----------
    value
        Value on day level to be converted to value on month level.

    Returns
    -------
    Value on month level.
    """
    return value * _M_PER_Y / _D_PER_Y


@overload
def per_d_to_per_m(value: int) -> float: ...


@overload
def per_d_to_per_m(value: float) -> float: ...


def per_d_to_per_m(value: float) -> float:
    """
    Converts values per day to values per month.

    Parameters
    ----------
    value
        Value per day to be converted to value per month.

    Returns
    -------
    Value per month.
    """
    return m_to_d(value)


@overload
def d_to_q(value: int) -> float: ...


@overload
def d_to_q(value: float) -> float: ...


def d_to_q(value: float) -> float:
    """
    Converts values on day level to values on quarter level.

    Parameters
    ----------
    value
        Value on day level to be converted to value on quarter level.

    Returns
    -------
    Value on quarter level.
    """
    return value * _Q_PER_Y / _D_PER_Y


@overload
def per_d_to_per_q(value: int) -> float: ...


@overload
def per_d_to_per_q(value: float) -> float: ...


def per_d_to_per_q(value: float) -> float:
    """
    Converts values per day to values per quarter.

    Parameters
    ----------
    value
        Value per day to be converted to value per quarter.

    Returns
    -------
    Value per quarter.
    """
    return q_to_d(value)


@overload
def d_to_w(value: int) -> float: ...


@overload
def d_to_w(value: float) -> float: ...


def d_to_w(value: float) -> float:
    """
    Converts values on day level to values on week level.

    Parameters
    ----------
    value
        Value on day level to be converted to value on week level.

    Returns
    -------
    Value on week level.
    """
    return value * _W_PER_Y / _D_PER_Y


@overload
def per_d_to_per_w(value: int) -> float: ...


@overload
def per_d_to_per_w(value: float) -> float: ...


def per_d_to_per_w(value: float) -> float:
    """
    Converts values per day to values per week.

    Parameters
    ----------
    value
        Value per day to be converted to value per week.

    Returns
    -------
    Value per week.
    """
    return w_to_d(value)
