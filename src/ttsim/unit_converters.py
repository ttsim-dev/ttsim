from __future__ import annotations

from typing import overload

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


@overload
def y_to_q(value: int) -> int: ...


@overload
def y_to_q(value: float) -> float: ...


def y_to_q(value: float) -> int | float:
    """
    Convert values (stocks) measured on the yearly level to values on the quarterly
    level.
    """
    return value * _Q_PER_Y


@overload
def per_y_to_per_q(value: int) -> float: ...


@overload
def per_y_to_per_q(value: float) -> float: ...


def per_y_to_per_q(value: float) -> float:
    """Convert flows: values per year to values per quarter."""
    return q_to_y(value)


@overload
def y_to_m(value: int) -> int: ...


@overload
def y_to_m(value: float) -> float: ...


def y_to_m(value: float) -> int | float:
    """
    Convert values (stocks) measured on the yearly level to values on the monthly
    level.
    """
    return value * _M_PER_Y


@overload
def per_y_to_per_m(value: int) -> float: ...


@overload
def per_y_to_per_m(value: float) -> float: ...


def per_y_to_per_m(value: float) -> float:
    """
    Convert flows: values per year to values per month."""
    return m_to_y(value)


@overload
def y_to_w(value: int) -> float: ...


@overload
def y_to_w(value: float) -> float: ...


def y_to_w(value: float) -> float:
    """
    Convert values (stocks) measured on the yearly level to values on the weekly
    level."""
    return value * _W_PER_Y


@overload
def per_y_to_per_w(value: int) -> float: ...


@overload
def per_y_to_per_w(value: float) -> float: ...


def per_y_to_per_w(value: float) -> float:
    """Convert flows: values per year to values per week."""
    return w_to_y(value)


@overload
def y_to_d(value: int) -> float: ...


@overload
def y_to_d(value: float) -> float: ...


def y_to_d(value: float) -> float:
    """
    Convert values (stocks) measured on the yearly level to values on the daily
    level.
    """
    return value * _D_PER_Y


@overload
def per_y_to_per_d(value: int) -> float: ...


@overload
def per_y_to_per_d(value: float) -> float: ...


def per_y_to_per_d(value: float) -> float:
    """Convert flows: values per year to values per day."""
    return d_to_y(value)


@overload
def q_to_y(value: int) -> float: ...


@overload
def q_to_y(value: float) -> float: ...


def q_to_y(value: float) -> float:
    """
    Convert values (stocks) measured on the quarterly level to values on the yearly
    level.
    """
    return value / _Q_PER_Y


@overload
def per_q_to_per_y(value: int) -> int: ...


@overload
def per_q_to_per_y(value: float) -> float: ...


def per_q_to_per_y(value: float) -> int | float:
    """Convert flows: values per quarter to values per year."""
    return y_to_q(value)


@overload
def q_to_m(value: int) -> float: ...


@overload
def q_to_m(value: float) -> float: ...


def q_to_m(value: float) -> float:
    """
    Convert values (stocks) measured on the quarterly level to values on the monthly
    level.
    """
    return value * _M_PER_Y / _Q_PER_Y


@overload
def per_q_to_per_m(value: int) -> float: ...


@overload
def per_q_to_per_m(value: float) -> float: ...


def per_q_to_per_m(value: float) -> float:
    """Convert flows: values per quarter to values per month."""
    return m_to_q(value)


@overload
def q_to_w(value: int) -> float: ...


@overload
def q_to_w(value: float) -> float: ...


def q_to_w(value: float) -> float:
    """
    Convert values (stocks) measured on the quarterly level to values on the weekly
    level.
    """
    return value * _W_PER_Y / _Q_PER_Y


@overload
def per_q_to_per_w(value: int) -> float: ...


@overload
def per_q_to_per_w(value: float) -> float: ...


def per_q_to_per_w(value: float) -> float:
    """Convert flows: values per quarter to values per week."""
    return w_to_q(value)


@overload
def q_to_d(value: int) -> float: ...


@overload
def q_to_d(value: float) -> float: ...


def q_to_d(value: float) -> float:
    """
    Convert values (stocks) measured on the quarterly level to values on the daily
    level.
    """
    return value * _D_PER_Y / _Q_PER_Y


@overload
def per_q_to_per_d(value: int) -> float: ...


@overload
def per_q_to_per_d(value: float) -> float: ...


def per_q_to_per_d(value: float) -> float:
    """Convert flows: values per quarter to values per day."""
    return d_to_q(value)


@overload
def m_to_y(value: int) -> float: ...


@overload
def m_to_y(value: float) -> float: ...


def m_to_y(value: float) -> float:
    """
    Convert values (stocks) measured on the monthly level to values on the yearly level.
    level.
    """
    return value / _M_PER_Y


@overload
def per_m_to_per_y(value: int) -> int: ...


@overload
def per_m_to_per_y(value: float) -> float: ...


def per_m_to_per_y(value: float) -> int | float:
    """Convert flows: values per month to values per year."""
    return y_to_m(value)


@overload
def m_to_q(value: int) -> float: ...


@overload
def m_to_q(value: float) -> float: ...


def m_to_q(value: float) -> float:
    """
    Convert values (stocks) measured on the monthly level to values on the quarterly
    level.
    """
    return value * _Q_PER_Y / _M_PER_Y


@overload
def per_m_to_per_q(value: int) -> float: ...


@overload
def per_m_to_per_q(value: float) -> float: ...


def per_m_to_per_q(value: float) -> float:
    """Convert flows: values per month to values per quarter."""
    return q_to_m(value)


@overload
def m_to_w(value: int) -> float: ...


@overload
def m_to_w(value: float) -> float: ...


def m_to_w(value: float) -> float:
    """
    Convert values (stocks) measured on the monthly level to values on the weekly level.
    """
    return value * _W_PER_Y / _M_PER_Y


@overload
def per_m_to_per_w(value: int) -> float: ...


@overload
def per_m_to_per_w(value: float) -> float: ...


def per_m_to_per_w(value: float) -> float:
    """Convert flows: values per month to values per week."""
    return w_to_m(value)


@overload
def m_to_d(value: int) -> float: ...


@overload
def m_to_d(value: float) -> float: ...


def m_to_d(value: float) -> float:
    """
    Convert values (stocks) measured on the monthly level to values on the daily level.
    """
    return value * _D_PER_Y / _M_PER_Y


@overload
def per_m_to_per_d(value: int) -> float: ...


@overload
def per_m_to_per_d(value: float) -> float: ...


def per_m_to_per_d(value: float) -> float:
    """Convert flows: values per month to values per day."""
    return d_to_m(value)


@overload
def w_to_y(value: int) -> float: ...


@overload
def w_to_y(value: float) -> float: ...


def w_to_y(value: float) -> float:
    """
    Convert values (stocks) measured on the weekly level to values on the yearly level.
    """
    return value / _W_PER_Y


@overload
def per_w_to_per_y(value: int) -> float: ...


@overload
def per_w_to_per_y(value: float) -> float: ...


def per_w_to_per_y(value: float) -> float:
    """Convert flows: values per week to values per year."""
    return y_to_w(value)


@overload
def w_to_q(value: int) -> float: ...


@overload
def w_to_q(value: float) -> float: ...


def w_to_q(value: float) -> float:
    """
    Convert values (stocks) measured on the weekly level to values on the quarterly
    level.
    """
    return value * _Q_PER_Y / _W_PER_Y


@overload
def per_w_to_per_q(value: int) -> float: ...


@overload
def per_w_to_per_q(value: float) -> float: ...


def per_w_to_per_q(value: float) -> float:
    """Convert flows: values per week to values per quarter."""
    return q_to_w(value)


@overload
def w_to_m(value: int) -> float: ...


@overload
def w_to_m(value: float) -> float: ...


def w_to_m(value: float) -> float:
    """
    Convert values (stocks) measured on the weekly level to values on the monthly level.
    """
    return value * _M_PER_Y / _W_PER_Y


@overload
def per_w_to_per_m(value: int) -> float: ...


@overload
def per_w_to_per_m(value: float) -> float: ...


def per_w_to_per_m(value: float) -> float:
    """Convert flows: values per week to values per month."""
    return m_to_w(value)


@overload
def w_to_d(value: int) -> float: ...


@overload
def w_to_d(value: float) -> float: ...


def w_to_d(value: float) -> float:
    """
    Convert values (stocks) measured on the weekly level to values on the daily level.
    """
    return value * _D_PER_Y / _W_PER_Y


@overload
def per_w_to_per_d(value: int) -> float: ...


@overload
def per_w_to_per_d(value: float) -> float: ...


def per_w_to_per_d(value: float) -> float:
    """Convert flows: values per week to values per day."""
    return d_to_w(value)


@overload
def d_to_y(value: int) -> float: ...


@overload
def d_to_y(value: float) -> float: ...


def d_to_y(value: float) -> float:
    """
    Convert values (stocks) measured on the daily level to values on the yearly level.
    """
    return value / _D_PER_Y


@overload
def per_d_to_per_y(value: int) -> float: ...


@overload
def per_d_to_per_y(value: float) -> float: ...


def per_d_to_per_y(value: float) -> float:
    """Convert flows: values per day to values per year."""
    return y_to_d(value)


@overload
def d_to_m(value: int) -> float: ...


@overload
def d_to_m(value: float) -> float: ...


def d_to_m(value: float) -> float:
    """
    Convert values (stocks) measured on the daily level to values on the monthly level.
    """
    return value * _M_PER_Y / _D_PER_Y


@overload
def per_d_to_per_m(value: int) -> float: ...


@overload
def per_d_to_per_m(value: float) -> float: ...


def per_d_to_per_m(value: float) -> float:
    """Convert flows: values per day to values per month."""
    return m_to_d(value)


@overload
def d_to_q(value: int) -> float: ...


@overload
def d_to_q(value: float) -> float: ...


def d_to_q(value: float) -> float:
    """
    Convert values (stocks) measured on the daily level to values on the quarterly
    level.
    """
    return value * _Q_PER_Y / _D_PER_Y


@overload
def per_d_to_per_q(value: int) -> float: ...


@overload
def per_d_to_per_q(value: float) -> float: ...


def per_d_to_per_q(value: float) -> float:
    """Convert flows: values per day to values per quarter."""
    return q_to_d(value)


@overload
def d_to_w(value: int) -> float: ...


@overload
def d_to_w(value: float) -> float: ...


def d_to_w(value: float) -> float:
    """
    Convert values (stocks) measured on the daily level to values on the weekly level.
    """
    return value * _W_PER_Y / _D_PER_Y


@overload
def per_d_to_per_w(value: int) -> float: ...


@overload
def per_d_to_per_w(value: float) -> float: ...


def per_d_to_per_w(value: float) -> float:
    """Convert flows: values per day to values per week."""
    return w_to_d(value)
