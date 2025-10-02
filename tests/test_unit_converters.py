from __future__ import annotations

import pytest

from ttsim.unit_converters import (
    d_to_m,
    d_to_q,
    d_to_w,
    d_to_y,
    m_to_d,
    m_to_q,
    m_to_w,
    m_to_y,
    per_d_to_per_m,
    per_d_to_per_q,
    per_d_to_per_w,
    per_d_to_per_y,
    per_m_to_per_d,
    per_m_to_per_q,
    per_m_to_per_w,
    per_m_to_per_y,
    per_q_to_per_d,
    per_q_to_per_m,
    per_q_to_per_w,
    per_q_to_per_y,
    per_w_to_per_d,
    per_w_to_per_m,
    per_w_to_per_q,
    per_w_to_per_y,
    per_y_to_per_d,
    per_y_to_per_m,
    per_y_to_per_q,
    per_y_to_per_w,
    q_to_d,
    q_to_m,
    q_to_w,
    q_to_y,
    w_to_d,
    w_to_m,
    w_to_q,
    w_to_y,
    y_to_d,
    y_to_m,
    y_to_q,
    y_to_w,
)


@pytest.mark.parametrize(
    ("yearly_value", "quarterly_value"),
    [
        (0, 0),
        (12, 3),
    ],
)
def test_per_y_to_per_q(yearly_value: float, quarterly_value: float) -> None:
    assert per_y_to_per_q(yearly_value) == quarterly_value


@pytest.mark.parametrize(
    ("yearly_value", "monthly_value"),
    [
        (0, 0),
        (12, 1),
    ],
)
def test_per_y_to_per_m(yearly_value: float, monthly_value: float) -> None:
    assert per_y_to_per_m(yearly_value) == monthly_value


@pytest.mark.parametrize(
    ("yearly_value", "weekly_value"),
    [
        (0, 0),
        (365.25, 7),
    ],
)
def test_per_y_to_per_w(yearly_value: float, weekly_value: float) -> None:
    assert per_y_to_per_w(yearly_value) == weekly_value


@pytest.mark.parametrize(
    ("yearly_value", "daily_value"),
    [
        (0, 0),
        (365.25, 1),
    ],
)
def test_per_y_to_per_d(yearly_value: float, daily_value: float) -> None:
    assert per_y_to_per_d(yearly_value) == daily_value


@pytest.mark.parametrize(
    ("quarterly_value", "yearly_value"),
    [
        (0, 0),
        (1, 4),
    ],
)
def test_per_q_to_per_y(quarterly_value: float, yearly_value: float) -> None:
    assert per_q_to_per_y(quarterly_value) == yearly_value


@pytest.mark.parametrize(
    ("quarterly_value", "monthly_value"),
    [
        (0, 0),
        (3, 1),
    ],
)
def test_per_q_to_per_m(quarterly_value: float, monthly_value: float) -> None:
    assert per_q_to_per_m(quarterly_value) == monthly_value


@pytest.mark.parametrize(
    ("quarterly_value", "weekly_value"),
    [
        (0, 0),
        (365.25 / 7 / 4, 1),
    ],
)
def test_per_q_to_per_w(quarterly_value: float, weekly_value: float) -> None:
    assert per_q_to_per_w(quarterly_value) == weekly_value


@pytest.mark.parametrize(
    ("quarterly_value", "daily_value"),
    [
        (0, 0),
        (365.25 / 4, 1),
    ],
)
def test_per_q_to_per_d(quarterly_value: float, daily_value: float) -> None:
    assert per_q_to_per_d(quarterly_value) == daily_value


@pytest.mark.parametrize(
    ("monthly_value", "yearly_value"),
    [
        (0, 0),
        (1, 12),
    ],
)
def test_per_m_to_per_y(monthly_value: float, yearly_value: float) -> None:
    assert per_m_to_per_y(monthly_value) == yearly_value


@pytest.mark.parametrize(
    ("monthly_value", "quarterly_value"),
    [
        (0, 0),
        (1, 3),
    ],
)
def test_per_m_to_per_q(monthly_value: float, quarterly_value: float) -> None:
    assert per_m_to_per_q(monthly_value) == quarterly_value


@pytest.mark.parametrize(
    ("monthly_value", "weekly_value"),
    [
        (0, 0),
        (365.25, 84),
    ],
)
def test_per_m_to_per_w(monthly_value: float, weekly_value: float) -> None:
    assert per_m_to_per_w(monthly_value) == weekly_value


@pytest.mark.parametrize(
    ("monthly_value", "daily_value"),
    [
        (0, 0),
        (365.25, 12),
    ],
)
def test_per_m_to_per_d(monthly_value: float, daily_value: float) -> None:
    assert per_m_to_per_d(monthly_value) == daily_value


@pytest.mark.parametrize(
    ("weekly_value", "yearly_value"),
    [
        (0, 0),
        (7, 365.25),
    ],
)
def test_per_w_to_per_y(weekly_value: float, yearly_value: float) -> None:
    assert per_w_to_per_y(weekly_value) == yearly_value


@pytest.mark.parametrize(
    ("weekly_value", "monthly_value"),
    [
        (0, 0),
        (84, 365.25),
    ],
)
def test_per_w_to_per_m(weekly_value: float, monthly_value: float) -> None:
    assert per_w_to_per_m(weekly_value) == monthly_value


@pytest.mark.parametrize(
    ("weekly_value", "quarterly_value"),
    [
        (0, 0),
        (7, 365.25 / 4),
    ],
)
def test_per_w_to_per_q(weekly_value: float, quarterly_value: float) -> None:
    assert per_w_to_per_q(weekly_value) == quarterly_value


@pytest.mark.parametrize(
    ("weekly_value", "daily_value"),
    [
        (0, 0),
        (7, 1),
    ],
)
def test_per_w_to_per_d(weekly_value: float, daily_value: float) -> None:
    assert per_w_to_per_d(weekly_value) == daily_value


@pytest.mark.parametrize(
    ("daily_value", "yearly_value"),
    [
        (0, 0),
        (1, 365.25),
    ],
)
def test_per_d_to_per_y(daily_value: float, yearly_value: float) -> None:
    assert per_d_to_per_y(daily_value) == yearly_value


@pytest.mark.parametrize(
    ("daily_value", "quarterly_value"),
    [
        (0, 0),
        (1, 365.25 / 4),
    ],
)
def test_per_d_to_per_q(daily_value: float, quarterly_value: float) -> None:
    assert per_d_to_per_q(daily_value) == quarterly_value


@pytest.mark.parametrize(
    ("daily_value", "monthly_value"),
    [
        (0, 0),
        (12, 365.25),
    ],
)
def test_per_d_to_per_m(daily_value: float, monthly_value: float) -> None:
    assert per_d_to_per_m(daily_value) == monthly_value


@pytest.mark.parametrize(
    ("daily_value", "weekly_value"),
    [
        (0, 0),
        (1, 7),
    ],
)
def test_per_d_to_per_w(daily_value: float, weekly_value: float) -> None:
    assert per_d_to_per_w(daily_value) == weekly_value


@pytest.mark.parametrize(
    ("yearly_value", "quarterly_value"),
    [
        (0, 0),
        (1, 4),
    ],
)
def test_y_to_q(yearly_value: float, quarterly_value: float) -> None:
    assert y_to_q(yearly_value) == quarterly_value


@pytest.mark.parametrize(
    ("yearly_value", "monthly_value"),
    [
        (0, 0),
        (1, 12),
    ],
)
def test_y_to_m(yearly_value: float, monthly_value: float) -> None:
    assert y_to_m(yearly_value) == monthly_value


@pytest.mark.parametrize(
    ("yearly_value", "weekly_value"),
    [
        (0, 0),
        (1, 365.25 / 7),
    ],
)
def test_y_to_w(yearly_value: float, weekly_value: float) -> None:
    assert y_to_w(yearly_value) == weekly_value


@pytest.mark.parametrize(
    ("yearly_value", "daily_value"),
    [
        (0, 0),
        (1, 365.25),
    ],
)
def test_y_to_d(yearly_value: float, daily_value: float) -> None:
    assert y_to_d(yearly_value) == daily_value


@pytest.mark.parametrize(
    ("quarterly_value", "yearly_value"),
    [
        (0, 0),
        (4, 1),
    ],
)
def test_q_to_y(quarterly_value: float, yearly_value: float) -> None:
    assert q_to_y(quarterly_value) == yearly_value


@pytest.mark.parametrize(
    ("quarterly_value", "monthly_value"),
    [
        (0, 0),
        (1, 3),
    ],
)
def test_q_to_m(quarterly_value: float, monthly_value: float) -> None:
    assert q_to_m(quarterly_value) == monthly_value


@pytest.mark.parametrize(
    ("quarterly_value", "weekly_value"),
    [
        (0, 0),
        (1, 365.25 / 7 / 4),
    ],
)
def test_q_to_w(quarterly_value: float, weekly_value: float) -> None:
    assert q_to_w(quarterly_value) == weekly_value


@pytest.mark.parametrize(
    ("quarterly_value", "daily_value"),
    [
        (0, 0),
        (1, 365.25 / 4),
    ],
)
def test_q_to_d(quarterly_value: float, daily_value: float) -> None:
    assert q_to_d(quarterly_value) == daily_value


@pytest.mark.parametrize(
    ("monthly_value", "yearly_value"),
    [
        (0, 0),
        (12, 1),
    ],
)
def test_m_to_y(monthly_value: float, yearly_value: float) -> None:
    assert m_to_y(monthly_value) == yearly_value


@pytest.mark.parametrize(
    ("monthly_value", "quarterly_value"),
    [
        (0, 0),
        (3, 1),
    ],
)
def test_m_to_q(monthly_value: float, quarterly_value: float) -> None:
    assert m_to_q(monthly_value) == quarterly_value


@pytest.mark.parametrize(
    ("monthly_value", "weekly_value"),
    [
        (0, 0),
        (1, 365.25 / 7 / 12),
    ],
)
def test_m_to_w(monthly_value: float, weekly_value: float) -> None:
    assert m_to_w(monthly_value) == weekly_value


@pytest.mark.parametrize(
    ("monthly_value", "daily_value"),
    [
        (0, 0),
        (1, 365.25 / 12),
    ],
)
def test_m_to_d(monthly_value: float, daily_value: float) -> None:
    assert m_to_d(monthly_value) == daily_value


@pytest.mark.parametrize(
    ("weekly_value", "yearly_value"),
    [
        (0, 0),
        (365.25 / 7, 1),
    ],
)
def test_w_to_y(weekly_value: float, yearly_value: float) -> None:
    assert w_to_y(weekly_value) == yearly_value


@pytest.mark.parametrize(
    ("weekly_value", "quarterly_value"),
    [
        (0, 0),
        (365.25 / 7 / 4, 1),
    ],
)
def test_w_to_q(weekly_value: float, quarterly_value: float) -> None:
    assert w_to_q(weekly_value) == quarterly_value


@pytest.mark.parametrize(
    ("weekly_value", "monthly_value"),
    [
        (0, 0),
        (365.25 / 7 / 12, 1),
    ],
)
def test_w_to_m(weekly_value: float, monthly_value: float) -> None:
    assert w_to_m(weekly_value) == monthly_value


@pytest.mark.parametrize(
    ("weekly_value", "daily_value"),
    [
        (0, 0),
        (1, 7),
    ],
)
def test_w_to_d(weekly_value: float, daily_value: float) -> None:
    assert w_to_d(weekly_value) == daily_value


@pytest.mark.parametrize(
    ("daily_value", "yearly_value"),
    [
        (0, 0),
        (365.25, 1),
    ],
)
def test_d_to_y(daily_value: float, yearly_value: float) -> None:
    assert d_to_y(daily_value) == yearly_value


@pytest.mark.parametrize(
    ("daily_value", "quarterly_value"),
    [
        (0, 0),
        (365.25 / 4, 1),
    ],
)
def test_d_to_q(daily_value: float, quarterly_value: float) -> None:
    assert d_to_q(daily_value) == quarterly_value


@pytest.mark.parametrize(
    ("daily_value", "monthly_value"),
    [
        (0, 0),
        (365.25 / 12, 1),
    ],
)
def test_d_to_m(daily_value: float, monthly_value: float) -> None:
    assert d_to_m(daily_value) == monthly_value


@pytest.mark.parametrize(
    ("daily_value", "weekly_value"),
    [
        (0, 0),
        (7, 1),
    ],
)
def test_d_to_w(daily_value: float, weekly_value: float) -> None:
    assert d_to_w(daily_value) == weekly_value
