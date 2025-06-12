from __future__ import annotations

import inspect

import pytest

from ttsim.interface_dag_elements.automatically_added_functions import (
    _create_function_for_time_unit,
    create_agg_by_group_functions,
    create_time_conversion_functions,
    d_to_m,
    d_to_q,
    d_to_w,
    d_to_y,
    m_to_d,
    m_to_q,
    m_to_w,
    m_to_y,
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
from ttsim.tt_dag_elements import policy_function


def return_one() -> int:
    return 1


def return_x_kin(x_kin: int) -> int:
    return x_kin


def return_n1__x_kin(n1__x_kin: int) -> int:
    return n1__x_kin


@pytest.mark.parametrize(
    ("yearly_value", "quarterly_value"),
    [
        (0, 0),
        (12, 3),
    ],
)
def test_y_to_q(yearly_value: float, quarterly_value: float) -> None:
    assert y_to_q(yearly_value) == quarterly_value


@pytest.mark.parametrize(
    ("yearly_value", "monthly_value"),
    [
        (0, 0),
        (12, 1),
    ],
)
def test_y_to_m(yearly_value: float, monthly_value: float) -> None:
    assert y_to_m(yearly_value) == monthly_value


@pytest.mark.parametrize(
    ("yearly_value", "weekly_value"),
    [
        (0, 0),
        (365.25, 7),
    ],
)
def test_y_to_w(yearly_value: float, weekly_value: float) -> None:
    assert y_to_w(yearly_value) == weekly_value


@pytest.mark.parametrize(
    ("yearly_value", "daily_value"),
    [
        (0, 0),
        (365.25, 1),
    ],
)
def test_y_to_d(yearly_value: float, daily_value: float) -> None:
    assert y_to_d(yearly_value) == daily_value


@pytest.mark.parametrize(
    ("quarterly_value", "yearly_value"),
    [
        (0, 0),
        (1, 4),
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
        (365.25 / 7 / 4, 1),
    ],
)
def test_q_to_w(quarterly_value: float, weekly_value: float) -> None:
    assert q_to_w(quarterly_value) == weekly_value


@pytest.mark.parametrize(
    ("quarterly_value", "daily_value"),
    [
        (0, 0),
        (365.25 / 4, 1),
    ],
)
def test_q_to_d(quarterly_value: float, daily_value: float) -> None:
    assert q_to_d(quarterly_value) == daily_value


@pytest.mark.parametrize(
    ("monthly_value", "yearly_value"),
    [
        (0, 0),
        (1, 12),
    ],
)
def test_m_to_y(monthly_value: float, yearly_value: float) -> None:
    assert m_to_y(monthly_value) == yearly_value


@pytest.mark.parametrize(
    ("monthly_value", "quarterly_value"),
    [
        (0, 0),
        (1, 3),
    ],
)
def test_m_to_q(monthly_value: float, quarterly_value: float) -> None:
    assert m_to_q(monthly_value) == quarterly_value


@pytest.mark.parametrize(
    ("monthly_value", "weekly_value"),
    [
        (0, 0),
        (365.25, 84),
    ],
)
def test_m_to_w(monthly_value: float, weekly_value: float) -> None:
    assert m_to_w(monthly_value) == weekly_value


@pytest.mark.parametrize(
    ("monthly_value", "daily_value"),
    [
        (0, 0),
        (365.25, 12),
    ],
)
def test_m_to_d(monthly_value: float, daily_value: float) -> None:
    assert m_to_d(monthly_value) == daily_value


@pytest.mark.parametrize(
    ("weekly_value", "yearly_value"),
    [
        (0, 0),
        (7, 365.25),
    ],
)
def test_w_to_y(weekly_value: float, yearly_value: float) -> None:
    assert w_to_y(weekly_value) == yearly_value


@pytest.mark.parametrize(
    ("weekly_value", "monthly_value"),
    [
        (0, 0),
        (84, 365.25),
    ],
)
def test_w_to_m(weekly_value: float, monthly_value: float) -> None:
    assert w_to_m(weekly_value) == monthly_value


@pytest.mark.parametrize(
    ("weekly_value", "quarterly_value"),
    [
        (0, 0),
        (7, 365.25 / 4),
    ],
)
def test_w_to_q(weekly_value: float, quarterly_value: float) -> None:
    assert w_to_q(weekly_value) == quarterly_value


@pytest.mark.parametrize(
    ("weekly_value", "daily_value"),
    [
        (0, 0),
        (7, 1),
    ],
)
def test_w_to_d(weekly_value: float, daily_value: float) -> None:
    assert w_to_d(weekly_value) == daily_value


@pytest.mark.parametrize(
    ("daily_value", "yearly_value"),
    [
        (0, 0),
        (1, 365.25),
    ],
)
def test_d_to_y(daily_value: float, yearly_value: float) -> None:
    assert d_to_y(daily_value) == yearly_value


@pytest.mark.parametrize(
    ("daily_value", "quarterly_value"),
    [
        (0, 0),
        (1, 365.25 / 4),
    ],
)
def test_d_to_q(daily_value: float, quarterly_value: float) -> None:
    assert d_to_q(daily_value) == quarterly_value


@pytest.mark.parametrize(
    ("daily_value", "monthly_value"),
    [
        (0, 0),
        (12, 365.25),
    ],
)
def test_d_to_m(daily_value: float, monthly_value: float) -> None:
    assert d_to_m(daily_value) == monthly_value


@pytest.mark.parametrize(
    ("daily_value", "weekly_value"),
    [
        (0, 0),
        (1, 7),
    ],
)
def test_d_to_w(daily_value: float, weekly_value: float) -> None:
    assert d_to_w(daily_value) == weekly_value


class TestCreateFunctionsForTimeUnits:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("test_y", ["test_m", "test_q", "test_w", "test_d"]),
            ("test_y_kin", ["test_m_kin", "test_q_kin", "test_w_kin", "test_d_kin"]),
            ("test_y_sn", ["test_m_sn", "test_q_sn", "test_w_sn", "test_d_sn"]),
            ("test_q", ["test_y", "test_m", "test_w", "test_d"]),
            ("test_q_kin", ["test_y_kin", "test_m_kin", "test_w_kin", "test_d_kin"]),
            ("test_q_sn", ["test_y_sn", "test_m_sn", "test_w_sn", "test_d_sn"]),
            ("test_m", ["test_y", "test_q", "test_w", "test_d"]),
            ("test_m_kin", ["test_y_kin", "test_q_kin", "test_w_kin", "test_d_kin"]),
            ("test_m_sn", ["test_y_sn", "test_q_sn", "test_w_sn", "test_d_sn"]),
            ("test_w", ["test_y", "test_m", "test_q", "test_d"]),
            ("test_w_kin", ["test_y_kin", "test_m_kin", "test_q_kin", "test_d_kin"]),
            ("test_w_sn", ["test_y_sn", "test_m_sn", "test_q_sn", "test_d_sn"]),
            ("test_d", ["test_y", "test_m", "test_q", "test_w"]),
            ("test_d_kin", ["test_y_kin", "test_m_kin", "test_q_kin", "test_w_kin"]),
            ("test_d_sn", ["test_y_sn", "test_m_sn", "test_q_sn", "test_w_sn"]),
        ],
    )
    def test_should_create_functions_for_other_time_units_for_functions(
        self, name: str, expected: list[str]
    ) -> None:
        time_conversion_functions = create_time_conversion_functions(
            qual_name_policy_environment={
                name: policy_function(leaf_name=name)(return_one)
            },
            processed_data_columns=set(),
            grouping_levels=("sn", "kin"),
        )

        for expected_name in expected:
            assert expected_name in time_conversion_functions

    def test_should_not_create_functions_automatically_that_exist_already(self) -> None:
        time_conversion_functions = create_time_conversion_functions(
            qual_name_policy_environment={
                "test1_d": policy_function(leaf_name="test1_d")(return_one)
            },
            processed_data_columns={"test2_y"},
            grouping_levels=("sn", "kin"),
        )

        assert "test1_d" not in time_conversion_functions
        assert "test2_y" not in time_conversion_functions

    def test_should_overwrite_functions_with_data_cols_that_only_differ_in_time_period(
        self,
    ) -> None:
        time_conversion_functions = create_time_conversion_functions(
            qual_name_policy_environment={
                "test_d": policy_function(leaf_name="test_d")(return_one)
            },
            processed_data_columns={"test_y"},
            grouping_levels=("sn", "kin"),
        )

        assert "test_d" in time_conversion_functions


class TestCreateFunctionForTimeUnit:
    def test_should_rename_parameter(self):
        function = _create_function_for_time_unit("test", d_to_m)

        parameter_spec = inspect.getfullargspec(function)
        assert parameter_spec.args == ["test"]

    def test_should_not_set_info_if_none(self):
        function = _create_function_for_time_unit("test", d_to_m)

        assert not hasattr(function, "__info__")

    def test_should_apply_converter(self):
        function = _create_function_for_time_unit("test", d_to_w)

        assert function(1) == 7


def test_should_not_create_cycle():
    # Check for:
    # https://github.com/iza-institute-of-labor-economics/gettsim/issues/621
    def x(test_m: int) -> int:
        return test_m

    time_conversion_functions = create_time_conversion_functions(
        qual_name_policy_environment={"test_d": policy_function(leaf_name="test_d")(x)},
        processed_data_columns=set(),
        grouping_levels=(),
    )

    assert "test_m" not in time_conversion_functions


@pytest.mark.parametrize(
    (
        "column_functions",
        "targets",
        "names__processed_data_columns",
        "expected",
    ),
    [
        (
            {"foo": policy_function(leaf_name="foo")(return_x_kin)},
            {},
            {"x"},
            ("x_kin"),
        ),
        (
            {"n2__foo": policy_function(leaf_name="foo")(return_n1__x_kin)},
            {},
            {"n1__x"},
            ("n1__x_kin"),
        ),
        (
            {},
            {"x_kin": None},
            {"x"},
            ("x_kin"),
        ),
    ],
)
def test_derived_aggregation_functions_are_in_correct_namespace(
    column_functions,
    targets,
    names__processed_data_columns,
    expected,
    backend,
):
    """Test that the derived aggregation functions are in the correct namespace.

    The namespace of the derived aggregation functions should be the same as the
    namespace of the function that is being aggregated.
    """
    result = create_agg_by_group_functions(
        column_functions=column_functions,
        names__processed_data_columns=names__processed_data_columns,
        targets=targets,
        grouping_levels=("kin",),
        backend=backend,
    )
    assert expected in result
