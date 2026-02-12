from __future__ import annotations

import inspect

import pytest

from ttsim.interface_dag_elements.automatically_added_functions import (
    _create_function_for_time_unit,
    create_agg_by_group_functions,
    create_time_conversion_functions,
)
from ttsim.tt import policy_function
from ttsim.unit_converters import (
    per_d_to_per_m,
    per_d_to_per_w,
)


def return_one() -> int:
    return 1


def return_x_kin(x_kin: int) -> int:
    return x_kin


def return_n1__x_kin(n1__x_kin: int) -> int:
    return n1__x_kin


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
def test_should_create_functions_for_other_time_units(
    name: str,
    expected: list[str],
) -> None:
    time_conversion_functions = create_time_conversion_functions(
        qname_policy_environment={
            name: policy_function(leaf_name=name)(return_one),
        },
        input_columns=set(),
        grouping_levels=("sn", "kin"),
    )

    for expected_name in expected:
        assert expected_name in time_conversion_functions


def test_should_not_create_functions_automatically_that_exist_already() -> None:
    time_conversion_functions = create_time_conversion_functions(
        qname_policy_environment={
            "test1_d": policy_function(leaf_name="test1_d")(return_one),
        },
        input_columns={"test2_y"},
        grouping_levels=("sn", "kin"),
    )

    assert "test1_d" not in time_conversion_functions
    assert "test2_y" not in time_conversion_functions


def test_should_overwrite_with_data_cols_differing_only_in_time_period() -> None:
    time_conversion_functions = create_time_conversion_functions(
        qname_policy_environment={
            "test_d": policy_function(leaf_name="test_d")(return_one),
        },
        input_columns={"test_y"},
        grouping_levels=("sn", "kin"),
    )

    assert "test_d" in time_conversion_functions


def test_create_function_for_time_unit_should_rename_parameter():
    function = _create_function_for_time_unit("test", per_d_to_per_m)

    parameter_spec = inspect.getfullargspec(function)
    assert parameter_spec.args == ["test"]


def test_create_function_for_time_unit_should_not_set_info_if_none():
    function = _create_function_for_time_unit("test", per_d_to_per_m)

    assert not hasattr(function, "__info__")


def test_create_function_for_time_unit_should_apply_converter():
    function = _create_function_for_time_unit("test", per_d_to_per_w)

    assert function(1) == 7  # ty: ignore[invalid-argument-type]


def test_time_conversions_should_not_create_cycle():
    # Check for:
    # https://github.com/iza-institute-of-labor-economics/gettsim/issues/621
    def x(test_m: int) -> int:
        return test_m

    time_conversion_functions = create_time_conversion_functions(
        qname_policy_environment={"test_d": policy_function(leaf_name="test_d")(x)},
        input_columns=set(),
        grouping_levels=(),
    )

    assert "test_m" not in time_conversion_functions


def test_grouping_functions_should_not_create_cycle():
    @policy_function()
    def x(x_hh: int) -> int:
        return x_hh

    @policy_function()
    def some_other_function_requiring_x_hh(x_hh: int) -> int:
        return x_hh

    grouping_functions = create_agg_by_group_functions(
        column_functions={
            "x": x,
            "some_other_function_requiring_x_hh": some_other_function_requiring_x_hh,
        },
        input_columns=set(),
        tt_targets=("some_other_function_requiring_x_hh",),
        grouping_levels=("hh",),
    )

    assert "x_hh" not in grouping_functions


@pytest.mark.parametrize(
    (
        "column_functions",
        "tt_targets",
        "input_columns",
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
    tt_targets,
    input_columns,
    expected,
):
    """Test that the derived aggregation functions are in the correct namespace.

    The namespace of the derived aggregation functions should be the same as the
    namespace of the function that is being aggregated.
    """
    result = create_agg_by_group_functions(
        column_functions=column_functions,
        input_columns=input_columns,
        tt_targets=tt_targets,
        grouping_levels=("kin",),
    )
    assert expected in result
