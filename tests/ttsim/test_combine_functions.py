import pandas as pd
import pytest

from ttsim import (
    AggType,
    PolicyEnvironment,
    agg_by_group_function,
    compute_taxes_and_transfers,
    policy_function,
    policy_input,
)
from ttsim.automatically_added_functions import create_agg_by_group_functions
from ttsim.combine_functions import _fail_if_targets_not_in_functions


@pytest.fixture
@policy_function(leaf_name="foo")
def function_with_bool_return(x: bool) -> bool:
    return x


@policy_input()
def x() -> int:
    pass


@policy_input()
def x_f() -> float:
    pass


@policy_input()
def x_b() -> bool:
    pass


@policy_input()
def p_id() -> int:
    pass


@policy_input()
def hh_id() -> int:
    pass


@agg_by_group_function(leaf_name="y_hh", agg_type=AggType.SUM)
def y_hh(hh_id: int, x: int) -> int:
    pass


@agg_by_group_function(leaf_name="y_hh", agg_type=AggType.SUM)
def y_hh_namespaced_input(hh_id: int, inputs__x: int) -> int:
    pass


@pytest.fixture
@policy_function(leaf_name="bar")
def function_with_int_return(x: int) -> int:
    return x


@pytest.fixture
@policy_function(leaf_name="baz")
def function_with_float_return(x: int) -> float:
    return x


def some_x(x):
    return x


def return_x_hh(x_hh: int) -> int:
    return x_hh


def return_y_hh(y_hh: int) -> int:
    return y_hh


def return_n1__x_hh(n1__x_hh: int) -> int:
    return n1__x_hh


@pytest.mark.parametrize(
    (
        "objects_tree",
        "targets_tree",
        "data_tree",
    ),
    [
        (
            # Aggregations derived from simple function arguments
            {
                "hh_id": hh_id,
                "p_id": p_id,
                "n1": {
                    "f": policy_function(leaf_name="f")(return_n1__x_hh),
                    "x": x,
                },
            },
            {"n1": {"f": None}},
            {
                "n1": {"x": pd.Series([1, 1, 1])},
                "hh_id": pd.Series([0, 0, 0]),
                "p_id": pd.Series([0, 1, 2]),
            },
        ),
        (
            # Aggregations derived from namespaced function arguments
            {
                "hh_id": hh_id,
                "p_id": p_id,
                "n1": {"f": policy_function(leaf_name="f")(return_x_hh), "x": x},
            },
            {"n1": {"f": None}},
            {
                "n1": {"x": pd.Series([1, 1, 1])},
                "hh_id": pd.Series([0, 0, 0]),
                "p_id": pd.Series([0, 1, 2]),
            },
        ),
        (
            # Aggregations derived from target
            {
                "hh_id": hh_id,
                "p_id": p_id,
                "n1": {
                    "f": policy_function(leaf_name="f")(some_x),
                    "x": x,
                },
            },
            {"n1": {"f_hh": None}},
            {
                "n1": {"x": pd.Series([1, 1, 1])},
                "hh_id": pd.Series([0, 0, 0]),
                "p_id": pd.Series([0, 1, 2]),
            },
        ),
        (
            # Explicit aggregation via objects tree with leaf name input
            {
                "hh_id": hh_id,
                "p_id": p_id,
                "n1": {
                    "f": policy_function(leaf_name="f")(some_x),
                    "x": x,
                },
                "y_hh": y_hh,
            },
            {"n1": {"f": None}},
            {
                "n1": {"x": pd.Series([1, 1, 1])},
                "hh_id": pd.Series([0, 0, 0]),
                "p_id": pd.Series([0, 1, 2]),
            },
        ),
        (
            # Explicit aggregation via objects tree with namespaced input
            {
                "hh_id": hh_id,
                "p_id": p_id,
                "n1": {
                    "f": policy_function(leaf_name="f")(return_y_hh),
                    "y_hh": y_hh_namespaced_input,
                },
                "inputs": {"x": x},
            },
            {"n1": {"f": None}},
            {
                "inputs": {"x": pd.Series([1, 1, 1])},
                "hh_id": pd.Series([0, 0, 0]),
                "p_id": pd.Series([0, 1, 2]),
            },
        ),
    ],
)
def test_create_agg_by_group_functions(
    objects_tree,
    targets_tree,
    data_tree,
):
    environment = PolicyEnvironment(raw_objects_tree=objects_tree)
    compute_taxes_and_transfers(
        environment=environment,
        data_tree=data_tree,
        targets_tree=targets_tree,
        groupings=("hh",),
    )


@pytest.mark.parametrize(
    "functions, targets, expected_error_match",
    [
        ({"foo": some_x}, {"bar": None}, "('bar',)"),
        ({"foo__baz": some_x}, {"foo__bar": None}, "('foo', 'bar')"),
    ],
)
def test_fail_if_targets_are_not_among_functions(
    functions, targets, expected_error_match
):
    with pytest.raises(ValueError) as e:
        _fail_if_targets_not_in_functions(functions, targets)
    assert expected_error_match in str(e.value)


@pytest.mark.parametrize(
    (
        "functions",
        "targets",
        "data",
        "expected",
    ),
    [
        (
            {"foo": policy_function(leaf_name="foo")(return_x_hh)},
            {},
            {"x": pd.Series([1])},
            ("x_hh"),
        ),
        (
            {"n2__foo": policy_function(leaf_name="foo")(return_n1__x_hh)},
            {},
            {"n1__x": pd.Series([1])},
            ("n1__x_hh"),
        ),
        (
            {},
            {"x_hh": None},
            {"x": pd.Series([1])},
            ("x_hh"),
        ),
    ],
)
def test_derived_aggregation_functions_are_in_correct_namespace(
    functions,
    targets,
    data,
    expected,
):
    """Test that the derived aggregation functions are in the correct namespace.

    The namespace of the derived aggregation functions should be the same as the
    namespace of the function that is being aggregated.
    """
    result = create_agg_by_group_functions(
        ttsim_functions_with_time_conversions=functions,
        data=data,
        targets=targets,
        groupings=("hh",),
    )
    assert expected in result
