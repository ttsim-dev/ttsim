from __future__ import annotations

import inspect

import pytest

from ttsim import (
    AggType,
    PolicyFunction,
    PolicyInput,
    agg_by_group_function,
    agg_by_p_id_function,
    policy_function,
    policy_input,
)
from ttsim.config import numpy_or_jax as np
from ttsim.tt_dag_elements.column_objects_param_function import (
    ParamFunction,
    param_function,
)

# ======================================================================================
# PolicyFunction and policy_function
# ======================================================================================


@policy_function()
def simple_policy_function(x):
    return x


@policy_function(leaf_name="simple_policy_function")
def policy_function_with_different_leaf_name(x):
    return x


@policy_function(start_date="2007-01-01", end_date="2011-12-31")
def policy_function_with_dates(x):
    return x


@pytest.mark.parametrize(
    "function",
    [
        simple_policy_function,
        policy_function_with_different_leaf_name,
    ],
)
def test_policy_function_type(function):
    assert isinstance(function, PolicyFunction)


@pytest.mark.parametrize(
    "function",
    [
        simple_policy_function,
        policy_function_with_different_leaf_name,
    ],
)
def test_policy_function_name(function):
    assert function.leaf_name == "simple_policy_function"


def test_policy_function_with_dates():
    assert str(policy_function_with_dates.start_date) == "2007-01-01"
    assert str(policy_function_with_dates.end_date) == "2011-12-31"


# ======================================================================================
# ParamFunction and param_function
# ======================================================================================


@param_function()
def simple_param_function(x):
    return x


@param_function(leaf_name="simple_param_function")
def param_function_with_different_leaf_name(x):
    return x


@param_function(start_date="2007-01-01", end_date="2011-12-31")
def param_function_with_dates(x):
    return x


@pytest.mark.parametrize(
    "function",
    [
        simple_param_function,
        param_function_with_different_leaf_name,
    ],
)
def test_param_function_type(function):
    assert isinstance(function, ParamFunction)


@pytest.mark.parametrize(
    "function",
    [
        simple_param_function,
        param_function_with_different_leaf_name,
    ],
)
def test_param_function_name(function):
    assert function.leaf_name == "simple_param_function"


def test_param_function_with_dates():
    assert str(param_function_with_dates.start_date) == "2007-01-01"
    assert str(param_function_with_dates.end_date) == "2011-12-31"


# ======================================================================================
# PolicyInput and policy_input
# ======================================================================================


@policy_input()
def simple_policy_input() -> float:
    pass


@policy_input(start_date="2007-01-01", end_date="2011-12-31")
def policy_input_with_dates() -> float:
    pass


@pytest.mark.parametrize(
    "function",
    [
        simple_policy_input,
        policy_input_with_dates,
    ],
)
def test_policy_input_type(function):
    assert isinstance(function, PolicyInput)


def test_policy_input_with_dates():
    assert str(policy_input_with_dates.start_date) == "2007-01-01"
    assert str(policy_input_with_dates.end_date) == "2011-12-31"


# ======================================================================================
# AggByGroupFunction and agg_by_group_function
# ======================================================================================


@agg_by_group_function(agg_type=AggType.COUNT)
def aggregate_by_group_count(group_id):
    pass


@agg_by_group_function(agg_type=AggType.SUM)
def aggregate_by_group_sum(group_id, source):
    pass


@pytest.mark.parametrize(
    (
        "function",
        "expected_group_id",
        "expected_other_arg",
    ),
    [
        (aggregate_by_group_count, "group_id", None),
        (aggregate_by_group_sum, "group_id", "source"),
    ],
)
def test_agg_by_group_function_type(function, expected_group_id, expected_other_arg):
    args = inspect.signature(function).parameters
    assert expected_group_id in args
    if expected_other_arg:
        assert expected_other_arg in args


def test_agg_by_group_count_other_arg_present():
    match = "There must be no argument besides identifiers for counting."
    with pytest.raises(ValueError, match=match):

        @agg_by_group_function(agg_type=AggType.COUNT)
        def aggregate_by_group_count_other_arg_present(group_id, wrong_arg):
            pass


def test_agg_by_group_sum_wrong_amount_of_args():
    match = "There must be exactly one argument besides identifiers for aggregations."
    with pytest.raises(ValueError, match=match):

        @agg_by_group_function(agg_type=AggType.SUM)
        def aggregate_by_group_sum_no_arg_present(group_id):
            pass

    with pytest.raises(ValueError, match=match):

        @agg_by_group_function(agg_type=AggType.SUM)
        def aggregate_by_group_sum_multiple_args_present(group_id, arg, another_arg):
            pass


def test_wrong_number_of_group_ids_present():
    match = "Require exactly one group identifier ending with '_id'"
    with pytest.raises(ValueError, match=match):

        @agg_by_group_function(agg_type=AggType.COUNT)
        def aggregate_by_group_count_multiple_group_ids_present(
            group_id, another_group_id
        ):
            pass

    with pytest.raises(ValueError, match=match):

        @agg_by_group_function(agg_type=AggType.COUNT)
        def aggregate_by_group_count_no_group_id_present():
            pass


# ======================================================================================
# AggByPIDFunction and agg_by_p_id_function
# ======================================================================================


@agg_by_p_id_function(agg_type=AggType.COUNT)
def aggregate_by_p_id_count(p_id, p_id_specifier):
    pass


@agg_by_p_id_function(agg_type=AggType.SUM)
def aggregate_by_p_id_sum(p_id, p_id_specifier, source):
    pass


@pytest.mark.parametrize(
    (
        "function",
        "expected_foreign_p_id",
        "expected_other_arg",
    ),
    [
        (aggregate_by_p_id_count, "p_id", None),
        (aggregate_by_p_id_sum, "p_id", "source"),
    ],
)
def test_agg_by_p_id_function_type(function, expected_foreign_p_id, expected_other_arg):
    args = inspect.signature(function).parameters
    assert expected_foreign_p_id in args
    if expected_other_arg:
        assert expected_other_arg in args


def test_agg_by_p_id_count_other_arg_present():
    match = "There must be no argument besides identifiers for counting."
    with pytest.raises(ValueError, match=match):

        @agg_by_p_id_function(agg_type=AggType.COUNT)
        def aggregate_by_p_id_count_other_arg_present(p_id, p_id_specifier, wrong_arg):
            pass


def test_agg_by_p_id_sum_wrong_amount_of_args():
    match = "There must be exactly one argument besides identifiers for aggregations."
    with pytest.raises(ValueError, match=match):

        @agg_by_p_id_function(agg_type=AggType.SUM)
        def aggregate_by_p_id_sum_no_arg_present(p_id, p_id_specifier):
            pass

        @agg_by_p_id_function(agg_type=AggType.SUM)
        def aggregate_by_p_id_sum_multiple_args_present(
            p_id, p_id_specifier, arg, another_arg
        ):
            pass


def test_agg_by_p_id_multiple_other_p_ids_present():
    match = "Require exactly one identifier starting with 'p_id_' for"
    with pytest.raises(ValueError, match=match):

        @agg_by_p_id_function(agg_type=AggType.SUM)
        def aggregate_by_p_id_multiple_other_p_ids_present(
            p_id, p_id_specifier_one, p_id_specifier_two
        ):
            pass


def test_agg_by_p_id_sum_with_all_missing_p_ids():
    aggregate_by_p_id_sum(
        p_id=np.array([180]), p_id_specifier=np.array([-1]), source=np.array([False])
    )
