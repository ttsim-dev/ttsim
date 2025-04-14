import pytest

from ttsim.ttsim_objects import (
    PolicyFunction,
    PolicyInput,
    policy_function,
    policy_input,
)

# ======================================================================================
# PolicyFunction and policy_function
# ======================================================================================


@policy_function()
def simple_policy_function(x):
    return x


@policy_function(leaf_name="simple_policy_function")
def policy_function_with_internal_name(x):
    return x


@policy_function(start_date="2007-01-01", end_date="2011-12-31")
def policy_function_with_dates(x):
    return x


@pytest.mark.parametrize(
    "function",
    [
        simple_policy_function,
        policy_function_with_internal_name,
    ],
)
def test_policy_function_type(function):
    assert isinstance(function, PolicyFunction)


@pytest.mark.parametrize(
    "function",
    [
        simple_policy_function,
        policy_function_with_internal_name,
    ],
)
def test_policy_function_name(function):
    assert function.leaf_name == "simple_policy_function"


def test_policy_function_with_dates():
    assert str(policy_function_with_dates.start_date) == "2007-01-01"
    assert str(policy_function_with_dates.end_date) == "2011-12-31"


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
