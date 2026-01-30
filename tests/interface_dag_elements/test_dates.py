from __future__ import annotations

import datetime

import pytest

from ttsim.interface_dag_elements.dates import (
    evaluation_date_from_evaluation_date_str,
    evaluation_date_use_other_info,
    policy_date,
)
from ttsim.interface_dag_elements.interface_node_objects import (
    InputDependentInterfaceFunction,
    InterfaceFunction,
    InterfaceInput,
)


# =============================================================================
# policy_date tests
# =============================================================================
def test_policy_date_is_interface_function():
    assert isinstance(policy_date, InterfaceFunction)


def test_policy_date_from_valid_string():
    result = policy_date("2024-06-15")
    assert result == datetime.date(2024, 6, 15)


def test_policy_date_from_leap_year_date():
    result = policy_date("2020-02-29")
    assert result == datetime.date(2020, 2, 29)


def test_policy_date_in_top_level_namespace():
    assert policy_date.in_top_level_namespace is True


@pytest.mark.parametrize(
    "date_str",
    [
        "2024-01-01",
        "1990-12-31",
        "2000-06-15",
    ],
)
def test_policy_date_various_valid_dates(date_str):
    result = policy_date(date_str)
    year, month, day = map(int, date_str.split("-"))
    assert result == datetime.date(year, month, day)


def test_policy_date_invalid_date_raises():
    with pytest.raises(ValueError):
        policy_date("2024-02-30")  # Invalid date


# =============================================================================
# evaluation_date_use_other_info tests
# =============================================================================
def test_evaluation_date_use_other_info_is_input_dependent():
    assert isinstance(evaluation_date_use_other_info, InputDependentInterfaceFunction)


def test_evaluation_date_use_other_info_returns_none():
    result = evaluation_date_use_other_info(backend="numpy")
    assert result is None


def test_evaluation_date_use_other_info_with_jax_backend():
    result = evaluation_date_use_other_info(backend="jax")
    assert result is None


def test_evaluation_date_use_other_info_has_correct_leaf_name():
    assert evaluation_date_use_other_info.leaf_name == "evaluation_date"


def test_evaluation_date_use_other_info_in_top_level_namespace():
    assert evaluation_date_use_other_info.in_top_level_namespace is True


def test_evaluation_date_use_other_info_include_condition():
    """This function should be included when evaluation_date_str is NOT present."""
    assert evaluation_date_use_other_info.include_if_no_input_present == [
        "evaluation_date_str"
    ]
    assert not evaluation_date_use_other_info.include_if_all_inputs_present
    assert not evaluation_date_use_other_info.include_if_any_input_present


def test_evaluation_date_use_other_info_condition_satisfied_when_no_eval_date_str():
    """Condition is satisfied when evaluation_date_str is not in input names."""
    result = evaluation_date_use_other_info.include_condition_satisfied(
        ["backend", "policy_date_str"]
    )
    assert result is True


def test_evaluation_date_use_other_info_condition_not_satisfied_when_eval_date_str():
    """Condition is not satisfied when evaluation_date_str is in input names."""
    result = evaluation_date_use_other_info.include_condition_satisfied(
        ["backend", "evaluation_date_str"]
    )
    assert result is False


# =============================================================================
# evaluation_date_from_evaluation_date_str tests
# =============================================================================
def test_evaluation_date_from_str_is_input_dependent():
    assert isinstance(
        evaluation_date_from_evaluation_date_str, InputDependentInterfaceFunction
    )


def test_evaluation_date_from_str_returns_date():
    result = evaluation_date_from_evaluation_date_str("2024-06-15")
    assert result == datetime.date(2024, 6, 15)


def test_evaluation_date_from_str_has_correct_leaf_name():
    assert evaluation_date_from_evaluation_date_str.leaf_name == "evaluation_date"


def test_evaluation_date_from_str_in_top_level_namespace():
    assert evaluation_date_from_evaluation_date_str.in_top_level_namespace is True


def test_evaluation_date_from_str_include_condition():
    """This function should be included when evaluation_date_str IS present."""
    assert evaluation_date_from_evaluation_date_str.include_if_all_inputs_present == [
        "evaluation_date_str"
    ]
    assert not evaluation_date_from_evaluation_date_str.include_if_no_input_present
    assert not evaluation_date_from_evaluation_date_str.include_if_any_input_present


def test_evaluation_date_from_str_condition_satisfied_when_eval_date_str_present():
    """Condition is satisfied when evaluation_date_str is in input names."""
    result = evaluation_date_from_evaluation_date_str.include_condition_satisfied(
        ["backend", "evaluation_date_str"]
    )
    assert result is True


def test_evaluation_date_from_str_condition_not_satisfied_when_no_eval_date_str():
    """Condition is not satisfied when evaluation_date_str is not in input names."""
    result = evaluation_date_from_evaluation_date_str.include_condition_satisfied(
        ["backend", "policy_date_str"]
    )
    assert result is False


# =============================================================================
# Input definitions tests
# =============================================================================
def test_policy_date_str_is_interface_input():
    from ttsim.interface_dag_elements.dates import policy_date_str

    assert isinstance(policy_date_str, InterfaceInput)
    assert policy_date_str.in_top_level_namespace is True


def test_evaluation_date_str_is_interface_input():
    from ttsim.interface_dag_elements.dates import evaluation_date_str

    assert isinstance(evaluation_date_str, InterfaceInput)
    assert evaluation_date_str.in_top_level_namespace is True


# =============================================================================
# Mutual exclusivity tests
# =============================================================================
def test_evaluation_date_functions_are_mutually_exclusive():
    """Only one evaluation_date function can be included at a time."""
    # When evaluation_date_str is present
    inputs_with_eval_str = ["backend", "evaluation_date_str", "policy_date_str"]

    use_other_satisfied = evaluation_date_use_other_info.include_condition_satisfied(
        inputs_with_eval_str
    )
    from_str_satisfied = (
        evaluation_date_from_evaluation_date_str.include_condition_satisfied(
            inputs_with_eval_str
        )
    )

    # Only from_str should be satisfied
    assert from_str_satisfied is True
    assert use_other_satisfied is False

    # When evaluation_date_str is NOT present
    inputs_without_eval_str = ["backend", "policy_date_str"]

    use_other_satisfied = evaluation_date_use_other_info.include_condition_satisfied(
        inputs_without_eval_str
    )
    from_str_satisfied = (
        evaluation_date_from_evaluation_date_str.include_condition_satisfied(
            inputs_without_eval_str
        )
    )

    # Only use_other should be satisfied
    assert use_other_satisfied is True
    assert from_str_satisfied is False
