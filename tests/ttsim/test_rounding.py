from __future__ import annotations

import pandas as pd
import pytest
from pandas._testing import assert_series_equal

from ttsim import (
    RoundingSpec,
    main,
    policy_environment,
    policy_function,
    policy_input,
)
from ttsim.config import IS_JAX_INSTALLED
from ttsim.config import numpy_or_jax as np

if IS_JAX_INSTALLED:
    DTYPE = "float32"
else:
    DTYPE = "float64"


@policy_input()
def x() -> int:
    pass


@policy_input()
def p_id() -> int:
    pass


rounding_specs_and_exp_results = [
    (
        RoundingSpec(base=1, direction="up"),
        np.array([100.24, 100.78]),
        np.array([101.0, 101.0]),
    ),
    (
        RoundingSpec(base=1, direction="down"),
        np.array([100.24, 100.78]),
        np.array([100.0, 100.0]),
    ),
    (
        RoundingSpec(base=1, direction="nearest"),
        np.array([100.24, 100.78]),
        np.array([100.0, 101.0]),
    ),
    (
        RoundingSpec(base=5, direction="up"),
        np.array([100.24, 100.78]),
        np.array([105.0, 105.0]),
    ),
    (
        RoundingSpec(base=0.1, direction="down"),
        np.array([100.24, 100.78]),
        np.array([100.2, 100.7]),
    ),
    (
        RoundingSpec(base=0.001, direction="nearest"),
        np.array([100.24, 100.78]),
        np.array([100.24, 100.78]),
    ),
    (
        RoundingSpec(base=1, direction="up", to_add_after_rounding=10),
        np.array([100.24, 100.78]),
        np.array([111.0, 111.0]),
    ),
    (
        RoundingSpec(base=1, direction="down", to_add_after_rounding=10),
        np.array([100.24, 100.78]),
        np.array([110.0, 110.0]),
    ),
    (
        RoundingSpec(base=1, direction="nearest", to_add_after_rounding=10),
        np.array([100.24, 100.78]),
        np.array([110.0, 111.0]),
    ),
]


def test_decorator():
    rs = RoundingSpec(base=1, direction="up")

    @policy_function(rounding_spec=rs)
    def test_func():
        return 0

    assert test_func.rounding_spec == rs


def test_malformed_rounding_specs():
    with pytest.raises(AssertionError):

        @policy_function(rounding_spec={"base": 1, "direction": "updsf"})
        def test_func():
            return 0

        policy_environment(
            active_tree_with_column_objects_and_param_functions={
                "x.py": {"test_func": test_func}
            },
        )


@pytest.mark.parametrize(
    "rounding_spec, input_values, exp_output",
    rounding_specs_and_exp_results,
)
def test_rounding(rounding_spec, input_values, exp_output):
    """Check if rounding is correct."""

    # Define function that should be rounded
    @policy_function(rounding_spec=rounding_spec)
    def test_func(x):
        return x

    input_data__tree = {
        "p_id": np.array([1, 2]),
        "namespace": {"x": np.array(input_values)},
    }
    policy_environment = {"namespace": {"test_func": test_func, "x": x}, "p_id": p_id}

    nested_results = main(
        inputs={
            "input_data__tree": input_data__tree,
            "policy_environment": policy_environment,
            "targets_tree": {"namespace": {"test_func": None}},
            "rounding": True,
        },
        targets=["nested_results"],
    )["nested_results"]
    assert_series_equal(
        pd.Series(nested_results["namespace"]["test_func"]),
        pd.Series(exp_output, dtype=DTYPE),
        check_names=False,
    )


def test_rounding_with_time_conversion():
    """Check if rounding is correct for time-converted functions."""

    # Define function that should be rounded
    @policy_function(rounding_spec=RoundingSpec(base=1, direction="down"))
    def test_func_m(x):
        return x

    data = {
        "p_id": np.array([1, 2]),
        "x": np.array([1.2, 1.5]),
    }

    policy_environment = {
        "test_func_m": test_func_m,
        "x": x,
        "p_id": p_id,
    }

    nested_results = main(
        inputs={
            "input_data__tree": data,
            "policy_environment": policy_environment,
            "targets_tree": {"test_func_y": None},
            "rounding": True,
        },
        targets=["nested_results"],
    )["nested_results"]
    assert_series_equal(
        pd.Series(nested_results["test_func_y"]),
        pd.Series([12.0, 12.0], dtype=DTYPE),
        check_names=False,
    )


@pytest.mark.parametrize(
    "rounding_spec, input_values_exp_output, ignore_since_no_rounding",
    rounding_specs_and_exp_results,
)
def test_no_rounding(
    rounding_spec,
    input_values_exp_output,
    ignore_since_no_rounding,  # noqa: ARG001
):
    # Define function that should be rounded
    @policy_function(rounding_spec=rounding_spec)
    def test_func(x):
        return x

    data = {"p_id": np.array([1, 2])}
    data["x"] = np.array(input_values_exp_output)
    policy_environment = {
        "test_func": test_func,
        "x": x,
        "p_id": p_id,
    }

    nested_results = main(
        inputs={
            "input_data__tree": data,
            "policy_environment": policy_environment,
            "targets_tree": {"test_func": None},
            "rounding": False,
        },
        targets=["nested_results"],
    )["nested_results"]
    assert_series_equal(
        pd.Series(nested_results["test_func"]),
        pd.Series(input_values_exp_output, dtype=DTYPE),
        check_names=False,
    )


@pytest.mark.parametrize(
    "rounding_spec, input_values, exp_output",
    rounding_specs_and_exp_results,
)
def test_rounding_callable(rounding_spec, input_values, exp_output):
    """Check if callable is rounded correctly."""

    def test_func(income):
        return income

    func_with_rounding = rounding_spec.apply_rounding(test_func)

    assert_series_equal(
        pd.Series(func_with_rounding(input_values)),
        pd.Series(exp_output),
        check_names=False,
    )


@pytest.mark.parametrize(
    "rounding_spec, input_values, exp_output",
    rounding_specs_and_exp_results,
)
def test_rounding_spec(rounding_spec, input_values, exp_output):
    """Test RoundingSpec directly."""

    def test_func(income):
        return income

    rounded_func = rounding_spec.apply_rounding(test_func)
    result = rounded_func(input_values)

    assert_series_equal(
        pd.Series(result),
        pd.Series(exp_output),
        check_names=False,
    )


@pytest.mark.parametrize(
    "base, direction, to_add_after_rounding",
    [
        (1, "upper", 0),
        ("0.1", "down", 0),
        (5, "closest", 0),
        (5, "up", "0"),
    ],
)
def test_rounding_spec_validation(base, direction, to_add_after_rounding):
    """Test validation of RoundingSpec parameters."""
    with pytest.raises(ValueError):
        RoundingSpec(
            base=base,
            direction=direction,
            to_add_after_rounding=to_add_after_rounding,
        )
