import pandas as pd
import pytest
from pandas._testing import assert_series_equal

from ttsim.compute_taxes_and_transfers import (
    _add_rounding_to_functions,
    _apply_rounding_spec,
    compute_taxes_and_transfers,
)
from ttsim.function_types import policy_function
from ttsim.policy_environment import PolicyEnvironment
from ttsim.rounding import RoundingDirection, RoundingSpec

rounding_specs_and_exp_results = [
    (
        RoundingSpec(base=1, direction=RoundingDirection.UP),
        [100.24, 100.78],
        [101.0, 101.0],
    ),
    (
        RoundingSpec(base=1, direction=RoundingDirection.DOWN),
        [100.24, 100.78],
        [100.0, 100.0],
    ),
    (
        RoundingSpec(base=1, direction=RoundingDirection.NEAREST),
        [100.24, 100.78],
        [100.0, 101.0],
    ),
    (
        RoundingSpec(base=5, direction=RoundingDirection.UP),
        [100.24, 100.78],
        [105.0, 105.0],
    ),
    (
        RoundingSpec(base=0.1, direction=RoundingDirection.DOWN),
        [100.24, 100.78],
        [100.2, 100.7],
    ),
    (
        RoundingSpec(base=0.001, direction=RoundingDirection.NEAREST),
        [100.24, 100.78],
        [100.24, 100.78],
    ),
    (
        RoundingSpec(base=1, direction=RoundingDirection.UP, to_add_after_rounding=10),
        [100.24, 100.78],
        [111.0, 111.0],
    ),
    (
        RoundingSpec(
            base=1, direction=RoundingDirection.DOWN, to_add_after_rounding=10
        ),
        [100.24, 100.78],
        [110.0, 110.0],
    ),
    (
        RoundingSpec(
            base=1, direction=RoundingDirection.NEAREST, to_add_after_rounding=10
        ),
        [100.24, 100.78],
        [110.0, 111.0],
    ),
]


def test_decorator():
    rs = RoundingSpec(base=1, direction=RoundingDirection.UP)

    @policy_function(rounding_spec=rs)
    def test_func():
        return 0

    assert test_func.rounding_spec == rs


def test_malformed_rounding_specs():
    with pytest.raises(KeyError):

        @policy_function(rounding_spec={"base": 1, "direction": "up"})
        def test_func():
            return 0

        PolicyEnvironment({"test_func": test_func})


@pytest.mark.parametrize(
    "base, direction, to_add_after_rounding",
    [
        (1, "upper", 0),
        ("0.1", RoundingDirection.DOWN, 0),
        (5, "closest", 0),
        (5, RoundingDirection.UP, "0"),
    ],
)
def test_rounding_specs_wrong_format(base, direction, to_add_after_rounding):
    with pytest.raises(ValueError):

        @policy_function(params_key_for_rounding="params_key_test")
        def test_func():
            return 0

        rounding_specs = {
            "params_key_test": {
                "rounding": {
                    "test_func": {
                        "base": base,
                        "direction": direction,
                        "to_add_after_rounding": to_add_after_rounding,
                    }
                }
            }
        }

        environment = PolicyEnvironment({"test_func": test_func}, rounding_specs)

        compute_taxes_and_transfers(
            data_tree={"p_id": pd.Series([1, 2])},
            environment=environment,
            targets_tree={"test_func": None},
        )


@pytest.mark.parametrize(
    "base, direction, to_add_after_rounding, input_values, exp_output",
    rounding_specs_and_exp_results,
)
def test_rounding(base, direction, to_add_after_rounding, input_values, exp_output):
    """Check if rounding is correct."""

    # Define function that should be rounded
    @policy_function(params_key_for_rounding="params_key_test")
    def test_func(income):
        return income

    data = {
        "p_id": pd.Series([1, 2]),
        "namespace": {"income": pd.Series(input_values)},
    }
    rounding_specs = {
        "params_key_test": {
            "rounding": {
                "namespace__test_func": {
                    "base": base,
                    "direction": direction,
                }
            }
        }
    }

    if to_add_after_rounding:
        rounding_specs["params_key_test"]["rounding"]["namespace__test_func"][
            "to_add_after_rounding"
        ] = to_add_after_rounding

    environment = PolicyEnvironment(
        {"namespace": {"test_func": test_func}}, rounding_specs
    )

    calc_result = compute_taxes_and_transfers(
        data_tree=data,
        environment=environment,
        targets_tree={"namespace": {"test_func": None}},
    )
    assert_series_equal(
        pd.Series(calc_result["namespace"]["test_func"]),
        pd.Series(exp_output),
        check_names=False,
    )


def test_rounding_with_time_conversion():
    """Check if rounding is correct for time-converted functions."""

    # Define function that should be rounded
    @policy_function(params_key_for_rounding="params_key_test")
    def test_func_m(income):
        return income

    data = {
        "p_id": pd.Series([1, 2]),
        "income": pd.Series([1.2, 1.5]),
    }
    rounding_specs = {
        "params_key_test": {
            "rounding": {
                "test_func_m": {
                    "base": 1,
                    "direction": "down",
                }
            }
        }
    }
    environment = PolicyEnvironment({"test_func_m": test_func_m}, rounding_specs)

    calc_result = compute_taxes_and_transfers(
        data_tree=data,
        environment=environment,
        targets_tree={"test_func_y": None},
    )
    assert_series_equal(
        pd.Series(calc_result["test_func_y"]),
        pd.Series([12.0, 12.0]),
        check_names=False,
    )


@pytest.mark.parametrize(
    """
    base,
    direction,
    to_add_after_rounding,
    input_values_exp_output,
    ignore_since_not_rounded
    """,
    rounding_specs_and_exp_results,
)
def test_no_rounding(
    base,
    direction,
    to_add_after_rounding,
    input_values_exp_output,
    ignore_since_not_rounded,  # noqa: ARG001
):
    # Define function that should be rounded
    @policy_function(params_key_for_rounding="params_key_test")
    def test_func(income):
        return income

    data = {"p_id": pd.Series([1, 2])}
    data["income"] = pd.Series(input_values_exp_output)
    rounding_specs = {
        "params_key_test": {
            "rounding": {"test_func": {"base": base, "direction": direction}}
        }
    }
    environment = PolicyEnvironment({"test_func": test_func}, rounding_specs)

    if to_add_after_rounding:
        rounding_specs["params_key_test"]["rounding"]["test_func"][
            "to_add_after_rounding"
        ] = to_add_after_rounding

    calc_result = compute_taxes_and_transfers(
        data_tree=data,
        environment=environment,
        targets_tree={"test_func": None},
        rounding=False,
    )
    assert_series_equal(
        pd.Series(calc_result["test_func"]),
        pd.Series(input_values_exp_output),
        check_names=False,
    )


@pytest.mark.parametrize(
    "base, direction, to_add_after_rounding, input_values, exp_output",
    rounding_specs_and_exp_results,
)
def test_rounding_callable(
    base, direction, to_add_after_rounding, input_values, exp_output
):
    """Check if callable is rounded correctly.

    Tests `_apply_rounding_spec` directly.
    """

    def test_func(income):
        return income

    func_with_rounding = _apply_rounding_spec(
        base=base,
        direction=direction,
        to_add_after_rounding=to_add_after_rounding if to_add_after_rounding else 0,
        name="test_func",
    )(test_func)

    assert_series_equal(
        func_with_rounding(pd.Series(input_values)),
        pd.Series(exp_output),
        check_names=False,
    )


@pytest.mark.parametrize(
    "params, match",
    [
        ({}, "Rounding specifications for function"),
        ({"eink_st": {}}, "Rounding specifications for function"),
        ({"eink_st": {"rounding": {}}}, "Rounding specifications for function"),
        (
            {"eink_st": {"rounding": {"eink_st_func": {}}}},
            "Both 'base' and 'direction' are expected",
        ),
    ],
)
def test_raise_if_missing_rounding_spec(params, match):
    @policy_function(params_key_for_rounding="eink_st")
    def eink_st_func(arg_1: float) -> float:
        return arg_1

    with pytest.raises(KeyError, match=match):
        _add_rounding_to_functions(
            functions={"eink_st_func": eink_st_func},
            params=params,
        )


@pytest.mark.parametrize(
    "base, direction, to_add_after_rounding, input_values, exp_output",
    rounding_specs_and_exp_results,
)
def test_rounding_spec(
    base, direction, to_add_after_rounding, input_values, exp_output
):
    """Test RoundingSpec directly."""

    def test_func(income):
        return income

    to_add = to_add_after_rounding if to_add_after_rounding is not None else 0
    rounding_spec = RoundingSpec(
        base=base,
        direction=direction,
        to_add_after_rounding=to_add,
    )

    rounded_func = rounding_spec.apply_rounding(test_func)
    result = rounded_func(pd.Series(input_values))

    assert_series_equal(
        pd.Series(result),
        pd.Series(exp_output),
        check_names=False,
    )


@pytest.mark.parametrize(
    "base, direction, to_add_after_rounding",
    [
        (1, "upper", 0),
        ("0.1", RoundingDirection.DOWN, 0),
        (5, "closest", 0),
        (5, RoundingDirection.UP, "0"),
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
