from __future__ import annotations

import warnings

import pandas as pd
import pytest

from ttsim import main
from ttsim.column_objects_param_function import (
    policy_function,
)
from ttsim.fail_if import (
    FunctionsAndDataColumnsOverlapWarning,
)


@policy_function()
def some_func(p_id: int) -> int:
    return p_id


@policy_function()
def another_func(some_func: int) -> int:
    return some_func


def test_warn_if__functions_and_data_columns_overlap():
    with pytest.warns(FunctionsAndDataColumnsOverlapWarning):
        main(
            inputs={
                "input_data__tree": {
                    "p_id": pd.Series([0]),
                    "some_func": pd.Series([1]),
                },
                "policy_environment": {
                    "some_func": some_func,
                    "some_target": another_func,
                },
                "targets_tree": {"some_target": None},
                "rounding": False,
                # "jit": jit,
            },
            targets=["warn_if__functions_and_data_columns_overlap"],
        )


def test_warn_if__functions_and_columns_overlap_no_warning_if_no_overlap():
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=FunctionsAndDataColumnsOverlapWarning)
        main(
            inputs={
                "input_data__tree": {
                    "p_id": pd.Series([0]),
                    "x": pd.Series([1]),
                },
                "policy_environment": {"some_func": some_func},
                "targets_tree": {"some_func": None},
                "rounding": False,
                # "jit": jit,
            },
            targets=["warn_if__functions_and_data_columns_overlap"],
        )
