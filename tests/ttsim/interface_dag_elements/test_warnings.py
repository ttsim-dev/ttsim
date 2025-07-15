from __future__ import annotations

import datetime
import warnings

import pandas as pd

from ttsim import main
from ttsim.interface_dag_elements.warn_if import FunctionsAndDataColumnsOverlapWarning
from ttsim.tt_dag_elements.column_objects_param_function import policy_function


@policy_function()
def some_func(p_id: int) -> int:
    return p_id


@policy_function()
def another_func(some_func: int) -> int:
    return some_func


def test_warn_if_functions_and_data_columns_overlap(backend):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        main(
            input_data={
                "tree": {
                    "p_id": pd.Series([0]),
                    "some_func": pd.Series([1]),
                }
            },
            policy_environment={
                "some_func": some_func,
                "some_target": another_func,
            },
            tt_targets={"tree": {"some_target": None}},
            date=datetime.date(2025, 1, 1),
            rounding=False,
            backend=backend,
            main_target="results__df_with_nested_columns",
        )
        # Check that we got exactly one warning
        assert len(w) == 1
        # Check that it's the right type of warning
        assert w[0].category.__name__ == "FunctionsAndDataColumnsOverlapWarning"


def test_warn_if_functions_and_columns_overlap_no_warning_if_no_overlap(backend):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            category=FunctionsAndDataColumnsOverlapWarning,
        )
        main(
            input_data={
                "tree": {
                    "p_id": pd.Series([0]),
                    "x": pd.Series([1]),
                }
            },
            policy_environment={"some_func": some_func},
            tt_targets={"tree": {"some_func": None}},
            date=datetime.date(2025, 1, 1),
            rounding=False,
            backend=backend,
            main_target="results__df_with_nested_columns",
        )
