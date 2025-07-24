from __future__ import annotations

import datetime
import warnings

import pandas as pd
import pytest

from ttsim import main
from ttsim.main_target import MainTarget
from ttsim.tt.column_objects_param_function import policy_function
from ttsim.tt.param_objects import ScalarParam


@policy_function()
def some_func(p_id: int) -> int:
    return p_id


@policy_function()
def another_func(some_func: int) -> int:
    return some_func


def test_warn_if_functions_and_data_columns_overlap(backend):
    with pytest.warns(match="Your data provides the column:"):
        main(
            main_target="warn_if__functions_and_data_columns_overlap",
            input_data={
                "tree": {
                    "p_id": pd.Series([0]),
                    "some_func": pd.Series([1]),
                }
            },
            policy_environment={
                "some_func": some_func,
                "another_func": another_func,
            },
            tt_targets={"tree": {"another_func": None}},
            evaluation_date=datetime.date(2025, 1, 1),
            rounding=False,
            include_fail_nodes=False,
            backend=backend,
        )


def test_warn_if_functions_and_columns_overlap_no_warning_if_no_overlap(backend):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        main(
            main_target="warn_if__functions_and_data_columns_overlap",
            input_data={
                "tree": {
                    "p_id": pd.Series([0]),
                    "x": pd.Series([1]),
                }
            },
            policy_environment={"some_func": some_func},
            tt_targets={"tree": {"some_func": None}},
            policy_date=datetime.date(2025, 1, 1),
            rounding=False,
            include_fail_nodes=False,
            backend=backend,
        )
        assert not w, f"Expected no warning, but got at least: {w[0].message}"


def test_warn_if_evaluation_date_set_in_multiple_places(backend):
    policy_environment = {
        "policy_year": ScalarParam(value=2025, leaf_name="policy_year"),
        "policy_month": ScalarParam(value=1, leaf_name="policy_month"),
        "policy_day": ScalarParam(value=1, leaf_name="policy_day"),
        "evaluation_year": ScalarParam(value=2025, leaf_name="evaluation_year"),
        "evaluation_month": ScalarParam(value=1, leaf_name="evaluation_month"),
        "evaluation_day": ScalarParam(value=1, leaf_name="evaluation_day"),
        "some_func": some_func,
        "another_func": another_func,
    }
    with pytest.warns(match="You have specified the evaluation date in more than one"):
        main(
            main_target="warn_if__evaluation_date_set_in_multiple_places",
            policy_environment=policy_environment,
            evaluation_date=datetime.date(2025, 1, 1),
            backend=backend,
        )


def test_warn_if_evaluation_date_set_in_multiple_places_implicitly_added(backend, xnp):
    policy_environment = {
        "policy_year": ScalarParam(value=2025, leaf_name="policy_year"),
        "policy_month": ScalarParam(value=1, leaf_name="policy_month"),
        "policy_day": ScalarParam(value=1, leaf_name="policy_day"),
        "evaluation_year": ScalarParam(value=2025, leaf_name="evaluation_year"),
        "evaluation_month": ScalarParam(value=1, leaf_name="evaluation_month"),
        "evaluation_day": ScalarParam(value=1, leaf_name="evaluation_day"),
        "some_func": some_func,
        "another_func": another_func,
    }
    with pytest.warns(match="You have specified the evaluation date in more than one"):
        main(
            main_target=MainTarget.raw_results.columns,
            policy_environment=policy_environment,
            evaluation_date=datetime.date(2025, 1, 1),
            processed_data={"p_id": xnp.array([0])},
            backend=backend,
        )


def test_do_not_need_to_warn_if_evaluation_date_is_set_only_once(backend, xnp):
    policy_environment = {
        "policy_year": ScalarParam(value=2025, leaf_name="policy_year"),
        "policy_month": ScalarParam(value=1, leaf_name="policy_month"),
        "policy_day": ScalarParam(value=1, leaf_name="policy_day"),
        "some_func": some_func,
        "another_func": another_func,
    }
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        main(
            main_target=MainTarget.raw_results.columns,
            policy_environment=policy_environment,
            evaluation_date=datetime.date(2025, 1, 1),
            processed_data={"p_id": xnp.array([0])},
            backend=backend,
        )
        assert not w, f"Expected no warning, but got at least: {w[0].message}"
