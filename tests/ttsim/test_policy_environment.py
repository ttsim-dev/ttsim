"""Some tests for the policy_environment module."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import optree
import pandas as pd
import pytest
from mettsim.config import METTSIM_ROOT

from ttsim import (
    ScalarParam,
    main,
    policy_function,
)
from ttsim.orig_policy_objects import (
    column_objects_and_param_functions,
    param_specs,
)
from ttsim.policy_environment import (
    _active_column_objects_and_param_functions,
    _active_param_objects,
    _get_param_value,
)

if TYPE_CHECKING:
    from ttsim.tt_dag_elements.typing import (
        NestedColumnObjectsParamFunctions,
    )


@pytest.fixture(scope="module")
def some_params_spec_with_updates_previous():
    return [
        {
            "a": 1,
            "b": 2,
        },
        {
            "updates_previous": True,
            "b": 4,
        },
    ]


@pytest.fixture(scope="module")
def some_int_param():
    return ScalarParam(
        value=1,
        leaf_name="some_int_param",
        start_date="2025-01-01",
        end_date="2025-12-31",
        name="some_int_param",
        description="Some int param",
        unit=None,
        reference_period=None,
        note=None,
        reference=None,
    )


def test_add_jahresanfang():
    _orig_tree_with_params = param_specs(root=Path(__file__).parent / "test_parameters")
    k = ("test_add_jahresanfang.yaml", "foo")
    _active_ttsim_tree_with_params = _active_param_objects(
        orig={k: _orig_tree_with_params[k]},
        date=pd.to_datetime("2020-07-01").date(),
    )
    assert _active_ttsim_tree_with_params["foo"].value == 2
    assert _active_ttsim_tree_with_params["foo_jahresanfang"].value == 1


def test_input_is_recognized_as_potential_group_id():
    grouping_levels = main(
        inputs={
            "root": METTSIM_ROOT,
            "date": datetime.date(2020, 1, 1),
        },
        targets=["grouping_levels"],
    )["grouping_levels"]
    assert "kin" in grouping_levels


def test_p_id_not_recognized_as_potential_group_id():
    grouping_levels = main(
        inputs={
            "root": METTSIM_ROOT,
            "date": datetime.date(2020, 1, 1),
        },
        targets=["grouping_levels"],
    )["grouping_levels"]
    assert "p" not in grouping_levels


@pytest.mark.parametrize(
    "date_string, expected",
    [
        ("2023-01-20", datetime.date(2023, 1, 20)),
    ],
)
def test_start_date_valid(date_string: str, expected: datetime.date):
    @policy_function(start_date=date_string)
    def test_func():
        pass

    assert test_func.start_date == expected


@pytest.mark.parametrize(
    "date_string",
    [
        "20230120",
        "20.1.2023",
        "20th January 2023",
    ],
)
def test_start_date_invalid(date_string: str):
    with pytest.raises(ValueError):

        @policy_function(start_date=date_string)
        def test_func():
            pass


def test_start_date_missing():
    @policy_function()
    def test_func():
        pass

    assert test_func.start_date == datetime.date(1900, 1, 1)


@pytest.mark.parametrize(
    "date_string, expected",
    [
        ("2023-01-20", datetime.date(2023, 1, 20)),
    ],
)
def test_end_date_valid(date_string: str, expected: datetime.date):
    @policy_function(end_date=date_string)
    def test_func():
        pass

    assert test_func.end_date == expected


@pytest.mark.parametrize(
    "date_string",
    [
        "20230120",
        "20.1.2023",
        "20th January 2023",
    ],
)
def test_end_date_invalid(date_string: str):
    with pytest.raises(ValueError):

        @policy_function(end_date=date_string)
        def test_func():
            pass


def test_end_date_missing():
    @policy_function()
    def test_func():
        pass

    assert test_func.end_date == datetime.date(2099, 12, 31)


def test_active_period_is_empty():
    with pytest.raises(ValueError):

        @policy_function(start_date="2023-01-20", end_date="2023-01-19")
        def test_func():
            pass


@pytest.mark.parametrize(
    "tree, last_day, function_name_last_day, function_name_next_day",
    [
        (
            {"housing_benefits": {"eligibility": {"requirement_fulfilled_fam": None}}},
            datetime.date(2019, 12, 31),
            "requirement_fulfilled_fam_not_considering_children",
            "requirement_fulfilled_fam_considering_children",
        ),
    ],
)
def test_active_tree_with_column_objects_and_param_functions(
    tree: NestedColumnObjectsParamFunctions,
    last_day: datetime.date,
    function_name_last_day: str,
    function_name_next_day: str,
):
    _orig_tree_with_column_objects_and_param_functions = (
        column_objects_and_param_functions(root=METTSIM_ROOT)
    )
    functions_last_day = _active_column_objects_and_param_functions(
        orig=_orig_tree_with_column_objects_and_param_functions,
        date=last_day,
    )
    functions_next_day = _active_column_objects_and_param_functions(
        orig=_orig_tree_with_column_objects_and_param_functions,
        date=last_day + datetime.timedelta(days=1),
    )

    accessor = optree.tree_accessors(tree, none_is_leaf=True)[0]

    assert accessor(functions_last_day).__name__ == function_name_last_day
    assert accessor(functions_next_day).__name__ == function_name_next_day


def test_get_params_contents_with_updated_previous(
    some_params_spec_with_updates_previous,
):
    params_contents = _get_param_value(some_params_spec_with_updates_previous)
    expected = {
        "a": 1,
        "b": 4,
    }
    assert params_contents == expected
