"""Some tests for the policy_environment module."""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

import optree
import pandas as pd
import pytest

from ttsim import OrigPolicyObjects, main
from ttsim.interface_dag_elements.orig_policy_objects import (
    column_objects_and_param_functions,
)
from ttsim.interface_dag_elements.policy_environment import (
    _active_column_objects_and_param_functions,
    _active_param_objects,
    _get_param_value,
    _get_param_value_piecewise,
)
from ttsim.tt import ScalarParam, policy_function

if TYPE_CHECKING:
    from types import ModuleType

    from ttsim.typing import (
        NestedColumnObjectsParamFunctions,
    )

from mettsim import middle_earth


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
        start_date=datetime.date(2025, 1, 1),
        end_date=datetime.date(2025, 12, 31),
        name={"de": "Some int param", "en": "Some int param"},
        description={"de": "Some int param", "en": "Some int param"},
        unit=None,
        reference_period=None,
        note=None,
        reference=None,
    )


def test_add_jahresanfang(xnp: ModuleType):
    spec = {
        "name": {"de": "Test", "en": "Check"},
        "description": {"de": "Nichts zu sehen", "en": "Nothing to do"},
        "type": "scalar",
        "add_jahresanfang": True,
        datetime.date(2020, 1, 1): {"value": 1},
        datetime.date(2020, 7, 1): {"value": 2},
    }
    _active_ttsim_tree_with_params = _active_param_objects(
        orig={("spam.yaml", "foo"): spec},  # ty: ignore[invalid-argument-type]
        policy_date=pd.to_datetime("2020-07-01").date(),
        xnp=xnp,
    )
    assert _active_ttsim_tree_with_params["foo"].value == 2
    assert _active_ttsim_tree_with_params["foo_jahresanfang"].value == 1


def test_input_is_recognized_as_potential_group_id(backend):
    assert "kin" in main(
        main_target="labels__grouping_levels",
        orig_policy_objects=OrigPolicyObjects.root(middle_earth.ROOT_PATH),
        policy_date=datetime.date(2020, 1, 1),
        backend=backend,
    )


def test_p_id_not_recognized_as_potential_group_id(backend):
    assert "p" not in main(
        main_target="labels__grouping_levels",
        orig_policy_objects=OrigPolicyObjects.root(middle_earth.ROOT_PATH),
        policy_date=datetime.date(2020, 1, 1),
        backend=backend,
    )


@pytest.mark.parametrize(
    ("date_string", "expected"),
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
    with pytest.raises(
        ValueError,
        match=r"neither matches the format YYYY-MM-DD nor is a datetime.date",
    ):

        @policy_function(start_date=date_string)
        def test_func():
            pass


def test_start_date_missing():
    @policy_function()
    def test_func():
        pass

    assert test_func.start_date == datetime.date(1900, 1, 1)


@pytest.mark.parametrize(
    ("date_string", "expected"),
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
    with pytest.raises(
        ValueError,
        match=r"neither matches the format YYYY-MM-DD nor is a datetime.date",
    ):

        @policy_function(end_date=date_string)
        def test_func():
            pass


def test_end_date_missing():
    @policy_function()
    def test_func():
        pass

    assert test_func.end_date == datetime.date(2099, 12, 31)


def test_active_period_is_empty():
    with pytest.raises(ValueError, match="must be before the end date"):

        @policy_function(start_date="2023-01-20", end_date="2023-01-19")
        def test_func():
            pass


@pytest.mark.parametrize(
    ("tree", "last_day", "function_name_last_day", "function_name_next_day"),
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
    orig = column_objects_and_param_functions(root=middle_earth.ROOT_PATH)
    functions_last_day = _active_column_objects_and_param_functions(
        orig=orig,
        policy_date=last_day,
    )
    functions_next_day = _active_column_objects_and_param_functions(
        orig=orig,
        policy_date=last_day + datetime.timedelta(days=1),
    )

    accessor = optree.tree_accessors(tree, none_is_leaf=True)[0]  # ty: ignore[invalid-argument-type]

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


def test_get_param_value_piecewise_no_updates():
    specs = [
        {
            "reference": "some ref",
            "intervals": [
                {"interval": "[0, 100)", "slope": 0.1, "intercept": 0},
                {"interval": "[100, inf)", "slope": 0.2, "intercept": 10},
            ],
        },
    ]
    result = _get_param_value_piecewise(specs)
    assert len(result) == 2
    assert result[0]["slope"] == 0.1
    assert result[1]["slope"] == 0.2


def test_get_param_value_piecewise_with_updates_previous():
    specs = [
        {
            "reference": "base ref",
            "intervals": [
                {"interval": "[0, 16956)", "intercept": 0, "slope": 0},
                {"interval": "[16956, 31528)", "slope": 0.119},
                {"interval": "[31528, inf)", "slope": 0.055},
            ],
        },
        {
            "updates_previous": True,
            "reference": "update ref",
            "intervals": [
                {"interval": "[0, 17543)"},
                {"interval": "[17543, 32619)", "slope": 0.119},
            ],
        },
    ]
    result = _get_param_value_piecewise(specs)
    assert len(result) == 3
    # First interval: no coefficients specified, so none inherited
    assert "intercept" not in result[0]
    assert "slope" not in result[0]
    # Second interval: explicit slope
    assert result[1]["slope"] == 0.119
    # Third interval: carried over from base, trimmed
    assert result[2]["slope"] == 0.055
    assert "32619" in result[2]["interval"]


def test_get_param_value_piecewise_updates_previous_on_first_raises():
    specs = [
        {
            "updates_previous": True,
            "intervals": [
                {"interval": "[0, 100)", "slope": 0.1},
            ],
        },
    ]
    with pytest.raises(ValueError, match="updates_previous"):
        _get_param_value_piecewise(specs)


def test_bare_list_format_raises_type_error(xnp: ModuleType):
    spec = {
        "name": {"de": "Test", "en": "Test"},
        "description": {"de": "Test", "en": "Test"},
        "type": "piecewise_linear",
        datetime.date(2020, 1, 1): [
            {"interval": "[0, inf)", "slope": 0.1, "intercept": 0},
        ],
    }
    with pytest.raises(TypeError, match="Bare list format is no longer supported"):
        _active_param_objects(
            orig={("spam.yaml", "foo"): spec},  # ty: ignore[invalid-argument-type]
            policy_date=datetime.date(2020, 7, 1),
            xnp=xnp,
        )
