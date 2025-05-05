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
    GroupCreationFunction,
    PolicyEnvironment,
    group_creation_function,
    policy_function,
    set_up_policy_environment,
)
from ttsim.policy_environment import (
    ConflictingTimeDependentObjectsError,
    _fail_if_name_of_last_branch_element_not_leaf_name_of_function,
    _load_parameter_group_from_yaml,
    active_ttsim_objects_tree,
    fail_because_of_clashes,
)

if TYPE_CHECKING:
    from ttsim.typing import FlatTTSIMObjectDict, NestedTTSIMObjectDict

YAML_PATH = Path(__file__).parent / "test_parameters"


def return_one():
    return 1


def return_two():
    return 2


def return_three():
    return 3


@group_creation_function()
def fam_id() -> int:
    pass


class TestPolicyEnvironment:
    def test_func_exists_in_tree(self):
        function = policy_function(leaf_name="foo")(return_one)
        environment = PolicyEnvironment({"foo": function})

        assert environment.raw_objects_tree["foo"] == function

    def test_func_does_not_exist_in_tree(self):
        environment = PolicyEnvironment({}, {})

        assert "foo" not in environment.raw_objects_tree

    @pytest.mark.parametrize(
        "environment",
        [
            PolicyEnvironment({}, {}),
            PolicyEnvironment({"foo": policy_function(leaf_name="foo")(return_one)}),
            PolicyEnvironment(
                {
                    "foo": policy_function(leaf_name="foo")(return_one),
                    "bar": policy_function(leaf_name="bar")(return_two),
                }
            ),
        ],
    )
    def test_upsert_functions(self, environment: PolicyEnvironment):
        new_function = policy_function(leaf_name="foo")(return_three)
        new_environment = environment.upsert_objects({"foo": new_function})

        assert new_environment.raw_objects_tree["foo"] == new_function

    @pytest.mark.parametrize(
        "environment",
        [
            PolicyEnvironment({}, {}),
            PolicyEnvironment({}, {"foo": {"bar": 1}}),
        ],
    )
    def test_replace_all_parameters(self, environment: PolicyEnvironment):
        new_params = {"foo": {"bar": 2}}
        new_environment = environment.replace_all_parameters(new_params)

        assert new_environment.params == new_params


def test_leap_year_correctly_handled():
    set_up_policy_environment(date="2020-02-29", root=METTSIM_ROOT)


def test_fail_if_invalid_date():
    with pytest.raises(ValueError):
        set_up_policy_environment(date="2020-02-30", root=METTSIM_ROOT)


def test_fail_if_invalid_access_different_date():
    with pytest.raises(ValueError):
        _load_parameter_group_from_yaml(
            date=pd.to_datetime("01-01-2020").date(),
            group="invalid_access_diff_date",
            parameters=None,
            yaml_path=YAML_PATH,
        )


def test_access_different_date_vorjahr():
    params = _load_parameter_group_from_yaml(
        date=pd.to_datetime("01-01-2020").date(),
        group="test_access_diff_date_vorjahr",
        parameters=None,
        yaml_path=YAML_PATH,
    )
    assert params["foo"] == 2020
    assert params["foo_vorjahr"] == 2019


def test_access_different_date_jahresanfang():
    params = _load_parameter_group_from_yaml(
        date=pd.to_datetime("07-01-2020").date(),
        group="test_access_diff_date_jahresanfang",
        parameters=None,
        yaml_path=YAML_PATH,
    )
    assert params["foo"] == 2021
    assert params["foo_jahresanfang"] == 2020


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
def test_load_functions_tree_for_date(
    tree: NestedTTSIMObjectDict,
    last_day: datetime.date,
    function_name_last_day: str,
    function_name_next_day: str,
):
    functions_last_day = active_ttsim_objects_tree(root=METTSIM_ROOT, date=last_day)
    functions_next_day = active_ttsim_objects_tree(
        root=METTSIM_ROOT, date=last_day + datetime.timedelta(days=1)
    )

    accessor = optree.tree_accessors(tree, none_is_leaf=True)[0]

    assert accessor(functions_last_day).__name__ == function_name_last_day
    assert accessor(functions_next_day).__name__ == function_name_next_day


@pytest.mark.parametrize(
    "functions_tree",
    [
        {"foo": policy_function(leaf_name="bar")(return_one)},
    ],
)
def test_fail_if_name_of_last_branch_element_not_leaf_name_of_function(
    functions_tree: NestedTTSIMObjectDict,
):
    with pytest.raises(KeyError):
        _fail_if_name_of_last_branch_element_not_leaf_name_of_function(functions_tree)


def test_dont_destroy_group_by_functions():
    functions_tree = {
        "foo": group_creation_function()(return_one),
    }
    environment = PolicyEnvironment(functions_tree)
    assert isinstance(environment.raw_objects_tree["foo"], GroupCreationFunction)


def test_creating_environment_fails_when_group_ids_are_outside_top_level_namespace():
    with pytest.raises(
        ValueError, match="Group identifiers must live in the top-level namespace. Got:"
    ):
        PolicyEnvironment({"n1": {"fam_id": fam_id}})


def test_upserting_group_ids_outside_top_level_namespace_fails():
    with pytest.raises(
        ValueError, match="Group identifiers must live in the top-level namespace. Got:"
    ):
        PolicyEnvironment({}).upsert_objects({"n1": {"fam_id": fam_id}})


def test_input_is_recognized_as_potential_group_id():
    environment = set_up_policy_environment(root=METTSIM_ROOT, date="2020-01-01")
    assert "kin" in environment.grouping_levels


def test_p_id_not_recognized_as_potential_group_id():
    environment = set_up_policy_environment(root=METTSIM_ROOT, date="2020-01-01")
    assert "p" not in environment.grouping_levels


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


# End date -------------------------------------------------


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

    assert test_func.end_date == datetime.date(2100, 12, 31)


# Change name ----------------------------------------------


def test_dates_active_change_name_given():
    @policy_function(leaf_name="renamed_func")
    def test_func():
        pass

    assert test_func.leaf_name == "renamed_func"


def test_dates_active_change_name_missing():
    @policy_function()
    def test_func():
        pass

    assert test_func.leaf_name == "test_func"


# Empty interval -------------------------------------------


def test_dates_active_empty_interval():
    with pytest.raises(ValueError):

        @policy_function(start_date="2023-01-20", end_date="2023-01-19")
        def test_func():
            pass


# Conflicts ------------------------------------------------


def identity(x):
    return x


@pytest.mark.parametrize(
    "orig_ttsim_objects_tree",
    [
        # Same global module, no overlapping periods.
        {
            ("a",): policy_function(
                start_date="2023-01-01",
                end_date="2023-01-31",
                leaf_name="f",
            )(identity),
            ("b",): policy_function(
                start_date="2023-02-01",
                end_date="2023-02-28",
                leaf_name="f",
            )(identity),
        },
        # Same submodule, no overlapping periods.
        {
            ("c", "a"): policy_function(
                start_date="2023-01-01",
                end_date="2023-01-31",
                leaf_name="f",
            )(identity),
            ("c", "b"): policy_function(
                start_date="2023-01-01",
                end_date="2023-02-28",
                leaf_name="g",
            )(identity),
        },
        # Different modules, no overlapping periods.
        {
            ("c", "f"): policy_function(
                start_date="2023-01-01",
                end_date="2023-01-31",
            )(identity),
            ("d", "f"): policy_function(
                start_date="2023-02-01",
                end_date="2023-02-28",
            )(identity),
        },
        # Different paths, overlapping periods.
        {
            ("x", "c", "a"): policy_function(
                start_date="2023-01-01",
                end_date="2023-01-31",
                leaf_name="f",
            )(identity),
            ("y", "c", "b"): policy_function(
                start_date="2023-01-01",
                end_date="2023-02-28",
                leaf_name="g",
            )(identity),
        },
    ],
)
def test_dates_active_no_conflicts(orig_ttsim_objects_tree):
    fail_because_of_clashes(
        orig_ttsim_objects_tree=orig_ttsim_objects_tree,
        orig_yaml_tree={},
    )


@pytest.mark.parametrize(
    "orig_ttsim_objects_tree",
    [
        # Exact overlap.
        {
            ("a",): policy_function(
                start_date="2023-01-01",
                end_date="2023-01-31",
                leaf_name="f",
            )(identity),
            ("b",): policy_function(
                start_date="2023-01-01",
                end_date="2023-01-31",
                leaf_name="f",
            )(identity),
        },
        # Active period for "a" is subset of "b".
        {
            ("a"): policy_function(
                start_date="2023-01-01",
                end_date="2023-01-31",
                leaf_name="f",
            )(identity),
            ("b"): policy_function(
                start_date="2021-01-02",
                end_date="2023-02-01",
                leaf_name="f",
            )(identity),
        },
        # Some overlap.
        {
            ("a",): policy_function(
                start_date="2023-01-02",
                end_date="2023-02-01",
                leaf_name="f",
            )(identity),
            ("b",): policy_function(
                start_date="2022-01-01",
                end_date="2023-01-31",
                leaf_name="f",
            )(identity),
        },
        # Same as before, but defined in different modules.
        {
            ("c", "a"): policy_function(
                start_date="2023-01-02",
                end_date="2023-02-01",
                leaf_name="f",
            )(identity),
            ("d", "b"): policy_function(
                start_date="2022-01-01",
                end_date="2023-01-31",
                leaf_name="f",
            )(identity),
        },
        # Same as before, but defined in different modules without leaf name.
        {
            ("c", "f"): policy_function(
                start_date="2023-01-02",
                end_date="2023-02-01",
            )(identity),
            ("d", "f"): policy_function(
                start_date="2022-01-01",
                end_date="2023-01-31",
            )(identity),
        },
    ],
)
def test_dates_active_with_conflicts(orig_ttsim_objects_tree: FlatTTSIMObjectDict):
    with pytest.raises(ConflictingTimeDependentObjectsError):
        fail_because_of_clashes(
            orig_ttsim_objects_tree=orig_ttsim_objects_tree,
            orig_yaml_tree={},
        )
