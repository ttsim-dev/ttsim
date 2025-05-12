"""Some tests for the policy_environment module."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import optree
import pandas as pd
import pytest
import yaml
from mettsim.config import METTSIM_ROOT

from ttsim import (
    GroupCreationFunction,
    PolicyEnvironment,
    group_creation_function,
    policy_function,
    set_up_policy_environment,
)
from ttsim.loader import orig_params_tree, orig_ttsim_objects_tree
from ttsim.policy_environment import (
    ConflictingActivePeriodsError,
    ConflictingNamesError,
    _fail_if_name_of_last_branch_element_not_leaf_name_of_function,
    _get_params_contents,
    _parse_raw_parameter_group,
    active_ttsim_objects_tree,
    active_ttsim_params_tree,
    fail_because_of_clashes,
)

if TYPE_CHECKING:
    from ttsim.typing import (
        FlatOrigParamSpecDict,
        FlatTTSIMObjectDict,
        NestedTTSIMObjectDict,
    )


def return_one():
    return 1


def return_two():
    return 2


def return_three():
    return 3


@group_creation_function()
def fam_id() -> int:
    pass


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


def test_add_jahresanfang():
    _orig_params_tree = orig_params_tree(root=Path(__file__).parent / "test_parameters")
    k = ("test_add_jahresanfang.yaml", "foo")
    _active_ttsim_params_tree = active_ttsim_params_tree(
        orig_params_tree={k: _orig_params_tree[k]},
        date=pd.to_datetime("2020-07-01").date(),
    )
    assert _active_ttsim_params_tree["foo"].value == 2
    assert _active_ttsim_params_tree["foo_jahresanfang"].value == 1


def test_fail_if_invalid_access_different_date_old():
    with pytest.raises(ValueError):
        group = "invalid_access_diff_date"
        raw_group_data = yaml.load(
            (Path(__file__).parent / "test_parameters_old" / f"{group}.yaml").read_text(
                encoding="utf-8"
            ),
            Loader=yaml.CLoader,  # noqa: S506
        )
        _parse_raw_parameter_group(
            raw_group_data=raw_group_data,
            date=pd.to_datetime("2020-01-01").date(),
            group=group,
            parameters=None,
        )


def test_access_different_date_vorjahr_old():
    group = "test_access_diff_date_vorjahr"
    raw_group_data = yaml.load(
        (Path(__file__).parent / "test_parameters_old" / f"{group}.yaml").read_text(
            encoding="utf-8"
        ),
        Loader=yaml.CLoader,  # noqa: S506
    )
    params = _parse_raw_parameter_group(
        raw_group_data=raw_group_data,
        date=pd.to_datetime("2020-01-01").date(),
        group=group,
        parameters=None,
    )
    assert params["foo"] == 2020
    assert params["foo_vorjahr"] == 2019


def test_access_different_date_jahresanfang_old():
    group = "test_access_diff_date_jahresanfang"
    raw_group_data = yaml.load(
        (Path(__file__).parent / "test_parameters_old" / f"{group}.yaml").read_text(
            encoding="utf-8"
        ),
        Loader=yaml.CLoader,  # noqa: S506
    )
    params = _parse_raw_parameter_group(
        raw_group_data=raw_group_data,
        date=pd.to_datetime("2020-07-01").date(),
        group=group,
        parameters=None,
    )
    assert params["foo"] == 2021
    assert params["foo_jahresanfang"] == 2020


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


def identity(x):
    return x


@pytest.mark.parametrize(
    "orig_ttsim_objects_tree, orig_params_tree",
    [
        # Same global module, no overlapping periods, no name clashes.
        (
            {
                ("c", "a"): policy_function(
                    start_date="2023-01-01",
                    end_date="2023-01-31",
                    leaf_name="f",
                )(identity),
                ("c", "b"): policy_function(
                    start_date="2023-02-01",
                    end_date="2023-02-28",
                    leaf_name="f",
                )(identity),
            },
            {("c", "g"): {datetime.date(2023, 1, 1): {"value": 1}}},
        ),
        # Same submodule, overlapping periods, different leaf names so no name clashes.
        (
            {
                ("x", "c", "a"): policy_function(
                    start_date="2023-01-01",
                    end_date="2023-01-31",
                    leaf_name="f",
                )(identity),
                ("x", "c", "b"): policy_function(
                    start_date="2023-01-01",
                    end_date="2023-02-28",
                    leaf_name="g",
                )(identity),
            },
            {("x", "c", "h"): {datetime.date(2023, 1, 1): {"value": 2}}},
        ),
        # Different submodules, no overlapping periods, no name clashes.
        (
            {
                ("x", "c", "f"): policy_function(
                    start_date="2023-01-01",
                    end_date="2023-01-31",
                )(identity),
                ("x", "d", "f"): policy_function(
                    start_date="2023-02-01",
                    end_date="2023-02-28",
                )(identity),
            },
            {("x", "c", "g"): {datetime.date(2023, 1, 1): {"value": 3}}},
        ),
        # Different paths, overlapping periods, same names but no clashes.
        (
            {
                ("x", "a", "b"): policy_function(
                    start_date="2023-01-01",
                    end_date="2023-01-31",
                    leaf_name="f",
                )(identity),
                ("y", "a", "b"): policy_function(
                    start_date="2023-01-01",
                    end_date="2023-02-28",
                    leaf_name="f",
                )(identity),
            },
            {("z", "a", "f"): {datetime.date(2023, 1, 1): {"value": 4}}},
        ),
        # Different yaml files, no name clashes because of different names.
        (
            {},
            {
                ("x", "a", "f"): {datetime.date(2023, 1, 1): {"value": 5}},
                ("x", "b", "g"): {datetime.date(2023, 1, 1): {"value": 6}},
            },
        ),
    ],
)
def test_fail_because_of_clashes_no_conflicts(
    orig_ttsim_objects_tree: FlatTTSIMObjectDict,
    orig_params_tree: FlatOrigParamSpecDict,
):
    fail_because_of_clashes(
        orig_ttsim_objects_tree=orig_ttsim_objects_tree,
        orig_params_tree=orig_params_tree,
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
def test_fail_because_of_conflicting_active_periods(
    orig_ttsim_objects_tree: FlatTTSIMObjectDict,
):
    with pytest.raises(ConflictingActivePeriodsError):
        fail_because_of_clashes(
            orig_ttsim_objects_tree=orig_ttsim_objects_tree,
            orig_params_tree={},
        )


@pytest.mark.parametrize(
    "orig_ttsim_objects_tree, orig_params_tree",
    [
        # Same global module, no overlapping periods, name clashes leaf name / yaml.
        (
            {
                ("c", "a"): policy_function(
                    start_date="2023-01-01",
                    end_date="2023-01-31",
                    leaf_name="f",
                )(identity),
                ("c", "b"): policy_function(
                    start_date="2023-02-01",
                    end_date="2023-02-28",
                    leaf_name="f",
                )(identity),
            },
            {("c", "f"): {datetime.date(2023, 1, 1): {"value": 1}}},
        ),
        # Same paths, no overlapping periods, name clashes leaf name / yaml.
        (
            {
                ("x", "a", "b"): policy_function(
                    start_date="2023-01-01",
                    end_date="2023-01-31",
                    leaf_name="f",
                )(identity),
                ("x", "a", "c"): policy_function(
                    start_date="2023-02-01",
                    end_date="2023-02-28",
                    leaf_name="f",
                )(identity),
            },
            {("x", "a", "f"): {datetime.date(2023, 1, 1): {"value": 2}}},
        ),
        # Same paths, name clashes within params from different yaml files.
        (
            {},
            {
                ("x", "a", "f"): {datetime.date(2023, 1, 1): {"value": 3}},
                ("x", "b", "f"): {datetime.date(2023, 1, 1): {"value": 4}},
            },
        ),
    ],
)
def test_fail_because_of_conflicting_names(
    orig_ttsim_objects_tree: FlatTTSIMObjectDict,
    orig_params_tree: FlatOrigParamSpecDict,
):
    with pytest.raises(ConflictingNamesError):
        fail_because_of_clashes(
            orig_ttsim_objects_tree=orig_ttsim_objects_tree,
            orig_params_tree=orig_params_tree,
        )


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
def test_active_ttsim_objects_tree(
    tree: NestedTTSIMObjectDict,
    last_day: datetime.date,
    function_name_last_day: str,
    function_name_next_day: str,
):
    _orig_ttsim_objects_tree = orig_ttsim_objects_tree(root=METTSIM_ROOT)
    functions_last_day = active_ttsim_objects_tree(
        orig_ttsim_objects_tree=_orig_ttsim_objects_tree, date=last_day
    )
    functions_next_day = active_ttsim_objects_tree(
        orig_ttsim_objects_tree=_orig_ttsim_objects_tree,
        date=last_day + datetime.timedelta(days=1),
    )

    accessor = optree.tree_accessors(tree, none_is_leaf=True)[0]

    assert accessor(functions_last_day).__name__ == function_name_last_day
    assert accessor(functions_next_day).__name__ == function_name_next_day


def test_get_params_contents_with_updated_previous(
    some_params_spec_with_updates_previous,
):
    params_contents = _get_params_contents(some_params_spec_with_updates_previous)
    expected = {
        "updates_previous": True,
        "a": 1,
        "b": 4,
    }
    assert params_contents == expected
