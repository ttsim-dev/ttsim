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
    group_creation_function,
    main,
    policy_function,
)
from ttsim.column_objects_param_function import DEFAULT_END_DATE
from ttsim.loader import (
    orig_tree_with_column_objects_and_param_functions,
    orig_tree_with_params,
)
from ttsim.policy_environment import (
    ConflictingActivePeriodsError,
    _get_param_value,
    _param_with_active_periods,
    _ParamWithActivePeriod,
    active_tree_with_column_objects_and_param_functions,
    active_tree_with_params,
    fail_if_active_periods_overlap,
    fail_if_group_ids_are_outside_top_level_namespace,
    fail_if_name_of_last_branch_element_not_leaf_name_of_function,
    upsert_tree_into_policy_environment,
)

if TYPE_CHECKING:
    from ttsim.typing import (
        FlatColumnObjectsParamFunctions,
        FlatOrigParamSpecs,
        NestedColumnObjectsParamFunctions,
        NestedPolicyEnvironment,
        OrigParamSpec,
    )

GENERIC_PARAM_HEADER = {
    "name": {"de": "foo", "en": "foo"},
    "description": {"de": "foo", "en": "foo"},
    "unit": None,
    "reference_period": None,
}


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
    _orig_tree_with_params = orig_tree_with_params(
        root=Path(__file__).parent / "test_parameters"
    )
    k = ("test_add_jahresanfang.yaml", "foo")
    _active_ttsim_tree_with_params = active_tree_with_params(
        orig_tree_with_params={k: _orig_tree_with_params[k]},
        date=pd.to_datetime("2020-07-01").date(),
    )
    assert _active_ttsim_tree_with_params["foo"].value == 2
    assert _active_ttsim_tree_with_params["foo_jahresanfang"].value == 1


@pytest.mark.parametrize(
    "functions_tree",
    [
        {"foo": policy_function(leaf_name="bar")(return_one)},
    ],
)
def test_fail_if_name_of_last_branch_element_not_leaf_name_of_function(
    functions_tree: NestedColumnObjectsParamFunctions,
):
    with pytest.raises(KeyError):
        fail_if_name_of_last_branch_element_not_leaf_name_of_function(functions_tree)


def test_fail_if_group_ids_are_outside_top_level_namespace():
    with pytest.raises(
        ValueError, match="Group identifiers must live in the top-level namespace. Got:"
    ):
        fail_if_group_ids_are_outside_top_level_namespace({"n1": {"fam_id": fam_id}})


def test_upsert_tree_into_policy_environment_fail_with_group_ids_outside_top_level_namespace():  # noqa: E501
    with pytest.raises(
        ValueError, match="Group identifiers must live in the top-level namespace. Got:"
    ):
        upsert_tree_into_policy_environment(
            policy_environment={},
            tree_to_upsert={"n1": {"fam_id": fam_id}},
        )


@pytest.mark.parametrize(
    "policy_environment",
    [
        {},
        {"foo": policy_function(leaf_name="foo")(return_one)},
        {
            "foo": policy_function(leaf_name="foo")(return_one),
            "bar": policy_function(leaf_name="bar")(return_two),
        },
    ],
)
def test_upsert_tree_into_policy_environment(
    policy_environment: NestedPolicyEnvironment,
):
    new_function = policy_function(leaf_name="foo")(return_three)
    new_environment = upsert_tree_into_policy_environment(
        policy_environment=policy_environment, tree_to_upsert={"foo": new_function}
    )

    assert new_environment["foo"] == new_function


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


def identity(x):
    return x


@pytest.mark.parametrize(
    "orig_tree_with_column_objects_and_param_functions, orig_tree_with_params",
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
            {
                ("c", "g"): {
                    **GENERIC_PARAM_HEADER,
                    datetime.date(2023, 1, 1): {"value": 1},
                }
            },
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
            {
                ("x", "c", "h"): {
                    **GENERIC_PARAM_HEADER,
                    datetime.date(2023, 1, 1): {"value": 2},
                }
            },
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
            {
                ("x", "c", "g"): {
                    **GENERIC_PARAM_HEADER,
                    datetime.date(2023, 1, 1): {"value": 3},
                }
            },
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
            {
                ("z", "a", "f"): {
                    **GENERIC_PARAM_HEADER,
                    datetime.date(2023, 1, 1): {"value": 4},
                }
            },
        ),
        # Different yaml files, no name clashes because of different names.
        (
            {},
            {
                ("x", "a", "f"): {
                    **GENERIC_PARAM_HEADER,
                    datetime.date(2023, 1, 1): {"value": 5},
                },
                ("x", "b", "g"): {
                    **GENERIC_PARAM_HEADER,
                    datetime.date(2023, 1, 1): {"value": 6},
                },
            },
        ),
    ],
)
def test_fail_because_active_periods_overlap_passes(
    orig_tree_with_column_objects_and_param_functions: FlatColumnObjectsParamFunctions,
    orig_tree_with_params: FlatOrigParamSpecs,
):
    fail_if_active_periods_overlap(
        orig_tree_with_column_objects_and_param_functions,
        orig_tree_with_params,
    )


@pytest.mark.parametrize(
    "orig_tree_with_column_objects_and_param_functions, orig_tree_with_params",
    [
        # Exact overlap.
        (
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
            {},
        ),
        # Active period for "a" is subset of "b".
        (
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
            {},
        ),
        # Some overlap.
        (
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
            {},
        ),
        # Same as before, but defined in different modules.
        (
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
            {},
        ),
        # Same as before, but defined in different modules without leaf name.
        (
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
            {},
        ),
        # Same global module, no overlap in functions, name clashes leaf name / yaml.
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
            {
                ("c", "f"): {
                    **GENERIC_PARAM_HEADER,
                    datetime.date(2023, 1, 1): {"value": 1},
                }
            },
        ),
        # Same paths, no overlap in functions, name clashes leaf name / yaml.
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
            {
                ("x", "a", "f"): {
                    **GENERIC_PARAM_HEADER,
                    datetime.date(2023, 1, 1): {"value": 2},
                }
            },
        ),
        # Same paths, name clashes within params from different yaml files.
        (
            {},
            {
                ("x", "a", "f"): {
                    **GENERIC_PARAM_HEADER,
                    datetime.date(2023, 1, 1): {"value": 3},
                },
                ("x", "b", "f"): {
                    **GENERIC_PARAM_HEADER,
                    datetime.date(2023, 1, 1): {"value": 4},
                },
            },
        ),
    ],
)
def test_fail_because_active_periods_overlap_raises(
    orig_tree_with_column_objects_and_param_functions: FlatColumnObjectsParamFunctions,
    orig_tree_with_params: FlatOrigParamSpecs,
):
    with pytest.raises(ConflictingActivePeriodsError):
        fail_if_active_periods_overlap(
            orig_tree_with_column_objects_and_param_functions,
            orig_tree_with_params,
        )


@pytest.mark.parametrize(
    "orig_tree_with_column_objects_and_param_functions, orig_tree_with_params",
    [
        # Same leaf names across functions / parameters, but no overlapping periods.
        (
            {
                ("c", "a"): policy_function(
                    start_date="2012-01-01",
                    end_date="2015-12-31",
                    leaf_name="f",
                )(identity),
                ("c", "b"): policy_function(
                    start_date="2023-02-01",
                    end_date="2023-02-28",
                    leaf_name="f",
                )(identity),
            },
            {
                ("c", "f"): {
                    "name": {"de": "foo", "en": "foo"},
                    "description": {"de": "foo", "en": "foo"},
                    "unit": None,
                    "reference_period": None,
                    "type": "scalar",
                    datetime.date(1984, 1, 1): {"value": 1},
                    datetime.date(1985, 1, 1): {"value": 3},
                    datetime.date(1995, 1, 1): {"value": 5},
                    datetime.date(2012, 1, 1): {"note": "more complex, see function"},
                    datetime.date(2016, 1, 1): {"value": 10},
                    datetime.date(2023, 2, 1): {
                        "note": "more complex, see function",
                        "reference": "https://example.com/foo",
                    },
                    datetime.date(2023, 3, 1): {
                        "value": 13,
                        "note": "Complex didn't last long.",
                    },
                }
            },
        ),
        # Different periods specified in different files.
        (
            {},
            {
                ("c", "f"): {
                    "name": {"de": "foo", "en": "foo"},
                    "description": {"de": "foo", "en": "foo"},
                    "unit": None,
                    "reference_period": None,
                    "type": "scalar",
                    datetime.date(1984, 1, 1): {"value": 1},
                    datetime.date(1985, 1, 1): {"value": 3},
                    datetime.date(1995, 1, 1): {"value": 5},
                    datetime.date(2012, 1, 1): {"note": "more complex, see function"},
                },
                ("d", "f"): {
                    "name": {"de": "foo", "en": "foo"},
                    "description": {"de": "foo", "en": "foo"},
                    "unit": None,
                    "reference_period": None,
                    "type": "scalar",
                    datetime.date(2016, 1, 1): {"value": 10},
                    datetime.date(2023, 2, 1): {
                        "note": "more complex, see function",
                        "reference": "https://example.com/foo",
                    },
                    datetime.date(2023, 3, 1): {
                        "value": 13,
                        "note": "Complex didn't last long.",
                    },
                },
            },
        ),
    ],
)
def test_pass_because_no_overlap_functions_params(
    orig_tree_with_column_objects_and_param_functions: FlatColumnObjectsParamFunctions,
    orig_tree_with_params: FlatOrigParamSpecs,
):
    fail_if_active_periods_overlap(
        orig_tree_with_column_objects_and_param_functions,
        orig_tree_with_params,
    )


@pytest.mark.parametrize(
    "param_spec, leaf_name, expected",
    (
        (
            {
                "name": {"de": "spam", "en": "spam"},
                "description": {"de": "spam", "en": "spam"},
                "unit": None,
                "reference_period": None,
                "type": "scalar",
                datetime.date(1984, 1, 1): {"note": "completely empty"},
            },
            "spam",
            [],
        ),
        (
            {
                "name": {"de": "foo", "en": "foo"},
                "description": {"de": "foo", "en": "foo"},
                "unit": None,
                "reference_period": None,
                "type": "scalar",
                datetime.date(1984, 1, 1): {"value": 1},
            },
            "foo",
            [
                _ParamWithActivePeriod(
                    leaf_name="foo",
                    original_function_name="foo",
                    start_date=datetime.date(1984, 1, 1),
                    end_date=DEFAULT_END_DATE,
                    **GENERIC_PARAM_HEADER,
                )
            ],
        ),
        (
            {
                "name": {"de": "foo", "en": "foo"},
                "description": {"de": "foo", "en": "foo"},
                "unit": None,
                "reference_period": None,
                "type": "scalar",
                datetime.date(1984, 1, 1): {"value": 1},
                datetime.date(1985, 1, 1): {"note": "stop"},
            },
            "foo",
            [
                _ParamWithActivePeriod(
                    leaf_name="foo",
                    original_function_name="foo",
                    start_date=datetime.date(1984, 1, 1),
                    end_date=datetime.date(1984, 12, 31),
                    **GENERIC_PARAM_HEADER,
                )
            ],
        ),
        (
            {
                "name": {"de": "bar", "en": "bar"},
                "description": {"de": "bar", "en": "bar"},
                "unit": None,
                "reference_period": None,
                "type": "scalar",
                datetime.date(1984, 1, 1): {"value": 1},
                datetime.date(1985, 1, 1): {"value": 3},
                datetime.date(1995, 1, 1): {"value": 5},
                datetime.date(2012, 1, 1): {"note": "more complex, see function"},
                datetime.date(2016, 1, 1): {"value": 10},
                datetime.date(2023, 2, 1): {
                    "note": "more complex, see function",
                    "reference": "https://example.com/bar",
                },
                datetime.date(2023, 3, 1): {
                    "value": 13,
                    "note": "Complex didn't last long.",
                },
            },
            "bar",
            [
                _ParamWithActivePeriod(
                    leaf_name="bar",
                    original_function_name="bar",
                    start_date=datetime.date(2023, 3, 1),
                    end_date=DEFAULT_END_DATE,
                    name={"de": "bar", "en": "bar"},
                    description={"de": "bar", "en": "bar"},
                    unit=None,
                    reference_period=None,
                ),
                _ParamWithActivePeriod(
                    leaf_name="bar",
                    original_function_name="bar",
                    start_date=datetime.date(2016, 1, 1),
                    end_date=datetime.date(2023, 1, 31),
                    name={"de": "bar", "en": "bar"},
                    description={"de": "bar", "en": "bar"},
                    unit=None,
                    reference_period=None,
                ),
                _ParamWithActivePeriod(
                    leaf_name="bar",
                    original_function_name="bar",
                    start_date=datetime.date(1984, 1, 1),
                    end_date=datetime.date(2011, 12, 31),
                    name={"de": "bar", "en": "bar"},
                    description={"de": "bar", "en": "bar"},
                    unit=None,
                    reference_period=None,
                ),
            ],
        ),
    ),
)
def test_ttsim_param_with_active_periods(
    param_spec: OrigParamSpec,
    leaf_name: str,
    expected: list[_ParamWithActivePeriod],
):
    actual = _param_with_active_periods(
        param_spec=param_spec,
        leaf_name=leaf_name,
    )
    assert actual == expected


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
        orig_tree_with_column_objects_and_param_functions(root=METTSIM_ROOT)
    )
    functions_last_day = active_tree_with_column_objects_and_param_functions(
        orig_tree_with_column_objects_and_param_functions=_orig_tree_with_column_objects_and_param_functions,
        date=last_day,
    )
    functions_next_day = active_tree_with_column_objects_and_param_functions(
        orig_tree_with_column_objects_and_param_functions=_orig_tree_with_column_objects_and_param_functions,
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
