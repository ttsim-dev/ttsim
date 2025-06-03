from __future__ import annotations

import datetime
import re
from typing import TYPE_CHECKING

import pytest

from ttsim.column_objects_param_function import (
    DEFAULT_END_DATE,
    group_creation_function,
)
from ttsim.failures_and_warnings import (
    ConflictingActivePeriodsError,
    _param_with_active_periods,
    _ParamWithActivePeriod,
    assert_valid_ttsim_pytree,
    fail_if_active_periods_overlap,
    fail_if_group_ids_are_outside_top_level_namespace,
    fail_if_name_of_last_branch_element_not_leaf_name_of_function,
    policy_function,
)

if TYPE_CHECKING:
    from ttsim.typing import (
        FlatColumnObjectsParamFunctions,
        FlatOrigParamSpecs,
        NestedColumnObjectsParamFunctions,
        OrigParamSpec,
    )

GENERIC_PARAM_HEADER = {
    "name": {"de": "foo", "en": "foo"},
    "description": {"de": "foo", "en": "foo"},
    "unit": None,
    "reference_period": None,
}


def identity(x: int) -> int:
    return x


def return_one() -> int:
    return 1


def return_two() -> int:
    return 2


def return_three() -> int:
    return 3


@group_creation_function()
def fam_id() -> int:
    pass


@pytest.mark.parametrize(
    ("tree", "leaf_checker", "err_substr"),
    [
        (
            {"a": 1, "b": 2},
            lambda leaf: leaf is None,
            "Leaf at tree[a] is invalid: got 1 of type <class 'int'>.",
        ),
        (
            {"a": None, "b": {"c": None, "d": 1}},
            lambda leaf: leaf is None,
            "Leaf at tree[b][d] is invalid: got 1 of type <class 'int'>.",
        ),
        (
            [1, 2, 3],
            lambda leaf: leaf is None,
            "tree must be a dict, got <class 'list'>.",
        ),
        (
            {1: 2},
            lambda leaf: leaf is None,
            "Key 1 in tree must be a string but got <class 'int'>.",
        ),
    ],
)
def test_assert_valid_ttsim_pytree(tree, leaf_checker, err_substr):
    with pytest.raises(TypeError, match=re.escape(err_substr)):
        assert_valid_ttsim_pytree(
            tree=tree, leaf_checker=leaf_checker, tree_name="tree"
        )


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
def test_fail_because_active_periods_overlap_passes(
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
