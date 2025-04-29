from dataclasses import dataclass

import pytest

from ttsim.shared import (
    create_tree_from_path_and_value,
    get_name_of_group_by_id,
    get_re_pattern_for_all_time_units_and_groupings,
    get_re_pattern_for_specific_time_units_and_groupings,
    insert_path_and_value,
    merge_trees,
    partition_tree_by_reference_tree,
    upsert_path_and_value,
    upsert_tree,
)


@dataclass
class SampleDataClass:
    a: int


@pytest.mark.parametrize(
    "base, path_to_upsert, value_to_upsert, expected",
    [
        ({}, ["a"], 1, {"a": 1}),
        ({"a": 1}, ["a"], 2, {"a": 2}),
        ({}, ["a", "b"], 2, {"a": {"b": 2}}),
        ({"a": {"b": 1}}, ["a", "c"], 2, {"a": {"b": 1, "c": 2}}),
    ],
)
def test_upsert_path_and_value(base, path_to_upsert, value_to_upsert, expected):
    result = upsert_path_and_value(
        base=base, path_to_upsert=path_to_upsert, value_to_upsert=value_to_upsert
    )
    assert result == expected


@pytest.mark.parametrize(
    "base, path_to_insert, value_to_insert, expected",
    [
        ({}, ("a",), 1, {"a": 1}),
        ({"a": 1}, ("b",), 2, {"a": 1, "b": 2}),
    ],
)
def test_insert_path_and_value(base, path_to_insert, value_to_insert, expected):
    result = insert_path_and_value(
        base=base, path_to_insert=path_to_insert, value_to_insert=value_to_insert
    )
    assert result == expected


@pytest.mark.parametrize(
    "base, path_to_insert, value_to_insert",
    [
        ({"a": 1}, ("a",), 2),
    ],
)
def test_insert_path_and_value_invalid(base, path_to_insert, value_to_insert):
    with pytest.raises(ValueError, match="Conflicting paths in trees to merge."):
        insert_path_and_value(
            base=base, path_to_insert=path_to_insert, value_to_insert=value_to_insert
        )


@pytest.mark.parametrize(
    "paths, expected",
    [
        ("a", {"a": None}),
        (("a", "b"), {"a": {"b": None}}),
        (("a", "b", "c"), {"a": {"b": {"c": None}}}),
    ],
)
def test_create_tree_from_path_and_value(paths, expected):
    assert create_tree_from_path_and_value(paths) == expected


@pytest.mark.parametrize(
    "paths, value, expected",
    [
        ((), {"a": None}, {"a": None}),
        ((), {"a": 1}, {"a": 1}),
    ],
)
def test_create_tree_from_path_and_value_if_path_is_empty(paths, value, expected):
    assert create_tree_from_path_and_value(paths, value) == expected


@pytest.mark.parametrize(
    "left, right, expected",
    [
        ({}, {"a": 1}, {"a": 1}),
        ({"a": 1}, {"b": 2}, {"a": 1, "b": 2}),
        ({"a": {"b": 1}}, {"a": {"c": 2}}, {"a": {"b": 1, "c": 2}}),
        ({"a": {"b": 1}}, {"a": 3}, {"a": 3}),
        ({"a": 3}, {"a": {"b": 1}}, {"a": {"b": 1}}),
        ({"a": SampleDataClass(a=1)}, {}, {"a": SampleDataClass(a=1)}),
    ],
)
def test_merge_trees_valid(left, right, expected):
    assert merge_trees(left=left, right=right) == expected


@pytest.mark.parametrize(
    "left, right",
    [({"a": 1}, {"a": 2}), ({"a": 1}, {"a": 1}), ({"a": {"b": 1}}, {"a": {"b": 5}})],
)
def test_merge_trees_invalid(left, right):
    with pytest.raises(ValueError, match="Conflicting paths in trees to merge."):
        merge_trees(left=left, right=right)


@pytest.mark.parametrize(
    "base_dict, update_dict, expected",
    [
        ({}, {"a": 1}, {"a": 1}),
        ({"a": 1}, {"b": 2}, {"a": 1, "b": 2}),
        ({"a": 1}, {"a": 2}, {"a": 2}),
        ({"a": {"b": 1}}, {"a": {"c": 2}}, {"a": {"b": 1, "c": 2}}),
        ({"a": {"b": 1}}, {"a": 3}, {"a": 3}),
        ({"a": 3}, {"a": {"b": 1}}, {"a": {"b": 1}}),
        ({"a": SampleDataClass(a=1)}, {}, {"a": SampleDataClass(a=1)}),
    ],
)
def test_upsert_tree(base_dict, update_dict, expected):
    assert upsert_tree(base=base_dict, to_upsert=update_dict) == expected


@pytest.mark.parametrize(
    "tree_to_partition, reference_tree, expected",
    [
        (
            {
                "a": {
                    "b": 1,
                    "c": 1,
                },
                "b": 1,
            },
            {
                "a": {
                    "b": 1,
                },
                "b": 1,
            },
            (
                {"a": {"b": 1}, "b": 1},
                {"a": {"c": 1}},
            ),
        ),
        (
            {
                "a": {
                    "c": 1,
                },
            },
            {},
            (
                {},
                {"a": {"c": 1}},
            ),
        ),
        (
            {
                "a": {
                    "b": None,
                    "c": None,
                },
                "b": None,
            },
            {
                "a": {
                    "b": None,
                },
                "b": None,
            },
            (
                {"a": {"b": None}, "b": None},
                {"a": {"c": None}},
            ),
        ),
    ],
)
def test_partition_tree_by_reference_tree(tree_to_partition, reference_tree, expected):
    in_reference_tree, not_in_reference_tree = partition_tree_by_reference_tree(
        tree_to_partition=tree_to_partition, reference_tree=reference_tree
    )

    assert in_reference_tree == expected[0]
    assert not_in_reference_tree == expected[1]


@pytest.mark.parametrize(
    "target_name, expected",
    [
        (("namespace1__foo"), None),
        (("foo_kin"), "kin_id"),
        (("namespace1__foo_kin"), "kin_id"),
        (("namespace1__foo_fam"), "fam_id"),
    ],
)
def test_get_name_of_group_by_id(target_name, expected):
    assert (
        get_name_of_group_by_id(
            target_name=target_name,
            groupings=("kin", "fam"),
        )
        == expected
    )


@pytest.mark.parametrize(
    (
        "func_name",
        "time_units",
        "groupings",
        "expected_base_name",
        "expected_time_unit",
        "expected_grouping",
    ),
    [
        ("foo", ("m", "y"), ["kin"], "foo", None, None),
        ("foo_m_kin", ("m", "y"), ["kin"], "foo", "m", "kin"),
        ("foo_y_kin", ("m", "y"), ["kin"], "foo", "y", "kin"),
        ("foo_m", ("m", "y"), ["kin"], "foo", "m", None),
        ("foo_y", ("m", "y"), ["kin"], "foo", "y", None),
        ("foo_kin", ("m", "y"), ["kin"], "foo", None, "kin"),
        ("foo_kin_bar", ("m", "y"), ["kin"], "foo_kin_bar", None, None),
    ],
)
def test_get_re_pattern_for_time_units_and_groupings(
    func_name,
    time_units,
    groupings,
    expected_base_name,
    expected_time_unit,
    expected_grouping,
):
    result = get_re_pattern_for_all_time_units_and_groupings(
        time_units=time_units,
        groupings=groupings,
    )
    match = result.fullmatch(func_name)
    assert match.group("base_name") == expected_base_name
    assert match.group("time_unit") == expected_time_unit
    assert match.group("grouping") == expected_grouping


@pytest.mark.parametrize(
    (
        "base_name",
        "time_units",
        "groupings",
        "expected_match",
    ),
    [
        ("foo", ["m", "y"], ["kin"], "foo_m_kin"),
        ("foo", ["m", "y"], ["kin", "x"], "foo_m"),
        ("foo", ["m", "y"], ["kin", "x"], "foo_kin"),
    ],
)
def test_get_re_pattern_for_some_base_name(
    base_name, time_units, groupings, expected_match
):
    re_pattern = get_re_pattern_for_specific_time_units_and_groupings(
        base_name=base_name,
        all_time_units=time_units,
        groupings=groupings,
    )
    assert re_pattern.fullmatch(expected_match)
