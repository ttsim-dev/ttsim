from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import pytest

from ttsim.interface_dag_elements.shared import (
    create_tree_from_path_and_value,
    get_name_of_group_by_id,
    get_re_pattern_for_all_time_units_and_groupings,
    get_re_pattern_for_specific_time_units_and_groupings,
    insert_path_and_value,
    merge_trees,
    partition_tree_by_reference_tree,
    to_datetime,
    upsert_path_and_value,
    upsert_tree,
)

if TYPE_CHECKING:
    import re


@dataclass
class SampleDataClass:
    a: int


def test_leap_year_correctly_handled():
    to_datetime(date="2020-02-29")


def test_fail_if_invalid_date():
    with pytest.raises(ValueError, match=r"day .+ range"):
        to_datetime(date="2020-02-30")


@pytest.mark.parametrize(
    ("base", "path_to_upsert", "value_to_upsert", "expected"),
    [
        ({}, ["a"], 1, {"a": 1}),
        ({"a": 1}, ["a"], 2, {"a": 2}),
        ({}, ["a", "b"], 2, {"a": {"b": 2}}),
        ({"a": {"b": 1}}, ["a", "c"], 2, {"a": {"b": 1, "c": 2}}),
    ],
)
def test_upsert_path_and_value(base, path_to_upsert, value_to_upsert, expected):
    result = upsert_path_and_value(
        base=base,
        path_to_upsert=path_to_upsert,
        value_to_upsert=value_to_upsert,
    )
    assert result == expected


@pytest.mark.parametrize(
    ("base", "path_to_insert", "value_to_insert", "expected"),
    [
        ({}, ("a",), 1, {"a": 1}),
        ({"a": 1}, ("b",), 2, {"a": 1, "b": 2}),
    ],
)
def test_insert_path_and_value(base, path_to_insert, value_to_insert, expected):
    result = insert_path_and_value(
        base=base,
        path_to_insert=path_to_insert,
        value_to_insert=value_to_insert,
    )
    assert result == expected


@pytest.mark.parametrize(
    ("base", "path_to_insert", "value_to_insert"),
    [
        ({"a": 1}, ("a",), 2),
    ],
)
def test_insert_path_and_value_invalid(base, path_to_insert, value_to_insert):
    with pytest.raises(ValueError, match=r"Conflicting paths in trees to merge."):
        insert_path_and_value(
            base=base,
            path_to_insert=path_to_insert,
            value_to_insert=value_to_insert,
        )


@pytest.mark.parametrize(
    ("paths", "expected"),
    [
        ("a", {"a": None}),
        (("a", "b"), {"a": {"b": None}}),
        (("a", "b", "c"), {"a": {"b": {"c": None}}}),
    ],
)
def test_create_tree_from_path_and_value(paths, expected):
    assert create_tree_from_path_and_value(paths) == expected


@pytest.mark.parametrize(
    ("paths", "value", "expected"),
    [
        ((), {"a": None}, {"a": None}),
        ((), {"a": 1}, {"a": 1}),
    ],
)
def test_create_tree_from_path_and_value_if_path_is_empty(paths, value, expected):
    assert create_tree_from_path_and_value(paths, value) == expected


@pytest.mark.parametrize(
    ("left", "right", "expected"),
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
    ("left", "right"),
    [({"a": 1}, {"a": 2}), ({"a": 1}, {"a": 1}), ({"a": {"b": 1}}, {"a": {"b": 5}})],
)
def test_merge_trees_invalid(left, right):
    with pytest.raises(ValueError, match=r"Conflicting paths in trees to merge."):
        merge_trees(left=left, right=right)


@pytest.mark.parametrize(
    ("base_dict", "update_dict", "expected"),
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
    ("tree_to_partition", "reference_tree", "expected"),
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
        tree_to_partition=tree_to_partition,
        reference_tree=reference_tree,
    )

    assert in_reference_tree == expected[0]
    assert not_in_reference_tree == expected[1]


@pytest.mark.parametrize(
    ("target_name", "expected"),
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
            grouping_levels=("kin", "fam"),
        )
        == expected
    )


@pytest.mark.parametrize(
    (
        "func_name",
        "time_units",
        "grouping_levels",
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
    grouping_levels,
    expected_base_name,
    expected_time_unit,
    expected_grouping,
):
    result = get_re_pattern_for_all_time_units_and_groupings(
        time_units=time_units,
        grouping_levels=grouping_levels,
    )
    match = cast("re.Match[str]", result.fullmatch(func_name))
    assert match.group("base_name") == expected_base_name
    assert match.group("time_unit") == expected_time_unit
    assert match.group("grouping") == expected_grouping


@pytest.mark.parametrize(
    (
        "base_name",
        "time_units",
        "grouping_levels",
        "expected_match",
    ),
    [
        ("foo", ["m", "y"], ["kin"], "foo_m_kin"),
        ("foo", ["m", "y"], ["kin", "x"], "foo_m"),
        ("foo", ["m", "y"], ["kin", "x"], "foo_kin"),
    ],
)
def test_get_re_pattern_for_some_base_name(
    base_name,
    time_units,
    grouping_levels,
    expected_match,
):
    re_pattern = get_re_pattern_for_specific_time_units_and_groupings(
        base_name=base_name,
        all_time_units=time_units,
        grouping_levels=grouping_levels,
    )
    assert re_pattern.fullmatch(expected_match)


# =============================================================================
# Additional tests for utility functions
# =============================================================================


from ttsim.interface_dag_elements.shared import (
    get_base_name_and_grouping_suffix,
    partition_by_reference_dict,
    remove_group_suffix,
)


@pytest.mark.parametrize(
    ("col", "grouping_levels", "expected"),
    [
        ("foo_kin", ("kin", "fam"), "foo"),
        ("foo_fam", ("kin", "fam"), "foo"),
        ("foo_bar", ("kin", "fam"), "foo_bar"),  # No recognized suffix
        ("foo", ("kin", "fam"), "foo"),  # No suffix at all
        ("foo_m_kin", ("kin", "fam"), "foo_m"),  # With time unit
        ("kin", ("kin", "fam"), "kin"),  # Edge case: just the suffix name
    ],
)
def test_remove_group_suffix(col, grouping_levels, expected):
    """Test remove_group_suffix removes grouping suffixes correctly."""
    result = remove_group_suffix(col, grouping_levels)
    assert result == expected


@pytest.mark.parametrize(
    ("to_partition", "reference_dict", "expected_intersection", "expected_difference"),
    [
        # Basic partitioning
        ({"a": 1, "b": 2, "c": 3}, {"a": 10, "b": 20}, {"a": 1, "b": 2}, {"c": 3}),
        # No intersection
        ({"a": 1, "b": 2}, {"c": 3, "d": 4}, {}, {"a": 1, "b": 2}),
        # Full intersection
        ({"a": 1, "b": 2}, {"a": 10, "b": 20}, {"a": 1, "b": 2}, {}),
        # Empty to_partition
        ({}, {"a": 1}, {}, {}),
        # Empty reference
        ({"a": 1, "b": 2}, {}, {}, {"a": 1, "b": 2}),
        # Both empty
        ({}, {}, {}, {}),
    ],
)
def test_partition_by_reference_dict(
    to_partition, reference_dict, expected_intersection, expected_difference
):
    """Test partition_by_reference_dict partitions correctly."""
    intersection, difference = partition_by_reference_dict(to_partition, reference_dict)
    assert intersection == expected_intersection
    assert difference == expected_difference


def test_get_base_name_and_grouping_suffix_with_suffix():
    """Test get_base_name_and_grouping_suffix with grouping suffix present."""
    pattern = get_re_pattern_for_all_time_units_and_groupings(
        time_units=("m", "y"),
        grouping_levels=("kin", "fam"),
    )
    match = pattern.fullmatch("foo_m_kin")
    assert match is not None

    base_name, suffix = get_base_name_and_grouping_suffix(match)
    assert base_name == "foo"
    assert suffix == "_kin"


def test_get_base_name_and_grouping_suffix_without_suffix():
    """Test get_base_name_and_grouping_suffix without grouping suffix."""
    pattern = get_re_pattern_for_all_time_units_and_groupings(
        time_units=("m", "y"),
        grouping_levels=("kin", "fam"),
    )
    match = pattern.fullmatch("foo_m")
    assert match is not None

    base_name, suffix = get_base_name_and_grouping_suffix(match)
    assert base_name == "foo"
    assert suffix == ""


def test_get_base_name_and_grouping_suffix_no_time_unit():
    """Test get_base_name_and_grouping_suffix with only grouping suffix."""
    pattern = get_re_pattern_for_all_time_units_and_groupings(
        time_units=("m", "y"),
        grouping_levels=("kin", "fam"),
    )
    match = pattern.fullmatch("foo_kin")
    assert match is not None

    base_name, suffix = get_base_name_and_grouping_suffix(match)
    assert base_name == "foo"
    assert suffix == "_kin"


# =============================================================================
# Additional tests for tree operations
# =============================================================================


def test_upsert_tree_deeply_nested():
    """Test upsert_tree with deeply nested structures."""
    base = {"a": {"b": {"c": {"d": 1}}}}
    to_upsert = {"a": {"b": {"c": {"e": 2}}}}

    result = upsert_tree(base=base, to_upsert=to_upsert)

    assert result == {"a": {"b": {"c": {"d": 1, "e": 2}}}}


def test_merge_trees_with_none_values():
    """Test merge_trees preserves None values correctly."""
    left = {"a": None}
    right = {"b": None}

    result = merge_trees(left=left, right=right)

    assert result == {"a": None, "b": None}


def test_partition_tree_by_reference_tree_empty_reference():
    """Test partition_tree_by_reference_tree with empty reference tree."""
    tree_to_partition = {"a": {"b": 1, "c": 2}}
    reference_tree = {}

    in_ref, not_in_ref = partition_tree_by_reference_tree(
        tree_to_partition=tree_to_partition,
        reference_tree=reference_tree,
    )

    assert in_ref == {}
    assert not_in_ref == {"a": {"b": 1, "c": 2}}


def test_partition_tree_by_reference_tree_full_overlap():
    """Test partition_tree_by_reference_tree when trees fully overlap."""
    tree_to_partition = {"a": {"b": 1}}
    reference_tree = {"a": {"b": 2}}

    in_ref, not_in_ref = partition_tree_by_reference_tree(
        tree_to_partition=tree_to_partition,
        reference_tree=reference_tree,
    )

    assert in_ref == {"a": {"b": 1}}
    assert not_in_ref == {}


def test_to_datetime_with_datetime_object():
    """Test to_datetime passes through datetime.date objects."""
    import datetime

    date_obj = datetime.date(2024, 6, 15)
    result = to_datetime(date_obj)

    assert result == date_obj
    assert isinstance(result, datetime.date)


def test_to_datetime_with_valid_string():
    """Test to_datetime parses valid ISO date strings."""
    import datetime

    result = to_datetime("2024-06-15")

    assert result == datetime.date(2024, 6, 15)


def test_to_datetime_with_invalid_format_raises():
    """Test to_datetime raises for invalid date formats."""
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        to_datetime("06-15-2024")  # Wrong format

    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        to_datetime("2024/06/15")  # Wrong separator


def test_create_tree_from_path_and_value_with_dataclass():
    """Test create_tree_from_path_and_value with dataclass value."""
    result = create_tree_from_path_and_value(
        path=("a", "b"),
        value=SampleDataClass(a=42),
    )

    assert result == {"a": {"b": SampleDataClass(a=42)}}
