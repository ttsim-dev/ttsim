from dataclasses import dataclass

import pytest

from ttsim.shared import (
    all_variations_of_base_name,
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
    "target_name, group_by_functions, expected",
    [
        (("namespace1__foo"), {}, None),
        (("namespace1__foo_hh"), {}, "hh_id"),
        (
            ("namespace1__foo_hh"),
            {"namespace1__hh_id": None},
            "hh_id",
        ),
        (
            ("namespace1__foo_bg"),
            {"arbeitslosengeld_2__bg_id": None},
            "arbeitslosengeld_2__bg_id",
        ),
        (
            ("namespace1__foo_eg"),
            {"grundsicherung__eg_id": None},
            "grundsicherung__eg_id",
        ),
        (
            ("namespace1__foo_eg"),
            {"arbeitslosengeld_2__eg_id": None},
            "arbeitslosengeld_2__eg_id",
        ),
        (
            ("arbeitslosengeld_2__einkommen_eg"),
            {
                "arbeitslosengeld_2__eg_id": None,
                "grundsicherung__eg_id": None,
            },
            "arbeitslosengeld_2__eg_id",
        ),
    ],
)
def test_get_name_of_group_by_id(target_name, group_by_functions, expected):
    assert (
        get_name_of_group_by_id(
            target_name=target_name,
            group_by_functions=group_by_functions,
            groupings=("hh", "bg", "eg"),
        )
        == expected
    )


@pytest.mark.parametrize(
    "target_name, group_by_functions, expected_error_match",
    [
        (
            ("outermost__foo_bg"),
            {
                "outermost__bg_id": None,
                "outermost__nested__bg_id": None,
            },
            (
                r"Group-by-identifier for target:[\s\S]+"
                r"\('outermost', 'foo_bg'\)[\s\S]+is ambiguous[\s\S]+"
                r"Found candidates[\s\S]+"
                r"\('outermost', 'bg_id'\)[\s\S]+"
                r"\('outermost', 'nested', 'bg_id'\)"
            ),
        ),
        (
            ("outermost__foo_bg"),
            {
                "outermost__inner1__bg_id": None,
                "outermost__inner2__bg_id": None,
            },
            r"Group-by-identifier for target:[\s\S]+"
            r"\('outermost', 'foo_bg'\)[\s\S]+is ambiguous[\s\S]+"
            r"Found candidates[\s\S]+"
            r"\('outermost', 'inner1', 'bg_id'\)[\s\S]+"
            r"\('outermost', 'inner2', 'bg_id'\)",
        ),
        (
            ("new_transfer__einkommen_eg"),
            {
                "arbeitslosengeld_2__eg_id": None,
                "grundsicherung__eg_id": None,
            },
            r"Group-by-identifier for target:[\s\S]+"
            r"\('new_transfer', 'einkommen_eg'\)[\s\S]+is ambiguous[\s\S]+"
            r"Found candidates[\s\S]+"
            r"\('arbeitslosengeld_2', 'eg_id'\)[\s\S]+"
            r"\('grundsicherung', 'eg_id'\)",
        ),
    ],
)
def test_get_name_of_group_by_id_fails(
    target_name, group_by_functions, expected_error_match
):
    with pytest.raises(ValueError, match=expected_error_match):
        get_name_of_group_by_id(
            target_name=target_name,
            group_by_functions=group_by_functions,
            groupings=("hh", "bg", "eg"),
        )


@pytest.mark.parametrize(
    (
        "base_name",
        "supported_time_conversions",
        "groupings",
        "create_conversions_for_time_units",
        "expected",
    ),
    [
        (
            "income",
            ["y", "m"],
            ["hh"],
            True,
            {"income_m", "income_y", "income_m_hh", "income_y_hh"},
        ),
        (
            "income",
            ["y", "m"],
            ["hh", "x"],
            True,
            {
                "income_m",
                "income_y",
                "income_m_hh",
                "income_y_hh",
                "income_m_x",
                "income_y_x",
            },
        ),
        (
            "claims_benefits",
            ["y", "m"],
            ["hh", "x"],
            False,
            {"claims_benefits", "claims_benefits_hh", "claims_benefits_x"},
        ),
    ],
)
def test_all_variations_of_base_name(
    base_name,
    supported_time_conversions,
    groupings,
    create_conversions_for_time_units,
    expected,
):
    assert (
        all_variations_of_base_name(
            base_name=base_name,
            supported_time_conversions=supported_time_conversions,
            groupings=groupings,
            create_conversions_for_time_units=create_conversions_for_time_units,
        )
        == expected
    )


@pytest.mark.parametrize(
    (
        "func_name",
        "supported_time_units",
        "groupings",
        "expected_base_name",
        "expected_time_unit",
        "expected_aggregation",
    ),
    [
        ("foo", ("m", "y"), ["hh"], "foo", None, None),
        ("foo_m_hh", ("m", "y"), ["hh"], "foo", "m", "hh"),
        ("foo_y_hh", ("m", "y"), ["hh"], "foo", "y", "hh"),
        ("foo_m", ("m", "y"), ["hh"], "foo", "m", None),
        ("foo_y", ("m", "y"), ["hh"], "foo", "y", None),
        ("foo_hh", ("m", "y"), ["hh"], "foo", None, "hh"),
        ("foo_hh_bar", ("m", "y"), ["hh"], "foo_hh_bar", None, None),
    ],
)
def test_get_re_pattern_for_time_units_and_groupings(
    func_name,
    supported_time_units,
    groupings,
    expected_base_name,
    expected_time_unit,
    expected_aggregation,
):
    result = get_re_pattern_for_all_time_units_and_groupings(
        supported_time_units=supported_time_units,
        groupings=groupings,
    )
    match = result.fullmatch(func_name)
    assert match.group("base_name") == expected_base_name
    assert match.group("time_unit") == expected_time_unit
    assert match.group("aggregation") == expected_aggregation


@pytest.mark.parametrize(
    (
        "base_name",
        "supported_time_units",
        "groupings",
        "expected_match",
    ),
    [
        ("foo", ["m", "y"], ["hh"], "foo_m_hh"),
        ("foo", ["m", "y"], ["hh", "x"], "foo_m"),
        ("foo", ["m", "y"], ["hh", "x"], "foo_hh"),
    ],
)
def test_get_re_pattern_for_some_base_name(
    base_name, supported_time_units, groupings, expected_match
):
    re_pattern = get_re_pattern_for_specific_time_units_and_groupings(
        base_name=base_name,
        all_time_units=supported_time_units,
        groupings=groupings,
    )
    assert re_pattern.fullmatch(expected_match)
