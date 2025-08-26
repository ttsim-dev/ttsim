from __future__ import annotations

import datetime
import re
from copy import copy
from typing import TYPE_CHECKING, Any, TypeAlias, overload

import dags.tree as dt
import optree

if TYPE_CHECKING:
    from ttsim.typing import (
        DashedISOString,
        NestedColumnObjectsParamFunctions,
        NestedData,
        OrderedQNames,
        PolicyEnvironment,
        SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
        SpecEnvWithPartialledParamsAndScalars,
        SpecEnvWithProcessedParamsAndScalars,
    )

    SomeEnv: TypeAlias = (
        PolicyEnvironment
        | SpecEnvWithoutTreeLogicAndWithDerivedFunctions
        | SpecEnvWithProcessedParamsAndScalars
        | SpecEnvWithPartialledParamsAndScalars
    )

_DASHED_ISO_DATE_REGEX = re.compile(r"\d{4}-\d{2}-\d{2}")


def to_datetime(date: datetime.date | DashedISOString) -> datetime.date:
    if isinstance(date, datetime.date):
        return date
    if isinstance(date, str) and _DASHED_ISO_DATE_REGEX.fullmatch(date):
        return datetime.date.fromisoformat(date)
    raise ValueError(
        f"Date {date} neither matches the format YYYY-MM-DD nor is a datetime.date.",
    )


def get_re_pattern_for_all_time_units_and_groupings(
    time_units: OrderedQNames,
    grouping_levels: OrderedQNames,
) -> re.Pattern[str]:
    """Get a regex pattern for time units and grouping_levels.

    The pattern matches strings in any of these formats:
    - <base_name>  (may contain underscores)
    - <base_name>_<time_unit>
    - <base_name>_<grouping>
    - <base_name>_<time_unit>_<grouping>

    Parameters
    ----------
    time_units
        The supported time units.
    grouping_levels
        The supported grouping levels.

    Returns
    -------
    pattern
        The regex pattern.
    """
    re_units = "".join(time_units)
    re_groupings = "|".join(grouping_levels)
    return re.compile(
        f"(?P<base_name>.*?)"
        f"(?:_(?P<time_unit>[{re_units}]))?"
        f"(?:_(?P<grouping>{re_groupings}))?"
        f"$",
    )


def group_pattern(grouping_levels: OrderedQNames) -> re.Pattern[str]:
    return re.compile(
        f"(?P<base_name_with_time_unit>.*)_(?P<group>{'|'.join(grouping_levels)})$",
    )


def get_re_pattern_for_specific_time_units_and_groupings(
    base_name: str,
    all_time_units: OrderedQNames,
    grouping_levels: OrderedQNames,
) -> re.Pattern[str]:
    """Get a regex for a specific base name with optional time unit and aggregation.

    The pattern matches strings in any of these formats:
    - <specific_base_name>
    - <specific_base_name>_<time_unit>
    - <specific_base_name>_<grouping>
    - <specific_base_name>_<time_unit>_<grouping>

    Parameters
    ----------
    base_name
        The specific base name to match.
    time_units
        The supported time units.
    grouping_levels
        The supported grouping levels.

    Returns
    -------
    pattern
        The regex pattern.
    """
    re_units = "".join(all_time_units)
    re_groupings = "|".join(grouping_levels)
    return re.compile(
        f"(?P<base_name>{re.escape(base_name)})"
        f"(?:_(?P<time_unit>[{re_units}]))?"
        f"(?:_(?P<grouping>{re_groupings}))?"
        f"$",
    )


def get_base_name_and_grouping_suffix(match: re.Match[str]) -> tuple[str, str]:
    return (
        match.group("base_name"),
        f"_{match.group('grouping')}" if match.group("grouping") else "",
    )


def create_tree_from_path_and_value(
    path: tuple[str],
    value: Any = None,  # noqa: ANN401
) -> dict[str, Any]:
    """Create a nested dict with 'path' as keys and 'value' as leaf.

    Almost the same as `dt.unflatten_from_tree_paths({path: value})`, except that
    it can also deal with an empty tuple as the path and a dictionary as the value.

    Example:
        Input:
            path = ("a", "b", "c")
            value = None
        Result:
            {"a": {"b": {"c": None}}}

    Parameters
    ----------
    path
        The path to create the tree structure from.
    value (Optional)
        The value to insert into the tree structure.

    Returns
    -------
    The tree structure.
    """
    nested_dict = value
    for entry in reversed(path):
        nested_dict = {entry: nested_dict}
    return nested_dict


def merge_trees(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """
    Merge two pytrees, raising an error if a path is present in both trees.

    Parameters
    ----------
    left
        The first tree to be merged.
    right
        The second tree to be merged.

    Returns
    -------
    The merged pytree.
    """
    if set(optree.tree_paths(left)) & set(optree.tree_paths(right)):  # type: ignore[arg-type]
        raise ValueError("Conflicting paths in trees to merge.")

    return upsert_tree(base=left, to_upsert=right)


def upsert_tree(base: dict[str, Any], to_upsert: dict[str, Any]) -> dict[str, Any]:
    """
    Upsert a tree into another tree for trees defined by dictionaries only.

    Note: In case of conflicting trees, the to_upsert takes precedence.

    Example:
        Input:
            base = {"a": {"b": {"c": None}}}
            to_upsert = {"a": {"b": {"d": None}}}
        Output:
            {"a": {"b": {"c": None, "d": None}}}

    Parameters
    ----------
    base
        The base dictionary.
    to_upsert
        The dictionary to update the base dictionary.

    Returns
    -------
    The merged dictionary.
    """
    result = base.copy()

    for key, value in to_upsert.items():
        base_value = result.get(key)
        if key in result and isinstance(base_value, dict) and isinstance(value, dict):
            result[key] = upsert_tree(base=base_value, to_upsert=value)
        else:
            result[key] = value

    return result


def upsert_path_and_value(
    base: dict[str, Any],
    path_to_upsert: tuple[str],
    value_to_upsert: Any = None,  # noqa: ANN401
) -> dict[str, Any]:
    """Update tree with a path and value.

    The path is a list of strings that represent the keys in the nested dictionary. If
    the path does not exist, it will be created. If the path already exists, the value
    will be updated.
    """
    to_upsert = create_tree_from_path_and_value(
        path=path_to_upsert,
        value=value_to_upsert,
    )
    return upsert_tree(base=base, to_upsert=to_upsert)


def insert_path_and_value(
    base: dict[str, Any],
    path_to_insert: tuple[str],
    value_to_insert: Any = None,  # noqa: ANN401
) -> dict[str, Any]:
    """Insert a path and value into a tree.

    The path is a list of strings that represent the keys in the nested dictionary. The
    path must not exist in base.
    """
    to_insert = create_tree_from_path_and_value(
        path=path_to_insert,
        value=value_to_insert,
    )
    return merge_trees(left=base, right=to_insert)


def partition_tree_by_reference_tree(
    tree_to_partition: NestedColumnObjectsParamFunctions | NestedData,
    reference_tree: NestedColumnObjectsParamFunctions | NestedData,
) -> tuple[
    NestedColumnObjectsParamFunctions | NestedData,
    NestedColumnObjectsParamFunctions | NestedData,
]:
    """
    Partition a tree into two based on the presence of its paths in a reference tree.

    Parameters
    ----------
    tree_to_partition
        The tree to be partitioned.
    reference_tree
        The reference tree used to determine the partitioning.

    Returns
    -------
    A tuple containing:
    - The first tree with leaves present in both trees.
    - The second tree with leaves absent in the reference tree.
    """
    ref_paths = set(dt.tree_paths(reference_tree))
    flat = dt.flatten_to_tree_paths(tree_to_partition)
    intersection = dt.unflatten_from_tree_paths(
        {path: leaf for path, leaf in flat.items() if path in ref_paths},
    )
    difference = dt.unflatten_from_tree_paths(
        {path: leaf for path, leaf in flat.items() if path not in ref_paths},
    )

    return intersection, difference


def partition_by_reference_dict(
    to_partition: dict[str, Any],
    reference_dict: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Partition a dictionary into two based on the presence of its keys in a reference
    dictionary.

    Parameters
    ----------
    to_partition
        The dictionary to be partitioned.
    reference_dict
        The reference dictionary used to determine the partitioning.

    Returns
    -------
    A tuple containing:
    - The first dictionary with keys present in both dictionaries.
    - The second dictionary with keys absent in the reference dictionary.
    """
    intersection = {k: v for k, v in to_partition.items() if k in reference_dict}
    difference = {k: v for k, v in to_partition.items() if k not in reference_dict}
    return intersection, difference


def remove_group_suffix(col: str, grouping_levels: OrderedQNames) -> str:
    out = col
    for g in grouping_levels:
        out = out.removesuffix(f"_{g}")

    return out


def get_name_of_group_by_id(
    target_name: str,
    grouping_levels: OrderedQNames,
) -> str | None:
    """Get the group-by-identifier name for some target.

    The group-by-identifier is the name of the group identifier that is embedded in the
    name of the target. E.g., "income_kin" has "kin_id" as its group-by-identifier.

    Parameters
    ----------
    target_name
        The name of the target.
    grouping_levels
        The supported grouping levels.

    Returns
    -------
    The group-by-identifier, or an empty tuple if it is an individual-level variable.
    """
    for g in grouping_levels:
        if target_name.endswith(f"_{g}"):
            return f"{g}_id"
    return None


@overload
def copy_environment(env: PolicyEnvironment) -> PolicyEnvironment: ...


@overload
def copy_environment(
    env: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
) -> SpecEnvWithoutTreeLogicAndWithDerivedFunctions: ...


@overload
def copy_environment(
    env: SpecEnvWithProcessedParamsAndScalars,
) -> SpecEnvWithProcessedParamsAndScalars: ...


@overload
def copy_environment(
    env: SpecEnvWithPartialledParamsAndScalars,
) -> SpecEnvWithPartialledParamsAndScalars: ...


def copy_environment(env: SomeEnv) -> SomeEnv:
    """Create a copy of a policy environment or other tree structure.

    This function creates a copy of nested tree structures that may contain objects
    that cannot be deep-copied due to unpickleable elements such as function objects.

    The function uses optree.tree_map with shallow copy to create independent copies
    of the tree structure while preserving references to functions and other objects
    that don't need to be copied.

    Parameters
    ----------
    env
        The environment to copy. Can be a PolicyEnvironment or any of the
        specialized environment types (SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
        SpecEnvWithProcessedParamsAndScalars, SpecEnvWithPartialledParamsAndScalars).

    Returns
    -------
    A copy of *env*, which is a deep copy for all practical purposes.

    """
    return optree.tree_map(copy, env)
