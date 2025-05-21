from __future__ import annotations

import datetime
import inspect
import re
import textwrap
from typing import TYPE_CHECKING, Any

import dags.tree as dt
import optree

from ttsim.config import numpy_or_jax as np

if TYPE_CHECKING:
    from ttsim.column_objects_param_function import PolicyFunction
    from ttsim.typing import (
        DashedISOString,
        GenericCallable,
        NestedColumnObjectsParamFunctions,
        NestedData,
    )


_DASHED_ISO_DATE_REGEX = re.compile(r"\d{4}-\d{2}-\d{2}")


def to_datetime(date: datetime.date | DashedISOString) -> datetime.date:
    if isinstance(date, datetime.date):
        return date
    if isinstance(date, str) and _DASHED_ISO_DATE_REGEX.fullmatch(date):
        return datetime.date.fromisoformat(date)
    else:
        raise ValueError(
            f"Date {date} neither matches the format YYYY-MM-DD nor is a datetime.date."
        )


def validate_date_range(start: datetime.date, end: datetime.date) -> None:
    if start > end:
        raise ValueError(f"The start date {start} must be before the end date {end}.")


def get_re_pattern_for_all_time_units_and_groupings(
    groupings: tuple[str, ...], time_units: tuple[str, ...]
) -> re.Pattern[str]:
    """Get a regex pattern for time units and groupings.

    The pattern matches strings in any of these formats:
    - <base_name>  (may contain underscores)
    - <base_name>_<time_unit>
    - <base_name>_<grouping>
    - <base_name>_<time_unit>_<grouping>

    Parameters
    ----------
    groupings
        The supported groupings.
    time_units
        The supported time units.

    Returns
    -------
    pattern
        The regex pattern.
    """
    re_units = "".join(time_units)
    re_groupings = "|".join(groupings)
    return re.compile(
        f"(?P<base_name>.*?)"
        f"(?:_(?P<time_unit>[{re_units}]))?"
        f"(?:_(?P<grouping>{re_groupings}))?"
        f"$"
    )


def group_pattern(groupings: tuple[str, ...]) -> re.Pattern[str]:
    return re.compile(
        f"(?P<base_name_with_time_unit>.*)_(?P<group>{'|'.join(groupings)})$"
    )


def get_re_pattern_for_specific_time_units_and_groupings(
    base_name: str,
    all_time_units: tuple[str, ...],
    groupings: tuple[str, ...],
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
    groupings
        The supported groupings.

    Returns
    -------
    pattern
        The regex pattern.
    """
    re_units = "".join(all_time_units)
    re_groupings = "|".join(groupings)
    return re.compile(
        f"(?P<base_name>{re.escape(base_name)})"
        f"(?:_(?P<time_unit>[{re_units}]))?"
        f"(?:_(?P<grouping>{re_groupings}))?"
        f"$"
    )


def get_base_name_and_grouping_suffix(match: re.Match[str]) -> tuple[str, str]:
    return (
        match.group("base_name"),
        f"_{match.group('grouping')}" if match.group("grouping") else "",
    )


def fail_if_multiple_time_units_for_same_base_name_and_group(
    base_names_and_groups_to_variations: dict[tuple[str, str], list[str]],
) -> None:
    invalid = {
        b: q for b, q in base_names_and_groups_to_variations.items() if len(q) > 1
    }
    if invalid:
        raise ValueError(f"Multiple time units for base names: {invalid}")


class KeyErrorMessage(str):
    """Subclass str to allow for line breaks in KeyError messages."""

    __slots__ = ()

    def __repr__(self) -> str:
        return str(self)


def format_list_linewise(some_list: list[Any]) -> str:  # type: ignore[type-arg, unused-ignore]
    formatted_list = '",\n    "'.join(some_list)
    return textwrap.dedent(
        """
        [
            "{formatted_list}",
        ]
        """
    ).format(formatted_list=formatted_list)


def create_tree_from_path_and_value(
    path: tuple[str], value: Any = None
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
    base: dict[str, Any], path_to_upsert: tuple[str], value_to_upsert: Any = None
) -> dict[str, Any]:
    """Update tree with a path and value.

    The path is a list of strings that represent the keys in the nested dictionary. If
    the path does not exist, it will be created. If the path already exists, the value
    will be updated.
    """
    to_upsert = create_tree_from_path_and_value(
        path=path_to_upsert, value=value_to_upsert
    )
    return upsert_tree(base=base, to_upsert=to_upsert)


def insert_path_and_value(
    base: dict[str, Any], path_to_insert: tuple[str], value_to_insert: Any = None
) -> dict[str, Any]:
    """Insert a path and value into a tree.

    The path is a list of strings that represent the keys in the nested dictionary. The
    path must not exist in base.
    """
    to_insert = create_tree_from_path_and_value(
        path=path_to_insert, value=value_to_insert
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
        {path: leaf for path, leaf in flat.items() if path in ref_paths}
    )
    difference = dt.unflatten_from_tree_paths(
        {path: leaf for path, leaf in flat.items() if path not in ref_paths}
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


def format_errors_and_warnings(text: str, width: int = 79) -> str:
    """Format our own exception messages and warnings by dedenting paragraphs and
    wrapping at the specified width. Mainly required because of messages are written as
    part of indented blocks in our source code.

    Parameters
    ----------
    text : str
        The text which can include multiple paragraphs separated by two newlines.
    width : int
        The text will be wrapped by `width` characters.

    Returns
    -------
    Correctly dedented, wrapped text.

    """
    text = text.lstrip("\n")
    paragraphs = text.split("\n\n")
    wrapped_paragraphs = []
    for paragraph in paragraphs:
        dedented_paragraph = textwrap.dedent(paragraph)
        wrapped_paragraph = textwrap.fill(dedented_paragraph, width=width)
        wrapped_paragraphs.append(wrapped_paragraph)

    formatted_text = "\n\n".join(wrapped_paragraphs)

    return formatted_text


def get_names_of_required_arguments(function: PolicyFunction) -> list[str]:
    """Get argument names without defaults.

    The detection of argument names also works for partialed functions.

    Examples
    --------
    >>> def func(a, b): pass
    >>> get_names_of_required_arguments(func)
    ['a', 'b']
    >>> def g(c=0): pass
    >>> get_names_of_required_arguments(g)
    []
    >>> import functools
    >>> func_ = functools.partial(func, a=1)
    >>> get_names_of_required_arguments(func_)
    ['b']

    """
    parameters = inspect.signature(function).parameters

    return [p for p in parameters if parameters[p].default == parameters[p].empty]


def remove_group_suffix(col: str, groupings: tuple[str, ...]) -> str:
    out = col
    for g in groupings:
        out = out.removesuffix(f"_{g}")

    return out


def join(
    foreign_key: np.ndarray,
    primary_key: np.ndarray,
    target: np.ndarray,
    value_if_foreign_key_is_missing: float | bool,
) -> np.ndarray:
    """
    Given a foreign key, find the corresponding primary key, and return the target at
    the same index as the primary key. When using Jax, does not work on String Arrays.

    Parameters
    ----------
    foreign_key : np.ndarray[Key]
        The foreign keys.
    primary_key : np.ndarray[Key]
        The primary keys.
    target : np.ndarray[Out]
        The targets in the same order as the primary keys.
    value_if_foreign_key_is_missing : Out
        The value to return if no matching primary key is found.

    Returns
    -------
    The joined array.
    """
    # For each foreign key and for each primary key, check if they match
    matches_foreign_key = foreign_key[:, None] == primary_key

    # For each foreign key, add a column with True at the end, to later fall back to
    # the value for unresolved foreign keys
    padded_matches_foreign_key = np.pad(
        matches_foreign_key, ((0, 0), (0, 1)), "constant", constant_values=True
    )

    # For each foreign key, compute the index of the first matching primary key
    indices = np.argmax(padded_matches_foreign_key, axis=1)

    # Add the value for unresolved foreign keys at the end of the target array
    padded_targets = np.pad(
        target, (0, 1), "constant", constant_values=value_if_foreign_key_is_missing
    )

    # Return the target at the index of the first matching primary key
    return padded_targets.take(indices)


def assert_valid_ttsim_pytree(
    tree: Any, leaf_checker: GenericCallable, tree_name: str
) -> None:
    """
    Recursively assert that a pytree meets the following conditions:
      - The tree is a dictionary.
      - All keys are strings.
      - All leaves satisfy a provided condition (leaf_checker).

    Parameters
    ----------
    tree : Any
         The tree to validate.
    leaf_checker : GenericCallable
         A function that takes a leaf and returns True if it is valid.
    tree_name : str
         The name of the tree (used for error messages).

    Raises
    ------
    TypeError
        If any branch or leaf does not meet the expected requirements.
    """

    def _assert_valid_ttsim_pytree(subtree: Any, current_key: tuple[str, ...]) -> None:
        def format_key_path(key_tuple: tuple[str, ...]) -> str:
            return "".join(f"[{k}]" for k in key_tuple)

        if not isinstance(subtree, dict):
            path_str = format_key_path(current_key)
            msg = format_errors_and_warnings(
                f"{tree_name}{path_str} must be a dict, got {type(subtree)}."
            )
            raise TypeError(msg)

        for key, value in subtree.items():
            new_key_path = (*current_key, key)
            if not isinstance(key, str):
                msg = format_errors_and_warnings(
                    f"Key {key} in {tree_name}{format_key_path(current_key)} must be a "
                    f"string but got {type(key)}."
                )
                raise TypeError(msg)
            if isinstance(value, dict):
                _assert_valid_ttsim_pytree(value, new_key_path)
            else:
                if not leaf_checker(value):
                    msg = format_errors_and_warnings(
                        f"Leaf at {tree_name}{format_key_path(new_key_path)} is "
                        f"invalid: got {value} of type {type(value)}."
                    )
                    raise TypeError(msg)

    _assert_valid_ttsim_pytree(tree, current_key=())


def get_name_of_group_by_id(
    target_name: str,
    groupings: tuple[str, ...],
) -> str | None:
    """Get the group-by-identifier name for some target.

    The group-by-identifier is the name of the group identifier that is embedded in the
    name of the target. E.g., "income_kin" has "kin_id" as its group-by-identifier.

    Parameters
    ----------
    target_name
        The name of the target.
    groupings
        The supported groupings.

    Returns
    -------
    The group-by-identifier, or an empty tuple if it is an individual-level variable.
    """
    for g in groupings:
        if target_name.endswith(f"_{g}"):
            return f"{g}_id"
    return None
