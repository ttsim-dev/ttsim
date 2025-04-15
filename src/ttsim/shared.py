from __future__ import annotations

import datetime
import inspect
import itertools
import re
import textwrap
from typing import TYPE_CHECKING, Any, TypeVar

import dags.tree as dt
import numpy
import optree

if TYPE_CHECKING:
    from ttsim.ttsim_objects import PolicyFunction
    from ttsim.typing import (
        DashedISOString,
        GenericCallable,
        NestedDataDict,
        NestedTTSIMObjectDict,
        QualNameTTSIMFunctionDict,
    )


_DASHED_ISO_DATE_REGEX = re.compile(r"\d{4}-\d{2}-\d{2}")


def to_datetime(date: datetime.date | DashedISOString):
    if isinstance(date, datetime.date):
        return date
    if isinstance(date, str) and _DASHED_ISO_DATE_REGEX.fullmatch(date):
        return datetime.date.fromisoformat(date)
    else:
        raise ValueError(
            f"Date {date} neither matches the format YYYY-MM-DD nor is a datetime.date."
        )


def validate_date_range(start: datetime.date, end: datetime.date):
    if start > end:
        raise ValueError(f"The start date {start} must be before the end date {end}.")


def get_re_pattern_for_all_time_units_and_groupings(
    groupings: tuple[str, ...], supported_time_units: tuple[str, ...]
) -> re.Pattern:
    """Get a regex pattern for time units and groupings.

    The pattern matches strings in any of these formats:
    - <base_name>  (may contain underscores)
    - <base_name>_<time_unit>
    - <base_name>_<aggregation>
    - <base_name>_<time_unit>_<aggregation>

    Parameters
    ----------
    groupings
        The supported groupings.
    supported_time_units
        The supported time units.

    Returns
    -------
    pattern
        The regex pattern.
    """
    re_units = "".join(supported_time_units)
    re_groupings = "|".join(groupings)
    return re.compile(
        f"(?P<base_name>.*?)"
        f"(?:_(?P<time_unit>[{re_units}]))?"
        f"(?:_(?P<aggregation>{re_groupings}))?"
        f"$"
    )


def get_re_pattern_for_specific_time_units_and_groupings(
    base_name: str,
    all_time_units: tuple[str, ...],
    groupings: tuple[str, ...],
) -> re.Pattern:
    """Get a regex for a specific base name with optional time unit and aggregation.

    The pattern matches strings in any of these formats:
    - <specific_base_name>
    - <specific_base_name>_<time_unit>
    - <specific_base_name>_<aggregation>
    - <specific_base_name>_<time_unit>_<aggregation>

    Parameters
    ----------
    base_name
        The specific base name to match.
    supported_time_units
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
        f"(?:_(?P<aggregation>{re_groupings}))?"
        f"$"
    )


def all_variations_of_base_name(
    base_name: str,
    supported_time_conversions: list[str],
    groupings: list[str],
    create_conversions_for_time_units: bool,
) -> set[str]:
    """Get possible derived function names given a base function name.

    Examples
    --------
    >>> all_variations_of_base_name(
        base_name="income",
        supported_time_conversions=["y", "m"],
        groupings=["hh"],
        create_conversions_for_time_units=True,
    )
    {'income_m', 'income_y', 'income_hh_y', 'income_hh_m'}

    >>> all_variations_of_base_name(
        base_name="claims_benefits",
        supported_time_conversions=["y", "m"],
        groupings=["hh"],
        create_conversions_for_time_units=False,
    )
    {'claims_benefits_hh'}

    Parameters
    ----------
    base_name
        The base function name.
    supported_time_conversions
        The supported time conversions.
    groupings
        The supported groupings.
    create_conversions_for_time_units
        Whether to create conversions for time units.

    Returns
    -------
    The names of all potential targets based on the base name.
    """
    result = set()
    if create_conversions_for_time_units:
        for time_unit in supported_time_conversions:
            result.add(f"{base_name}_{time_unit}")
        for time_unit, aggregation in itertools.product(
            supported_time_conversions, groupings
        ):
            result.add(f"{base_name}_{time_unit}_{aggregation}")
    else:
        result.add(base_name)
        for aggregation in groupings:
            result.add(f"{base_name}_{aggregation}")
    return result


class KeyErrorMessage(str):
    """Subclass str to allow for line breaks in KeyError messages."""

    __slots__ = ()

    def __repr__(self):
        return str(self)


def format_list_linewise(list_):
    formatted_list = '",\n    "'.join(list_)
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


def merge_trees(left: dict, right: dict) -> dict:
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

    if set(optree.tree_paths(left)) & set(optree.tree_paths(right)):
        raise ValueError("Conflicting paths in trees to merge.")

    return upsert_tree(base=left, to_upsert=right)


def upsert_tree(base: dict, to_upsert: dict) -> dict:
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
    tree_to_partition: NestedTTSIMObjectDict | NestedDataDict,
    reference_tree: NestedTTSIMObjectDict | NestedDataDict,
) -> tuple[
    NestedTTSIMObjectDict | NestedDataDict,
    NestedTTSIMObjectDict | NestedDataDict,
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


def get_names_of_arguments_without_defaults(function: PolicyFunction) -> list[str]:
    """Get argument names without defaults.

    The detection of argument names also works for partialed functions.

    Examples
    --------
    >>> def func(a, b): pass
    >>> get_names_of_arguments_without_defaults(func)
    ['a', 'b']
    >>> import functools
    >>> func_ = functools.partial(func, a=1)
    >>> get_names_of_arguments_without_defaults(func_)
    ['b']

    """
    parameters = inspect.signature(function).parameters

    return [p for p in parameters if parameters[p].default == parameters[p].empty]


def remove_group_suffix(col, groupings):
    out = col
    for g in groupings:
        out = out.removesuffix(f"_{g}")

    return out


Key: TypeVar = TypeVar("Key")
Out: TypeVar = TypeVar("Out")


def join_numpy(
    foreign_key: numpy.ndarray[Key],
    primary_key: numpy.ndarray[Key],
    target: numpy.ndarray[Out],
    value_if_foreign_key_is_missing: Out,
) -> numpy.ndarray[Out]:
    """
    Given a foreign key, find the corresponding primary key, and return the target at
    the same index as the primary key.

    Parameters
    ----------
    foreign_key : numpy.ndarray[Key]
        The foreign keys.
    primary_key : numpy.ndarray[Key]
        The primary keys.
    target : numpy.ndarray[Out]
        The targets in the same order as the primary keys.
    value_if_foreign_key_is_missing : Out
        The value to return if no matching primary key is found.

    Returns
    -------
    The joined array.
    """
    if len(numpy.unique(primary_key)) != len(primary_key):
        keys, counts = numpy.unique(primary_key, return_counts=True)
        duplicate_primary_keys = keys[counts > 1]
        msg = format_errors_and_warnings(
            f"Duplicate primary keys: {duplicate_primary_keys}",
        )
        raise ValueError(msg)

    invalid_foreign_keys = foreign_key[
        (foreign_key >= 0) & (~numpy.isin(foreign_key, primary_key))
    ]

    if len(invalid_foreign_keys) > 0:
        msg = format_errors_and_warnings(
            f"Invalid foreign keys: {invalid_foreign_keys}",
        )
        raise ValueError(msg)

    # For each foreign key and for each primary key, check if they match
    matches_foreign_key = foreign_key[:, None] == primary_key

    # For each foreign key, add a column with True at the end, to later fall back to
    # the value for unresolved foreign keys
    padded_matches_foreign_key = numpy.pad(
        matches_foreign_key, ((0, 0), (0, 1)), "constant", constant_values=True
    )

    # For each foreign key, compute the index of the first matching primary key
    indices = numpy.argmax(padded_matches_foreign_key, axis=1)

    # Add the value for unresolved foreign keys at the end of the target array
    padded_targets = numpy.pad(
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
    group_by_functions: QualNameTTSIMFunctionDict,
    groupings: tuple[str, ...],
) -> str:
    """Get the group-by-identifier name for some target.

    The group-by-identifier is the name of the group identifier that is embedded in the
    name of the target. E.g., "einkommen_hh" has "hh_id" as its group-by-identifier. In
    this sense, the group-by-identifiers live in a global namespace. We generally expect
    them to be unique.

    There is an exception, though: It is enough for them to be unique within the
    uppermost namespace. In that case, however, they cannot be used outside of that
    namespace.

    Parameters
    ----------
    target_name
        The name of the target.
    group_by_functions
        The group-by functions.

    Returns
    -------
    The group-by-identifier, or an empty tuple if it is an individual-level variable.
    """
    for g in groupings:
        if target_name.endswith(f"_{g}") and g == "hh":
            # Hardcode because hh_id is not part of the functions tree
            return "hh_id"
        elif target_name.endswith(f"_{g}"):
            return _select_group_by_id_from_candidates(
                candidate_names=[
                    p for p in group_by_functions if p.endswith(f"{g}_id")
                ],
                target_name=target_name,
            )
    return None


def _select_group_by_id_from_candidates(
    candidate_names: list[str],
    target_name: str,
) -> str:
    """Select the group-by-identifier name from the candidates.

    If there are multiple candidates, the function takes the one that shares the
    first part of the path (uppermost level of namespace) with the aggregation target.

    Raises
    ------
    ValueError
        Raised if the group-by-identifier is ambiguous.

    Parameters
    ----------
    candidates
        The candidates.
    target_path
        The target path.
    nice_target_name
        The nice target name.

    Returns
    -------
    The group-by-identifier.
    """
    if len(candidate_names) > 1:
        candidate_names_in_matching_namespace = [
            p
            for p in candidate_names
            if dt.tree_path_from_qual_name(p)[0]
            == dt.tree_path_from_qual_name(target_name)[0]
        ]
        if len(candidate_names_in_matching_namespace) == 1:
            return candidate_names_in_matching_namespace[0]
        else:
            _fail_because_of_ambiguous_group_by_identifier(
                candidate_names_in_matching_namespace=candidate_names_in_matching_namespace,
                all_candidate_names=candidate_names,
                target_name=target_name,
            )
    else:
        return candidate_names[0]


def _fail_because_of_ambiguous_group_by_identifier(
    candidate_names_in_matching_namespace: list[str],
    all_candidate_names: list[str],
    target_name: str,
):
    if len(candidate_names_in_matching_namespace) == 0:
        paths = "\n    ".join(
            [str(dt.tree_path_from_qual_name(p)) for p in all_candidate_names]
        )
    else:
        paths = "\n    ".join(
            [
                str(dt.tree_path_from_qual_name(p))
                for p in candidate_names_in_matching_namespace
            ]
        )

    target_path = dt.tree_path_from_qual_name(target_name)
    msg = format_errors_and_warnings(
        f"""
        Group-by-identifier for target:\n\n    {target_path}\n
        is ambiguous. Group-by-identifiers must be

        1. unique at the uppermost level of the functions tree.
        2. inside the uppermost namespace if there are namespaced identifiers

        Found candidates:\n\n    {paths}
        """
    )
    raise ValueError(msg)
