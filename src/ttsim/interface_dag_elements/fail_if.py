from __future__ import annotations

import datetime
import itertools
import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import dags.tree as dt
import networkx as nx
import numpy
import optree
import pandas as pd

from ttsim.interface_dag_elements.interface_node_objects import interface_function
from ttsim.interface_dag_elements.shared import get_name_of_group_by_id
from ttsim.tt_dag_elements.column_objects_param_function import (
    DEFAULT_END_DATE,
    ColumnFunction,
    ColumnObject,
    FKType,
    ParamFunction,
    PolicyInput,
)
from ttsim.tt_dag_elements.param_objects import ParamObject

if TYPE_CHECKING:
    from types import ModuleType

    from ttsim.interface_dag_elements.typing import (
        FlatColumnObjectsParamFunctions,
        FlatOrigParamSpecs,
        GenericCallable,
        NestedData,
        NestedPolicyEnvironment,
        NestedStrings,
        NestedTargetDict,
        OrderedQNames,
        OrigParamSpec,
        QNameData,
        QNameDataColumns,
        QNamePolicyEnvironment,
        UnorderedQNames,
    )


class KeyErrorMessage(str):
    """Subclass str to allow for line breaks in KeyError messages."""

    __slots__ = ()

    def __repr__(self) -> str:
        return str(self)


class ConflictingActivePeriodsError(Exception):
    def __init__(
        self,
        affected_column_objects: list[ColumnObject],
        path: OrderedQNames,
        overlap_start: datetime.date,
        overlap_end: datetime.date,
    ) -> None:
        self.affected_column_objects = affected_column_objects
        self.path = path
        self.overlap_start = overlap_start
        self.overlap_end = overlap_end

    def __str__(self) -> str:
        overlapping_objects = [
            obj.__getattribute__("original_function_name")
            for obj in self.affected_column_objects
            if obj
        ]
        return f"""
        Functions with path

          {self.path}

        have overlapping start and end dates. The following functions are affected:

          {
            '''
          '''.join(overlapping_objects)
        }

        Overlap from {self.overlap_start} to {self.overlap_end}."""


@dataclass(frozen=True)
class _ParamWithActivePeriod(ParamObject):
    """A ParamObject object which mimics a ColumnObject regarding active periods.

    Only used here for checking overlap.
    """

    original_function_name: str


def assert_valid_ttsim_pytree(
    tree: Any,  # noqa: ANN401
    leaf_checker: GenericCallable,
    tree_name: str,
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

    def _assert_valid_ttsim_pytree(subtree: Any, current_key: tuple[str, ...]) -> None:  # noqa: ANN401
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


@interface_function()
def active_periods_overlap(
    orig_policy_objects__column_objects_and_param_functions: FlatColumnObjectsParamFunctions,
    orig_policy_objects__param_specs: FlatOrigParamSpecs,
) -> None:
    """Fail because active periods of objects / parameters overlap.

    Checks that objects or parameters with the same tree path / qualified name are not
    active at the same time.

    Raises
    ------
    ConflictingActivePeriodsError
        If multiple objects and/or parameters with the same leaf name are active at the
        same time.
    """
    # Create mapping from leaf names to objects.
    overlap_checker: dict[
        tuple[str, ...], list[ColumnObject | ParamFunction | _ParamWithActivePeriod]
    ] = {}
    for (
        orig_path,
        obj,
    ) in orig_policy_objects__column_objects_and_param_functions.items():
        path = (*orig_path[:-2], obj.leaf_name)
        if path in overlap_checker:
            overlap_checker[path].append(obj)
        else:
            overlap_checker[path] = [obj]

    for orig_path, obj in orig_policy_objects__param_specs.items():
        path = (*orig_path[:-2], orig_path[-1])
        if path in overlap_checker:
            overlap_checker[path].extend(
                _param_with_active_periods(param_spec=obj, leaf_name=orig_path[-1])
            )
        else:
            overlap_checker[path] = _param_with_active_periods(
                param_spec=obj, leaf_name=orig_path[-1]
            )

    # Check for overlapping start and end dates for time-dependent functions.
    for path, objects in overlap_checker.items():
        active_period = [(f.start_date, f.end_date) for f in objects]
        for (start1, end1), (start2, end2) in itertools.combinations(active_period, 2):
            if start1 <= end2 and start2 <= end1:
                raise ConflictingActivePeriodsError(
                    affected_column_objects=objects,
                    path=path,
                    overlap_start=max(start1, start2),
                    overlap_end=min(end1, end2),
                )


@interface_function()
def any_paths_are_invalid(
    policy_environment: NestedPolicyEnvironment,
    input_data__tree: NestedData,
    targets__tree: NestedTargetDict,
    names__top_level_namespace: UnorderedQNames,
) -> None:
    """Thin wrapper around `dt.fail_if_paths_are_invalid`."""
    return dt.fail_if_paths_are_invalid(
        functions=policy_environment,
        input_data__tree=input_data__tree,
        targets=targets__tree,
        names__top_level_namespace=names__top_level_namespace,
    )


@interface_function()
def data_paths_are_missing_in_paths_to_column_names(
    results__tree: NestedData,
    targets__tree: NestedStrings,
) -> None:
    """Fail if the data paths are missing in the paths to column names."""
    paths_in_data = dt.flatten_to_tree_paths(results__tree)
    paths_in_mapper = dt.flatten_to_tree_paths(targets__tree)
    missing_paths = [str(p) for p in paths_in_mapper if p not in paths_in_data]
    if missing_paths:
        msg = format_errors_and_warnings(
            "Converting the nested data to a DataFrame failed because the following "
            "paths are not mapped to a column name: "
            f"{format_list_linewise(list(missing_paths))}"
        )
        raise ValueError(msg)


@interface_function()
def input_data_tree_is_invalid(input_data__tree: NestedData, xnp: ModuleType) -> None:
    """
    Validate the basic structure of the data tree.

    1. It must be is a dictionary with string keys and Series or Array leaves.
    2. It must contain the `p_id` column.
    3. Each element of `p_id` must uniquely identify a row.

    Parameters
    ----------
    input_data__tree
        The data tree.

    Raises
    ------
    ValueError
        If any of the above conditions is not met.
    """
    assert_valid_ttsim_pytree(
        tree=input_data__tree,
        leaf_checker=lambda leaf: isinstance(
            leaf, int | pd.Series | numpy.ndarray | xnp.ndarray
        ),
        tree_name="input_data__tree",
    )
    p_id = input_data__tree.get("p_id", None)
    if p_id is None:
        raise ValueError("The input data must contain the `p_id` column.")

    # Check for non-unique p_ids
    p_id_counts: dict[int, int] = {}
    # Need the map because Jax loop items are 1-element arrays.
    for i in map(int, p_id):
        if i in p_id_counts:
            p_id_counts[i] += 1
        else:
            p_id_counts[i] = 1

    non_unique_p_ids = [i for i, count in p_id_counts.items() if count > 1]

    if non_unique_p_ids:
        message = (
            "The following `p_id`s are not unique in the input data:\n\n"
            f"{non_unique_p_ids}\n\n"
        )
        raise ValueError(message)


@interface_function()
def environment_is_invalid(
    policy_environment: NestedPolicyEnvironment,
) -> None:
    """Validate that the environment is a pytree with supported types."""
    assert_valid_ttsim_pytree(
        tree=policy_environment,
        leaf_checker=lambda leaf: isinstance(
            leaf, ColumnObject | ParamFunction | ParamObject
        ),
        tree_name="policy_environment",
    )


@interface_function()
def foreign_keys_are_invalid_in_data(
    names__root_nodes: UnorderedQNames,
    processed_data: QNameData,
    specialized_environment__with_derived_functions_and_processed_input_nodes: QNamePolicyEnvironment,
) -> None:
    """
    Check that all foreign keys are valid.

    Foreign keys must point to an existing `p_id` in the input data and must not refer
    to the `p_id` of the same row.

    We need processed_data because we cannot guarantee that `p_id` is present in the
    input data.
    """
    valid_ids = set(processed_data["p_id"].tolist()) | {-1}
    relevant_objects = {
        k: v
        for k, v in specialized_environment__with_derived_functions_and_processed_input_nodes.items()
        if isinstance(v, PolicyInput | ColumnFunction)
    }

    for fk_name, fk in relevant_objects.items():
        if fk.foreign_key_type == FKType.IRRELEVANT:
            continue
        elif fk_name in names__root_nodes:
            path = dt.tree_path_from_qual_name(fk_name)
            # Referenced `p_id` must exist in the input data
            if not all(i in valid_ids for i in processed_data[fk_name].tolist()):
                message = format_errors_and_warnings(
                    f"""
                    For {path}, the following are not a valid p_id in the input
                    data: {[i for i in processed_data[fk_name] if i not in valid_ids]}.
                    """
                )
                raise ValueError(message)

            if fk.foreign_key_type == FKType.MUST_NOT_POINT_TO_SELF:
                equal_to_pid_in_same_row = [
                    i
                    for i, j in zip(
                        processed_data[fk_name].tolist(),
                        processed_data["p_id"].tolist(),
                        strict=False,
                    )
                    if i == j
                ]
                if any(equal_to_pid_in_same_row):
                    message = format_errors_and_warnings(
                        f"""
                        For {path}, the following are equal to the p_id in the same
                        row: {equal_to_pid_in_same_row}.
                        """
                    )
                    raise ValueError(message)


@interface_function()
def group_ids_are_outside_top_level_namespace(
    policy_environment: NestedPolicyEnvironment,
) -> None:
    """Fail if group ids are outside the top level namespace."""
    group_ids_outside_top_level_namespace = {
        tree_path
        for tree_path in dt.flatten_to_tree_paths(policy_environment)
        if len(tree_path) > 1 and tree_path[-1].endswith("_id")
    }
    if group_ids_outside_top_level_namespace:
        raise ValueError(
            "Group identifiers must live in the top-level namespace. Got:\n\n"
            f"{group_ids_outside_top_level_namespace}\n\n"
            "To fix this error, move the group identifiers to the top-level namespace."
        )


@interface_function()
def group_variables_are_not_constant_within_groups(
    names__grouping_levels: OrderedQNames,
    names__root_nodes: UnorderedQNames,
    processed_data: QNameData,
) -> None:
    """
    Check that group variables are constant within each group.

    Parameters
    ----------
    data
        Dictionary of data.
    grouping_levels
        The grouping levels available in the policy environment.
    """
    faulty_data_columns = []

    for name in names__root_nodes:
        group_by_id = get_name_of_group_by_id(
            target_name=name,
            grouping_levels=names__grouping_levels,
        )
        if group_by_id in processed_data:
            group_by_id_series = pd.Series(processed_data[group_by_id])
            leaf_series = pd.Series(processed_data[name])
            unique_counts = leaf_series.groupby(group_by_id_series).nunique(
                dropna=False
            )
            if not (unique_counts == 1).all():
                faulty_data_columns.append(name)

    if faulty_data_columns:
        formatted = format_list_linewise(faulty_data_columns)
        msg = format_errors_and_warnings(
            f"""The following data inputs do not have a unique value within
                each group defined by the provided grouping IDs:

                {formatted}

                To fix this error, assign the same value to each group.
                """
        )
        raise ValueError(msg)


@interface_function()
def non_convertible_objects_in_results_tree(
    processed_data: QNameData,
    results__tree: NestedData,
    xnp: ModuleType,
) -> None:
    """Fail if results should be converted to a DataFrame but contain non-convertible
    objects.
    """
    _numeric_types = (int, float, bool, xnp.integer, xnp.floating, xnp.bool_)
    expected_object_length = len(next(iter(processed_data.values())))

    paths_with_incorrect_types: list[str] = []
    paths_with_incorrect_length: list[str] = []
    for path, data in dt.flatten_to_tree_paths(results__tree).items():
        if isinstance(data, xnp.ndarray) and len(data) not in {
            1,
            expected_object_length,
        }:
            paths_with_incorrect_length.append(str(path))
        elif isinstance(data, _numeric_types):
            continue
        else:
            paths_with_incorrect_types.append(str(path))

    if paths_with_incorrect_types:
        msg = format_errors_and_warnings(
            "The data contains objects that cannot be cast to a pandas.DataFrame "
            "column. Make sure that the requested targets return scalars or arrays of "
            "scalars only. The following paths contain incompatible objects: "
            f"{format_list_linewise(paths_with_incorrect_types)}"
        )
        raise TypeError(msg)
    if paths_with_incorrect_length:
        msg = format_errors_and_warnings(
            "The data contains paths that don't have the same length as the input data "
            "and are not scalars. The following paths are faulty: "
            f"{format_list_linewise(paths_with_incorrect_length)}"
        )
        raise ValueError(msg)


@interface_function()
def input_df_has_bool_or_numeric_column_names(
    input_data__df_and_mapper__df: pd.DataFrame,
) -> None:
    """Fail if the DataFrame has bool or numeric column names."""
    common_msg = format_errors_and_warnings(
        """DataFrame column names cannot be booleans or numbers. This restriction
        prevents ambiguity between actual column references and values intended for
        broadcasting.
        """
    )
    bool_column_names = [
        col for col in input_data__df_and_mapper__df.columns if isinstance(col, bool)
    ]
    numeric_column_names = [
        col
        for col in input_data__df_and_mapper__df.columns
        if isinstance(col, (int, float)) or (isinstance(col, str) and col.isnumeric())
    ]

    if bool_column_names or numeric_column_names:
        msg = format_errors_and_warnings(
            f"""
            {common_msg}

            Boolean column names: {bool_column_names}.
            Numeric column names: {numeric_column_names}.
            """
        )
        raise ValueError(msg)


@interface_function()
def input_df_mapper_columns_missing_in_df(
    input_data__df_and_mapper__df: pd.DataFrame,
    input_data__df_and_mapper__mapper: NestedStrings,
) -> None:
    """Fail if the input mapper has columns that are not in the input dataframe."""
    mapper_vals = dt.flatten_to_qual_names(input_data__df_and_mapper__mapper).values()
    missing_columns = [
        col for col in mapper_vals if col not in input_data__df_and_mapper__df.columns
    ]
    if missing_columns:
        msg = format_errors_and_warnings(
            "All columns in the input mapper must be present in the input dataframe. "
            f"The following columns are missing: {missing_columns}"
        )
        raise ValueError(msg)


@interface_function()
def input_df_mapper_has_incorrect_format(
    input_data__df_and_mapper__mapper: NestedStrings,
) -> None:
    """Fail if the input tree to column name mapping has an incorrect format."""
    if not isinstance(input_data__df_and_mapper__mapper, dict):
        msg = format_errors_and_warnings(
            """The inputs tree to column mapping must be a (nested) dictionary. Call
            `dags.tree.create_tree_with_input_types` to create a template."""
        )
        raise TypeError(msg)

    non_string_paths = [
        str(path)
        for path in optree.tree_paths(
            input_data__df_and_mapper__mapper,  # type: ignore[arg-type]
            none_is_leaf=True,
        )
        if not all(isinstance(part, str) for part in path)
    ]
    if non_string_paths:
        msg = format_errors_and_warnings(
            f"""All path elements of `inputs_tree_to_df_columns` must be strings.
            Found the following paths that contain non-string elements:

            {format_list_linewise(non_string_paths)}

            Call `dags.tree.create_tree_with_input_types` to create a template.
            """
        )
        raise TypeError(msg)

    incorrect_types = {
        k: type(v)
        for k, v in dt.flatten_to_qual_names(input_data__df_and_mapper__mapper).items()
        if not isinstance(v, str | int | float | bool)
    }
    if incorrect_types:
        formatted_incorrect_types = "\n".join(
            f"    - {k}: {v.__name__}" for k, v in incorrect_types.items()
        )
        msg = format_errors_and_warnings(
            f"""Values of the input tree to column mapping must be strings, integers,
            floats, or Booleans.
            Found the following incorrect types:

            {formatted_incorrect_types}
            """
        )
        raise TypeError(msg)


@interface_function()
def root_nodes_are_missing(
    specialized_environment__tax_transfer_dag: nx.DiGraph,
    processed_data: QNameData,
) -> None:
    """Fail if root nodes are missing.

    Parameters
    ----------
    specialized_environment__tax_transfer_dag
        The DAG of taxes and transfers functions.
    processed_data
        The data tree in qualified name representation.

    Raises
    ------
    ValueError
        If root nodes are missing.
    """
    # Obtain root nodes
    root_nodes = nx.subgraph_view(
        specialized_environment__tax_transfer_dag,
        filter_node=lambda n: specialized_environment__tax_transfer_dag.in_degree(n)
        == 0,
    ).nodes

    missing_nodes = [node for node in root_nodes if node not in processed_data]

    if missing_nodes:
        formatted = format_list_linewise(
            [str(dt.tree_path_from_qual_name(mn)) for mn in missing_nodes]
        )
        raise ValueError(f"The following data columns are missing.\n{formatted}")


@interface_function()
def targets_are_not_in_policy_environment_or_data(
    policy_environment: QNamePolicyEnvironment,
    names__processed_data_columns: QNameDataColumns,
    targets__qname: OrderedQNames,
) -> None:
    """Fail if some target is not among functions.

    Parameters
    ----------
    functions
        Dictionary containing functions to build the DAG.
    names__processed_data_columns
        The columns which are available in the data tree.
    targets
        The targets which should be computed. They limit the DAG in the way that only
        ancestors of these nodes need to be considered.

    Raises
    ------
    ValueError
        Raised if any member of `targets` is not among functions.

    """
    targets_not_in_policy_environment_or_data = [
        str(dt.tree_path_from_qual_name(n))
        for n in targets__qname
        if n not in policy_environment and n not in names__processed_data_columns
    ]
    if targets_not_in_policy_environment_or_data:
        formatted = format_list_linewise(targets_not_in_policy_environment_or_data)
        msg = format_errors_and_warnings(
            f"The following targets have no corresponding function:\n\n{formatted}"
        )
        raise ValueError(msg)


@interface_function()
def targets_tree_is_invalid(targets__tree: NestedTargetDict) -> None:
    """
    Validate that the targets tree is a dictionary with string keys and None leaves.
    """
    assert_valid_ttsim_pytree(
        tree=targets__tree,
        leaf_checker=lambda leaf: isinstance(leaf, (None | str)),
        tree_name="targets__tree",
    )


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


def format_list_linewise(some_list: list[Any]) -> str:  # type: ignore[type-arg, unused-ignore]
    formatted_list = '",\n    "'.join(some_list)
    return textwrap.dedent(
        """
        [
            "{formatted_list}",
        ]
        """
    ).format(formatted_list=formatted_list)


def _param_with_active_periods(
    param_spec: OrigParamSpec,
    leaf_name: str,
) -> list[_ParamWithActivePeriod]:
    """Return parameter with active periods."""

    def _remove_note_and_reference(entry: dict[str | int, Any]) -> dict[str | int, Any]:
        """Remove note and reference from a parameter specification."""
        entry.pop("note", None)
        entry.pop("reference", None)
        return entry

    relevant = sorted(
        [key for key in param_spec if isinstance(key, datetime.date)],
        reverse=True,
    )
    if not relevant:
        raise ValueError(f"No relevant dates found for {param_spec}")

    params_header = {
        "name": param_spec["name"],
        "description": param_spec["description"],
        "unit": param_spec["unit"],
        "reference_period": param_spec["reference_period"],
    }
    out = []
    start_date: datetime.date | None = None
    end_date = DEFAULT_END_DATE
    for date in relevant:
        if _remove_note_and_reference(param_spec[date]):
            start_date = date
        else:
            if start_date:
                out.append(
                    _ParamWithActivePeriod(
                        leaf_name=leaf_name,
                        start_date=start_date,
                        end_date=end_date,
                        original_function_name=leaf_name,
                        **params_header,
                    )
                )
            start_date = None
            end_date = date - datetime.timedelta(days=1)
    if start_date:
        out.append(
            _ParamWithActivePeriod(
                leaf_name=leaf_name,
                original_function_name=leaf_name,
                start_date=start_date,
                end_date=end_date,
                **params_header,
            )
        )

    return out
