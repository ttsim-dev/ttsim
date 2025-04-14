from __future__ import annotations

import functools
import inspect
import warnings
from typing import TYPE_CHECKING, Any

import dags
import dags.tree as dt
import networkx as nx
import pandas as pd

from ttsim.combine_functions import (
    combine_policy_functions_and_derived_functions,
)
from ttsim.config import numpy_or_jax as np
from ttsim.policy_environment import PolicyEnvironment
from ttsim.shared import (
    all_variations_of_base_name,
    assert_valid_ttsim_pytree,
    format_errors_and_warnings,
    format_list_linewise,
    get_name_of_group_by_id,
    get_names_of_arguments_without_defaults,
    get_re_pattern_for_all_time_units_and_groupings,
    merge_trees,
    partition_by_reference_dict,
)
from ttsim.time_conversion import TIME_UNITS
from ttsim.ttsim_objects import (
    FKType,
    GroupCreationFunction,
    PolicyInput,
    TTSIMFunction,
)

if TYPE_CHECKING:
    from ttsim.typing import (
        NestedDataDict,
        NestedTargetDict,
        QualNameDataDict,
        QualNamePolicyInputDict,
        QualNameTargetList,
        QualNameTTSIMFunctionDict,
    )


def compute_taxes_and_transfers(
    data_tree: NestedDataDict,
    environment: PolicyEnvironment,
    targets_tree: NestedTargetDict,
    supported_groupings: tuple[str, ...],
    rounding: bool = True,
    debug: bool = False,
) -> NestedDataDict:
    """Compute taxes and transfers.

    Parameters
    ----------
    data_tree : NestedDataDict
        Data provided by the user.
    environment: PolicyEnvironment
        The policy environment which contains all necessary functions and parameters.
    targets_tree : NestedTargetDict | None
        The targets tree.
    rounding : bool, default True
        Indicator for whether rounding should be applied as specified in the law.
    debug : bool
        If debug is 'True', `compute_taxes_and_transfers` returns the input data tree
        along with the computed targets.

    Returns
    -------
    results : NestedDataDict
        The computed variables as a tree.

    """

    # Check user inputs
    _fail_if_targets_tree_not_valid(targets_tree)
    _fail_if_data_tree_not_valid(data_tree)
    _fail_if_environment_not_valid(environment)

    # Transform functions tree to qualified names dict with qualified arguments
    top_level_namespace = _get_top_level_namespace(
        environment=environment,
        supported_time_conversions=tuple(TIME_UNITS.keys()),
        supported_groupings=supported_groupings,
    )
    # Flatten nested objects to qualified names
    targets = dt.qual_names(targets_tree)
    data = dt.flatten_to_qual_names(data_tree)
    aggregation_specs = dt.flatten_to_qual_names(environment.aggregation_specs_tree)
    functions: QualNameTTSIMFunctionDict = {}
    inputs: QualNamePolicyInputDict = {}
    for name, f_or_i in dt.flatten_to_qual_names(environment.raw_objects_tree).items():
        if isinstance(f_or_i, TTSIMFunction):
            functions[name] = dt.one_function_without_tree_logic(
                function=f_or_i,
                tree_path=dt.tree_path_from_qual_name(name),
                top_level_namespace=top_level_namespace,
            )
        elif isinstance(f_or_i, PolicyInput):
            inputs[name] = f_or_i
        else:
            raise TypeError(f"Unknown type: {type(f_or_i)}")

    # Add derived functions to the qualified functions tree.
    functions = combine_policy_functions_and_derived_functions(
        functions=functions,
        aggregation_specs_from_environment=aggregation_specs,
        targets=targets,
        data=data,
        inputs=inputs,
        top_level_namespace=top_level_namespace,
        supported_groupings=supported_groupings,
    )

    functions_overridden, functions_to_be_used = partition_by_reference_dict(
        to_partition=functions,
        reference_dict=data,
    )

    _warn_if_functions_overridden_by_data(functions_overridden)

    functions_with_rounding_specs = (
        _add_rounding_to_functions(functions=functions_to_be_used)
        if rounding
        else functions_to_be_used
    )
    functions_with_partialled_parameters = _partial_parameters_to_functions(
        functions=functions_with_rounding_specs,
        params=environment.params,
    )

    # Remove unnecessary elements from user-provided data.
    input_data = _create_input_data_for_concatenated_function(
        data=data,
        functions=functions_with_partialled_parameters,
        targets=targets,
    )

    _fail_if_group_variables_not_constant_within_groups(
        data=input_data,
        functions=functions,
        supported_groupings=supported_groupings,
    )
    # Hack until correct behavior is implemented
    _input_data_with_p_id = input_data.copy()
    _input_data_with_p_id["p_id"] = data["p_id"].copy()
    _fail_if_foreign_keys_are_invalid_in_data(
        data=_input_data_with_p_id,
        policy_inputs=inputs,
    )

    tax_transfer_function = dags.concatenate_functions(
        functions=functions_with_partialled_parameters,
        targets=targets,
        return_type="dict",
        aggregator=None,
        enforce_signature=True,
    )

    results = tax_transfer_function(**input_data)

    result_tree = dt.unflatten_from_qual_names(results)

    if debug:
        result_tree = merge_trees(
            left=result_tree,
            right=dt.unflatten_from_qual_names(input_data),
        )

    return result_tree


def _get_top_level_namespace(
    environment: PolicyEnvironment,
    supported_time_conversions: tuple[str, ...],
    supported_groupings: tuple[str, ...],
) -> set[str]:
    """Get the top level namespace.

    Parameters
    ----------
    environment:
        The policy environment.

    Returns
    -------
    top_level_namespace:
        The top level namespace.
    """
    direct_top_level_names = set(environment.raw_objects_tree.keys()) | set(
        environment.aggregation_specs_tree.keys()
    )
    re_pattern = get_re_pattern_for_all_time_units_and_groupings(
        supported_groupings=supported_groupings,
        supported_time_units=supported_time_conversions,
    )

    all_top_level_names = set()
    for name in direct_top_level_names:
        match = re_pattern.fullmatch(name)
        base_name = match.group("base_name")
        create_conversions_for_time_units = bool(match.group("time_unit"))

        all_top_level_names_for_name = all_variations_of_base_name(
            base_name=base_name,
            supported_time_conversions=supported_time_conversions,
            supported_groupings=supported_groupings,
            create_conversions_for_time_units=create_conversions_for_time_units,
        )
        all_top_level_names.update(all_top_level_names_for_name)

    return all_top_level_names


def _create_input_data_for_concatenated_function(
    data: QualNameDataDict,
    functions: QualNameTTSIMFunctionDict,
    targets: QualNameTargetList,
) -> QualNameDataDict:
    """Create input data for the concatenated function.

    1. Check that all root nodes are present in the user-provided data.
    2. Get only part of the data that is needed for the concatenated function.
    3. Convert pandas.Series to numpy.array.

    Parameters
    ----------
    data
        Data provided by the user.
    functions
        Nested function dictionary.
    targets
        Targets provided by the user.


    Returns
    -------
    Inputs for the concatenated function.

    """
    # Create dag using processed functions
    dag = dags.create_dag(functions=functions, targets=targets)

    # Create root nodes tree
    root_nodes = nx.subgraph_view(
        dag, filter_node=lambda n: dag.in_degree(n) == 0
    ).nodes

    _fail_if_root_nodes_are_missing(
        functions=functions,
        data=data,
        root_nodes=root_nodes,
    )

    # Get only part of the data tree that is needed
    return {k: np.array(v) for k, v in data.items() if k in root_nodes}


def _partial_parameters_to_functions(
    functions: QualNameTTSIMFunctionDict,
    params: dict[str, Any],
) -> QualNameTTSIMFunctionDict:
    """Round and partial parameters into functions.

    Parameters
    ----------
    functions
        The functions dict with qualified function names as keys and functions as
        values.
    params
        Dictionary of parameters.

    Returns
    -------
    Functions tree with parameters partialled.

    """
    # Partial parameters to functions such that they disappear in the DAG.
    # Note: Needs to be done after rounding such that dags recognizes partialled
    # parameters.
    processed_functions = {}
    for name, function in functions.items():
        arguments = get_names_of_arguments_without_defaults(function)
        partial_params = {
            arg: params[key]
            for arg in arguments
            for key in params
            if arg.endswith(f"{key}_params")
        }
        if partial_params:
            processed_functions[name] = functools.partial(function, **partial_params)
        else:
            processed_functions[name] = function

    return processed_functions


def _add_rounding_to_functions(
    functions: QualNameTTSIMFunctionDict,
) -> QualNameTTSIMFunctionDict:
    """Add appropriate rounding of outputs to function.

    Parameters
    ----------
    functions
        Functions to which rounding should be added.

    Returns
    -------
    Function with rounding added.

    """
    return {
        name: func.rounding_spec.apply_rounding(func)
        if getattr(func, "rounding_spec", False)
        else func
        for name, func in functions.items()
    }


def _fail_if_environment_not_valid(environment: Any) -> None:
    """
    Validate that the environment is a PolicyEnvironment.
    """
    if not isinstance(environment, PolicyEnvironment):
        raise TypeError(
            f"The environment must be a PolicyEnvironment, got {type(environment)}."
        )


def _fail_if_targets_tree_not_valid(targets_tree: NestedTargetDict) -> None:
    """
    Validate that the targets tree is a dictionary with string keys and None leaves.
    """
    assert_valid_ttsim_pytree(
        tree=targets_tree,
        leaf_checker=lambda leaf: leaf is None,
        tree_name="targets_tree",
    )


def _fail_if_data_tree_not_valid(data_tree: NestedDataDict) -> None:
    """
    Validate that the data tree is a dictionary with string keys and pd.Series or
    np.ndarray leaves.
    """
    assert_valid_ttsim_pytree(
        tree=data_tree,
        leaf_checker=lambda leaf: isinstance(leaf, pd.Series | np.ndarray),
        tree_name="data_tree",
    )
    _fail_if_p_id_is_non_unique(data_tree)


def _fail_if_group_variables_not_constant_within_groups(
    data: QualNameDataDict,
    functions: QualNameTTSIMFunctionDict,
    supported_groupings: tuple[str, ...],
) -> None:
    """
    Check that group variables are constant within each group.

    If the user provides a supported grouping ID (see SUPPORTED_GROUPINGS in config.py),
    the function checks that the corresponding data is constant within each group.

    Parameters
    ----------
    data
        Dictionary of data.
    functions
        Dictionary of functions.
    """
    group_by_functions = {
        k: v
        for k, v in functions.items()
        if isinstance(getattr(v, "__wrapped__", v), GroupCreationFunction)
    }

    faulty_data_columns = []

    for name, data_column in data.items():
        group_by_id = get_name_of_group_by_id(
            target_name=name,
            group_by_functions=group_by_functions,
            supported_groupings=supported_groupings,
        )
        if group_by_id in data:
            group_by_id_series = pd.Series(data[group_by_id])
            leaf_series = pd.Series(data_column)
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


def _fail_if_p_id_is_non_unique(data_tree: NestedDataDict) -> None:
    """Check that pid is unique."""
    p_id = data_tree.get("p_id", None)
    if p_id is None:
        raise ValueError("The input data must contain the p_id.")

    # Check for non-unique p_ids
    p_id_counts = {}
    for i in p_id:
        if i in p_id_counts:
            p_id_counts[i] += 1
        else:
            p_id_counts[i] = 1

    non_unique_p_ids = [i for i, count in p_id_counts.items() if count > 1]

    if non_unique_p_ids:
        message = (
            f"The following p_ids are non-unique in the input data:{non_unique_p_ids}"
        )
        raise ValueError(message)


def _fail_if_foreign_keys_are_invalid_in_data(
    data: QualNameDataDict,
    policy_inputs: QualNamePolicyInputDict,
) -> None:
    """
    Check that all foreign keys are valid.

    Foreign keys must point to an existing `p_id` in the input data and must not refer
    to the `p_id` of the same row.
    """

    valid_ids = set(data["p_id"]) | {-1}

    for fk_name, fk in policy_inputs.items():
        if fk.foreign_key_type == FKType.IRRELEVANT:
            continue
        elif fk_name in data:
            path = dt.tree_path_from_qual_name(fk_name)
            # Referenced `p_id` must exist in the input data
            if not all(i in valid_ids for i in data[fk_name]):
                message = format_errors_and_warnings(
                    f"""
                    For {path}, the following are not a valid p_id in the input
                    data: {[i for i in data[fk_name] if i not in valid_ids]}.
                    """
                )
                raise ValueError(message)

            if fk.foreign_key_type == FKType.MUST_NOT_POINT_TO_SELF:
                equal_to_pid_in_same_row = [
                    i for i, j in zip(data[fk_name], data["p_id"]) if i == j
                ]
                if any(equal_to_pid_in_same_row):
                    message = format_errors_and_warnings(
                        f"""
                        For {path}, the following are equal to the p_id in the same
                        row: {equal_to_pid_in_same_row}.
                        """
                    )
                    raise ValueError(message)


def _warn_if_functions_overridden_by_data(
    functions_overridden: QualNameTTSIMFunctionDict,
) -> None:
    """Warn if functions are overridden by data."""
    if len(functions_overridden) > 0:
        warnings.warn(
            FunctionsAndColumnsOverlapWarning(functions_overridden.keys()),
            stacklevel=3,
        )


class FunctionsAndColumnsOverlapWarning(UserWarning):
    """
    Warning that functions which compute columns overlap with existing columns.

    Parameters
    ----------
    columns_overriding_functions : set[str]
        Names of columns in the data that override hard-coded functions.
    """

    def __init__(self, columns_overriding_functions: set[str]) -> None:
        n_cols = len(columns_overriding_functions)
        if n_cols == 1:
            first_part = format_errors_and_warnings("Your data provides the column:")
            second_part = format_errors_and_warnings(
                """
                This is already present among the hard-coded functions of the taxes and
                transfers system. If you want this data column to be used instead of
                calculating it within GETTSIM you need not do anything. If you want this
                data column to be calculated by hard-coded functions, remove it from the
                *data* you pass to GETTSIM. You need to pick one option for each column
                that appears in the list above.
                """
            )
        else:
            first_part = format_errors_and_warnings("Your data provides the columns:")
            second_part = format_errors_and_warnings(
                """
                These are already present among the hard-coded functions of the taxes
                and transfers system. If you want a data column to be used instead of
                calculating it within GETTSIM you do not need to do anything. If you
                want data columns to be calculated by hard-coded functions, remove them
                from the *data* you pass to GETTSIM. You need to pick one option for
                each column that appears in the list above.
                """
            )
        formatted = format_list_linewise(list(columns_overriding_functions))
        how_to_ignore = format_errors_and_warnings(
            """
            If you want to ignore this warning, add the following code to your script
            before calling GETTSIM:

                import warnings
                from gettsim import FunctionsAndColumnsOverlapWarning

                warnings.filterwarnings(
                    "ignore",
                    category=FunctionsAndColumnsOverlapWarning
                )
            """
        )
        super().__init__(f"{first_part}\n{formatted}\n{second_part}\n{how_to_ignore}")


def _fail_if_root_nodes_are_missing(
    functions: QualNameTTSIMFunctionDict,
    data: QualNameDataDict,
    root_nodes: list[str],
) -> None:
    """Fail if root nodes are missing.

    Fails if there are root nodes in the DAG (i.e. nodes without predecessors that do
    not depend on parameters only) that are not present in the data tree.

    Parameters
    ----------
    functions
        Dictionary of functions that are overridden by data.
    root_nodes
        List of root nodes.
    data
        Dictionary of data.

    Raises
    ------
    ValueError
        If root nodes are missing.
    """
    missing_nodes = []

    for node in root_nodes:
        if node in functions:
            func = functions[node]
            if _func_depends_on_parameters_only(func):
                # Function depends on parameters only, so it does not have to be present
                # in the data tree.
                continue
        elif node in data:
            # Root node is present in the data tree.
            continue
        else:
            missing_nodes.append(str(node))

    if missing_nodes:
        formatted = format_list_linewise(
            [str(dt.tree_path_from_qual_name(mn)) for mn in missing_nodes]
        )
        raise ValueError(f"The following data columns are missing.\n{formatted}")


def _func_depends_on_parameters_only(
    func: TTSIMFunction,
) -> bool:
    """Check if a function depends on parameters only."""
    return (
        len(
            [a for a in inspect.signature(func).parameters if not a.endswith("_params")]
        )
        == 0
    )
