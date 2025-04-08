from __future__ import annotations

import functools
import inspect
import warnings
from typing import TYPE_CHECKING, Any, Literal, get_args

import dags
import dags.tree as dt
import networkx as nx
import pandas as pd

from _gettsim.config import (
    DEFAULT_TARGETS,
    FOREIGN_KEYS,
    TYPES_INPUT_VARIABLES,
)
from ttsim.combine_functions import (
    combine_policy_functions_and_derived_functions,
)
from ttsim.config import numpy_or_jax as np
from ttsim.function_types import (
    DerivedAggregationFunction,
    GroupByFunction,
    PolicyFunction,
    TTSIMFunction,
)
from ttsim.policy_environment import PolicyEnvironment
from ttsim.shared import (
    KeyErrorMessage,
    assert_valid_ttsim_pytree,
    format_errors_and_warnings,
    format_list_linewise,
    get_name_of_group_by_id,
    get_names_of_arguments_without_defaults,
    merge_trees,
    partition_by_reference_dict,
)
from ttsim.typing import (
    check_series_has_expected_type,
    convert_series_to_internal_type,
)

if TYPE_CHECKING:
    from ttsim.typing import (
        NestedDataDict,
        NestedTargetDict,
        QualNameDataDict,
        QualNameFunctionsDict,
        QualNameTargetList,
    )


def compute_taxes_and_transfers(
    data_tree: NestedDataDict,
    environment: PolicyEnvironment,
    targets_tree: NestedTargetDict | None = None,
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
        The targets tree. By default, ``targets_tree`` is ``None`` and all key outputs
        as defined by `gettsim.config.DEFAULT_TARGETS` are returned.
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
    # Use default targets if no targets are provided.
    targets_tree = targets_tree if targets_tree else DEFAULT_TARGETS

    # Check user inputs
    _fail_if_targets_tree_not_valid(targets_tree)
    _fail_if_data_tree_not_valid(data_tree)
    _fail_if_environment_not_valid(environment)

    # Transform functions tree to qualified names dict with qualified arguments
    top_level_namespace = (
        set(environment.functions_tree.keys())
        | set(data_tree.keys())
        | set(TYPES_INPUT_VARIABLES.keys())
        | set(environment.aggregation_specs_tree.keys())
    )
    functions = dt.functions_without_tree_logic(
        functions=environment.functions_tree, top_level_namespace=top_level_namespace
    )

    targets = dt.qual_names(targets_tree)
    data = dt.flatten_to_qual_names(data_tree)
    aggregation_specs = dt.flatten_to_qual_names(environment.aggregation_specs_tree)

    # Add derived functions to the qualified functions tree.
    functions = combine_policy_functions_and_derived_functions(
        functions=functions,
        aggregation_specs_from_environment=aggregation_specs,
        targets=targets,
        data=data,
        top_level_namespace=top_level_namespace,
    )

    functions_overridden, functions_not_overridden = partition_by_reference_dict(
        to_partition=functions,
        reference_dict=data,
    )

    _warn_if_functions_overridden_by_data(functions_overridden)
    data_with_correct_types = _convert_data_to_correct_types(
        data=data,
        functions_overridden=functions_overridden,
    )

    functions_with_rounding_specs = (
        _add_rounding_to_functions(
            functions=functions_not_overridden,
            params=environment.params,
        )
        if rounding
        else functions_not_overridden
    )
    functions_with_partialled_parameters = _partial_parameters_to_functions(
        functions=functions_with_rounding_specs,
        params=environment.params,
    )

    # Remove unnecessary elements from user-provided data.
    input_data = _create_input_data_for_concatenated_function(
        data=data_with_correct_types,
        functions=functions_with_partialled_parameters,
        targets=targets,
    )

    _fail_if_group_variables_not_constant_within_groups(
        data=input_data,
        functions=functions,
    )
    _fail_if_foreign_keys_are_invalid(
        data=input_data,
        p_id=data.get("p_id", None),
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


def _convert_data_to_correct_types(
    data: QualNameDataDict, functions_overridden: QualNameFunctionsDict
) -> QualNameDataDict:
    """Convert all data columns to the type that is expected by GETTSIM.

    Parameters
    ----------
    data
        Data provided by the user.
    functions_overridden
        Functions that are overridden by data.

    Returns
    -------
    Data with correct types.

    """
    collected_errors = ["The data types of the following columns are invalid:\n"]
    collected_conversions = [
        "The data types of the following input variables have been converted:"
    ]
    general_warning = (
        "Note that the automatic conversion of data types is unsafe and that"
        " its correctness cannot be guaranteed."
        " The best solution is to convert all columns to the expected data"
        " types yourself."
    )

    data_with_correct_types = {}

    for name, series in data.items():
        internal_type = None

        # Look for column in TYPES_INPUT_VARIABLES
        types_qualified_input_variables = dt.flatten_to_qual_names(
            TYPES_INPUT_VARIABLES
        )
        if name in types_qualified_input_variables:
            internal_type = types_qualified_input_variables[name]
        # Look for column in functions_tree_overridden
        elif name in functions_overridden:
            func = functions_overridden[name]
            func_is_group_by_function = isinstance(
                getattr(func, "__wrapped__", func), GroupByFunction
            )
            func_is_policy_function = isinstance(
                getattr(func, "__wrapped__", func), PolicyFunction
            ) and not isinstance(
                getattr(func, "__wrapped__", func), DerivedAggregationFunction
            )
            skip_vectorization = (
                func.skip_vectorization if func_is_policy_function else True
            )
            return_annotation_is_array = (
                func_is_group_by_function or func_is_policy_function
            ) and skip_vectorization
            if return_annotation_is_array:
                # Assumes that things are annotated with numpy.ndarray([dtype]), might
                # require a change if using proper numpy.typing. Not changing for now
                # as we will likely switch to JAX completely.
                internal_type = get_args(func.__annotations__["return"])[0]
            elif "return" in func.__annotations__:
                internal_type = func.__annotations__["return"]
            else:
                pass
        else:
            pass

        # Make conversion if necessary
        if internal_type and not check_series_has_expected_type(
            series=series, internal_type=internal_type
        ):
            try:
                converted_leaf = convert_series_to_internal_type(
                    series=series, internal_type=internal_type
                )
                data_with_correct_types[name] = converted_leaf
                collected_conversions.append(
                    f" - {name} from {series.dtype} to {internal_type.__name__}"
                )
            except ValueError as e:
                collected_errors.append(f"\n - {name}: {e}")
        else:
            data_with_correct_types[name] = series

    # If any error occured raise Error
    if len(collected_errors) > 1:
        msg = """
            Note that conversion from floating point to integers or Booleans inherently
            suffers from approximation error. It might well be that your data seemingly
            obey the restrictions when scrolling through them, but in fact they do not
            (for example, because 1e-15 is displayed as 0.0). \n The best solution is to
            convert all columns to the expected data types yourself.
            """
        collected_errors = "\n".join(collected_errors)
        raise ValueError(format_errors_and_warnings(collected_errors + msg))
    # Otherwise raise warning which lists all successful conversions
    elif len(collected_conversions) > 1:
        collected_conversions = format_list_linewise(collected_conversions)
        warnings.warn(
            collected_conversions + "\n" + "\n" + general_warning,
            stacklevel=2,
        )

    return data_with_correct_types


def _create_input_data_for_concatenated_function(
    data: QualNameDataDict,
    functions: QualNameFunctionsDict,
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
    functions: QualNameFunctionsDict,
    params: dict[str, Any],
) -> QualNameFunctionsDict:
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
    functions: QualNameFunctionsDict,
    params: dict[str, Any],
) -> QualNameFunctionsDict:
    """Add appropriate rounding of outputs to function.

    Parameters
    ----------
    functions
        Functions to which rounding should be added.
    params : dict
        Dictionary of parameters

    Returns
    -------
    Function with rounding added.

    """
    rounded_functions = {}
    for name, func in functions.items():
        if getattr(func, "params_key_for_rounding", False):
            params_key = func.params_key_for_rounding
            # Check if there are any rounding specifications in params files.
            if not (
                params_key in params
                and "rounding" in params[params_key]
                and name in params[params_key]["rounding"]
            ):
                path = dt.tree_path_from_qual_name(name)
                raise KeyError(
                    KeyErrorMessage(
                        f"""
                        Rounding specifications for function

                            {path}

                        are expected in the parameter dictionary at:\n
                        [{params_key!r}]['rounding'][{name!r}].\n
                        These nested keys do not exist. If this function should not be
                        rounded, remove the respective decorator.
                        """
                    )
                )
            rounding_spec = params[params_key]["rounding"][name]
            # Check if expected parameters are present in rounding specifications.
            if not ("base" in rounding_spec and "direction" in rounding_spec):
                raise KeyError(
                    KeyErrorMessage(
                        "Both 'base' and 'direction' are expected as rounding "
                        "parameters in the parameter dictionary. \n "
                        "At least one of them is missing at:\n"
                        f"[{params_key!r}]['rounding'][{name!r}]."
                    )
                )
            # Add rounding.
            rounded_functions[name] = _apply_rounding_spec(
                base=rounding_spec["base"],
                direction=rounding_spec["direction"],
                to_add_after_rounding=rounding_spec.get("to_add_after_rounding", 0),
                name=name,
            )(func)
        else:
            rounded_functions[name] = func

    return rounded_functions


def _apply_rounding_spec(
    base: float,
    direction: Literal["up", "down", "nearest"],
    to_add_after_rounding: float,
    name: str,
) -> callable:
    """Decorator to round the output of a function.

    Parameters
    ----------
    base
        Precision of rounding (e.g. 0.1 to round to the first decimal place)
    direction
        Whether the series should be rounded up, down or to the nearest number
    to_add_after_rounding
        Number to be added after the rounding step
    name:
        Name of the function to be rounded.

    Returns
    -------
    Series with (potentially) rounded numbers

    """

    path = dt.tree_path_from_qual_name(name)

    def inner(func):
        # Make sure that signature is preserved.
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            out = func(*args, **kwargs)

            # Check inputs.
            if type(base) not in [int, float]:
                raise ValueError(f"base needs to be a number, got {base!r} for {path}")
            if type(to_add_after_rounding) not in [int, float]:
                raise ValueError(
                    f"Additive part needs to be a number, got"
                    f" {to_add_after_rounding!r} for {path}"
                )

            if direction == "up":
                rounded_out = base * np.ceil(out / base)
            elif direction == "down":
                rounded_out = base * np.floor(out / base)
            elif direction == "nearest":
                rounded_out = base * (out / base).round()
            else:
                raise ValueError(
                    "direction must be one of 'up', 'down', or 'nearest'"
                    f", got {direction!r} for {path}"
                )

            rounded_out += to_add_after_rounding
            return rounded_out

        return wrapper

    return inner


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
    _fail_if_pid_is_non_unique(data_tree)


def _fail_if_group_variables_not_constant_within_groups(
    data: QualNameDataDict,
    functions: QualNameFunctionsDict,
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
        if isinstance(getattr(v, "__wrapped__", v), GroupByFunction)
    }

    faulty_data_columns = []

    for name, data_column in data.items():
        group_by_id = get_name_of_group_by_id(
            target_name=name,
            group_by_functions=group_by_functions,
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


def _fail_if_pid_is_non_unique(data_tree: NestedDataDict) -> None:
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


def _fail_if_foreign_keys_are_invalid(
    data: QualNameDataDict,
    p_id: pd.Series,
) -> None:
    """
    Check that all foreign keys are valid.

    Foreign keys must point to an existing `p_id` in the input data and must not refer
    to the `p_id` of the same row.
    """
    valid_ids = set(p_id) | {-1}

    for name, data_column in data.items():
        foreign_key_col = dt.tree_path_from_qual_name(name) in FOREIGN_KEYS
        path = dt.tree_path_from_qual_name(name)
        if not foreign_key_col:
            continue

        # Referenced `p_id` must exist in the input data
        if not all(i in valid_ids for i in data_column):
            message = format_errors_and_warnings(
                f"""
                For {path}, the following are not a valid p_id in the input
                data: {[i for i in data_column if i not in valid_ids]}.
                """
            )
            raise ValueError(message)

        equal_to_pid_in_same_row = [i for i, j in zip(data_column, p_id) if i == j]
        if any(equal_to_pid_in_same_row):
            message = format_errors_and_warnings(
                f"""
                For {path}, the following are equal to the p_id in the same
                row: {[i for i, j in zip(data_column, p_id) if i == j]}.
                """
            )
            raise ValueError(message)


def _warn_if_functions_overridden_by_data(
    functions_overridden: QualNameFunctionsDict,
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
    functions: QualNameFunctionsDict,
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
        formatted = format_list_linewise(missing_nodes)
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
