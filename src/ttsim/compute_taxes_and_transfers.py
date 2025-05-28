from __future__ import annotations

import functools
import inspect
import warnings
from typing import TYPE_CHECKING

import dags.tree as dt
import networkx as nx
import pandas as pd
from dags import concatenate_functions, create_dag, get_free_arguments

from ttsim.automatically_added_functions import (
    TIME_UNIT_LABELS,
    create_agg_by_group_functions,
    create_time_conversion_functions,
)
from ttsim.column_objects_param_function import (
    ColumnFunction,
    FKType,
    ParamFunction,
    PolicyInput,
)
from ttsim.config import IS_JAX_INSTALLED
from ttsim.config import numpy_or_jax as np
from ttsim.param_objects import ParamObject, RawParam
from ttsim.policy_environment import fail_if_environment_not_valid, grouping_levels
from ttsim.shared import (
    assert_valid_ttsim_pytree,
    fail_if_multiple_time_units_for_same_base_name_and_group,
    format_errors_and_warnings,
    format_list_linewise,
    get_base_name_and_grouping_suffix,
    get_name_of_group_by_id,
    get_re_pattern_for_all_time_units_and_groupings,
    group_pattern,
    merge_trees,
    partition_by_reference_dict,
)

if TYPE_CHECKING:
    from ttsim.typing import (
        NestedData,
        NestedParamObjects,
        NestedPolicyEnvironment,
        NestedTargetDict,
        QualNameColumnFunctions,
        QualNameColumnObjectsParamFunctions,
        QualNameData,
        QualNamePolicyEnvironment,
        QualNameProcessedParams,
        QualNameTargetList,
    )


def compute_taxes_and_transfers(
    data_tree: NestedData,
    policy_environment: NestedPolicyEnvironment,
    targets_tree: NestedTargetDict,
    rounding: bool = True,
    debug: bool = False,
    jit: bool = False,
) -> NestedData:
    """Compute taxes and transfers.

    Parameters
    ----------
    data_tree : NestedData
        Data provided by the user.
    environment: NestedPolicyEnvironment
        The policy environment which contains all necessary functions and parameters.
    targets_tree : NestedTargetDict | None
        The targets tree.
    rounding : bool, default True
        Indicator for whether rounding should be applied as specified in the law.
    debug : bool
        If debug is 'True', `compute_taxes_and_transfers` returns the input data tree
        along with the computed targets.
    jit : bool
        If jit is 'True', the function is compiled using JAX's JIT compilation. To use
        this feature, JAX must be installed.

    Returns
    -------
    results : NestedData
        The computed variables as a tree.

    """

    # Check user inputs
    _fail_if_targets_tree_not_valid(targets_tree)
    _fail_if_data_tree_not_valid(data_tree)
    fail_if_environment_not_valid(policy_environment)

    top_level_namespace = _get_top_level_namespace(
        policy_environment=policy_environment,
        time_units=tuple(TIME_UNIT_LABELS.keys()),
    )
    # Check that all paths in the params tree are valid
    dt.fail_if_paths_are_invalid(
        functions=policy_environment,
        data_tree=data_tree,
        targets=targets_tree,
        top_level_namespace=top_level_namespace,
    )
    data = dt.flatten_to_qual_names(data_tree)
    column_objects_param_functions = remove_tree_logic_from_policy_environment(
        policy_environment=policy_environment,
        top_level_namespace=top_level_namespace,
    )
    # Process parameters
    processed_tree_with_params = _process_tree_with_params(
        tree_with_params=policy_environment.tree_with_params,
        param_functions={
            k: v
            for k, v in column_objects_param_functions.items()
            if isinstance(v, ParamFunction)
        },
    )

    # Flatten nested objects to qualified names
    all_targets = set(dt.qual_names(targets_tree))
    params_targets = {t for t in all_targets if t in processed_tree_with_params}
    function_targets = all_targets - params_targets

    # Add derived functions to the qualified functions tree.
    functions = combine_policy_functions_and_derived_functions(
        column_objects_param_functions=column_objects_param_functions,
        targets=function_targets,
        data=data,
        groupings=grouping_levels(policy_environment),
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
    functions_with_partialled_parameters = _partial_params_to_functions(
        functions=functions_with_rounding_specs,
        params=processed_tree_with_params,
    )
    # Remove unnecessary elements from user-provided data.
    input_data = _create_input_data_for_concatenated_function(
        data=data,
        functions=functions_with_partialled_parameters,
        targets=function_targets,
    )

    _fail_if_group_variables_not_constant_within_groups(
        data=input_data,
        groupings=grouping_levels(policy_environment),
    )
    _input_data_with_p_id = {
        "p_id": data["p_id"],
        **input_data,
    }
    _fail_if_foreign_keys_are_invalid_in_data(
        data=_input_data_with_p_id,
        column_objects_param_functions=column_objects_param_functions,
    )
    if debug:
        function_targets = {
            *function_targets,
            *functions_with_partialled_parameters.keys(),
        }

    tax_transfer_function = concatenate_functions(
        functions=functions_with_partialled_parameters,
        targets=list(function_targets),
        return_type="dict",
        aggregator=None,
        enforce_signature=True,
        set_annotations=False,
    )

    if jit:
        if not IS_JAX_INSTALLED:
            raise ImportError(
                "JAX is not installed. Please install JAX to use JIT compilation."
            )
        import jax

        static_args = {
            argname: data["p_id"].max() + 1
            for argname in inspect.signature(tax_transfer_function).parameters
            if argname.endswith("_num_segments")
        }
        tax_transfer_function = functools.partial(tax_transfer_function, **static_args)
        tax_transfer_function = jax.jit(tax_transfer_function)
    results = tax_transfer_function(**input_data)

    results_tree = dt.unflatten_from_qual_names(
        {
            **results,
            **{pt: processed_tree_with_params[pt] for pt in params_targets},
        }
    )

    if debug:
        results_tree = merge_trees(
            left=results_tree,
            right=dt.unflatten_from_qual_names(input_data),
        )

    return results_tree


def _get_top_level_namespace(
    policy_environment: NestedPolicyEnvironment,
    time_units: tuple[str, ...],
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

    direct_top_level_names = set(policy_environment.keys())

    # Do not create variations for lower-level namespaces.
    top_level_objects_for_variations = direct_top_level_names - {
        k for k, v in policy_environment.items() if isinstance(v, dict)
    }

    pattern_all = get_re_pattern_for_all_time_units_and_groupings(
        groupings=grouping_levels(policy_environment),
        time_units=time_units,
    )
    bngs_to_variations = {}
    all_top_level_names = direct_top_level_names.copy()
    for name in top_level_objects_for_variations:
        match = pattern_all.fullmatch(name)
        # We must not find multiple time units for the same base name and group.
        bngs = get_base_name_and_grouping_suffix(match)
        if match.group("time_unit"):
            if bngs not in bngs_to_variations:
                bngs_to_variations[bngs] = [name]
            else:
                bngs_to_variations[bngs].append(name)
            for time_unit in time_units:
                all_top_level_names.add(f"{bngs[0]}_{time_unit}{bngs[1]}")
    fail_if_multiple_time_units_for_same_base_name_and_group(bngs_to_variations)

    gp = group_pattern(grouping_levels(policy_environment))
    potential_base_names = {n for n in all_top_level_names if not gp.match(n)}

    for name in potential_base_names:
        for g in grouping_levels(policy_environment):
            all_top_level_names.add(f"{name}_{g}")

    # Add num_segments to grouping variables
    for g in grouping_levels(policy_environment):
        all_top_level_names.add(f"{g}_id_num_segments")
    return all_top_level_names


def remove_tree_logic_from_policy_environment(
    policy_environment: NestedPolicyEnvironment,
    top_level_namespace: set[str],
) -> QualNamePolicyEnvironment:
    """Map qualified names to column objects / param functions without tree logic."""
    out = {}
    for name, obj in dt.flatten_to_qual_names(policy_environment).items():
        if isinstance(obj, ParamObject):
            out[name] = obj
        else:
            out[name] = obj.remove_tree_logic(
                tree_path=dt.tree_path_from_qual_name(name),
                top_level_namespace=top_level_namespace,
            )
    return out


def combine_policy_functions_and_derived_functions(
    column_objects_param_functions: QualNameColumnObjectsParamFunctions,
    targets: QualNameTargetList,
    data: QualNameData,
    groupings: tuple[str, ...],
) -> QualNameColumnFunctions:
    """Return a mapping of qualified names to functions operating on columns.

    Anything that is not a ColumnFunction is filtered out (e.g., ParamFunctions,
    PolicyInputs).

    Derived functions are time converted functions and aggregation functions (aggregate
    by p_id or by group).

    Check that all targets have a corresponding function in the functions tree or can
    be taken from the data.

    Parameters
    ----------
    column_objects_param_functions
        Dict with qualified function names as keys and functions with qualified
        arguments as values.
    targets
        The list of targets with qualified names.
    data
        Dict with qualified data names as keys and pandas Series as values.
    top_level_namespace
        Set of top-level namespaces.

    Returns
    -------
    The qualified functions dict with derived functions.

    """
    # Create functions for different time units
    time_conversion_functions = create_time_conversion_functions(
        column_objects={
            k: v
            for k, v in column_objects_param_functions.items()
            if not isinstance(v, ParamFunction)
        },
        data=data,
        groupings=groupings,
    )
    out = {
        **{
            qn: f
            for qn, f in column_objects_param_functions.items()
            if isinstance(f, ColumnFunction)
        },
        **time_conversion_functions,
    }
    # Create aggregation functions by group.
    aggregate_by_group_functions = create_agg_by_group_functions(
        ttsim_functions_with_time_conversions=out,
        data=data,
        targets=targets,
        groupings=groupings,
    )
    out = {**aggregate_by_group_functions, **out}

    _fail_if_function_targets_not_in_functions(
        functions=out,
        targets=targets,
    )

    return out


def _fail_if_function_targets_not_in_functions(
    functions: QualNameColumnFunctions,
    targets: QualNameTargetList,
) -> None:
    """Fail if some target is not among functions.

    Parameters
    ----------
    functions
        Dictionary containing functions to build the DAG.
    targets
        The targets which should be computed. They limit the DAG in the way that only
        ancestors of these nodes need to be considered.

    Raises
    ------
    ValueError
        Raised if any member of `targets` is not among functions.

    """
    targets_not_in_functions_tree = [
        str(dt.tree_path_from_qual_name(n)) for n in targets if n not in functions
    ]
    if targets_not_in_functions_tree:
        formatted = format_list_linewise(targets_not_in_functions_tree)
        msg = format_errors_and_warnings(
            f"The following targets have no corresponding function:\n\n{formatted}"
        )
        raise ValueError(msg)


def _create_input_data_for_concatenated_function(
    data: QualNameData,
    functions: QualNameColumnFunctions,
    targets: QualNameTargetList,
) -> QualNameData:
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
    dag = create_dag(functions=functions, targets=targets)

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


def _process_tree_with_params(
    tree_with_params: NestedParamObjects,
    param_functions: QualNameColumnFunctions,
) -> QualNameProcessedParams:
    """Return a mapping of qualified names to processed parameter values.

    Notes:

    - This gets rid of the ParamObject objects and all meta-information like
      extended names, descriptions, units, etc.
    - RawParams are filtered out, converted values are left.

    """

    qual_name_params = dt.flatten_to_qual_names(tree_with_params)

    # Construct a function for the processing of all params.
    process = concatenate_functions(
        functions=param_functions,
        targets=None,
        return_type="dict",
        aggregator=None,
        enforce_signature=False,
        set_annotations=False,
    )
    # Call the processing function.
    processed = process(**{k: v.value for k, v in qual_name_params.items()})

    # Return the processed parameters
    return merge_trees(
        left={
            k: v.value
            for k, v in qual_name_params.items()
            if not isinstance(v, RawParam)
        },
        right=processed,
    )


def _partial_params_to_functions(
    functions: QualNameColumnFunctions,
    params: QualNameProcessedParams,
) -> QualNameColumnFunctions:
    """Partial parameters to functions such that they disappear from the DAG.

    Note: Needs to be done after rounding such that dags recognizes partialled
    parameters.

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
    processed_functions = {}
    for name, func in functions.items():
        partial_params = {}
        for arg in [a for a in get_free_arguments(func) if a in params]:
            partial_params[arg] = params[arg]
        if partial_params:
            processed_functions[name] = functools.partial(func, **partial_params)
        else:
            processed_functions[name] = func

    return processed_functions


def _add_rounding_to_functions(
    functions: QualNameColumnFunctions,
) -> QualNameColumnFunctions:
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


def _fail_if_targets_tree_not_valid(targets_tree: NestedTargetDict) -> None:
    """
    Validate that the targets tree is a dictionary with string keys and None leaves.
    """
    assert_valid_ttsim_pytree(
        tree=targets_tree,
        leaf_checker=lambda leaf: isinstance(leaf, (None | str)),
        tree_name="targets_tree",
    )


def _fail_if_data_tree_not_valid(data_tree: NestedData) -> None:
    """
    Validate that the data tree is a dictionary with string keys and pd.Series or
    np.ndarray leaves.
    """
    assert_valid_ttsim_pytree(
        tree=data_tree,
        leaf_checker=lambda leaf: isinstance(leaf, int | pd.Series | np.ndarray),
        tree_name="data_tree",
    )
    _fail_if_p_id_is_non_unique(data_tree)


def _fail_if_group_variables_not_constant_within_groups(
    data: QualNameData,
    groupings: tuple[str, ...],
) -> None:
    """
    Check that group variables are constant within each group.

    Parameters
    ----------
    data
        Dictionary of data.
    groupings
        The groupings available in the policy environment.
    """
    faulty_data_columns = []

    for name, data_column in data.items():
        group_by_id = get_name_of_group_by_id(
            target_name=name,
            groupings=groupings,
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


def _fail_if_p_id_is_non_unique(data_tree: NestedData) -> None:
    """Check that pid is unique."""
    p_id = data_tree.get("p_id", None)
    if p_id is None:
        raise ValueError("The input data must contain the p_id.")

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
            f"The following p_ids are non-unique in the input data:{non_unique_p_ids}"
        )
        raise ValueError(message)


def _fail_if_foreign_keys_are_invalid_in_data(
    data: QualNameData,
    column_objects_param_functions: QualNameColumnObjectsParamFunctions,
) -> None:
    """
    Check that all foreign keys are valid.

    Foreign keys must point to an existing `p_id` in the input data and must not refer
    to the `p_id` of the same row.
    """

    valid_ids = set(data["p_id"].tolist()) | {-1}
    relevant_objects = {
        k: v
        for k, v in column_objects_param_functions.items()
        if isinstance(v, PolicyInput | ColumnFunction)
    }

    for fk_name, fk in relevant_objects.items():
        if fk.foreign_key_type == FKType.IRRELEVANT:
            continue
        elif fk_name in data:
            path = dt.tree_path_from_qual_name(fk_name)
            # Referenced `p_id` must exist in the input data
            if not all(i in valid_ids for i in data[fk_name].tolist()):
                message = format_errors_and_warnings(
                    f"""
                    For {path}, the following are not a valid p_id in the input
                    data: {[i for i in data[fk_name] if i not in valid_ids]}.
                    """
                )
                raise ValueError(message)

            if fk.foreign_key_type == FKType.MUST_NOT_POINT_TO_SELF:
                equal_to_pid_in_same_row = [
                    i
                    for i, j in zip(data[fk_name].tolist(), data["p_id"].tolist())
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


def _warn_if_functions_overridden_by_data(
    functions_overridden: QualNameColumnFunctions,
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
                calculating it within TTSIM you need not do anything. If you want this
                data column to be calculated by hard-coded functions, remove it from the
                *data* you pass to TTSIM. You need to pick one option for each column
                that appears in the list above.
                """
            )
        else:
            first_part = format_errors_and_warnings("Your data provides the columns:")
            second_part = format_errors_and_warnings(
                """
                These are already present among the hard-coded functions of the taxes
                and transfers system. If you want a data column to be used instead of
                calculating it within TTSIM you do not need to do anything. If you
                want data columns to be calculated by hard-coded functions, remove them
                from the *data* you pass to TTSIM. You need to pick one option for
                each column that appears in the list above.
                """
            )
        formatted = format_list_linewise(list(columns_overriding_functions))
        how_to_ignore = format_errors_and_warnings(
            """
            If you want to ignore this warning, add the following code to your script
            before calling TTSIM:

                import warnings
                from ttsim import FunctionsAndColumnsOverlapWarning

                warnings.filterwarnings(
                    "ignore",
                    category=FunctionsAndColumnsOverlapWarning
                )
            """
        )
        super().__init__(f"{first_part}\n{formatted}\n{second_part}\n{how_to_ignore}")


def _fail_if_root_nodes_are_missing(
    functions: QualNameColumnFunctions,
    data: QualNameData,
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
    missing_nodes = [
        node
        for node in root_nodes
        if node not in functions
        and node not in data
        and not node.endswith("_num_segments")
    ]

    if missing_nodes:
        formatted = format_list_linewise(
            [str(dt.tree_path_from_qual_name(mn)) for mn in missing_nodes]
        )
        raise ValueError(f"The following data columns are missing.\n{formatted}")
