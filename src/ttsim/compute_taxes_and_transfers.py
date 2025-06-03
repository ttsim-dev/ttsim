from __future__ import annotations

import datetime
import functools
from typing import TYPE_CHECKING, Any

import dags.tree as dt
import networkx as nx
from dags import concatenate_functions, create_dag, get_free_arguments

from ttsim.automatically_added_functions import (
    TIME_UNIT_LABELS,
    create_agg_by_group_functions,
    create_time_conversion_functions,
)
from ttsim.column_objects_param_function import (
    ColumnFunction,
    ColumnObject,
    ParamFunction,
)
from ttsim.config import numpy_or_jax as np
from ttsim.failures_and_warnings import (
    fail_if_multiple_time_units_for_same_base_name_and_group,
)
from ttsim.param_objects import ParamObject, RawParam
from ttsim.policy_environment import grouping_levels
from ttsim.shared import (
    get_base_name_and_grouping_suffix,
    get_re_pattern_for_all_time_units_and_groupings,
    group_pattern,
    merge_trees,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ttsim.typing import (
        NestedData,
        NestedPolicyEnvironment,
        NestedTargetDict,
        QualNameColumnFunctions,
        QualNameColumnFunctionsWithProcessedParamsAndScalars,
        QualNameData,
        QualNameDataColumns,
        QualNamePolicyEnvironment,
        QualNameTargetList,
    )


_DUMMY_COLUMN_OBJECT = ColumnObject(
    leaf_name="dummy",
    start_date=datetime.date(1900, 1, 1),
    end_date=datetime.date(2099, 12, 31),
)


def column_results(
    qual_name_input_data: QualNameData,
    tax_transfer_function: Callable[[QualNameData], QualNameData],
) -> QualNameData:
    return tax_transfer_function(qual_name_input_data)


def qual_name_data(data_tree: NestedData) -> QualNameData:
    return dt.flatten_to_qual_names(data_tree)


def qual_name_data_columns(qual_name_data: QualNameData) -> set[str]:
    return set(qual_name_data.keys())


def nested_results(qual_name_results: QualNameData) -> NestedData:
    return dt.unflatten_from_qual_names(qual_name_results)


def qual_name_results(
    column_results: QualNameData,
    qual_name_param_targets: QualNameTargetList,
    column_functions_with_processed_params_and_scalars: QualNameColumnFunctionsWithProcessedParamsAndScalars,
    qual_name_own_targets: QualNameTargetList,
    qual_name_data: QualNameData,
    qual_name_targets: QualNameTargetList,
) -> QualNameData:
    unordered = {
        **column_results,
        **{
            pt: column_functions_with_processed_params_and_scalars[pt]
            for pt in qual_name_param_targets
        },
        **{ot: qual_name_data[ot] for ot in qual_name_own_targets},
    }
    return {k: unordered[k] for k in qual_name_targets}


def tax_transfer_dag(
    required_column_functions: QualNameColumnFunctions,
    qual_name_column_targets: QualNameTargetList,
) -> nx.DiGraph:
    """Thin wrapper around `create_dag`."""
    return create_dag(
        functions=required_column_functions,
        targets=qual_name_column_targets,
    )


def tax_transfer_function(
    tax_transfer_dag: nx.DiGraph,
    required_column_functions: QualNameColumnFunctions,
    qual_name_column_targets: QualNameTargetList,
    # backend: numpy | jax,
) -> Callable[[QualNameData], QualNameData]:
    """Returns a function that takes a dictionary of arrays and unpacks them as keyword arguments."""

    ttf_with_keyword_args = concatenate_functions(
        dag=tax_transfer_dag,
        functions=required_column_functions,
        targets=list(qual_name_column_targets),
        return_type="dict",
        aggregator=None,
        enforce_signature=True,
        set_annotations=False,
    )

    # if backend == jax:
    #     if not IS_JAX_INSTALLED:
    #         raise ImportError(
    #             "JAX is not installed. Please install JAX to use JIT compilation."
    #         )
    #     import jax

    #     static_args = {
    #         argname: data_tree["p_id"].max() + 1
    #         for argname in inspect.signature(ttf_with_keyword_args).parameters
    #         if argname.endswith("_num_segments")
    #     }
    #     ttf_with_keyword_args=functools.partial(ttf_with_keyword_args, **static_args)
    #     ttf_with_keyword_args = jax.jit(ttf_with_keyword_args)

    def wrapper(qual_name_data: QualNameData) -> QualNameData:
        return ttf_with_keyword_args(**qual_name_data)

    return wrapper


def qual_name_targets(targets_tree: NestedTargetDict) -> QualNameTargetList:
    """All targets in their qualified name-representation."""
    return dt.qual_names(targets_tree)


def qual_name_column_targets(
    required_column_functions: QualNameColumnFunctions,
    qual_name_targets: QualNameTargetList,
) -> QualNameTargetList:
    """All targets that are column functions."""
    return [t for t in qual_name_targets if t in required_column_functions]


def qual_name_param_targets(
    flat_policy_environment_with_derived_functions_and_without_overridden_functions: QualNamePolicyEnvironment,
    qual_name_targets: QualNameTargetList,
    qual_name_column_targets: QualNameTargetList,
) -> QualNameTargetList:
    possible_targets = set(qual_name_targets) - set(qual_name_column_targets)
    return [
        t
        for t in qual_name_targets
        if t in possible_targets
        and t
        in flat_policy_environment_with_derived_functions_and_without_overridden_functions
    ]


def qual_name_own_targets(
    qual_name_targets: QualNameTargetList,
    qual_name_column_targets: QualNameTargetList,
    qual_name_param_targets: QualNameTargetList,
) -> QualNameTargetList:
    possible_targets = (
        set(qual_name_targets)
        - set(qual_name_column_targets)
        - set(qual_name_param_targets)
    )
    return [t for t in qual_name_targets if t in possible_targets]


def flat_policy_environment_with_derived_functions_and_without_overridden_functions(
    policy_environment: NestedPolicyEnvironment,
    qual_name_data: QualNameData,
    qual_name_data_columns: QualNameDataColumns,
    targets_tree: NestedTargetDict,
    top_level_namespace: set[str],
) -> QualNamePolicyEnvironment:
    """Return a flat policy environment with derived functions.

    Three steps:
    1. Remove all tree logic from the policy environment.
    2. Add derived functions to the policy environment.
    3. Remove all functions that are overridden by data columns.

    """
    flat = _remove_tree_logic_from_policy_environment(
        policy_environment=policy_environment,
        top_level_namespace=top_level_namespace,
    )
    flat_with_derived = _add_derived_functions(
        qual_name_policy_environment=flat,
        targets=dt.qual_names(targets_tree),
        qual_name_data_columns=qual_name_data_columns,
        groupings=grouping_levels(policy_environment),
    )
    out = {}
    for n, f in flat_with_derived.items():
        # Put scalar data into the policy environment, else skip the key
        if n in qual_name_data:
            if isinstance(qual_name_data[n], int | float | bool):
                out[n] = qual_name_data[n]
        else:
            out[n] = f

    return out


def top_level_namespace(
    policy_environment: NestedPolicyEnvironment,
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

    time_units = tuple(TIME_UNIT_LABELS.keys())
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


def _remove_tree_logic_from_policy_environment(
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


def _add_derived_functions(
    qual_name_policy_environment: QualNamePolicyEnvironment,
    targets: QualNameTargetList,
    qual_name_data_columns: QualNameDataColumns,
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
        Dict with qualified data names as keys and arrays as values.
    top_level_namespace
        Set of top-level namespaces.

    Returns
    -------
    The qualified functions dict with derived functions.

    """
    # Create functions for different time units
    time_conversion_functions = create_time_conversion_functions(
        qual_name_policy_environment=qual_name_policy_environment,
        qual_name_data_columns=qual_name_data_columns,
        groupings=groupings,
    )
    column_functions = {
        k: v
        for k, v in {
            **qual_name_policy_environment,
            **time_conversion_functions,
        }.items()
        if isinstance(v, ColumnFunction)
    }

    # Create aggregation functions by group.
    aggregate_by_group_functions = create_agg_by_group_functions(
        column_functions=column_functions,
        qual_name_data_columns=qual_name_data_columns,
        targets=targets,
        groupings=groupings,
    )
    out = {
        **qual_name_policy_environment,
        **time_conversion_functions,
        **aggregate_by_group_functions,
    }

    return out


def qual_name_input_data(
    tax_transfer_dag: nx.DiGraph,
    qual_name_data: QualNameData,
) -> QualNameData:
    """Create input data for the concatenated function.

    1. Check that all root nodes are present in the user-provided data.
    2. Get only part of the data that is needed for the concatenated function.
    3. Convert inputs to np.array

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

    # Obtain root nodes
    root_nodes = nx.subgraph_view(
        tax_transfer_dag, filter_node=lambda n: tax_transfer_dag.in_degree(n) == 0
    ).nodes

    # Restrict the passed data to the subset that is actually used.
    return {k: np.array(v) for k, v in qual_name_data.items() if k in root_nodes}


def _apply_rounding(element: Any) -> Any:
    return (
        element.rounding_spec.apply_rounding(element)
        if getattr(element, "rounding_spec", False)
        else element
    )


def column_functions_with_processed_params_and_scalars(
    flat_policy_environment_with_derived_functions_and_without_overridden_functions: QualNamePolicyEnvironment,
) -> QualNameColumnFunctionsWithProcessedParamsAndScalars:
    """Process the parameters and param functions, remove RawParams from the tree."""
    params = {
        k: v
        for k, v in flat_policy_environment_with_derived_functions_and_without_overridden_functions.items()
        if isinstance(v, ParamObject)
    }
    scalars = {
        k: v
        for k, v in flat_policy_environment_with_derived_functions_and_without_overridden_functions.items()
        if isinstance(v, float | int | bool)
    }
    param_functions = {
        k: v
        for k, v in flat_policy_environment_with_derived_functions_and_without_overridden_functions.items()
        if isinstance(v, ParamFunction)
    }
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
    processed_param_functions = process(
        **{k: v.value for k, v in params.items()},
        **scalars,
    )
    processed_params = merge_trees(
        left={k: v.value for k, v in params.items() if not isinstance(v, RawParam)},
        right=processed_param_functions,
    )
    return {
        **{
            k: v
            for k, v in flat_policy_environment_with_derived_functions_and_without_overridden_functions.items()
            if not isinstance(v, RawParam)
        },
        **processed_params,
    }


def required_column_functions(
    column_functions_with_processed_params_and_scalars: QualNameColumnFunctionsWithProcessedParamsAndScalars,
    rounding: bool,
) -> QualNameColumnFunctions:
    """Partial parameters to functions such that they disappear from the DAG.

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
    for name, _func in column_functions_with_processed_params_and_scalars.items():
        if isinstance(_func, ColumnFunction):
            func = _apply_rounding(_func) if rounding else _func
            partial_params = {}
            for arg in [
                a
                for a in get_free_arguments(func)
                if not isinstance(
                    column_functions_with_processed_params_and_scalars.get(
                        a, _DUMMY_COLUMN_OBJECT
                    ),
                    ColumnObject,
                )
            ]:
                partial_params[arg] = (
                    column_functions_with_processed_params_and_scalars[arg]
                )
            if partial_params:
                processed_functions[name] = functools.partial(func, **partial_params)
            else:
                processed_functions[name] = func

    return processed_functions
