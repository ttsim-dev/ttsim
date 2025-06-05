from __future__ import annotations

import datetime
import functools
from typing import TYPE_CHECKING, Any

import dags.tree as dt
import networkx as nx
from dags import concatenate_functions, create_dag, get_free_arguments

from ttsim.automatically_added_functions import (
    create_agg_by_group_functions,
    create_time_conversion_functions,
)
from ttsim.config import numpy_or_jax as np
from ttsim.shared import (
    merge_trees,
)
from ttsim.tt_dag_elements.column_objects_param_function import (
    ColumnFunction,
    ColumnObject,
    ParamFunction,
)
from ttsim.tt_dag_elements.param_objects import ParamObject, RawParam

if TYPE_CHECKING:
    from collections.abc import Callable

    from ttsim.tt_dag_elements.typing import (
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


def qual_name_data(input_data__tree: NestedData) -> QualNameData:
    return dt.flatten_to_qual_names(input_data__tree)


def qual_name_data_columns(qual_name_data: QualNameData) -> set[str]:
    return set(qual_name_data.keys())


def nested_results(qual_name_results: QualNameData) -> NestedData:
    return dt.unflatten_from_qual_names(qual_name_results)


def qual_name_results(
    column_results: QualNameData,
    targets__processed__params: QualNameTargetList,
    column_functions_with_processed_params_and_scalars: QualNameColumnFunctionsWithProcessedParamsAndScalars,
    targets__processed__from_input_data: QualNameTargetList,
    qual_name_data: QualNameData,
    targets__qname: QualNameTargetList,
) -> QualNameData:
    unordered = {
        **column_results,
        **{
            pt: column_functions_with_processed_params_and_scalars[pt]
            for pt in targets__processed__params
        },
        **{ot: qual_name_data[ot] for ot in targets__processed__from_input_data},
    }
    return {k: unordered[k] for k in targets__qname}


def tax_transfer_dag(
    required_column_functions: QualNameColumnFunctions,
    targets__processed__columns: QualNameTargetList,
) -> nx.DiGraph:
    """Thin wrapper around `create_dag`."""
    return create_dag(
        functions=required_column_functions,
        targets=targets__processed__columns,
    )


def tax_transfer_function(
    tax_transfer_dag: nx.DiGraph,
    required_column_functions: QualNameColumnFunctions,
    targets__processed__columns: QualNameTargetList,
    # backend: numpy | jax,
) -> Callable[[QualNameData], QualNameData]:
    """Returns a function that takes a dictionary of arrays and unpacks them as keyword arguments."""

    ttf_with_keyword_args = concatenate_functions(
        dag=tax_transfer_dag,
        functions=required_column_functions,
        targets=list(targets__processed__columns),
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
    #         argname: input_data__tree["p_id"].max() + 1
    #         for argname in inspect.signature(ttf_with_keyword_args).parameters
    #         if argname.endswith("_num_segments")
    #     }
    #     ttf_with_keyword_args=functools.partial(ttf_with_keyword_args, **static_args)
    #     ttf_with_keyword_args = jax.jit(ttf_with_keyword_args)

    def wrapper(qual_name_data: QualNameData) -> QualNameData:
        return ttf_with_keyword_args(**qual_name_data)

    return wrapper


def flat_policy_environment_with_derived_functions_and_without_overridden_functions(
    policy_environment: NestedPolicyEnvironment,
    qual_name_data: QualNameData,
    qual_name_data_columns: QualNameDataColumns,
    targets__tree: NestedTargetDict,
    names__top_level_namespace: set[str],
    names__grouping_levels: tuple[str, ...],
) -> QualNamePolicyEnvironment:
    """Return a flat policy environment with derived functions.

    Three steps:
    1. Remove all tree logic from the policy environment.
    2. Add derived functions to the policy environment.
    3. Remove all functions that are overridden by data columns.

    """
    flat = _remove_tree_logic_from_policy_environment(
        policy_environment=policy_environment,
        names__top_level_namespace=names__top_level_namespace,
    )
    flat_with_derived = _add_derived_functions(
        qual_name_policy_environment=flat,
        targets=dt.qual_names(targets__tree),
        qual_name_data_columns=qual_name_data_columns,
        groupings=names__grouping_levels,
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


def _remove_tree_logic_from_policy_environment(
    policy_environment: NestedPolicyEnvironment,
    names__top_level_namespace: set[str],
) -> QualNamePolicyEnvironment:
    """Map qualified names to column objects / param functions without tree logic."""
    out = {}
    for name, obj in dt.flatten_to_qual_names(policy_environment).items():
        if isinstance(obj, ParamObject):
            out[name] = obj
        else:
            out[name] = obj.remove_tree_logic(
                tree_path=dt.tree_path_from_qual_name(name),
                names__top_level_namespace=names__top_level_namespace,
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
    names__top_level_namespace
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
