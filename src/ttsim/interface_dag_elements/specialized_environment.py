from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Literal

import dags.tree as dt
from dags import concatenate_functions, create_dag, get_free_arguments

from ttsim.interface_dag_elements.automatically_added_functions import (
    create_agg_by_group_functions,
    create_time_conversion_functions,
)
from ttsim.interface_dag_elements.interface_node_objects import (
    interface_function,
    interface_input,
)
from ttsim.interface_dag_elements.shared import merge_trees
from ttsim.tt.column_objects_param_function import (
    ColumnFunction,
    ColumnObject,
    ParamFunction,
    PolicyInput,
)
from ttsim.tt.param_objects import ParamObject, RawParam

if TYPE_CHECKING:
    import datetime
    from types import ModuleType

    import networkx as nx

    from ttsim.typing import (
        OrderedQNames,
        PolicyEnvironment,
        QNameData,
        SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
        SpecEnvWithPartialledParamsAndScalars,
        SpecEnvWithProcessedParamsAndScalars,
        UnorderedQNames,
    )


@interface_input(in_top_level_namespace=True)
def rounding() -> bool:
    """Whether to apply rounding to policy functions."""


@interface_function()
def without_tree_logic_and_with_derived_functions(
    policy_environment: PolicyEnvironment,
    tt_targets__qname: OrderedQNames,
    labels__input_columns: UnorderedQNames,
    labels__top_level_namespace: UnorderedQNames,
    labels__grouping_levels: OrderedQNames,
) -> SpecEnvWithoutTreeLogicAndWithDerivedFunctions:
    """Return a flat policy environment which includes derived functions.

    Two steps:
    1. Remove all tree logic from the policy environment.
    2. Add derived functions to the policy environment.

    """
    qname_env_without_tree_logic = _remove_tree_logic_from_policy_environment(
        qname_env=dt.flatten_to_qnames(policy_environment),
        labels__top_level_namespace=labels__top_level_namespace,
    )
    return _add_derived_functions(
        qname_env_without_tree_logic=qname_env_without_tree_logic,
        tt_targets=tt_targets__qname,
        input_columns=labels__input_columns,
        grouping_levels=labels__grouping_levels,
    )


def _remove_tree_logic_from_policy_environment(
    qname_env: dict[str, ColumnObject | ParamFunction | ParamObject],
    labels__top_level_namespace: UnorderedQNames,
) -> dict[str, ColumnObject | ParamFunction | ParamObject]:
    """Map qualified names to column objects / param functions without tree logic."""
    out = {}
    for name, obj in qname_env.items():
        if hasattr(obj, "remove_tree_logic"):
            out[name] = obj.remove_tree_logic(
                tree_path=dt.tree_path_from_qname(name),
                top_level_namespace=labels__top_level_namespace,
            )
        else:
            out[name] = obj
    return out


def _add_derived_functions(
    qname_env_without_tree_logic: dict[str, ColumnObject | ParamFunction | ParamObject],
    tt_targets: OrderedQNames,
    input_columns: UnorderedQNames,
    grouping_levels: OrderedQNames,
) -> SpecEnvWithoutTreeLogicAndWithDerivedFunctions:
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
    tt_targets
        The list of targets with qualified names.
    data
        Dict with qualified data names as keys and arrays as values.
    labels__top_level_namespace
        Set of top-level namespaces.

    Returns
    -------
    The specialized environment with derived functions (aggregations and time
    conversions), and without tree logic, i.e. absolute qualified names in all keys
    and function arguments.

    """
    # Create functions for different time units
    time_conversion_functions = create_time_conversion_functions(
        qname_policy_environment=qname_env_without_tree_logic,
        input_columns=input_columns,
        grouping_levels=grouping_levels,
    )
    column_functions = {
        k: v
        for k, v in {
            **qname_env_without_tree_logic,
            **time_conversion_functions,
        }.items()
        if isinstance(v, ColumnFunction)
    }

    # Create aggregation functions by group.
    aggregate_by_group_functions = create_agg_by_group_functions(
        column_functions=column_functions,
        input_columns=input_columns,
        tt_targets=tt_targets,
        grouping_levels=grouping_levels,
    )
    return {
        **qname_env_without_tree_logic,
        **time_conversion_functions,
        **aggregate_by_group_functions,
    }


@interface_function()
def with_processed_params_and_scalars(
    without_tree_logic_and_with_derived_functions: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
    processed_data: QNameData,
    backend: Literal["numpy", "jax"],
    xnp: ModuleType,
    dnp: ModuleType,
    evaluation_date: datetime.date | None,
) -> SpecEnvWithProcessedParamsAndScalars:
    """
    The policy environment where all parameters and param functions have been processed.

    All RawParams have been removed (note that a RawParam object is pointless without a
    param function making use of it).
    """

    all_nodes = {}
    for n, f in without_tree_logic_and_with_derived_functions.items():
        if n in processed_data:
            # Put scalars into the policy environment.
            if isinstance(processed_data[n], int | float | bool):
                all_nodes[n] = processed_data[n]
            # Else, remove the node. Will be an input of the taxes-transfers function.
        else:
            # Leave nodes not in the data what they are.
            all_nodes[n] = f

    must_set_evaluation_date = (
        # Never need to do anything if the evaluation date is set in the data.
        "evaluation_year" not in processed_data
        and (
            # PolicyInput as a placeholder
            isinstance(all_nodes.get("evaluation_year"), PolicyInput)
            # No evaluation_year in the environment (can happen in tests).
            or "evaluation_year" not in all_nodes
        )
    )
    if must_set_evaluation_date:
        if evaluation_date is None:
            all_nodes["evaluation_year"] = all_nodes["policy_year"]
            all_nodes["evaluation_month"] = all_nodes["policy_month"]
            all_nodes["evaluation_day"] = all_nodes["policy_day"]
        else:
            all_nodes["evaluation_year"] = evaluation_date.year
            all_nodes["evaluation_month"] = evaluation_date.month
            all_nodes["evaluation_day"] = evaluation_date.day

    params = {k: v for k, v in all_nodes.items() if isinstance(v, ParamObject)}
    scalars = {k: v for k, v in all_nodes.items() if isinstance(v, float | int | bool)}
    param_functions = {
        k: v for k, v in all_nodes.items() if isinstance(v, ParamFunction)
    }
    # Construct a function for processing all param_functions.
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
        xnp=xnp,
        dnp=dnp,
        backend=backend,
    )
    processed_params = merge_trees(
        left={k: v.value for k, v in params.items() if not isinstance(v, RawParam)},
        right=processed_param_functions,
    )
    return {
        **{k: v for k, v in all_nodes.items() if not isinstance(v, RawParam)},
        **processed_params,
    }


@interface_function()
def with_partialled_params_and_scalars(
    with_processed_params_and_scalars: SpecEnvWithProcessedParamsAndScalars,
    rounding: bool,
    num_segments: int,
    backend: Literal["numpy", "jax"],
    xnp: ModuleType,
    dnp: ModuleType,
) -> SpecEnvWithPartialledParamsAndScalars:
    """
    The policy environment where all parameters and scalars have been partialed into
    the column functions.

    """
    column_functions = {
        k: v
        for k, v in with_processed_params_and_scalars.items()
        if isinstance(v, ColumnFunction)
    }
    all_partial_params = {
        **{
            k: v
            for k, v in with_processed_params_and_scalars.items()
            if not isinstance(v, ColumnObject)
        },
        "num_segments": num_segments,
        "backend": backend,
        "xnp": xnp,
        "dnp": dnp,
    }

    processed_functions = {}
    for name, col_func in column_functions.items():
        vect_col_func = (
            col_func.vectorize(backend=backend, xnp=xnp)
            if hasattr(col_func, "vectorize")
            else col_func
        )
        rounded_col_func = (
            _apply_rounding(vect_col_func, xnp) if rounding else vect_col_func
        )
        partial_params_of_this_column_function = {
            arg: all_partial_params[arg]
            for arg in get_free_arguments(rounded_col_func)
            if arg in all_partial_params
        }
        if partial_params_of_this_column_function:
            processed_functions[name] = functools.partial(
                rounded_col_func, **partial_params_of_this_column_function
            )
        else:
            processed_functions[name] = rounded_col_func

    return processed_functions


def _apply_rounding(element: ColumnFunction, xnp: ModuleType) -> ColumnFunction:
    return (
        element.rounding_spec.apply_rounding(element, xnp=xnp)
        if getattr(element, "rounding_spec", False)
        else element
    )


@interface_function()
def tt_dag(
    with_partialled_params_and_scalars: SpecEnvWithPartialledParamsAndScalars,
    labels__column_targets: OrderedQNames,
) -> nx.DiGraph:
    """The taxes-transfers DAG."""
    return create_dag(
        functions=with_partialled_params_and_scalars,
        targets=labels__column_targets,
    )
