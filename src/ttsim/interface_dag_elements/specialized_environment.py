from __future__ import annotations

import datetime
import functools
from types import ModuleType
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
from ttsim.tt_dag_elements.column_objects_param_function import (
    ColumnFunction,
    ColumnObject,
    ParamFunction,
    PolicyFunction,
)
from ttsim.tt_dag_elements.param_objects import ParamObject, RawParam

if TYPE_CHECKING:
    from collections.abc import Callable

    import networkx as nx

    from ttsim.interface_dag_elements.typing import (
        NestedPolicyEnvironment,
        NestedStrings,
        OrderedQNames,
        QNameCombinedEnvironment0,
        QNameCombinedEnvironment1,
        QNameCombinedEnvironment2,
        QNameData,
        QNameDataColumns,
        QNamePolicyEnvironment,
        UnorderedQNames,
    )


_DUMMY_COLUMN_OBJECT = ColumnObject(
    leaf_name="dummy",
    start_date=datetime.date(1900, 1, 1),
    end_date=datetime.date(2099, 12, 31),
)


@interface_input(in_top_level_namespace=True)
def rounding() -> bool:
    """Whether to apply rounding to policy functions."""


@interface_function()
def with_derived_functions_and_processed_input_nodes(
    policy_environment: NestedPolicyEnvironment,
    processed_data: QNameData,
    labels__processed_data_columns: QNameDataColumns,
    targets__tree: NestedStrings,
    labels__top_level_namespace: UnorderedQNames,
    labels__grouping_levels: OrderedQNames,
    backend: str,
    xnp: ModuleType,
) -> QNameCombinedEnvironment0:
    """Return a flat policy environment with derived functions.

    Three steps:
    1. Remove all tree logic from the policy environment.
    2. Add derived functions to the policy environment.
    3. Remove all functions that are overridden by data columns.

    """
    flat_vectorized = {
        k: f.vectorize(
            backend=backend,
            xnp=xnp,
        )
        if isinstance(f, PolicyFunction)
        else f
        for k, f in dt.flatten_to_qual_names(policy_environment).items()
    }
    flat_without_tree_logic = _remove_tree_logic_from_policy_environment(
        policy_environment=flat_vectorized,
        labels__top_level_namespace=labels__top_level_namespace,
    )
    flat_with_derived = _add_derived_functions(
        qual_name_policy_environment=flat_without_tree_logic,
        targets=dt.qual_names(targets__tree),
        labels__processed_data_columns=labels__processed_data_columns,
        grouping_levels=labels__grouping_levels,
    )
    out = {}
    for n, f in flat_with_derived.items():
        # Put scalars into the policy environment, else remove the element because it
        # will be passed into the `tax_transfer_function` as an input.
        if n in processed_data:
            if isinstance(processed_data[n], int | float | bool):
                out[n] = processed_data[n]
        else:
            out[n] = f
    # The number of segments for jax' segment sum. After processing the data, we know
    # that the number of ids is at most the length of the data.
    out["num_segments"] = len(next(iter(processed_data.values())))
    out["backend"] = backend
    return out


def _remove_tree_logic_from_policy_environment(
    policy_environment: QNamePolicyEnvironment,
    labels__top_level_namespace: UnorderedQNames,
) -> QNamePolicyEnvironment:
    """Map qualified names to column objects / param functions without tree logic."""
    out = {}
    for name, obj in policy_environment.items():
        if hasattr(obj, "remove_tree_logic"):
            out[name] = obj.remove_tree_logic(
                tree_path=dt.tree_path_from_qual_name(name),
                top_level_namespace=labels__top_level_namespace,
            )
        else:
            out[name] = obj
    return out


def _add_derived_functions(
    qual_name_policy_environment: QNamePolicyEnvironment,
    targets: OrderedQNames,
    labels__processed_data_columns: QNameDataColumns,
    grouping_levels: OrderedQNames,
) -> UnorderedQNames:
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
    labels__top_level_namespace
        Set of top-level namespaces.

    Returns
    -------
    The qualified functions dict with derived functions.

    """
    # Create functions for different time units
    time_conversion_functions = create_time_conversion_functions(
        qual_name_policy_environment=qual_name_policy_environment,
        processed_data_columns=labels__processed_data_columns,
        grouping_levels=grouping_levels,
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
        labels__processed_data_columns=labels__processed_data_columns,
        targets=targets,
        grouping_levels=grouping_levels,
    )
    return {
        **qual_name_policy_environment,
        **time_conversion_functions,
        **aggregate_by_group_functions,
    }


@interface_function()
def with_processed_params_and_scalars(
    with_derived_functions_and_processed_input_nodes: QNameCombinedEnvironment0,
) -> QNameCombinedEnvironment1:
    """Process the parameters and param functions, remove RawParams from the tree."""
    params = {
        k: v
        for k, v in with_derived_functions_and_processed_input_nodes.items()
        if isinstance(v, ParamObject)
    }
    scalars = {
        k: v
        for k, v in with_derived_functions_and_processed_input_nodes.items()
        if isinstance(v, float | int | bool) or k == "backend"
    }
    modules = {
        k: v
        for k, v in with_derived_functions_and_processed_input_nodes.items()
        if isinstance(v, ModuleType)
    }
    param_functions = {
        k: v
        for k, v in with_derived_functions_and_processed_input_nodes.items()
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
        **modules,
    )
    processed_params = merge_trees(
        left={k: v.value for k, v in params.items() if not isinstance(v, RawParam)},
        right=processed_param_functions,
    )
    return {
        **{
            k: v
            for k, v in with_derived_functions_and_processed_input_nodes.items()
            if not isinstance(v, RawParam)
        },
        **processed_params,
    }


@interface_function()
def with_partialled_params_and_scalars(
    with_processed_params_and_scalars: QNameCombinedEnvironment1,
    rounding: bool,
    xnp: ModuleType,
) -> QNameCombinedEnvironment2:
    """Partial parameters to functions such that they disappear from the DAG.

    Parameters
    ----------
    with_processed_params_and_scalars
        The tree with qualified names as keys and column objects or processed
        parameters / scalars as values.
    rounding
        Whether to apply rounding to functions.

    Returns
    -------
    Tree with column functions that depend on columns only.

    """
    processed_functions = {}
    for name, _func in with_processed_params_and_scalars.items():
        if isinstance(_func, ColumnFunction):
            func = _apply_rounding(_func, xnp) if rounding else _func
            partial_params = {}
            for arg in [
                a
                for a in get_free_arguments(func)
                if not isinstance(
                    with_processed_params_and_scalars.get(a, _DUMMY_COLUMN_OBJECT),
                    ColumnObject,
                )
            ]:
                partial_params[arg] = with_processed_params_and_scalars[arg]
            if partial_params:
                processed_functions[name] = functools.partial(func, **partial_params)
            else:
                processed_functions[name] = func

    return processed_functions


def _apply_rounding(element: ColumnFunction, xnp: ModuleType) -> ColumnFunction:
    return (
        element.rounding_spec.apply_rounding(element, xnp=xnp)
        if getattr(element, "rounding_spec", False)
        else element
    )


@interface_function()
def tax_transfer_dag(
    with_partialled_params_and_scalars: QNameCombinedEnvironment2,
    labels__column_targets: OrderedQNames,
) -> nx.DiGraph:
    """Thin wrapper around `create_dag`."""
    return create_dag(
        functions=with_partialled_params_and_scalars,
        targets=labels__column_targets,
    )


@interface_function()
def tax_transfer_function(
    tax_transfer_dag: nx.DiGraph,
    with_partialled_params_and_scalars: QNameCombinedEnvironment2,
    labels__column_targets: OrderedQNames,
    backend: Literal["numpy", "jax"],
) -> Callable[[QNameData], QNameData]:
    """Returns a function that takes a dictionary of arrays and unpacks them as keyword arguments."""
    ttf_with_keyword_args = concatenate_functions(
        dag=tax_transfer_dag,
        functions=with_partialled_params_and_scalars,
        targets=list(labels__column_targets),
        return_type="dict",
        aggregator=None,
        enforce_signature=True,
        set_annotations=False,
    )

    if backend == "jax":
        import jax

        ttf_with_keyword_args = jax.jit(ttf_with_keyword_args)

    def wrapper(processed_data: QNameData) -> QNameData:
        return ttf_with_keyword_args(**processed_data)

    return wrapper
