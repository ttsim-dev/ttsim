from __future__ import annotations

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
        OrderedQNames,
        QNameData,
        QNamePolicyEnvironment,
        QNameSpecializedEnvironment0,
        QNameSpecializedEnvironment1,
        QNameSpecializedEnvironment2,
        UnorderedQNames,
    )


@interface_input(in_top_level_namespace=True)
def rounding() -> bool:
    """Whether to apply rounding to policy functions."""


@interface_function()
def without_tree_logic_and_with_derived_functions(
    policy_environment: NestedPolicyEnvironment,
    targets__qname: OrderedQNames,
    labels__input_columns: UnorderedQNames,
    labels__top_level_namespace: UnorderedQNames,
    labels__grouping_levels: OrderedQNames,
    backend: str,
    xnp: ModuleType,
) -> QNameSpecializedEnvironment0:
    """Return a flat policy environment with derived functions.

    Three steps:
    1. Vectorize policy functions.
    2. Remove all tree logic from the policy environment.
    3. Add derived functions to the policy environment.

    """
    qname_env_vectorized = {
        k: f.vectorize(
            backend=backend,
            xnp=xnp,
        )
        if isinstance(f, PolicyFunction)
        else f
        for k, f in dt.flatten_to_qnames(policy_environment).items()
    }
    qname_env_without_tree_logic = _remove_tree_logic_from_policy_environment(
        qname_env_vectorized=qname_env_vectorized,
        labels__top_level_namespace=labels__top_level_namespace,
    )
    return _add_derived_functions(
        qname_env_without_tree_logic=qname_env_without_tree_logic,
        targets=targets__qname,
        input_columns=labels__input_columns,
        grouping_levels=labels__grouping_levels,
    )


def _remove_tree_logic_from_policy_environment(
    qname_env_vectorized: QNamePolicyEnvironment,
    labels__top_level_namespace: UnorderedQNames,
) -> QNameSpecializedEnvironment0:
    """Map qualified names to column objects / param functions without tree logic."""
    out = {}
    for name, obj in qname_env_vectorized.items():
        if hasattr(obj, "remove_tree_logic"):
            out[name] = obj.remove_tree_logic(
                tree_path=dt.tree_path_from_qname(name),
                top_level_namespace=labels__top_level_namespace,
            )
        else:
            out[name] = obj
    return out


def _add_derived_functions(
    qname_env_without_tree_logic: QNameSpecializedEnvironment0,
    targets: OrderedQNames,
    input_columns: UnorderedQNames,
    grouping_levels: OrderedQNames,
) -> QNameSpecializedEnvironment0:
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
    The specialized environment with vectorized column functions, derived functions
    (aggregations and time conversions), and without tree logic, i.e. absolute
    qualified names in all keys and function arguments.

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
        targets=targets,
        grouping_levels=grouping_levels,
    )
    return {
        **qname_env_without_tree_logic,
        **time_conversion_functions,
        **aggregate_by_group_functions,
    }


@interface_function()
def with_processed_params_and_scalars(
    without_tree_logic_and_with_derived_functions: QNameSpecializedEnvironment0,
    processed_data: QNameData,
) -> QNameSpecializedEnvironment1:
    """Process the parameters and param functions, remove RawParams from the tree.

    Parameters
    ----------
    without_tree_logic_and_with_derived_functions
        The specialized environment with vectorized column functions, derived functions
        (aggregations and time conversions), and without tree logic, i.e. absolute
        qualified names in all keys and function arguments.
    processed_data
        The processed data.

    Returns
    -------
    The specialized environment with processed parameters and scalars. Input nodes
    that come in via the processed data are removed from the environment.
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
    # The number of segments for jax' segment sum. After processing the data, we know
    # that the number of ids is at most the length of the data.
    if processed_data:
        all_nodes["num_segments"] = len(next(iter(processed_data.values())))
    else:
        all_nodes["num_segments"] = 0

    params = {k: v for k, v in all_nodes.items() if isinstance(v, ParamObject)}
    scalars = {
        k: v
        for k, v in all_nodes.items()
        if isinstance(v, float | int | bool) or k == "backend"
    }
    modules = {k: v for k, v in all_nodes.items() if isinstance(v, ModuleType)}
    param_functions = {
        k: v for k, v in all_nodes.items() if isinstance(v, ParamFunction)
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
        **{k: v for k, v in all_nodes.items() if not isinstance(v, RawParam)},
        **processed_params,
    }


@interface_function()
def with_partialled_params_and_scalars(
    with_processed_params_and_scalars: QNameSpecializedEnvironment1,
    rounding: bool,
    xnp: ModuleType,
) -> QNameSpecializedEnvironment2:
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
                if (
                    a in with_processed_params_and_scalars
                    and not isinstance(
                        with_processed_params_and_scalars[a], ColumnObject
                    )
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
    with_partialled_params_and_scalars: QNameSpecializedEnvironment2,
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
    with_partialled_params_and_scalars: QNameSpecializedEnvironment2,
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
