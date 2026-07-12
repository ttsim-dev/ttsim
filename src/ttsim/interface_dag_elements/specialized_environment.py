from __future__ import annotations

import datetime
import functools
from collections.abc import Callable
from types import ModuleType
from typing import Literal, cast

import dags.tree as dt
import networkx as nx
from dags import (
    concatenate_functions,
    create_dag,
    get_annotations,
    get_free_arguments,
    with_signature,
)

from ttsim.interface_dag_elements.automatically_added_functions import (
    create_agg_by_group_functions,
    create_time_conversion_functions,
)
from ttsim.interface_dag_elements.interface_node_objects import (
    interface_function,
    interface_input,
)
from ttsim.interface_dag_elements.param_currency_conversion import (
    restate_converter_outputs_in_run_currency,
)
from ttsim.interface_dag_elements.shared import (
    FRAMEWORK_PARTIAL_ARGUMENTS,
    merge_trees,
)
from ttsim.tt.column_objects_param_function import (
    AggByGroupFunction,
    AggByPIDFunction,
    ColumnFunction,
    ColumnObject,
    GroupCreationFunction,
    ParamFunction,
    PolicyFunction,
    PolicyInput,
)
from ttsim.tt.param_objects import (
    ParamObject,
    RawParam,
)
from ttsim.tt.type_resolution import is_column_annotation
from ttsim.typing import (
    OrderedQNames,
    PolicyEnvironment,
    QNameData,
    QNameStrings,
    QNameTTTargets,
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
    tt_targets__qname: QNameTTTargets,
    labels__input_columns: UnorderedQNames,
    labels__top_level_namespace: UnorderedQNames,
    labels__grouping_levels: OrderedQNames,
) -> SpecEnvWithoutTreeLogicAndWithDerivedFunctions:
    """Return a flat policy environment which includes derived functions."""
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
            out[name] = obj.remove_tree_logic(  # ty: ignore[call-non-callable]
                tree_path=dt.tree_path_from_qname(name),
                top_level_namespace=labels__top_level_namespace,
            )
        else:
            out[name] = obj
    return out


def _add_derived_functions(
    qname_env_without_tree_logic: dict[str, ColumnObject | ParamFunction | ParamObject],
    tt_targets: QNameStrings,
    input_columns: UnorderedQNames,
    grouping_levels: OrderedQNames,
) -> SpecEnvWithoutTreeLogicAndWithDerivedFunctions:
    """Return a mapping of qualified names to functions operating on columns.

    Anything that is not a ColumnFunction is filtered out (e.g., ParamFunctions,
    PolicyInputs). Derived functions are time-converted functions and aggregation
    functions (aggregate by p_id or by group).
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
        qname_policy_environment=qname_env_without_tree_logic,
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
    currency: str,
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

    # Register scalars in `processed_data` whose qname is not a DAG node so
    # the derived consumer that depends on them can partial them out. The
    # canonical case is the time-conversion source unit `x_y` for a
    # `policy_input` `x_m`: the time-conversion machinery produces
    # conversions *away* from `x_y` (`x_m`, `x_w`, `x_d`) but never creates
    # `x_y` itself as a self-node, so without this loop the derived
    # consumer's `x_y` argument would remain an unbound root node. The
    # `qname not in all_nodes` guard skips entries the loop above already
    # inserted as processed-data overrides of existing DAG nodes.
    for qname, value in processed_data.items():
        if qname not in all_nodes and isinstance(value, int | float | bool):
            all_nodes[qname] = value

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
    restate_converter_outputs_in_run_currency(
        outputs=processed_param_functions,
        params=params,
        param_functions=param_functions,
        run_currency=currency,
        xnp=xnp,
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
    len_p_id: int,
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
    # Names live in `FRAMEWORK_PARTIAL_ARGUMENTS` (shared with the unit checks);
    # iterating that constant below keeps the two in sync — a new argument added
    # there without a value here fails loudly.
    framework_argument_values = {
        "len_p_id": len_p_id,
        # Aggregation functions take a jax `num_segments` argument; the number of
        # distinct groups is at most `len_p_id`, so feed it that safe upper bound.
        "num_segments": len_p_id,
        "backend": backend,
        "xnp": xnp,
        "dnp": dnp,
    }
    all_partial_params = {
        **{
            k: v
            for k, v in with_processed_params_and_scalars.items()
            if not isinstance(v, ColumnObject)
        },
        **{
            name: framework_argument_values[name]
            for name in FRAMEWORK_PARTIAL_ARGUMENTS
        },
    }

    processed_functions = {}
    for name, col_func in column_functions.items():
        vect_col_func = (
            col_func.vectorize(backend=backend, xnp=xnp)  # ty: ignore[call-non-callable]
            if hasattr(col_func, "vectorize")
            else col_func
        )
        rounded_col_func = (
            _apply_rounding(element=vect_col_func, xnp=xnp)
            if rounding
            else vect_col_func
        )
        # Functions that are natively vectorized (aggregations, group creation, and
        # `PolicyFunction`s marked ``vectorization_strategy="not_required"``) expect
        # their `Column`-typed arguments to be full arrays. Wrap them so such scalars
        # are broadcast to the population length at call time.
        final_col_func = (
            _broadcast_scalar_columns_at_call_time(rounded_col_func, xnp=xnp)
            if _vectorization_not_required(col_func)
            else rounded_col_func
        )
        # Functions that are natively vectorized (aggregations, group creation, and
        # `PolicyFunction`s marked ``vectorization_strategy="not_required"``) expect
        # their `Column`-typed arguments to be full arrays. Wrap them so such scalars
        # are broadcast to the population length at call time.
        final_col_func = (
            _broadcast_scalar_columns_at_call_time(rounded_col_func, xnp=xnp)
            if _vectorization_not_required(col_func)
            else rounded_col_func
        )
        partial_params_of_this_column_function = {
            arg: all_partial_params[arg]
            for arg in get_free_arguments(final_col_func)
            if arg in all_partial_params
        }
        if partial_params_of_this_column_function:
            processed_functions[name] = functools.partial(
                final_col_func, **partial_params_of_this_column_function
            )
        else:
            processed_functions[name] = final_col_func

    return processed_functions


def _apply_rounding(
    element: ColumnFunction, xnp: ModuleType
) -> ColumnFunction | Callable[..., object]:
    """Apply the element's rounding spec, if any.

    `RoundingSpec.apply_rounding` returns a plain wrapper function, not a
    `ColumnFunction`, so the return type is the union of the unrounded
    `ColumnFunction` and the rounded plain callable.
    """
    return (
        element.rounding_spec.apply_rounding(element, xnp=xnp)  # ty: ignore[unresolved-attribute]
        if getattr(element, "rounding_spec", False)
        else element
    )


def _vectorization_not_required(col_func: ColumnObject) -> bool:
    """Whether `col_func` is not auto-vectorized and hence expects array-valued columns.

    Such functions (aggregations, person-pointer aggregations, endogenous group
    creation, and `PolicyFunction`s declared ``vectorization_strategy="not_required"``)
    run array operations that are meaningless on a scalar, so any scalar bound to one
    of their `Column`-typed arguments must be materialised as an array before the body
    runs.
    """
    if isinstance(
        col_func,
        AggByGroupFunction | AggByPIDFunction | GroupCreationFunction,
    ):
        return True
    return (
        isinstance(col_func, PolicyFunction)
        and col_func.vectorization_strategy == "not_required"
    )


def _broadcast_scalar_columns_at_call_time(
    func: Callable[..., object],
    xnp: ModuleType,
) -> Callable[..., object]:
    """Wrap a non-vectorized column function to broadcast scalar columns at call time.

    The wrapper carries the same signature and annotations as `func` (so dependency
    resolution and the DAG annotation-consistency check see no change) plus `len_p_id`
    when `func` does not already require it. `len_p_id` is the population length
    ``n_obs``, partialled in as a scalar so the wrapper introduces no DAG dependency.

    At call time, every argument bound to a `Column`-typed parameter that arrives as a
    scalar is broadcast to ``(n_obs,)`` before `func` runs. The extra `len_p_id` value
    is used only to size the broadcast and is not forwarded unless `func` declares it.
    """
    annotations = get_annotations(func)
    func_args = get_free_arguments(func)
    column_args = [
        arg for arg in func_args if is_column_annotation(annotations.get(arg))
    ]
    wrapper_arg_annotations: dict[str, str] = {
        arg: annotations[arg] for arg in func_args
    }
    wrapper_arg_annotations.setdefault("len_p_id", "int")

    def broadcast_scalar_columns(**kwargs: object) -> object:
        n_obs = cast("int", kwargs["len_p_id"])
        call_kwargs = {arg: kwargs[arg] for arg in func_args}
        for arg in column_args:
            if isinstance(call_kwargs[arg], int | float | bool):
                call_kwargs[arg] = xnp.broadcast_to(call_kwargs[arg], (n_obs,))
        return func(**call_kwargs)

    return with_signature(
        broadcast_scalar_columns,
        args=wrapper_arg_annotations,
        return_annotation=annotations["return"],
        enforce=False,
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
