from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import dags
import dags.tree as dt

from _gettsim.aggregation import (
    AggregateByGroupSpec,
    AggregateByPIDSpec,
    all_by_p_id,
    any_by_p_id,
    count_by_p_id,
    grouped_all,
    grouped_any,
    grouped_count,
    grouped_max,
    grouped_mean,
    grouped_min,
    grouped_sum,
    max_by_p_id,
    mean_by_p_id,
    min_by_p_id,
    sum_by_p_id,
)
from _gettsim.config import (
    SUPPORTED_GROUPINGS,
)
from _gettsim.function_types import DerivedFunction, GroupByFunction
from _gettsim.shared import (
    format_errors_and_warnings,
    format_list_linewise,
    get_name_of_group_by_id,
    get_names_of_arguments_without_defaults,
    remove_group_suffix,
)
from _gettsim.time_conversion import create_time_conversion_functions

if TYPE_CHECKING:
    from _gettsim.gettsim_typing import (
        QualifiedAggregationSpecsDict,
        QualifiedDataDict,
        QualifiedFunctionsDict,
        QualifiedTargetsDict,
    )


def combine_policy_functions_and_derived_functions(
    functions_dict: QualifiedFunctionsDict,
    aggregation_specs_from_environment: QualifiedAggregationSpecsDict,
    targets_dict: QualifiedTargetsDict,
    data_dict: QualifiedDataDict,
    top_level_namespace: set[str],
) -> QualifiedFunctionsDict:
    """Add derived functions to the qualified functions dict.

    Derived functions are time converted functions and aggregation functions (aggregate
    by p_id or by group).

    Checks that all targets have a corresponding function in the functions tree or can
    be taken from the data.

    Parameters
    ----------
    functions_dict
        Dict with qualified function names as keys and functions with qualified
        arguments as values.
    aggregation_specs_from_environment
        Dict with qualified aggregation spec names as keys and aggregation specs as
        values.
    targets_dict
        Dict with qualified target names as keys and None as values.
    data_dict
        Dict with qualified data names as keys and pandas Series as values.
    top_level_namespace
        Set of top-level namespaces.

    Returns
    -------
    The qualified functions dict with derived functions.

    """
    # Create parent-child relationships
    aggregate_by_p_id_functions = _create_aggregation_functions(
        functions_dict=functions_dict,
        aggregations_dict=aggregation_specs_from_environment,
        top_level_namespace=top_level_namespace,
        aggregation_type="p_id",
    )
    current_functions_dict = {
        **aggregate_by_p_id_functions,
        **functions_dict,
    }

    # Create functions for different time units
    time_conversion_functions = create_time_conversion_functions(
        functions_dict=current_functions_dict,
        data_dict=data_dict,
    )
    current_functions_dict = {
        **time_conversion_functions,
        **current_functions_dict,
    }

    # Create aggregation functions
    aggregate_by_group_functions = _create_aggregate_by_group_functions(
        functions_dict=current_functions_dict,
        targets_dict=targets_dict,
        data_dict=data_dict,
        aggregations_dict_from_environment=aggregation_specs_from_environment,
        top_level_namespace=top_level_namespace,
    )
    current_functions_dict = {
        **aggregate_by_group_functions,
        **current_functions_dict,
    }

    _fail_if_targets_not_in_functions(
        functions_dict=current_functions_dict,
        targets_dict=targets_dict,
    )

    return current_functions_dict


def _create_aggregate_by_group_functions(
    functions_dict: QualifiedFunctionsDict,
    targets_dict: QualifiedTargetsDict,
    data_dict: QualifiedDataDict,
    aggregations_dict_from_environment: QualifiedAggregationSpecsDict,
    top_level_namespace: set[str],
) -> QualifiedFunctionsDict:
    """Create aggregation functions."""

    # Create aggregation functions from environment
    aggregation_functions_from_environment = _create_aggregation_functions(
        functions_dict=functions_dict,
        aggregations_dict=aggregations_dict_from_environment,
        aggregation_type="group",
        top_level_namespace=top_level_namespace,
    )

    current_functions_dict = {
        **aggregation_functions_from_environment,
        **functions_dict,
    }

    # Create derived aggregation functions
    derived_aggregation_specs_dict = _create_derived_aggregations_specs_dict(
        functions_dict=current_functions_dict,
        targets_dict=targets_dict,
        data_dict=data_dict,
    )
    derived_aggregation_functions = _create_aggregation_functions(
        functions_dict=current_functions_dict,
        aggregations_dict=derived_aggregation_specs_dict,
        aggregation_type="group",
        top_level_namespace=top_level_namespace,
    )

    return {
        **derived_aggregation_functions,
        **current_functions_dict,
    }


def _create_aggregation_functions(
    functions_dict: QualifiedFunctionsDict,
    aggregations_dict: QualifiedAggregationSpecsDict,
    aggregation_type: Literal["group", "p_id"],
    top_level_namespace: set[str],
) -> QualifiedFunctionsDict:
    """Create aggregation functions for one aggregation type.

    Parameters
    ----------
    functions_dict
        Dict with qualified function names as keys and functions with qualified
        arguments as values.
    aggregations_dict
        Dict with qualified aggregation spec names as keys and aggregation specs as
        values.
    aggregation_type
        The aggregation type.
    top_level_namespace
        Set of top-level namespaces.

    Returns
    -------
    The qualified functions dict with derived functions.
    """

    group_by_functions = {
        name: func
        for name, func in functions_dict.items()
        if isinstance(func, GroupByFunction)
    }

    expected_aggregation_spec_type = (
        AggregateByGroupSpec if aggregation_type == "group" else AggregateByPIDSpec
    )

    aggregation_functions = {}
    for target_name, aggregation_spec in aggregations_dict.items():
        # Skip if aggregation spec is not the current aggregation type
        if not isinstance(aggregation_spec, expected_aggregation_spec_type):
            continue

        if aggregation_type == "group":
            group_by_id_name = get_name_of_group_by_id(
                target_name=target_name,
                group_by_functions=group_by_functions,
            )

            if not group_by_id_name:
                msg = format_errors_and_warnings(
                    "Name of aggregated column needs to have a suffix "
                    "indicating the group over which it is aggregated. "
                    f"{dt.tree_path_from_qual_name(target_name)} does not do so."
                )
                raise ValueError(msg)

            derived_func = _create_one_aggregate_by_group_func(
                aggregation_target=target_name,
                aggregation_spec=aggregation_spec,
                group_by_id=group_by_id_name,
                top_level_namespace=top_level_namespace,
            )
        else:
            derived_func = _create_one_aggregate_by_p_id_func(
                aggregation_target=target_name,
                aggregation_spec=aggregation_spec,
                top_level_namespace=top_level_namespace,
            )

        aggregation_functions[target_name] = derived_func

    return aggregation_functions


def _create_derived_aggregations_specs_dict(
    functions_dict: QualifiedFunctionsDict,
    targets_dict: QualifiedTargetsDict,
    data_dict: QualifiedDataDict,
) -> QualifiedAggregationSpecsDict:
    """Create automatic aggregation specs derived from functions and data.

    Aggregation specifications are created automatically for summation aggregations.

    Example
    -------
    If
    - `func_hh` is an argument of the functions in `functions_dict`, or a target
    - and not represented by a function in `functions_dict` or a data column in
        the input data
    then an automatic aggregation specification is created for the sum aggregation of
    `func` by household.

    Parameters
    ----------
    functions_dict
        The functions dict with qualified function names as keys and functions as
        values.
    targets_dict
        The targets dict with qualified target names as keys and None as values.
    data_dict
        The data dict with qualified data names as keys and pandas Series as values.

    Returns
    -------
    The aggregation specifications derived from the functions and data.
    """
    potential_aggregation_function_names = set(
        *targets_dict,
        *_get_potential_aggregation_function_names_from_function_arguments(
            functions_dict=functions_dict,
        ),
    )

    # Create source tree for aggregations. Source can be any already existing function
    # or data column.
    aggregation_sources = {
        **functions_dict,
        **data_dict,
    }

    # Create aggregation specs.
    derived_aggregations_specs = {}
    for target_name in potential_aggregation_function_names:
        # Don't create aggregation functions for unsupported groupings or functions that
        # already exist in the source tree.
        aggregation_specs_needed = (
            any(target_name.endswith(f"_{g}") for g in SUPPORTED_GROUPINGS)
            and target_name not in aggregation_sources
        )

        if aggregation_specs_needed:
            derived_aggregations_specs[target_name] = AggregateByGroupSpec(
                aggr="sum",
                source_col=remove_group_suffix(target_name),
            )
        else:
            continue

    return derived_aggregations_specs


def _get_potential_aggregation_function_names_from_function_arguments(
    functions_dict: QualifiedFunctionsDict,
) -> set[str]:
    """Get potential aggregation function names from function arguments.

    Parameters
    ----------
    functions_dict
        Dictionary containing functions to build the DAG.

    Returns
    -------
    Set of potential aggregation targets.
    """
    current_set = set()
    for func in functions_dict.values():
        for name in get_names_of_arguments_without_defaults(func):
            current_set.add(name)
    return current_set


def _annotations_for_aggregation(
    aggregation_method: str,
    source_col: str,
    namespace: tuple[str],
    functions_tree: NestedFunctionDict,
    types_input_variables: dict[str, Any],
) -> dict[str, Any]:
    """Create annotations for derived aggregation functions."""
    annotations = {}

    path_to_source_col = (
        _get_tree_path_from_source_col_name(
            name=source_col,
            namespace=namespace,
        )
        if aggregation_method != "count"
        else None
    )

    flat_functions = dt.flatten_to_tree_paths(functions_tree)
    flat_types_input_variables = dt.flatten_to_tree_paths(types_input_variables)

    if aggregation_method == "count":
        annotations["return"] = int
    elif path_to_source_col in flat_functions:
        # Source col is a function in the functions tree
        source_function = flat_functions[path_to_source_col]
        if "return" in source_function.__annotations__:
            annotations[source_col] = source_function.__annotations__["return"]
            annotations["return"] = _select_return_type(
                aggregation_method, annotations[source_col]
            )
        else:
            # TODO(@hmgaudecker): Think about how type annotations of aggregations
            # of user-provided input variables are handled
            # https://github.com/iza-institute-of-labor-economics/gettsim/issues/604
            pass
    elif path_to_source_col in flat_types_input_variables:
        # Source col is a basic input variable
        annotations[source_col] = flat_types_input_variables[path_to_source_col]
        annotations["return"] = _select_return_type(
            aggregation_method, annotations[source_col]
        )
    else:
        # TODO(@hmgaudecker): Think about how type annotations of aggregations of
        # user-provided input variables are handled
        # https://github.com/iza-institute-of-labor-economics/gettsim/issues/604
        pass
    return annotations


def _select_return_type(aggregation_method: str, source_col_type: type) -> type:
    # Find out return type
    if (source_col_type == int) and (aggregation_method in ["any", "all"]):
        return_type = bool
    elif (source_col_type == bool) and (aggregation_method in ["sum"]):
        return_type = int
    else:
        return_type = source_col_type

    return return_type


def _create_one_aggregate_by_group_func(
    aggregation_target: str,
    aggregation_spec: AggregateByGroupSpec,
    group_by_id: str,
    top_level_namespace: set[str],
) -> DerivedFunction:
    """Create an aggregation function based on aggregation specification.

    Parameters
    ----------
    aggregation_target
        Leaf name of the aggregation target.
    aggregation_spec
        The aggregation specification.
    annotations
        The annotations for the derived function.
    group_by_id
        The group-by-identifier.
    top_level_namespace
        Set of top-level namespaces.

    Returns
    -------
    The derived function.

    """

    aggregation_method = aggregation_spec.aggr
    source_col = aggregation_spec.source_col

    if aggregation_method == "count":
        derived_from = group_by_id
        mapper = {"group_by_id": group_by_id}

        def agg_func(group_by_id):
            return grouped_count(group_by_id)

    else:
        derived_from = (source_col, group_by_id)
        mapper = {
            "source_col": source_col,
            "group_by_id": group_by_id,
        }
        if aggregation_method == "sum":

            def agg_func(source_col, group_by_id):
                return grouped_sum(source_col, group_by_id)

        elif aggregation_method == "mean":

            def agg_func(source_col, group_by_id):
                return grouped_mean(source_col, group_by_id)

        elif aggregation_method == "max":

            def agg_func(source_col, group_by_id):
                return grouped_max(source_col, group_by_id)

        elif aggregation_method == "min":

            def agg_func(source_col, group_by_id):
                return grouped_min(source_col, group_by_id)

        elif aggregation_method == "any":

            def agg_func(source_col, group_by_id):
                return grouped_any(source_col, group_by_id)

        elif aggregation_method == "all":

            def agg_func(source_col, group_by_id):
                return grouped_all(source_col, group_by_id)

        else:
            msg = format_errors_and_warnings(
                f"Aggregation method {aggregation_method} is not implemented."
            )
            raise ValueError(msg)

    wrapped_func = dt.one_function_without_tree_logic(
        function=dags.rename_arguments(
            func=agg_func,
            mapper=mapper,
        ),
        tree_path=dt.tree_path_from_qual_name(aggregation_target),
        top_level_namespace=top_level_namespace,
    )

    return DerivedFunction(
        function=wrapped_func,
        leaf_name=aggregation_target,
        derived_from=derived_from,
    )


def _create_one_aggregate_by_p_id_func(
    aggregation_target: str,
    aggregation_spec: AggregateByPIDSpec,
    top_level_namespace: set[str],
) -> DerivedFunction:
    """Create one function that links variables across persons.

    Parameters
    ----------
    aggregation_target
        Name of the aggregation target.
    aggregation_spec
        The aggregation specification.
    top_level_namespace
        Set of top-level namespaces.

    Returns
    -------
    The derived function.

    """
    aggregation_method = aggregation_spec.aggr
    p_id_to_aggregate_by = aggregation_spec.p_id_to_aggregate_by
    source_col = aggregation_spec.source_col

    if aggregation_method == "count":
        derived_from = p_id_to_aggregate_by
        mapper = {
            "p_id_to_aggregate_by": p_id_to_aggregate_by,
            "p_id_to_store_by": "p_id",
        }

        def agg_func(p_id_to_aggregate_by, p_id_to_store_by):
            return count_by_p_id(p_id_to_aggregate_by, p_id_to_store_by)

    else:
        derived_from = (source_col, p_id_to_aggregate_by)
        mapper = {
            "p_id_to_aggregate_by": p_id_to_aggregate_by,
            "p_id_to_store_by": "p_id",
            "column": source_col,
        }

        if aggregation_method == "sum":

            def agg_func(column, p_id_to_aggregate_by, p_id_to_store_by):
                return sum_by_p_id(column, p_id_to_aggregate_by, p_id_to_store_by)

        elif aggregation_method == "mean":

            def agg_func(column, p_id_to_aggregate_by, p_id_to_store_by):
                return mean_by_p_id(column, p_id_to_aggregate_by, p_id_to_store_by)

        elif aggregation_method == "max":

            def agg_func(column, p_id_to_aggregate_by, p_id_to_store_by):
                return max_by_p_id(column, p_id_to_aggregate_by, p_id_to_store_by)

        elif aggregation_method == "min":

            def agg_func(column, p_id_to_aggregate_by, p_id_to_store_by):
                return min_by_p_id(column, p_id_to_aggregate_by, p_id_to_store_by)

        elif aggregation_method == "any":

            def agg_func(column, p_id_to_aggregate_by, p_id_to_store_by):
                return any_by_p_id(column, p_id_to_aggregate_by, p_id_to_store_by)

        elif aggregation_method == "all":

            def agg_func(column, p_id_to_aggregate_by, p_id_to_store_by):
                return all_by_p_id(column, p_id_to_aggregate_by, p_id_to_store_by)

        else:
            msg = format_errors_and_warnings(
                f"Aggregation method {aggregation_method} is not implemented."
            )
            raise ValueError(msg)

    wrapped_func = dt.one_function_without_tree_logic(
        function=dags.rename_arguments(
            func=agg_func,
            mapper=mapper,
        ),
        tree_path=dt.tree_path_from_qual_name(aggregation_target),
        top_level_namespace=top_level_namespace,
    )

    return DerivedFunction(
        function=wrapped_func,
        leaf_name=aggregation_target,
        derived_from=derived_from,
    )


def _get_tree_path_from_source_col_name(
    name: str,
    namespace: tuple[str],
) -> tuple[str]:
    """Get the tree path of a source column name that may be qualified or simple.

    This function returns the tree path of a source column name that may be a qualified
    or simple name. If the name is qualified, the path implied by the name is returned.
    Else, the current path plus the simple name is returned.

    Parameters
    ----------
    name
        The qualified or simple name.
    namespace
        The namespace where 'name' is located.

    Returns
    -------
    The path of 'name' in the tree.
    """
    if dt.QUAL_NAME_DELIMITER in name:
        # 'name' is already namespaced.
        new_tree_path = name.split(dt.QUAL_NAME_DELIMITER)
    else:
        # 'name' is not namespaced.
        new_tree_path = [*namespace, name]

    return tuple(new_tree_path)


def _fail_if_targets_not_in_functions(
    functions_dict: QualifiedFunctionsDict, targets_dict: QualifiedTargetsDict
) -> None:
    """Fail if some target is not among functions.

    Parameters
    ----------
    functions_dict
        Dictionary containing functions to build the DAG.
    targets_dict
        The targets which should be computed. They limit the DAG in the way that only
        ancestors of these nodes need to be considered.

    Raises
    ------
    ValueError
        Raised if any member of `targets_dict` is not among functions.

    """
    targets_not_in_functions_tree = [n for n in targets_dict if n not in functions_dict]
    if targets_not_in_functions_tree:
        formatted = format_list_linewise(targets_not_in_functions_tree)
        msg = format_errors_and_warnings(
            f"The following targets have no corresponding function:\n\n{formatted}"
        )
        raise ValueError(msg)
