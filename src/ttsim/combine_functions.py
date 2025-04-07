from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Literal

import dags
import dags.tree as dt

from _gettsim.config import (
    SUPPORTED_GROUPINGS,
    TYPES_INPUT_VARIABLES,
)
from ttsim.aggregation import (
    AggregateByGroupSpec,
    AggregateByPIDSpec,
    AggregationType,
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
from ttsim.function_types import DerivedAggregationFunction, GroupByFunction
from ttsim.shared import (
    format_errors_and_warnings,
    format_list_linewise,
    get_name_of_group_by_id,
    get_names_of_arguments_without_defaults,
    remove_group_suffix,
)
from ttsim.time_conversion import create_time_conversion_functions

if TYPE_CHECKING:
    from collections.abc import Callable

    from ttsim.typing import (
        QualNameAggregationSpecsDict,
        QualNameDataDict,
        QualNameFunctionsDict,
        QualNameTargetList,
    )


def combine_policy_functions_and_derived_functions(
    functions: QualNameFunctionsDict,
    aggregation_specs_from_environment: QualNameAggregationSpecsDict,
    targets: QualNameTargetList,
    data: QualNameDataDict,
    top_level_namespace: set[str],
) -> QualNameFunctionsDict:
    """Add derived functions to the qualified functions dict.

    Derived functions are time converted functions and aggregation functions (aggregate
    by p_id or by group).

    Checks that all targets have a corresponding function in the functions tree or can
    be taken from the data.

    Parameters
    ----------
    functions
        Dict with qualified function names as keys and functions with qualified
        arguments as values.
    aggregation_specs_from_environment
        Dict with qualified aggregation spec names as keys and aggregation specs as
        values.
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
    # Create parent-child relationships
    aggregate_by_p_id_functions = _create_aggregation_functions(
        functions=functions,
        aggregation_functions_to_create=aggregation_specs_from_environment,
        top_level_namespace=top_level_namespace,
        aggregation_type="p_id",
    )
    current_functions = {**aggregate_by_p_id_functions, **functions}

    # Create functions for different time units
    time_conversion_functions = create_time_conversion_functions(
        functions=current_functions,
        data=data,
    )
    current_functions = {**time_conversion_functions, **current_functions}

    # Create aggregation functions
    aggregate_by_group_functions = _create_aggregate_by_group_functions(
        functions=current_functions,
        targets=targets,
        data=data,
        aggregations_from_environment=aggregation_specs_from_environment,
        top_level_namespace=top_level_namespace,
    )
    current_functions = {**aggregate_by_group_functions, **current_functions}

    _fail_if_targets_not_in_functions(functions=current_functions, targets=targets)

    return current_functions


def _create_aggregate_by_group_functions(
    functions: QualNameFunctionsDict,
    targets: QualNameTargetList,
    data: QualNameDataDict,
    aggregations_from_environment: QualNameAggregationSpecsDict,
    top_level_namespace: set[str],
) -> QualNameFunctionsDict:
    """Create aggregation functions."""
    # Create the aggregation functions that were explicitly specified.
    aggregation_functions_from_environment = _create_aggregation_functions(
        functions=functions,
        aggregation_functions_to_create=aggregations_from_environment,
        aggregation_type="group",
        top_level_namespace=top_level_namespace,
    )

    functions_with_aggregation_functions_from_environment = {
        **aggregation_functions_from_environment,
        **functions,
    }

    # Create derived aggregation functions.
    derived_aggregation_specs = _create_derived_aggregations_specs(
        functions=functions_with_aggregation_functions_from_environment,
        targets=targets,
        data=data,
        top_level_namespace=top_level_namespace,
    )
    aggregation_functions_derived_from_names = _create_aggregation_functions(
        functions=functions_with_aggregation_functions_from_environment,
        aggregation_functions_to_create=derived_aggregation_specs,
        aggregation_type="group",
        top_level_namespace=top_level_namespace,
    )
    return {
        **aggregation_functions_derived_from_names,
        **aggregation_functions_from_environment,
    }


def _create_aggregation_functions(
    functions: QualNameFunctionsDict,
    aggregation_functions_to_create: QualNameAggregationSpecsDict,
    aggregation_type: Literal["group", "p_id"],
    top_level_namespace: set[str],
) -> QualNameFunctionsDict:
    """Create aggregation functions for one aggregation type.

    Parameters
    ----------
    functions
        Dict with qualified function names as keys and functions with qualified
        arguments as values.
    aggregation_functions_to_create
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
        for name, func in functions.items()
        if isinstance(getattr(func, "__wrapped__", func), GroupByFunction)
    }

    expected_aggregation_spec_type = (
        AggregateByGroupSpec if aggregation_type == "group" else AggregateByPIDSpec
    )

    aggregation_functions = {}
    for target_name, aggregation_spec in aggregation_functions_to_create.items():
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
                functions=functions,
                top_level_namespace=top_level_namespace,
            )
        else:
            derived_func = _create_one_aggregate_by_p_id_func(
                aggregation_target=target_name,
                aggregation_spec=aggregation_spec,
                functions=functions,
                top_level_namespace=top_level_namespace,
            )

        aggregation_functions[target_name] = derived_func

    return _annotate_aggregation_functions(
        functions=functions,
        aggregation_functions=aggregation_functions,
        types_input_variables=dt.flatten_to_qual_names(TYPES_INPUT_VARIABLES),
    )


def _create_derived_aggregations_specs(
    functions: QualNameFunctionsDict,
    targets: QualNameTargetList,
    data: QualNameDataDict,
    top_level_namespace: set[str],
) -> QualNameAggregationSpecsDict:
    """Create automatic aggregation specs derived from functions and data.

    Aggregation specifications are created automatically for summation aggregations.

    Example
    -------
    If
    - `func_hh` is an argument of the functions in `functions`, or a target
    - and not represented by a function in `functions` or a data column in
        the input data
    then an automatic aggregation specification is created for the sum aggregation of
    `func` by household.

    Parameters
    ----------
    functions
        The functions dict with qualified function names as keys and functions as
        values.
    targets
        The list of targets with qualified names.
    data
        The data dict with qualified data names as keys and pandas Series as values.

    Returns
    -------
    The aggregation specifications derived from the functions and data.
    """
    potential_aggregation_function_names = {
        *targets,
        *_get_potential_aggregation_function_names_from_function_arguments(
            functions=functions,
        ),
    }

    # Create source tree for aggregations. Source can be any already existing function
    # or data column.
    aggregation_sources = {
        **functions,
        **data,
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
                aggr=AggregationType.SUM,
                source=_get_name_of_aggregation_source(
                    target_name=target_name,
                    top_level_namespace=top_level_namespace,
                ),
            )
        else:
            continue

    return derived_aggregations_specs


def _get_potential_aggregation_function_names_from_function_arguments(
    functions: QualNameFunctionsDict,
) -> set[str]:
    """Get potential aggregation function names from function arguments.

    Parameters
    ----------
    functions
        Dictionary containing functions to build the DAG.

    Returns
    -------
    Set of potential aggregation targets.
    """
    current_set = set()
    for func in functions.values():
        for name in get_names_of_arguments_without_defaults(func):
            current_set.add(name)
    return current_set


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
    functions: QualNameFunctionsDict,
    top_level_namespace: set[str],
) -> DerivedAggregationFunction:
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
    functions
        The functions dict with qualified function names as keys and functions as
        values.
    top_level_namespace
        Set of top-level namespaces.

    Returns
    -------
    The derived function.

    """

    aggregation_method = aggregation_spec.aggr
    source = aggregation_spec.source

    if aggregation_method == "count":
        mapper = {"group_by_id": group_by_id}

        def agg_func(group_by_id):
            return grouped_count(group_by_id)

    else:
        mapper = {
            "source": source,
            "group_by_id": group_by_id,
        }
        if aggregation_method == "sum":

            def agg_func(source, group_by_id):
                return grouped_sum(source, group_by_id)

        elif aggregation_method == "mean":

            def agg_func(source, group_by_id):
                return grouped_mean(source, group_by_id)

        elif aggregation_method == "max":

            def agg_func(source, group_by_id):
                return grouped_max(source, group_by_id)

        elif aggregation_method == "min":

            def agg_func(source, group_by_id):
                return grouped_min(source, group_by_id)

        elif aggregation_method == "any":

            def agg_func(source, group_by_id):
                return grouped_any(source, group_by_id)

        elif aggregation_method == "all":

            def agg_func(source, group_by_id):
                return grouped_all(source, group_by_id)

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

    qual_name_source = (
        _get_qual_name_of_source_col(
            source=source,
            wrapped_func=wrapped_func,
        )
        if source
        else None
    )

    return DerivedAggregationFunction(
        function=wrapped_func,
        source=qual_name_source,
        source_function=functions.get(qual_name_source, None),
        aggregation_target=aggregation_target,
        aggregation_method=aggregation_method,
    )


def _create_one_aggregate_by_p_id_func(
    aggregation_target: str,
    aggregation_spec: AggregateByPIDSpec,
    functions: QualNameFunctionsDict,
    top_level_namespace: set[str],
) -> DerivedAggregationFunction:
    """Create one function that links variables across persons.

    Parameters
    ----------
    aggregation_target
        Name of the aggregation target.
    aggregation_spec
        The aggregation specification.
    functions
        The functions dict with qualified function names as keys and functions as
        values.
    top_level_namespace
        Set of top-level namespaces.

    Returns
    -------
    The derived function.

    """
    aggregation_method = aggregation_spec.aggr
    p_id_to_aggregate_by = aggregation_spec.p_id_to_aggregate_by
    source = aggregation_spec.source if aggregation_method != "count" else None

    if aggregation_method == "count":
        mapper = {
            "p_id_to_aggregate_by": p_id_to_aggregate_by,
            "p_id_to_store_by": "p_id",
        }

        def agg_func(p_id_to_aggregate_by, p_id_to_store_by):
            return count_by_p_id(p_id_to_aggregate_by, p_id_to_store_by)

    else:
        mapper = {
            "p_id_to_aggregate_by": p_id_to_aggregate_by,
            "p_id_to_store_by": "p_id",
            "column": source,
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

    qual_name_source = (
        _get_qual_name_of_source_col(
            source=source,
            wrapped_func=wrapped_func,
        )
        if source
        else None
    )

    return DerivedAggregationFunction(
        function=wrapped_func,
        source=qual_name_source,
        source_function=functions.get(qual_name_source, None),
        aggregation_target=aggregation_target,
        aggregation_method=aggregation_method,
    )


def _annotate_aggregation_functions(
    functions: QualNameFunctionsDict,
    types_input_variables: dict[str, type],
    aggregation_functions: QualNameFunctionsDict,
) -> QualNameFunctionsDict:
    """Annotate aggregation functions.

    Add type annotations to the aggregation functions based on the type annotations of
    the source columns and the aggregation method.

    Parameters
    ----------
    functions
        Dict with qualified function names as keys and functions with qualified
        arguments as values.
    aggregation_functions
        Dict with qualified aggregation function names as keys and aggregation functions
        as values.
    types_input_variables
        Dict with qualified data names as keys and types as values.

    Returns
    -------
    The annotated aggregation functions.

    """
    annotated_functions = {}
    for aggregation_target, aggregation_function in aggregation_functions.items():
        source = aggregation_function.source
        aggregation_method = aggregation_function.aggregation_method

        annotations = {}
        if aggregation_method == "count":
            annotations["return"] = int
        elif source in functions:
            source_function = functions[source]
            if "return" in source_function.__annotations__:
                annotations[source] = source_function.__annotations__["return"]
                annotations["return"] = _select_return_type(
                    aggregation_method, annotations[source]
                )
            else:
                # TODO(@hmgaudecker): Think about how type annotations of aggregations
                # of user-provided input variables are handled
                # https://github.com/iza-institute-of-labor-economics/gettsim/issues/604
                pass
        elif source in types_input_variables:
            annotations[source] = types_input_variables[source]
            annotations["return"] = _select_return_type(
                aggregation_method, annotations[source]
            )
        else:
            # TODO(@hmgaudecker): Think about how type annotations of aggregations of
            # user-provided input variables are handled
            # https://github.com/iza-institute-of-labor-economics/gettsim/issues/604
            pass

        aggregation_function.__annotations__ = annotations
        annotated_functions[aggregation_target] = aggregation_function

    return annotated_functions


def _fail_if_targets_not_in_functions(
    functions: QualNameFunctionsDict, targets: QualNameTargetList
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


def _get_qual_name_of_source_col(
    source: str,
    wrapped_func: Callable,
) -> str | None:
    """Get the qualified source column name."""
    parameters = inspect.signature(wrapped_func).parameters
    matches = [p for p in parameters if p.endswith(source)]
    if len(matches) == 1:
        return matches[0]
    else:
        return None


def _get_name_of_aggregation_source(
    target_name: str,
    top_level_namespace: set[str],
) -> str:
    """Get the name of the source column for an aggregation target.

    This function allows for source and target name to be from different namespaces.

    Example 1
    ---------
    > target_name = "arbeitslosengeld_2__vermögen_bg"
    > top_level_namespace = {"vermögen", "arbeitslosengeld_2"}
    > _get_name_of_aggregation_source(target_name, top_level_namespace)
    "vermögen"

    Example 2
    ---------
    > target_name = "arbeitslosengeld_2__vermögen_bg"
    > top_level_namespace = {"arbeitslosengeld_2"}
    > _get_name_of_aggregation_source(target_name, top_level_namespace)
    "arbeitslosengeld_2__vermögen"
    """
    leaf_name = remove_group_suffix(dt.tree_path_from_qual_name(target_name)[-1])
    if leaf_name in top_level_namespace:
        return leaf_name
    else:
        return remove_group_suffix(target_name)
