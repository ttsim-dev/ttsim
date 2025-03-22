from __future__ import annotations

from typing import TYPE_CHECKING

import dags.tree as dt

from _gettsim.function_types.policy_function import PolicyFunction

if TYPE_CHECKING:
    from collections.abc import Callable


class DerivedAggregationFunction(PolicyFunction):
    """
    A function that is an aggregation of another function.

    Parameters
    ----------
    function:
        The function to wrap. Argument values of the `@policy_function` are reused
        unless explicitly overwritten.
    aggregation_target:
        The qualified name of the aggregation target.
    source:
        The column from which the new function is derived.
    aggregation_method:
        The method of aggregation used.
    """

    def __init__(
        self,
        *,
        function: Callable,
        source_function: PolicyFunction
        | DerivedTimeConversionFunction
        | DerivedAggregationFunction,
        source_function_name: str,
        aggregation_target: str,
        aggregation_method: str,
    ):
        super().__init__(
            function=function,
            leaf_name=dt.tree_path_from_qual_name(aggregation_target)[-1],
            start_date=source_function.start_date,
            end_date=source_function.end_date,
            params_key_for_rounding=None,
            skip_vectorization=True,
        )

        self.source = source_function_name
        self.aggregation_method = aggregation_method


class DerivedTimeConversionFunction(PolicyFunction):
    """
    A function that is a time conversion of another function.

    Parameters
    ----------
    function:
        The function to wrap. Argument values of the `@policy_function` are reused
        unless explicitly overwritten.
    aggregation_target:
        The qualified name of the aggregation target.
    source:
        The column from which the new function is derived.
    """

    def __init__(
        self,
        *,
        function: Callable,
        source_function: PolicyFunction
        | DerivedTimeConversionFunction
        | DerivedAggregationFunction,
        source_function_name: str,
        conversion_target: str,
    ):
        super().__init__(
            function=function,
            leaf_name=dt.tree_path_from_qual_name(conversion_target)[-1],
            start_date=source_function.start_date,
            end_date=source_function.end_date,
            params_key_for_rounding=None,
            skip_vectorization=True,
        )

        self.source = source_function_name
