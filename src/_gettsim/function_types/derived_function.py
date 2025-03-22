from __future__ import annotations

from typing import TYPE_CHECKING

import dags.tree as dt

from _gettsim.function_types.policy_function import PolicyFunction

if TYPE_CHECKING:
    from collections.abc import Callable


class DerivedFunction(PolicyFunction):
    """
    A function that is derived from another via aggregation, time conversion, etc.

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
        aggregation_target: str,
        source: str,
        aggregation_method: str,
    ):
        super().__init__(
            function=function,
            leaf_name=dt.tree_path_from_qual_name(aggregation_target)[-1],
            start_date=None,  # Improve this
            end_date=None,  # Improve this
            params_key_for_rounding=None,
            skip_vectorization=True,
        )

        self.source = source
        self.aggregation_method = aggregation_method
