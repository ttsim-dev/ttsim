from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import dags.tree as dt

from ttsim.interface_dag_elements.fail_if import (
    format_errors_and_warnings,
    format_list_linewise,
)
from ttsim.interface_dag_elements.interface_node_objects import interface_function

if TYPE_CHECKING:
    from ttsim.interface_dag_elements.typing import (
        NestedPolicyEnvironment,
        OrderedQNames,
        QNameDataColumns,
    )


class FunctionsAndDataColumnsOverlapWarning(UserWarning):
    """
    Warning that functions which compute columns overlap with existing columns.

    Parameters
    ----------
    columns_overriding_functions : UnorderedQNames
        Names of columns in the data that override hard-coded functions.
    """

    def __init__(self, columns_overriding_functions: OrderedQNames) -> None:
        n_cols = len(columns_overriding_functions)
        if n_cols == 1:
            first_part = format_errors_and_warnings("Your data provides the column:")
            second_part = format_errors_and_warnings(
                """
                This is already present among the hard-coded functions of the taxes and
                transfers system. If you want this data column to be used instead of
                calculating it within TTSIM you need not do anything. If you want this
                data column to be calculated by hard-coded functions, remove it from the
                *data* you pass to TTSIM. You need to pick one option for each column
                that appears in the list above.
                """,
            )
        else:
            first_part = format_errors_and_warnings("Your data provides the columns:")
            second_part = format_errors_and_warnings(
                """
                These are already present among the hard-coded functions of the taxes
                and transfers system. If you want a data column to be used instead of
                calculating it within TTSIM you do not need to do anything. If you
                want data columns to be calculated by hard-coded functions, remove them
                from the *data* you pass to TTSIM. You need to pick one option for
                each column that appears in the list above.
                """,
            )
        formatted = format_list_linewise(columns_overriding_functions)
        how_to_ignore = format_errors_and_warnings(
            """
            In order to not perform this check, you can ... TODO
            """,
        )
        super().__init__(f"{first_part}\n{formatted}\n{second_part}\n{how_to_ignore}")


@interface_function()
def functions_and_data_columns_overlap(
    policy_environment: NestedPolicyEnvironment,
    labels__processed_data_columns: QNameDataColumns,
) -> None:
    """Warn if functions are overridden by data."""
    overridden_elements = sorted(
        {
            col
            for col in labels__processed_data_columns
            if col in dt.flatten_to_qual_names(policy_environment)
        },
    )
    if len(overridden_elements) > 0:
        warnings.warn(
            FunctionsAndDataColumnsOverlapWarning(overridden_elements),
            stacklevel=2,
        )
