from __future__ import annotations

from typing import TYPE_CHECKING

import dags.tree as dt
import optree
import pandas as pd

from ttsim.shared import format_errors_and_warnings, format_list_linewise

if TYPE_CHECKING:
    from ttsim.typing import NestedData, NestedInputsPathsToDfColumns


def create_data_tree_from_df(
    inputs_tree_to_df_columns: NestedInputsPathsToDfColumns,
    df: pd.DataFrame,
) -> NestedData:
    """Transform a pandas DataFrame to a nested dictionary expected by TTSIM.
    `
        Args
        ----
            inputs_tree_to_df_columns:
                A nested dictionary that defines the structure of the output tree. Keys
                are strings that define the nested structure. Values can be:

                - Strings that reference column names in the DataFrame.
                - Numeric or boolean values (which will be broadcasted to match the
                  DataFrame length)
            df:
                The pandas DataFrame containing the source data.

        Returns
        -------
            A nested dictionary structure containing the data organized according to the
            mapping definition.

        Examples
        --------
            >>> df = pd.DataFrame({
            ...     "a": [1, 2, 3],
            ...     "b": [4, 5, 6],
            ...     "c": [7, 8, 9],
            ... })
            >>> inputs_tree_to_df_columns = {
            ...     "n1": {
            ...         "n2": "a",
            ...         "n3": "b",
            ...     },
            ...     "n4": 3,
            ... }
            >>> result = create_data_tree(
            ...     inputs_tree_to_df_columns=inputs_tree_to_df_columns,
            ...     df=df,
            ... )
            >>> result
            {
                "n1": {
                    "n2": pd.Series([1, 2, 3]),
                    "n3": pd.Series([4, 5, 6]),
                },
                "n4": pd.Series([3, 3, 3]),
            }


    """
    _fail_if_df_has_bool_or_numeric_column_names(df)
    _fail_if_mapper_has_incorrect_format(inputs_tree_to_df_columns)

    qualified_inputs_tree_to_df_columns = dt.flatten_to_qual_names(
        inputs_tree_to_df_columns
    )

    name_to_input_series = {}
    for (
        qualified_input_name,
        input_value,
    ) in qualified_inputs_tree_to_df_columns.items():
        if input_value in df.columns:
            name_to_input_series[qualified_input_name] = df[input_value]
        else:
            name_to_input_series[qualified_input_name] = pd.Series(
                [input_value] * len(df),
                index=df.index,
            )

    return dt.unflatten_from_qual_names(name_to_input_series)


def _fail_if_mapper_has_incorrect_format(
    inputs_tree_to_df_columns: NestedInputsPathsToDfColumns,
) -> None:
    """Fail if the input tree to column name mapping has an incorrect format."""
    if not isinstance(inputs_tree_to_df_columns, dict):
        msg = format_errors_and_warnings(
            """The input tree to column mapping must be a (nested) dictionary. Call
            `dags.tree.create_tree_with_input_types` to create a template."""
        )
        raise TypeError(msg)

    non_string_paths = [
        str(path)
        for path in optree.tree_paths(inputs_tree_to_df_columns, none_is_leaf=True)  # type: ignore[arg-type]
        if not all(isinstance(part, str) for part in path)
    ]
    if non_string_paths:
        msg = format_errors_and_warnings(
            f"""All path elements of `inputs_tree_to_df_columns` must be strings.
            Found the following paths that contain non-string elements:

            {format_list_linewise(non_string_paths)}

            Call `dags.tree.create_tree_with_input_types` to create a template.
            """
        )
        raise TypeError(msg)

    incorrect_types = {
        k: type(v)
        for k, v in dt.flatten_to_qual_names(inputs_tree_to_df_columns).items()
        if not isinstance(v, str | int | bool)
    }
    if incorrect_types:
        formatted_incorrect_types = "\n".join(
            f"    - {k}: {v.__name__}" for k, v in incorrect_types.items()
        )
        msg = format_errors_and_warnings(
            f"""Values of the input tree to column mapping must be strings, integers,
            or booleans.
            Found the following incorrect types:

            {formatted_incorrect_types}
            """
        )
        raise TypeError(msg)


def _fail_if_df_has_bool_or_numeric_column_names(df: pd.DataFrame) -> None:
    """Fail if the DataFrame has bool or numeric column names."""
    common_msg = format_errors_and_warnings(
        """DataFrame column names cannot be booleans or numbers. This restriction
        prevents ambiguity between actual column references and values intended for
        broadcasting.
        """
    )
    bool_column_names = [col for col in df.columns if isinstance(col, bool)]
    numeric_column_names = [
        col
        for col in df.columns
        if isinstance(col, (int, float)) or (isinstance(col, str) and col.isnumeric())
    ]

    if bool_column_names or numeric_column_names:
        msg = format_errors_and_warnings(
            f"""
            {common_msg}

            Boolean column names: {bool_column_names}.
            Numeric column names: {numeric_column_names}.
            """
        )
        raise ValueError(msg)
