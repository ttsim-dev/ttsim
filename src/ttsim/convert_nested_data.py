from __future__ import annotations

from typing import TYPE_CHECKING

import dags.tree as dt
import numpy as np
import optree
import pandas as pd

from ttsim.shared import format_errors_and_warnings, format_list_linewise

if TYPE_CHECKING:
    from ttsim.typing import NestedData, NestedStrings, QualNameData


def nested_data_to_df_with_nested_columns(
    nested_data_to_convert: NestedData,
    data_with_p_id: NestedData | QualNameData,
) -> pd.DataFrame:
    """Convert a nested data structure to a DataFrame.

    Args:
        nested_data_with_p_id:
            A nested data structure.
        nested_data_paths_to_outputs_df_columns:
            A tree that maps paths (sequence of keys) to data columns names.

    Returns:
        A DataFrame.
    """
    flat_data_to_convert = dt.flatten_to_tree_paths(nested_data_to_convert)

    return pd.DataFrame(
        flat_data_to_convert, index=pd.Index(data_with_p_id["p_id"], name="p_id")
    )


def nested_data_to_df_with_mapped_columns(
    nested_data_to_convert: NestedData,
    nested_outputs_df_column_names: NestedStrings,
    data_with_p_id: NestedData | QualNameData,
) -> pd.DataFrame:
    """Convert a nested data structure to a DataFrame.

    Args:
        nested_data_to_convert:
            A nested data structure.
        nested_data_paths_to_outputs_df_columns:
            A tree that maps paths (sequence of keys) to data columns names.
        data_with_p_id:
            Some data structure with a "p_id" column.

    Returns:
        A DataFrame.
    """
    flat_data_to_convert = dt.flatten_to_tree_paths(nested_data_to_convert)
    flat_df_columns = dt.flatten_to_tree_paths(nested_outputs_df_column_names)

    fail_if_data_paths_are_missing_in_paths_to_column_names(
        available_paths=list(flat_df_columns.keys()),
        required_paths=list(flat_data_to_convert.keys()),
    )
    fail_if_incompatible_objects_in_nested_data(flat_data_to_convert)

    return pd.DataFrame(
        {flat_df_columns[path]: data for path, data in flat_data_to_convert.items()},
        index=pd.Index(data_with_p_id["p_id"], name="p_id"),
    )


def dataframe_to_nested_data(
    inputs_tree_to_df_columns: NestedStrings,
    df: pd.DataFrame,
) -> NestedData:
    """Transform a pandas DataFrame to a nested dictionary expected by TTSIM.
    `
        Args
        ----
            inputs_tree_to_df_columns:
                A nested dictionary that defines the structure of the inputs tree. The
                elements of the tree paths are strings. Leaves can be:

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


def fail_if_incompatible_objects_in_nested_data(
    paths_to_data: QualNameData,
) -> None:
    """Fail if the nested data contains incompatible objects."""
    _numeric_types = (int, float, bool, np.integer, np.floating, np.bool_)

    faulty_paths = []
    for path, data in paths_to_data.items():
        if isinstance(data, (pd.Series, np.ndarray, list)):
            if all(isinstance(item, _numeric_types) for item in data):
                continue
            else:
                faulty_paths.append(str(path))
        elif isinstance(data, _numeric_types):
            continue
        else:
            faulty_paths.append(str(path))
    if faulty_paths:
        msg = format_errors_and_warnings(
            "The data returned contains objects that cannot be cast to "
            "a pandas.DataFrame column. Make sure that the requested targets return "
            "scalars (int, bool, float - or their numpy equivalents) only."
            "The following paths contain non-scalar objects: "
            f"{format_list_linewise(faulty_paths)}"
        )
        raise TypeError(msg)


def fail_if_data_paths_are_missing_in_paths_to_column_names(
    available_paths: list[str],
    required_paths: list[str],
) -> None:
    """Fail if the data paths are missing in the paths to column names."""
    missing_paths = [
        str(path)
        for path in required_paths
        if path not in available_paths and path != ("p_id",)
    ]
    if missing_paths:
        msg = format_errors_and_warnings(
            "Converting the nested data to a DataFrame failed because the following "
            "paths are not mapped to a column name: "
            f"{format_list_linewise(list(missing_paths))}"
        )
        raise ValueError(msg)


def _fail_if_mapper_has_incorrect_format(
    inputs_tree_to_df_columns: NestedStrings,
) -> None:
    """Fail if the input tree to column name mapping has an incorrect format."""
    if not isinstance(inputs_tree_to_df_columns, dict):
        msg = format_errors_and_warnings(
            """The inputs tree to column mapping must be a (nested) dictionary. Call
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
        if not isinstance(v, str | int | float | bool)
    }
    if incorrect_types:
        formatted_incorrect_types = "\n".join(
            f"    - {k}: {v.__name__}" for k, v in incorrect_types.items()
        )
        msg = format_errors_and_warnings(
            f"""Values of the input tree to column mapping must be strings, integers,
            floats, or Booleans.
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
