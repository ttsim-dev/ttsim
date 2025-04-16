import dags.tree as dt
import optree
import pandas as pd

from ttsim.typing import NestedDataDict, NestedInputToSeriesNameDict


def df_to_data_tree(
    df: pd.DataFrame,
    input_tree_to_column_name_mapping: NestedInputToSeriesNameDict,
) -> NestedDataDict:
    """Convert a pandas DataFrame to a nested data dictionary.

    This function transforms a flat DataFrame into a nested dictionary structure based
    on the provided mapping. The mapping can reference DataFrame columns, pandas Series,
    or numeric values.

    Args:
        df:
            The pandas DataFrame containing the source data.
        input_tree_to_column_name_mapping:
            A nested dictionary that defines the structure of the output. Keys are
            strings that define the nested structure. Values can be: - Strings that
            reference column names in the DataFrame - pandas Series objects - Numeric
            values (which will be broadcast to match the DataFrame length)

    Examples
    --------
        >>> df = pd.DataFrame({
        ...     "a": [1, 2, 3],
        ...     "b": [4, 5, 6],
        ...     "c": [7, 8, 9],
        ... })
        >>> input_tree_to_column_name_mapping = {
        ...     "n1": {
        ...         "n2": "a",
        ...         "n3": "b",
        ...     },
        ...     "n4": pd.Series([1, 2, 3]),
        ...     "n5": 3,
        ... }
        >>> result = df_to_data_tree(df, input_tree_to_column_name_mapping)
        >>> result
        {
            "n1": {
                "n2": pd.Series([1, 2, 3]),
                "n3": pd.Series([4, 5, 6]),
            },
            "n4": pd.Series([1, 2, 3]),
            "n5": pd.Series([3, 3, 3]),
        }



    Returns:
        A nested data dictionary.
    """
    _fail_if_mapper_has_incorrect_format(input_tree_to_column_name_mapping)
    _fail_if_df_has_bool_or_numeric_column_names(df)

    qualified_input_tree_to_column_name_mapping = dt.flatten_to_qual_names(
        input_tree_to_column_name_mapping
    )

    name_to_input_series = {}
    for (
        qualified_input_name,
        input_value,
    ) in qualified_input_tree_to_column_name_mapping.items():
        if input_value in df.columns:
            name_to_input_series[qualified_input_name] = df[input_value]
        elif isinstance(input_value, pd.Series):
            name_to_input_series[qualified_input_name] = input_value
        else:
            name_to_input_series[qualified_input_name] = pd.Series(
                [input_value] * len(df),
                index=df.index,
            )

    return dt.unflatten_from_qual_names(name_to_input_series)


def _fail_if_mapper_has_incorrect_format(
    input_tree_to_column_name_mapping: NestedInputToSeriesNameDict,
) -> None:
    """Fail if the input tree to column name mapping has an incorrect format."""
    if not isinstance(input_tree_to_column_name_mapping, dict):
        msg = "The input tree to column name mapping must be a dictionary."
        raise TypeError(msg)

    inputs = optree.tree_flatten(input_tree_to_column_name_mapping)[0]

    # Check that all Series have the same length
    series_inputs = [x for x in inputs if isinstance(x, pd.Series)]
    if series_inputs:
        expected_length = len(series_inputs[0])
        mismatched_lengths = [
            len(series) for series in series_inputs if len(series) != expected_length
        ]
        if mismatched_lengths:
            raise ValueError(
                "All pd.Series in inputs must have the same length. "
                f"Found series with lengths {mismatched_lengths} "
                f"but expected length {expected_length}."
            )


def _fail_if_df_has_bool_or_numeric_column_names(df: pd.DataFrame) -> None:
    """Fail if the DataFrame has bool or numeric column names."""
    common_msg = "The DataFrame must not have bool or numeric column names."
    bool_column_names = [col for col in df.columns if col.dtype == "bool"]
    numeric_column_names = [col for col in df.columns if col.isnumeric()]

    if bool_column_names or numeric_column_names:
        msg = f"""
        {common_msg}
        Boolean column names: {bool_column_names}.
        Numeric column names: {numeric_column_names}.
        """
        raise ValueError(msg)
