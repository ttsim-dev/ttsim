import dags.tree as dt
import optree
import pandas as pd

from ttsim.typing import NestedDataDict, NestedInputToSeriesNameDict


def create_data_tree(
    input_tree_to_column_map: NestedInputToSeriesNameDict,
    df: pd.DataFrame | None = None,
) -> NestedDataDict:
    """Create a data tree from user data.

    This function creates a data tree using the `input_tree_to_column_map`. It can
    either transform data from a pandas DataFrame or use numeric values directly from
    the mapping to construct the nested dictionary structure.

    Examples
    --------
        >>> df = pd.DataFrame({
        ...     "a": [1, 2, 3],
        ...     "b": [4, 5, 6],
        ...     "c": [7, 8, 9],
        ... })
        >>> input_tree_to_column_map = {
        ...     "n1": {
        ...         "n2": "a",
        ...         "n3": "b",
        ...     },
        ...     "n4": pd.Series([1, 2, 3]),
        ...     "n5": 3,
        ... }
        >>> result = create_data_tree(
        ...     input_tree_to_column_map=input_tree_to_column_map,
        ...     df=df,
        ... )
        >>> result
        {
            "n1": {
                "n2": pd.Series([1, 2, 3]),
                "n3": pd.Series([4, 5, 6]),
            },
            "n4": pd.Series([1, 2, 3]),
            "n5": pd.Series([3, 3, 3]),
        }

    Args:
        input_tree_to_column_map:
            A nested dictionary that defines the structure of the output tree. Keys are
            strings that define the nested structure. Values can be:

            - Strings that reference column names in the DataFrame (when df is provided)
            - pandas Series objects
            - Numeric or boolean values (which will be used directly if df is None, or
              broadcast to match the DataFrame length if df is provided)
        df:
            Optional. The pandas DataFrame containing the source data.

    Returns:
        A nested dictionary structure containing the data organized according to the
        mapping definition.
    """
    _fail_if_mapper_has_incorrect_format(input_tree_to_column_map)
    qualified_input_tree_to_column_map = dt.flatten_to_qual_names(
        input_tree_to_column_map
    )
    name_to_input_series = {}

    if df:
        _fail_if_df_has_bool_or_numeric_column_names(df)

    df_columns = set(df.columns) if df else set()

    for qualified_input_name, input_value in qualified_input_tree_to_column_map.items():
        if input_value in df_columns:
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
    input_tree_to_column_map: NestedInputToSeriesNameDict,
) -> None:
    """Fail if the input tree to column name mapping has an incorrect format."""
    if not isinstance(input_tree_to_column_map, dict):
        msg = "The input tree to column mapping must be a dictionary."
        raise TypeError(msg)

    inputs = optree.tree_flatten(input_tree_to_column_map)[0]

    # Check that all Series have the same length
    series_inputs = [x for x in inputs if isinstance(x, pd.Series)]
    if series_inputs:
        expected_length = len(series_inputs[0])
        mismatched_lengths = [
            len(series) for series in series_inputs if len(series) != expected_length
        ]
        if mismatched_lengths:
            raise ValueError(
                "All provided pd.Series must have the same length. "
                f"Found series with lengths {mismatched_lengths} "
                f"but expected length {expected_length}."
            )


def _fail_if_df_has_bool_or_numeric_column_names(df: pd.DataFrame) -> None:
    """Fail if the DataFrame has bool or numeric column names."""
    common_msg = "The DataFrame must not have bool or numeric column names."
    bool_column_names = [col for col in df.columns if isinstance(col, bool)]
    numeric_column_names = [
        col
        for col in df.columns
        if isinstance(col, (int, float)) or (isinstance(col, str) and col.isnumeric())
    ]

    if bool_column_names or numeric_column_names:
        msg = f"""
        {common_msg}
        Boolean column names: {bool_column_names}.
        Numeric column names: {numeric_column_names}.
        """
        raise ValueError(msg)
