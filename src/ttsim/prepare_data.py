import dags.tree as dt
import optree
import pandas as pd

from ttsim.typing import NestedDataDict, NestedInputToSeriesNameDict


def create_data_tree_from_df(
    input_tree_to_column_map: NestedInputToSeriesNameDict,
    df: pd.DataFrame,
) -> NestedDataDict:
    """Transform a pandas DataFrame to a nested dictionary expected by TTSIM.
    `
        Args
        ----
            input_tree_to_column_map:
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
            >>> input_tree_to_column_map = {
            ...     "n1": {
            ...         "n2": "a",
            ...         "n3": "b",
            ...     },
            ...     "n4": 3,
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
                "n4": pd.Series([3, 3, 3]),
            }


    """
    _fail_if_df_has_bool_or_numeric_column_names(df)
    _fail_if_mapper_has_incorrect_format(input_tree_to_column_map)

    qualified_input_tree_to_column_map = dt.flatten_to_qual_names(
        input_tree_to_column_map
    )

    name_to_input_series = {}
    for qualified_input_name, input_value in qualified_input_tree_to_column_map.items():
        if input_value in df.columns:
            name_to_input_series[qualified_input_name] = df[input_value]
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

    inputs = optree.tree_flatten(input_tree_to_column_map, none_is_leaf=True)[0]
    if not all(isinstance(x, str | int | bool) for x in inputs):
        found_types = {type(x) for x in inputs if not isinstance(x, str | int | bool)}
        msg = f"""Values of the input tree to column mapping must be strings, integers,
        or booleans.
        Found values of type {found_types}.
        """
        raise TypeError(msg)


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
