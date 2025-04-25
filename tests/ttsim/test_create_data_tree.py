import dags.tree as dt
import pandas as pd
import pytest

from ttsim import create_data_tree_from_df


@pytest.mark.parametrize(
    (
        "input_tree_to_column_map",
        "df",
        "expected_output",
    ),
    [
        (
            {
                "n1": {
                    "n2": "a",
                },
            },
            pd.DataFrame({"a": [1, 2, 3]}),
            {"n1": {"n2": pd.Series([1, 2, 3])}},
        ),
        (
            {
                "n1": {
                    "n2": "a",
                },
                "n3": "b",
            },
            pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
            {"n1": {"n2": pd.Series([1, 2, 3])}, "n3": pd.Series([4, 5, 6])},
        ),
        (
            {
                "n1": {
                    "n2": "a",
                },
                "n3": 3,
            },
            pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
            {"n1": {"n2": pd.Series([1, 2, 3])}, "n3": pd.Series([3, 3, 3])},
        ),
    ],
)
def test_create_data_tree_from_df(
    input_tree_to_column_map,
    df,
    expected_output,
):
    result = create_data_tree_from_df(
        input_tree_to_column_map=input_tree_to_column_map,
        df=df,
    )
    flat_result = dt.flatten_to_qual_names(result)
    flat_expected_output = dt.flatten_to_qual_names(expected_output)

    assert set(flat_result.keys()) == set(flat_expected_output.keys())
    for key in flat_result:
        pd.testing.assert_series_equal(
            flat_result[key], flat_expected_output[key], check_names=False
        )


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame({True: [1, 2]}),
        pd.DataFrame({1: [1, 2]}),
    ],
)
def test_create_data_tree_fails_if_df_has_bool_or_numeric_column_names(df):
    with pytest.raises(
        ValueError, match="The DataFrame must not have bool or numeric column names."
    ):
        create_data_tree_from_df(input_tree_to_column_map={}, df=df)


@pytest.mark.parametrize(
    (
        "input_tree_to_column_map",
        "expected_error_message",
    ),
    [
        (
            [],
            "The input tree to column mapping must be a dictionary.",
        ),
        (
            {
                "n1": {
                    "n2": pd.Series([1, 2, 3]),
                },
            },
            "Found values of type {<class 'pandas.core.series.Series'>}.",
        ),
        (
            {
                "n1": {
                    "n2": None,
                },
            },
            "Found values of type {<class 'NoneType'>}.",
        ),
    ],
)
def test_create_data_tree_fails_if_mapper_has_incorrect_format(
    input_tree_to_column_map, expected_error_message
):
    with pytest.raises(TypeError, match=expected_error_message):
        create_data_tree_from_df(
            input_tree_to_column_map=input_tree_to_column_map, df=pd.DataFrame()
        )
