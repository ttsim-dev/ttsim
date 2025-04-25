import dags.tree as dt
import pandas as pd
import pytest

from ttsim.shared import create_data_tree_from_df


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
