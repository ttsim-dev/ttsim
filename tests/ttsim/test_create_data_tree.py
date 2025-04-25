import pandas as pd
import pytest

from ttsim.create_data_tree_from_df import create_data_tree


@pytest.mark.parametrize(
    (
        "input_tree_to_column_map",
        "df",
        "expected_output",
    ),
    [
        (
            {
                "a": {
                    "b": "a",
                },
            },
            pd.DataFrame({"a": [1, 2, 3]}),
            {"a": {"b": pd.Series([1, 2, 3])}},
        ),
        (
            {
                "a": {
                    "b": "a",
                },
            },
            pd.DataFrame({"a": [1, 2, 3]}),
            {"a": {"b": pd.Series([1, 2, 3])}},
        ),
    ],
)
def test_create_data_tree(
    input_tree_to_column_map,
    df,
    expected_output,
):
    result = create_data_tree(
        input_tree_to_column_map=input_tree_to_column_map,
        df=df,
    )
    assert result == expected_output
