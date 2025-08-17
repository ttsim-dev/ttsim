from __future__ import annotations

import numpy
import pandas as pd
import pytest

from ttsim.interface_dag_elements.results import _restore_original_row_order


@pytest.mark.parametrize(
    (
        "df",
        "input_data__flat",
        "expected_row_order",
        "expected_index",
        "should_be_unchanged",
    ),
    [
        # No sort indices - should return DataFrame unchanged
        (
            pd.DataFrame(
                {"col1": [10, 20, 30], "col2": [100, 200, 300]},
                index=pd.Index([3, 1, 2], name="p_id"),
            ),
            {},  # No sort indices
            [10, 20, 30],  # Should remain unchanged
            [3, 1, 2],  # Original index preserved
            True,  # DataFrame should be identical
        ),
        # With sort indices - should restore original order
        (
            pd.DataFrame(
                {"col1": [10, 20, 30], "col2": [100, 200, 300]},
                index=pd.Index([1, 2, 3], name="p_id"),  # Sorted p_id values
            ),
            {
                ("__original_sort_indices__",): numpy.array([1, 2, 0])
            },  # Original positions: row 1→0, row 2→1, row 0→2
            [
                30,
                10,
                20,
            ],  # Should be reordered to: original row 0, original row 1, original row 2
            [3, 1, 2],  # Restored original p_id order (not reset index)
            False,  # DataFrame should be modified
        ),
        # More complex reordering
        (
            pd.DataFrame(
                {"col1": [100, 200, 300, 400], "col2": [10, 20, 30, 40]},
                index=pd.Index([1, 2, 3, 4], name="p_id"),  # Sorted p_id values
            ),
            {
                ("__original_sort_indices__",): numpy.array([3, 0, 2, 1])
            },  # Complex shuffle: original row 3→0, row 0→1, row 2→2, row 1→3
            [200, 400, 300, 100],  # Restored order: orig row 0, 1, 2, 3
            [2, 4, 3, 1],  # Restored original p_id order
            False,  # DataFrame should be modified
        ),
    ],
)
def test_restore_original_row_order(
    df, input_data__flat, expected_row_order, expected_index, should_be_unchanged
):
    """Test the _restore_original_row_order helper function."""
    original_df = df.copy()
    result_df = _restore_original_row_order(df, input_data__flat)

    # Check that the first column values are in the expected order
    assert result_df["col1"].tolist() == expected_row_order

    # Check that index name is preserved
    assert result_df.index.name == "p_id"

    # Check the index values
    assert result_df.index.tolist() == expected_index

    # If should be unchanged, verify it's identical to original
    if should_be_unchanged:
        pd.testing.assert_frame_equal(result_df, original_df)
