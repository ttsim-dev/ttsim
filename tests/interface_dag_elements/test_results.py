from __future__ import annotations

import numpy
import pytest

from ttsim.interface_dag_elements.results import tree


@pytest.mark.parametrize(
    (
        "raw_results_columns",
        "raw_results_params",
        "raw_results_from_input_data",
        "input_data__sort_indices",
        "expected_result",
    ),
    [
        (
            # Raw results columns with sorted data
            {"result_col": numpy.array([200, 300, 100])},
            # Raw results params (empty)
            {},
            # Raw results from input data (empty)
            {},
            # Sort indices
            numpy.array([1, 2, 0]),
            # Expected result with restored row order
            {"result_col": numpy.array([100, 200, 300])},
        ),
        (
            # Nested raw results
            {"level1__result_col": numpy.array([200, 300, 100])},
            {"level1__scalar_value": 42},
            {},
            numpy.array([1, 2, 0]),
            {
                "level1": {
                    "result_col": numpy.array([100, 200, 300]),
                    "scalar_value": 42,
                }
            },
        ),
        (
            # Mixed data types - param values and row data
            {},
            {
                "param_value": numpy.array(
                    [1, 2]
                ),  # Length != num_rows, should not be reordered
            },
            {
                "row_data": numpy.array(
                    [200, 300, 100]
                ),  # Length == num_rows, should be reordered
            },
            numpy.array([1, 2, 0]),
            {
                "param_value": numpy.array([1, 2]),  # Unchanged
                "row_data": numpy.array([100, 200, 300]),  # Reordered to original
            },
        ),
    ],
)
def test_restore_original_row_order(
    raw_results_columns,
    raw_results_params,
    raw_results_from_input_data,
    input_data__sort_indices,
    expected_result,
):
    """Test that the tree function restores original row order correctly."""
    result = tree(
        raw_results_columns,
        raw_results_params,
        raw_results_from_input_data,
        input_data__sort_indices,
    )

    # Check the structure and values recursively
    def assert_nested_equal(actual, expected):
        if isinstance(expected, dict):
            assert isinstance(actual, dict)
            assert set(actual.keys()) == set(expected.keys())
            for key in expected:
                assert_nested_equal(actual[key], expected[key])
        elif isinstance(expected, numpy.ndarray):
            numpy.testing.assert_array_equal(actual, expected)
        else:
            assert actual == expected

    assert_nested_equal(result, expected_result)
