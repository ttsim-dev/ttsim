from __future__ import annotations

import dags.tree as dt
import numpy as np
import pandas as pd
import pytest

from ttsim import (
    ConsecutiveInt1dLookupTableParam,
    ConsecutiveInt1dLookupTableParamValue,
    DictParam,
    PiecewisePolynomialParam,
    PiecewisePolynomialParamValue,
    ScalarParam,
    dataframe_to_nested_data,
    main,
    nested_data_to_dataframe,
    param_function,
    policy_function,
)

_GENERIC_PARAM_SPEC = {
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "unit": None,
    "reference_period": None,
    "name": {
        "de": "generic_param",
        "en": "generic_param",
    },
    "description": {
        "de": "generic_param",
        "en": "generic_param",
    },
}


@policy_function()
def int_policy_function() -> int:
    return 1


@param_function()
def int_param_function() -> int:
    return 1


_SOME_SCALAR_PARAM = ScalarParam(
    leaf_name="some_scalar_param",
    value=1,
    **_GENERIC_PARAM_SPEC,
)


_SOME_PIECEWISE_POLYNOMIAL_PARAM = PiecewisePolynomialParam(
    leaf_name="some_piecewise_polynomial_param",
    value=PiecewisePolynomialParamValue(
        thresholds=np.array([1, 2, 3]),
        intercepts=np.array([1, 2, 3]),
        rates=np.array([1, 2, 3]),
    ),
    **_GENERIC_PARAM_SPEC,
)


_SOME_CONSECUTIVE_INT_1D_LOOKUP_TABLE_PARAM = ConsecutiveInt1dLookupTableParam(
    leaf_name="some_consecutive_int_1d_lookup_table_param",
    value=ConsecutiveInt1dLookupTableParamValue(
        base_to_subtract=1,
        values_to_look_up=np.array([1, 2, 3]),
    ),
    **_GENERIC_PARAM_SPEC,
)


_SOME_DICT_PARAM = DictParam(
    leaf_name="some_dict_param",
    value={"a": 1, "b": 2},
    **_GENERIC_PARAM_SPEC,
)


@pytest.fixture
def minimal_data_tree():
    return {
        "hh_id": np.array([1, 2, 3]),
        "p_id": np.array([1, 2, 3]),
    }


@pytest.mark.parametrize(
    (
        "inputs_tree_to_df_columns",
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
def test_dataframe_to_nested_data(
    inputs_tree_to_df_columns,
    df,
    expected_output,
):
    result = dataframe_to_nested_data(
        inputs_tree_to_df_columns=inputs_tree_to_df_columns,
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
        ValueError, match="DataFrame column names cannot be booleans or numbers."
    ):
        dataframe_to_nested_data(inputs_tree_to_df_columns={}, df=df)


@pytest.mark.parametrize(
    (
        "inputs_tree_to_df_columns",
        "expected_error_message",
    ),
    [
        (
            [],
            "The inputs tree to column mapping must be a \\(nested\\) dictionary.",
        ),
        (
            {
                "n1": {
                    "n2": pd.Series([1, 2, 3]),
                },
            },
            "n1__n2: Series",
        ),
        (
            {
                "n1": {
                    "n2": None,
                },
            },
            "n1__n2: NoneType",
        ),
        (
            {
                "n1": {
                    True: 2,
                },
            },
            "All path elements of `inputs_tree_to_df_columns` must be strings.",
        ),
    ],
)
def test_create_data_tree_fails_if_mapper_has_incorrect_format(
    inputs_tree_to_df_columns, expected_error_message
):
    with pytest.raises(TypeError, match=expected_error_message):
        dataframe_to_nested_data(
            inputs_tree_to_df_columns=inputs_tree_to_df_columns, df=pd.DataFrame()
        )


@pytest.mark.parametrize(
    (
        "environment",
        "targets_tree_to_outputs_df_columns",
        "expected_output",
    ),
    [
        # Two policy functions
        (
            {
                "some_policy_function": int_policy_function,
                "another_policy_function": int_policy_function,
            },
            {
                "some_policy_function": "res1",
                "another_policy_function": "res2",
            },
            pd.DataFrame(
                {"res1": np.array([1, 1, 1]), "res2": np.array([1, 1, 1])},
                index=pd.Index([1, 2, 3], name="p_id"),
            ),
        ),
        # One policy function
        (
            {
                "some_policy_function": int_policy_function,
            },
            {
                "some_policy_function": "res1",
            },
            pd.DataFrame(
                {"res1": np.array([1, 1, 1])},
                index=pd.Index([1, 2, 3], name="p_id"),
            ),
        ),
        # One param function
        (
            {
                "some_param_function": int_param_function,
            },
            {
                "some_param_function": "res1",
            },
            pd.DataFrame(
                {"res1": np.array([1, 1, 1])},
                index=pd.Index([1, 2, 3], name="p_id"),
            ),
        ),
        # One param function and one policy function
        (
            {
                "some_param_function": int_param_function,
                "some_policy_function": int_policy_function,
            },
            {
                "some_param_function": "res1",
                "some_policy_function": "res2",
            },
            pd.DataFrame(
                {"res1": np.array([1, 1, 1]), "res2": np.array([1, 1, 1])},
                index=pd.Index([1, 2, 3], name="p_id"),
            ),
        ),
        # One scalar param
        (
            {
                "some_scalar_param": _SOME_SCALAR_PARAM,
            },
            {"some_scalar_param": "res1"},
            pd.DataFrame(
                {"res1": np.array([1, 1, 1])},
                index=pd.Index([1, 2, 3], name="p_id"),
            ),
        ),
        # One scalar param and one policy function
        (
            {
                "some_scalar_param": _SOME_SCALAR_PARAM,
                "some_policy_function": int_policy_function,
            },
            {
                "some_scalar_param": "res1",
                "some_policy_function": "res2",
            },
            pd.DataFrame(
                {"res1": np.array([1, 1, 1]), "res2": np.array([1, 1, 1])},
                index=pd.Index([1, 2, 3], name="p_id"),
            ),
        ),
    ],
)
def test_nested_data_to_dataframe(
    environment,
    minimal_data_tree,
    targets_tree_to_outputs_df_columns,
    expected_output,
):
    nested_results = main(
        inputs={
            "data_tree": minimal_data_tree,
            "policy_environment": environment,
            "targets_tree": targets_tree_to_outputs_df_columns,
            "rounding": False,
        },
        targets=["nested_results"],
    )["nested_results"]
    result_df = nested_data_to_dataframe(
        nested_data_to_convert=nested_results,
        nested_outputs_df_column_names=targets_tree_to_outputs_df_columns,
        data_with_p_id=minimal_data_tree,
    )
    pd.testing.assert_frame_equal(result_df, expected_output, check_like=True)


@pytest.mark.parametrize(
    (
        "environment",
        "targets_tree_to_outputs_df_columns",
    ),
    [
        (
            {
                "some_piecewise_polynomial_param": _SOME_PIECEWISE_POLYNOMIAL_PARAM,
            },
            {"some_piecewise_polynomial_param": "res1"},
        ),
        (
            {
                "some_consecutive_int_1d_lookup_table_param": (
                    _SOME_CONSECUTIVE_INT_1D_LOOKUP_TABLE_PARAM
                ),
            },
            {"some_consecutive_int_1d_lookup_table_param": "res1"},
        ),
    ],
)
def test_nested_data_to_dataframe_fails_if_noncompatible_objects_are_returned(
    environment,
    targets_tree_to_outputs_df_columns,
    minimal_data_tree,
):
    nested_results = main(
        inputs={
            "data_tree": minimal_data_tree,
            "policy_environment": environment,
            "targets_tree": targets_tree_to_outputs_df_columns,
            "rounding": False,
        },
        targets=["nested_results"],
    )["nested_results"]
    with pytest.raises(
        TypeError, match=r"The following paths contain non-scalar\nobjects"
    ):
        nested_data_to_dataframe(
            nested_data_to_convert=nested_results,
            nested_outputs_df_column_names=targets_tree_to_outputs_df_columns,
            data_with_p_id=minimal_data_tree,
        )


@pytest.mark.parametrize(
    (
        "environment",
        "targets_tree_to_outputs_df_columns",
    ),
    [
        (
            {
                "some_dict_param": _SOME_DICT_PARAM,
            },
            {"some_dict_param": "res1"},
        ),
    ],
)
def test_nested_data_to_dataframe_fails_because_raw_param_dict_is_returned(
    environment,
    targets_tree_to_outputs_df_columns,
    minimal_data_tree,
):
    nested_results = main(
        inputs={
            "data_tree": minimal_data_tree,
            "policy_environment": environment,
            "targets_tree": targets_tree_to_outputs_df_columns,
            "rounding": False,
        },
        targets=["nested_results"],
    )["nested_results"]
    with pytest.raises(
        ValueError,
        match="failed because the following paths\nare not mapped to a column name",
    ):
        nested_data_to_dataframe(
            nested_data_to_convert=nested_results,
            nested_outputs_df_column_names=targets_tree_to_outputs_df_columns,
            data_with_p_id=minimal_data_tree,
        )
