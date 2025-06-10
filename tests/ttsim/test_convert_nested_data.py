from __future__ import annotations

import dags.tree as dt
import numpy as np
import pandas as pd
import pytest

from ttsim import (
    main,
)
from ttsim.interface_dag_elements.data_converters import (
    dataframe_to_nested_data,
    nested_data_to_df_with_mapped_columns,
)
from ttsim.tt_dag_elements import (
    ScalarParam,
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
        mapper=inputs_tree_to_df_columns,
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
    (
        "environment",
        "targets__tree",
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
    targets__tree,
    expected_output,
):
    results__tree = main(
        inputs={
            "input_data__tree": minimal_data_tree,
            "policy_environment": environment,
            "targets__tree": targets__tree,
            "rounding": False,
        },
        targets=["results__tree"],
    )["results__tree"]
    result_df = nested_data_to_df_with_mapped_columns(
        nested_data_to_convert=results__tree,
        nested_outputs_df_column_names=targets__tree,
        data_with_p_id=minimal_data_tree,
    )
    pd.testing.assert_frame_equal(result_df, expected_output, check_like=True)
