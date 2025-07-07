from __future__ import annotations

import datetime

import numpy
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from ttsim import (
    Output,
    main,
)
from ttsim.interface_dag_elements.data_converters import (
    df_with_mapped_columns_to_flat_data,
    df_with_nested_columns_to_flat_data,
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
        "hh_id": numpy.array([1, 2, 3]),
        "p_id": numpy.array([1, 2, 3]),
    }


@pytest.mark.parametrize(
    (
        "inputs_tree_to_df_columns",
        "df",
        "expected",
    ),
    [
        (
            {
                "n1": {
                    "n2": "a",
                },
            },
            pd.DataFrame({"a": [1, 2, 3]}),
            {("n1", "n2"): pd.Series([1, 2, 3])},
        ),
        (
            {
                "n1": {
                    "n2": "a",
                },
                "n3": "b",
            },
            pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
            {("n1", "n2"): pd.Series([1, 2, 3]), ("n3",): pd.Series([4, 5, 6])},
        ),
        (
            {
                "n1": {
                    "n2": "a",
                },
                "n3": 3,
            },
            pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
            {("n1", "n2"): pd.Series([1, 2, 3]), ("n3",): pd.Series([3, 3, 3])},
        ),
    ],
)
def test_df_with_mapped_columns_to_flat_data(
    inputs_tree_to_df_columns,
    df,
    expected,
):
    result = df_with_mapped_columns_to_flat_data(
        mapper=inputs_tree_to_df_columns,
        df=df,
        xnp=numpy,
    )

    assert set(result.keys()) == set(expected.keys())
    for key in result:
        pd.testing.assert_series_equal(
            pd.Series(result[key]),
            expected[key],
            check_names=False,
        )


def test_df_with_mapped_columns_to_flat_data_fails_if_mapper_value_not_in_df(xnp):
    with pytest.raises(ValueError, match="Value of mapper path"):
        df_with_mapped_columns_to_flat_data(
            mapper={
                "n1": "a",
                "n2": "b",
            },
            df=pd.DataFrame({"a": [1, 2, 3]}),
            xnp=xnp,
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
                {"res1": numpy.array([1, 1, 1]), "res2": numpy.array([1, 1, 1])},
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
                {"res1": numpy.array([1, 1, 1])},
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
                {"res1": numpy.array([1, 1, 1])},
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
                {"res1": numpy.array([1, 1, 1]), "res2": numpy.array([1, 1, 1])},
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
                {"res1": numpy.array([1, 1, 1])},
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
                {"res1": numpy.array([1, 1, 1]), "res2": numpy.array([1, 1, 1])},
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
    backend,
):
    results__tree = main(
        input_data={"tree": minimal_data_tree},
        policy_environment=environment,
        date=datetime.date(2024, 1, 1),
        targets={"tree": targets__tree},
        rounding=False,
        backend=backend,
        output=Output.name("results__tree"),
    )
    result_df = nested_data_to_df_with_mapped_columns(
        nested_data_to_convert=results__tree,
        nested_outputs_df_column_names=targets__tree,
        data_with_p_id=minimal_data_tree,
    )
    pd.testing.assert_frame_equal(
        result_df,
        expected_output,
        check_like=True,
        check_dtype=False,
        check_index_type=False,
    )


@pytest.mark.parametrize(
    (
        "df",
        "expected",
    ),
    [
        (
            pd.DataFrame({("a", "b"): [1, 2, 3], ("c",): [4, 5, 6]}),
            {("a", "b"): [1, 2, 3], ("c",): [4, 5, 6]},
        ),
        (
            pd.DataFrame({("a", "b"): [1, 2, 3], ("b",): [4, 5, 6]}),
            {("a", "b"): [1, 2, 3], ("b",): [4, 5, 6]},
        ),
    ],
)
def test_df_with_nested_columns_to_flat_data(df, expected):
    result = df_with_nested_columns_to_flat_data(
        df=df,
        xnp=numpy,
    )

    assert set(result.keys()) == set(expected.keys())
    for key in result:
        assert_array_equal(result[key], expected[key])
