from __future__ import annotations

import datetime

import numpy
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from ttsim import (
    InputData,
    TTTargets,
    main,
)
from ttsim.interface_dag_elements.data_converters import (
    df_with_mapped_columns_to_flat_data,
    df_with_nested_columns_to_flat_data,
    nested_data_to_df_with_mapped_columns,
    nested_data_to_df_with_qname_columns,
)
from ttsim.tt import (
    ScalarParam,
    Unit,
    param_function,
    policy_function,
)

_GENERIC_PARAM_SPEC = {
    "start_date": datetime.date(2024, 1, 1),
    "end_date": datetime.date(2024, 12, 31),
    "unit": "DIMENSIONLESS",
    "name": {
        "de": "generic_param",
        "en": "generic_param",
    },
    "description": {
        "de": "generic_param",
        "en": "generic_param",
    },
}


@policy_function(unit=Unit.DIMENSIONLESS)
def int_policy_function() -> int:
    return 1


@policy_function(unit=Unit.DIMENSIONLESS)
def another_int_policy_function() -> int:
    return 1


@param_function(unit=Unit.DIMENSIONLESS)
def int_param_function() -> int:
    return 1


_SOME_SCALAR_PARAM = ScalarParam(value=1, **_GENERIC_PARAM_SPEC)  # ty: ignore[invalid-argument-type]


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
        backend="numpy",
        xnp=numpy,
    )

    assert set(result.keys()) == set(expected.keys())
    for key in result:
        pd.testing.assert_series_equal(
            pd.Series(result[key]),
            expected[key],
            check_names=False,
        )


@pytest.mark.parametrize(
    (
        "environment",
        "tt_targets__tree",
        "expected_output",
    ),
    [
        # Two policy functions
        (
            {
                "int_policy_function": int_policy_function,
                "another_int_policy_function": another_int_policy_function,
            },
            {
                "int_policy_function": "res1",
                "another_int_policy_function": "res2",
            },
            pd.DataFrame(
                {"res1": numpy.array([1, 1, 1]), "res2": numpy.array([1, 1, 1])},
                index=pd.Index([1, 2, 3], name="p_id"),
            ),
        ),
        # One policy function
        (
            {
                "int_policy_function": int_policy_function,
            },
            {
                "int_policy_function": "res1",
            },
            pd.DataFrame(
                {"res1": numpy.array([1, 1, 1])},
                index=pd.Index([1, 2, 3], name="p_id"),
            ),
        ),
        # One param function
        (
            {
                "int_param_function": int_param_function,
            },
            {
                "int_param_function": "res1",
            },
            pd.DataFrame(
                {"res1": numpy.array([1, 1, 1])},
                index=pd.Index([1, 2, 3], name="p_id"),
            ),
        ),
        # One param function and one policy function
        (
            {
                "int_param_function": int_param_function,
                "int_policy_function": int_policy_function,
            },
            {
                "int_param_function": "res1",
                "int_policy_function": "res2",
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
                "int_policy_function": int_policy_function,
            },
            {
                "some_scalar_param": "res1",
                "int_policy_function": "res2",
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
    tt_targets__tree,
    expected_output,
    backend,
):
    results__tree = main(
        main_target="results__tree",
        input_data=InputData.tree(tree=minimal_data_tree),
        policy_environment=environment,
        policy_date=datetime.date(2025, 1, 1),
        evaluation_date=datetime.date(2024, 1, 1),
        tt_targets=TTTargets.tree(tt_targets__tree),
        rounding=False,
        backend=backend,
    )
    result_df = nested_data_to_df_with_mapped_columns(
        nested_data_to_convert=results__tree,
        nested_outputs_df_column_names=tt_targets__tree,
        data_with_p_id=minimal_data_tree,
    )
    pd.testing.assert_frame_equal(
        result_df,
        expected_output,
        check_like=True,
        check_dtype=False,
        check_index_type=False,
    )


def test_nested_data_to_df_with_qname_columns_flattens_paths_with_double_underscore():
    nested = {
        "a": numpy.array([10, 20, 30]),
        "b": {
            "c": numpy.array([1, 2, 3]),
            "d": {"e": numpy.array([0.1, 0.2, 0.3])},
        },
    }
    index = pd.Index([100, 200, 300], name="p_id")
    result = nested_data_to_df_with_qname_columns(nested, index=index)
    assert list(result.columns) == ["a", "b__c", "b__d__e"]
    assert_array_equal(result["a"].to_numpy(), [10, 20, 30])
    assert_array_equal(result["b__c"].to_numpy(), [1, 2, 3])
    assert_array_equal(result["b__d__e"].to_numpy(), [0.1, 0.2, 0.3])
    pd.testing.assert_index_equal(result.index, index)


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
        backend="numpy",
        xnp=numpy,
    )

    assert set(result.keys()) == set(expected.keys())
    for key in result:
        assert_array_equal(result[key], expected[key])
