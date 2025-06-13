from __future__ import annotations

import copy

import numpy
import pytest

try:
    import jax_datetime

    my_datetime = jax_datetime.to_datetime
except ImportError:
    my_datetime = lambda x: x


from ttsim.tt_dag_elements.aggregation import (
    grouped_all,
    grouped_any,
    grouped_count,
    grouped_max,
    grouped_mean,
    grouped_min,
    grouped_sum,
    sum_by_p_id,
)


def parameterize_based_on_dict(test_cases, keys_of_test_cases=None):
    """Apply pytest.mark.parametrize based on a dictionary."""
    test_cases = copy.copy(test_cases)
    if keys_of_test_cases:
        # Only use requested keys
        test_cases = {
            k: {
                k_inner: v_inner
                for k_inner, v_inner in v.items()
                if k_inner in keys_of_test_cases
            }
            for k, v in test_cases.items()
            if all(e in v for e in keys_of_test_cases)
        }

    # Return parametrization
    return pytest.mark.parametrize(
        argnames=(argnames := sorted({k for v in test_cases.values() for k in v})),
        argvalues=[[v.get(k) for k in argnames] for v in test_cases.values()],
        ids=test_cases.keys(),
    )


test_grouped_specs = {
    "constant_column": {
        "column_to_aggregate": numpy.array([1, 1, 1, 1, 1]),
        "group_id": numpy.array([0, 0, 1, 1, 1]),
        "expected_res_count": numpy.array([2, 2, 3, 3, 3]),
        "expected_res_sum": numpy.array([2, 2, 3, 3, 3]),
        "expected_res_max": numpy.array([1, 1, 1, 1, 1]),
        "expected_res_min": numpy.array([1, 1, 1, 1, 1]),
        "expected_res_any": numpy.array([True, True, True, True, True]),
        "expected_res_all": numpy.array([True, True, True, True, True]),
    },
    "constant_column_group_id_unsorted": {
        "column_to_aggregate": numpy.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        "group_id": numpy.array([0, 1, 0, 1, 0]),
        "expected_res_count": numpy.array([3, 2, 3, 2, 3]),
        "expected_res_sum": numpy.array([3.0, 2.0, 3.0, 2.0, 3.0]),
        "expected_res_mean": numpy.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        "expected_res_max": numpy.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        "expected_res_min": numpy.array([1.0, 1.0, 1.0, 1.0, 1.0]),
    },
    "int_column": {
        "column_to_aggregate": numpy.array([0, 1, 2, 3, 4]),
        "group_id": numpy.array([0, 0, 1, 1, 1]),
        "expected_res_sum": numpy.array([1, 1, 9, 9, 9]),
        "expected_res_mean": numpy.array([0.5, 0.5, 3, 3, 3]),
        "expected_res_max": numpy.array([1, 1, 4, 4, 4]),
        "expected_res_min": numpy.array([0, 0, 2, 2, 2]),
        "expected_res_any": numpy.array([True, True, True, True, True]),
        "expected_res_all": numpy.array([False, False, True, True, True]),
    },
    "unique_group_ids_with_gaps": {
        "column_to_aggregate": numpy.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "group_id": numpy.array([0, 0, 3, 3, 3]),
        "expected_res_count": numpy.array([2, 2, 3, 3, 3]),
        "expected_res_sum": numpy.array([1.0, 1.0, 9.0, 9.0, 9.0]),
        "expected_res_mean": numpy.array([0.5, 0.5, 3.0, 3.0, 3.0]),
        "expected_res_max": numpy.array([1.0, 1.0, 4.0, 4.0, 4.0]),
        "expected_res_min": numpy.array([0.0, 0.0, 2.0, 2.0, 2.0]),
    },
    "float_column": {
        "column_to_aggregate": numpy.array([0.0, 1.5, 2.0, 3.0, 4.0]),
        "group_id": numpy.array([0, 0, 3, 3, 3]),
        "expected_res_sum": numpy.array([1.5, 1.5, 9.0, 9.0, 9.0]),
        "expected_res_mean": numpy.array([0.75, 0.75, 3.0, 3.0, 3.0]),
        "expected_res_max": numpy.array([1.5, 1.5, 4.0, 4.0, 4.0]),
        "expected_res_min": numpy.array([0.0, 0.0, 2.0, 2.0, 2.0]),
    },
    "more_than_two_groups": {
        "column_to_aggregate": numpy.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "group_id": numpy.array([1, 0, 1, 1, 3]),
        "expected_res_count": numpy.array([3, 1, 3, 3, 1]),
        "expected_res_sum": numpy.array([5.0, 1.0, 5.0, 5.0, 4.0]),
        "expected_res_mean": numpy.array([5.0 / 3.0, 1.0, 5.0 / 3.0, 5.0 / 3.0, 4.0]),
        "expected_res_max": numpy.array([3.0, 1.0, 3.0, 3.0, 4.0]),
        "expected_res_min": numpy.array([0.0, 1.0, 0.0, 0.0, 4.0]),
    },
    "bool_column": {
        "column_to_aggregate": numpy.array([True, False, True, False, False]),
        "group_id": numpy.array([0, 0, 1, 1, 1]),
        "expected_res_any": numpy.array([True, True, True, True, True]),
        "expected_res_all": numpy.array([False, False, False, False, False]),
        "expected_res_sum": numpy.array([1, 1, 1, 1, 1]),
        "expected_res_mean": numpy.array([0.5, 0.5, 1 / 3, 1 / 3, 1 / 3]),
    },
    "group_id_unsorted_bool": {
        "column_to_aggregate": numpy.array([True, False, True, True, True]),
        "group_id": numpy.array([0, 1, 0, 1, 0]),
        "expected_res_any": numpy.array([True, True, True, True, True]),
        "expected_res_all": numpy.array([True, False, True, False, True]),
        "expected_res_sum": numpy.array([3, 1, 3, 1, 3]),
    },
    "unique_group_ids_with_gaps_bool": {
        "column_to_aggregate": numpy.array([True, False, False, False, False]),
        "group_id": numpy.array([0, 0, 3, 3, 3]),
        "expected_res_any": numpy.array([True, True, False, False, False]),
        "expected_res_all": numpy.array([False, False, False, False, False]),
        "expected_res_sum": numpy.array([1, 1, 0, 0, 0]),
    },
    "sum_by_p_id_float": {
        "column_to_aggregate": numpy.array([10.0, 20.0, 30.0, 40.0, 50.0]),
        "p_id_to_aggregate_by": numpy.array([-1, -1, 8, 8, 10]),
        "p_id_to_store_by": numpy.array([7, 8, 9, 10, 11]),
        "expected_res": numpy.array([0.0, 70.0, 0.0, 50.0, 0.0]),
        "expected_type": numpy.floating,
    },
    "sum_by_p_id_int": {
        "column_to_aggregate": numpy.array([10, 20, 30, 40, 50]),
        "p_id_to_aggregate_by": numpy.array([-1, -1, 8, 8, 10]),
        "p_id_to_store_by": numpy.array([7, 8, 9, 10, 11]),
        "expected_res": numpy.array([0, 70, 0, 50, 0]),
        "expected_type": numpy.integer,
    },
}

test_grouped_raises_specs = {
    "dtype_boolean": {
        "column_to_aggregate": numpy.array([True, True, True, False, False]),
        "group_id": numpy.array([0, 0, 1, 1, 1]),
        "error_max": TypeError,
        "error_min": TypeError,
        "exception_match": "grouped_",
    },
    "float_group_id": {
        "column_to_aggregate": numpy.array([0, 1, 2, 3, 4]),
        "group_id": numpy.array([0, 0, 3.5, 3.5, 3.5]),
        "p_id_to_store_by": numpy.array([0, 1, 2, 3, 4]),
        "error_sum": TypeError,
        "error_mean": TypeError,
        "error_max": TypeError,
        "error_min": TypeError,
        "error_sum_by_p_id": TypeError,
        "exception_match": "The dtype of id columns must be integer.",
    },
    "dtype_float": {
        "column_to_aggregate": numpy.array([1.5, 2, 3.5, 4, 5]),
        "group_id": numpy.array([0, 0, 1, 1, 1]),
        "error_any": TypeError,
        "error_all": TypeError,
        "exception_match": "grouped_",
    },
    "float_group_id_bool": {
        "column_to_aggregate": numpy.array([True, True, True, False, False]),
        "group_id": numpy.array([0, 0, 3.5, 3.5, 3.5]),
        "error_any": TypeError,
        "error_all": TypeError,
        "exception_match": "The dtype of id columns must be integer.",
    },
}
test_grouped_specs["datetime"] = {
    "column_to_aggregate": numpy.array(
        [
            numpy.datetime64("2000"),
            numpy.datetime64("2001"),
            numpy.datetime64("2002"),
            numpy.datetime64("2003"),
            numpy.datetime64("2004"),
        ]
    ),
    "group_id": numpy.array([1, 0, 1, 1, 1]),
    "expected_res_max": numpy.array(
        [
            numpy.datetime64("2004"),
            numpy.datetime64("2001"),
            numpy.datetime64("2004"),
            numpy.datetime64("2004"),
            numpy.datetime64("2004"),
        ]
    ),
    "expected_res_min": numpy.array(
        [
            numpy.datetime64("2000"),
            numpy.datetime64("2001"),
            numpy.datetime64("2000"),
            numpy.datetime64("2000"),
            numpy.datetime64("2000"),
        ]
    ),
}

test_grouped_raises_specs["dtype_string"] = {
    "column_to_aggregate": numpy.array(["0", "1", "2", "3", "4"]),
    "group_id": numpy.array([0, 0, 1, 1, 1]),
    "error_sum": TypeError,
    "error_mean": TypeError,
    "error_max": TypeError,
    "error_min": TypeError,
    "error_any": TypeError,
    "error_all": TypeError,
    "exception_match": "grouped_",
}
test_grouped_raises_specs["datetime"] = {
    "column_to_aggregate": numpy.array(
        [
            numpy.datetime64("2000"),
            numpy.datetime64("2001"),
            numpy.datetime64("2002"),
            numpy.datetime64("2003"),
            numpy.datetime64("2004"),
        ]
    ),
    "group_id": numpy.array([0, 0, 1, 1, 1]),
    "error_sum": TypeError,
    "error_mean": TypeError,
    "error_any": TypeError,
    "error_all": TypeError,
    "exception_match": "grouped_",
}


@parameterize_based_on_dict(
    test_grouped_specs,
    keys_of_test_cases=["group_id", "expected_res_count"],
)
def test_grouped_count(group_id, expected_res_count, backend):
    result = grouped_count(
        group_id=group_id,
        num_segments=len(group_id),
        backend=backend,
    )
    numpy.testing.assert_array_almost_equal(result, expected_res_count)


def _run_agg_by_group(agg_func, column_to_aggregate, group_id, backend):
    return agg_func(
        column=column_to_aggregate,
        group_id=group_id,
        num_segments=len(group_id),
        backend=backend,
    )


@parameterize_based_on_dict(
    test_grouped_specs,
    keys_of_test_cases=[
        "column_to_aggregate",
        "group_id",
        "expected_res_sum",
    ],
)
def test_grouped_sum(column_to_aggregate, group_id, expected_res_sum, backend):
    result = _run_agg_by_group(
        agg_func=grouped_sum,
        column_to_aggregate=column_to_aggregate,
        group_id=group_id,
        backend=backend,
    )
    numpy.testing.assert_array_almost_equal(result, expected_res_sum)


@parameterize_based_on_dict(
    test_grouped_specs,
    keys_of_test_cases=[
        "column_to_aggregate",
        "group_id",
        "expected_res_mean",
    ],
)
def test_grouped_mean(column_to_aggregate, group_id, expected_res_mean, backend):
    result = _run_agg_by_group(
        agg_func=grouped_mean,
        column_to_aggregate=column_to_aggregate,
        group_id=group_id,
        backend=backend,
    )
    numpy.testing.assert_array_almost_equal(result, expected_res_mean)


@parameterize_based_on_dict(
    {k: v for k, v in test_grouped_specs.items() if "datetime" not in k},
    keys_of_test_cases=[
        "column_to_aggregate",
        "group_id",
        "expected_res_max",
    ],
)
def test_grouped_max(column_to_aggregate, group_id, expected_res_max, backend):
    result = _run_agg_by_group(
        agg_func=grouped_max,
        column_to_aggregate=column_to_aggregate,
        group_id=group_id,
        backend=backend,
    )
    numpy.testing.assert_array_equal(result, expected_res_max)


@pytest.mark.skipif_jax
@parameterize_based_on_dict(
    {k: v for k, v in test_grouped_specs.items() if "datetime" in k},
    keys_of_test_cases=[
        "column_to_aggregate",
        "group_id",
        "expected_res_max",
    ],
)
def test_grouped_max_datetime(column_to_aggregate, group_id, expected_res_max, backend):
    result = _run_agg_by_group(
        agg_func=grouped_max,
        column_to_aggregate=my_datetime(column_to_aggregate),
        group_id=group_id,
        backend=backend,
    )
    numpy.testing.assert_array_equal(result, expected_res_max)


@parameterize_based_on_dict(
    {k: v for k, v in test_grouped_specs.items() if "datetime" not in k},
    keys_of_test_cases=[
        "column_to_aggregate",
        "group_id",
        "expected_res_min",
    ],
)
def test_grouped_min(column_to_aggregate, group_id, expected_res_min, backend):
    result = _run_agg_by_group(
        agg_func=grouped_min,
        column_to_aggregate=column_to_aggregate,
        group_id=group_id,
        backend=backend,
    )
    numpy.testing.assert_array_equal(result, expected_res_min)


@pytest.mark.skipif_jax
@parameterize_based_on_dict(
    {k: v for k, v in test_grouped_specs.items() if "datetime" in k},
    keys_of_test_cases=[
        "column_to_aggregate",
        "group_id",
        "expected_res_min",
    ],
)
def test_grouped_min_datetime(column_to_aggregate, group_id, expected_res_min, backend):
    result = _run_agg_by_group(
        agg_func=grouped_min,
        column_to_aggregate=my_datetime(column_to_aggregate),
        group_id=group_id,
        backend=backend,
    )
    numpy.testing.assert_array_equal(result, expected_res_min)


@parameterize_based_on_dict(
    test_grouped_specs,
    keys_of_test_cases=[
        "column_to_aggregate",
        "group_id",
        "expected_res_any",
    ],
)
def test_grouped_any(column_to_aggregate, group_id, expected_res_any, backend):
    result = _run_agg_by_group(
        agg_func=grouped_any,
        column_to_aggregate=column_to_aggregate,
        group_id=group_id,
        backend=backend,
    )
    numpy.testing.assert_array_almost_equal(result, expected_res_any)


@parameterize_based_on_dict(
    test_grouped_specs,
    keys_of_test_cases=[
        "column_to_aggregate",
        "group_id",
        "expected_res_all",
    ],
)
def test_grouped_all(column_to_aggregate, group_id, expected_res_all, backend):
    result = _run_agg_by_group(
        agg_func=grouped_all,
        column_to_aggregate=column_to_aggregate,
        group_id=group_id,
        backend=backend,
    )
    numpy.testing.assert_array_almost_equal(result, expected_res_all)


@parameterize_based_on_dict(
    test_grouped_raises_specs,
    keys_of_test_cases=[
        "column_to_aggregate",
        "group_id",
        "error_sum",
        "exception_match",
    ],
)
@pytest.mark.skipif_jax
def test_grouped_sum_raises(
    column_to_aggregate, group_id, error_sum, exception_match, backend
):
    with pytest.raises(
        error_sum,
        match=exception_match,
    ):
        grouped_sum(
            column=column_to_aggregate,
            group_id=group_id,
            num_segments=len(group_id),
            backend=backend,
        )


@parameterize_based_on_dict(
    test_grouped_raises_specs,
    keys_of_test_cases=[
        "column_to_aggregate",
        "group_id",
        "error_mean",
        "exception_match",
    ],
)
@pytest.mark.skipif_jax
def test_grouped_mean_raises(
    column_to_aggregate, group_id, error_mean, exception_match, backend
):
    with pytest.raises(
        error_mean,
        match=exception_match,
    ):
        grouped_mean(
            column=column_to_aggregate,
            group_id=group_id,
            num_segments=len(group_id),
            backend=backend,
        )


@parameterize_based_on_dict(
    test_grouped_raises_specs,
    keys_of_test_cases=[
        "column_to_aggregate",
        "group_id",
        "error_max",
        "exception_match",
    ],
)
@pytest.mark.skipif_jax
def test_grouped_max_raises(
    column_to_aggregate, group_id, error_max, exception_match, backend
):
    with pytest.raises(
        error_max,
        match=exception_match,
    ):
        grouped_max(
            column=column_to_aggregate,
            group_id=group_id,
            num_segments=len(group_id),
            backend=backend,
        )


@parameterize_based_on_dict(
    test_grouped_raises_specs,
    keys_of_test_cases=[
        "column_to_aggregate",
        "group_id",
        "error_min",
        "exception_match",
    ],
)
@pytest.mark.skipif_jax
def test_grouped_min_raises(
    column_to_aggregate, group_id, error_min, exception_match, backend
):
    with pytest.raises(
        error_min,
        match=exception_match,
    ):
        grouped_min(
            column=column_to_aggregate,
            group_id=group_id,
            num_segments=len(group_id),
            backend=backend,
        )


@parameterize_based_on_dict(
    test_grouped_raises_specs,
    keys_of_test_cases=[
        "column_to_aggregate",
        "group_id",
        "error_any",
        "exception_match",
    ],
)
@pytest.mark.skipif_jax
def test_grouped_any_raises(
    column_to_aggregate, group_id, error_any, exception_match, backend
):
    with pytest.raises(
        error_any,
        match=exception_match,
    ):
        grouped_any(
            column=column_to_aggregate,
            group_id=group_id,
            num_segments=len(group_id),
            backend=backend,
        )


@parameterize_based_on_dict(
    test_grouped_raises_specs,
    keys_of_test_cases=[
        "column_to_aggregate",
        "group_id",
        "error_all",
        "exception_match",
    ],
)
@pytest.mark.skipif_jax
def test_grouped_all_raises(
    column_to_aggregate, group_id, error_all, exception_match, backend
):
    with pytest.raises(
        error_all,
        match=exception_match,
    ):
        grouped_all(
            column=column_to_aggregate,
            group_id=group_id,
            num_segments=len(group_id),
            backend=backend,
        )


@parameterize_based_on_dict(
    test_grouped_specs,
    keys_of_test_cases=[
        "column_to_aggregate",
        "p_id_to_aggregate_by",
        "p_id_to_store_by",
        "expected_res",
        "expected_type",
    ],
)
def test_sum_by_p_id(
    column_to_aggregate,
    p_id_to_aggregate_by,
    p_id_to_store_by,
    expected_res,
    expected_type,
    backend,
):
    result = sum_by_p_id(
        column=column_to_aggregate,
        p_id_to_aggregate_by=p_id_to_aggregate_by,
        p_id_to_store_by=p_id_to_store_by,
        num_segments=len(p_id_to_aggregate_by),
        backend=backend,
    )
    numpy.testing.assert_array_almost_equal(result, expected_res)
    assert numpy.issubdtype(result.dtype.type, expected_type), (
        "The dtype of the result is not as expected."
    )


@parameterize_based_on_dict(
    test_grouped_raises_specs,
    keys_of_test_cases=[
        "column_to_aggregate",
        "group_id",
        "p_id_to_store_by",
        "error_sum_by_p_id",
        "exception_match",
    ],
)
@pytest.mark.skipif_jax
def test_sum_by_p_id_raises(
    column_to_aggregate,
    group_id,
    p_id_to_store_by,
    error_sum_by_p_id,
    exception_match,
    backend,
):
    with pytest.raises(
        error_sum_by_p_id,
        match=exception_match,
    ):
        sum_by_p_id(
            column=column_to_aggregate,
            p_id_to_aggregate_by=group_id,
            p_id_to_store_by=p_id_to_store_by,
            num_segments=len(group_id),
            backend=backend,
        )
