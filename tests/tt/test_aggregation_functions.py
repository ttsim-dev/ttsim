from __future__ import annotations

import copy

import numpy
import pytest
from beartype.roar import BeartypeCallHintViolation

try:
    import jax_datetime  # ty: ignore[unresolved-import]

    my_datetime = jax_datetime.to_datetime
except ImportError:
    my_datetime = lambda x: x  # noqa: E731


from ttsim.tt.aggregation import (
    all_by_p_id,
    any_by_p_id,
    count_by_p_id,
    grouped_all,
    grouped_any,
    grouped_count,
    grouped_max,
    grouped_mean,
    grouped_min,
    grouped_sum,
    max_by_p_id,
    mean_by_p_id,
    min_by_p_id,
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
    "count_by_p_id": {
        "p_id_to_aggregate_by": numpy.array([-1, -1, 8, 8, 10]),
        "p_id_to_store_by": numpy.array([7, 8, 9, 10, 11]),
        "expected_res_count_by_p_id": numpy.array([0, 2, 0, 1, 0]),
    },
    "mean_by_p_id_float": {
        "column_to_aggregate": numpy.array([10.0, 20.0, 30.0, 40.0, 50.0]),
        "p_id_to_aggregate_by": numpy.array([-1, -1, 8, 8, 10]),
        "p_id_to_store_by": numpy.array([7, 8, 9, 10, 11]),
        "expected_res_mean_by_p_id": numpy.array([0.0, 35.0, 0.0, 50.0, 0.0]),
    },
    "max_by_p_id_float": {
        "column_to_aggregate": numpy.array([10.0, 20.0, 30.0, 40.0, 50.0]),
        "p_id_to_aggregate_by": numpy.array([-1, -1, 8, 8, 10]),
        "p_id_to_store_by": numpy.array([7, 8, 9, 10, 11]),
        "expected_res_max_by_p_id": numpy.array([0.0, 40.0, 0.0, 50.0, 0.0]),
    },
    "min_by_p_id_float": {
        "column_to_aggregate": numpy.array([10.0, 20.0, 30.0, 40.0, 50.0]),
        "p_id_to_aggregate_by": numpy.array([-1, -1, 8, 8, 10]),
        "p_id_to_store_by": numpy.array([7, 8, 9, 10, 11]),
        "expected_res_min_by_p_id": numpy.array([0.0, 30.0, 0.0, 50.0, 0.0]),
    },
    "any_by_p_id_bool": {
        "column_to_aggregate": numpy.array([True, False, True, False, True]),
        "p_id_to_aggregate_by": numpy.array([-1, -1, 8, 8, 10]),
        "p_id_to_store_by": numpy.array([7, 8, 9, 10, 11]),
        "expected_res_any_by_p_id": numpy.array([False, True, False, True, False]),
    },
    "all_by_p_id_bool": {
        "column_to_aggregate": numpy.array([True, False, True, False, True]),
        "p_id_to_aggregate_by": numpy.array([-1, -1, 8, 8, 10]),
        "p_id_to_store_by": numpy.array([7, 8, 9, 10, 11]),
        "expected_res_all_by_p_id": numpy.array([False, False, False, True, False]),
    },
}

# With the package-wide beartype claw on, the aggregation dispatcher's column
# aliases are the runtime contract: a wrong-dtype argument is rejected by the
# claw with `BeartypeCallHintViolation` before any function body runs. Each
# `exception_match` pins the offending parameter name so the case still proves
# the correct argument is rejected.
test_grouped_raises_specs = {
    "dtype_boolean": {
        "column_to_aggregate": numpy.array([True, True, True, False, False]),
        "group_id": numpy.array([0, 0, 1, 1, 1]),
        "error_max": BeartypeCallHintViolation,
        "error_min": BeartypeCallHintViolation,
        "exception_match": "parameter column",
    },
    "float_group_id": {
        "column_to_aggregate": numpy.array([0, 1, 2, 3, 4]),
        "group_id": numpy.array([0, 0, 3.5, 3.5, 3.5]),
        "p_id_to_store_by": numpy.array([0, 1, 2, 3, 4]),
        "error_sum": BeartypeCallHintViolation,
        "error_mean": BeartypeCallHintViolation,
        "error_max": BeartypeCallHintViolation,
        "error_min": BeartypeCallHintViolation,
        "error_sum_by_p_id": BeartypeCallHintViolation,
        "exception_match": "parameter (group_id|p_id_to_aggregate_by)",
    },
    "dtype_float": {
        "column_to_aggregate": numpy.array([1.5, 2, 3.5, 4, 5]),
        "group_id": numpy.array([0, 0, 1, 1, 1]),
        "error_any": BeartypeCallHintViolation,
        "error_all": BeartypeCallHintViolation,
        "exception_match": "parameter column",
    },
    "float_group_id_bool": {
        "column_to_aggregate": numpy.array([True, True, True, False, False]),
        "group_id": numpy.array([0, 0, 3.5, 3.5, 3.5]),
        "error_any": BeartypeCallHintViolation,
        "error_all": BeartypeCallHintViolation,
        "exception_match": "parameter group_id",
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
        ],
    ),
    "group_id": numpy.array([1, 0, 1, 1, 1]),
    "expected_res_max": numpy.array(
        [
            numpy.datetime64("2004"),
            numpy.datetime64("2001"),
            numpy.datetime64("2004"),
            numpy.datetime64("2004"),
            numpy.datetime64("2004"),
        ],
    ),
    "expected_res_min": numpy.array(
        [
            numpy.datetime64("2000"),
            numpy.datetime64("2001"),
            numpy.datetime64("2000"),
            numpy.datetime64("2000"),
            numpy.datetime64("2000"),
        ],
    ),
}

test_grouped_raises_specs["dtype_string"] = {
    "column_to_aggregate": numpy.array(["0", "1", "2", "3", "4"]),
    "group_id": numpy.array([0, 0, 1, 1, 1]),
    "error_sum": BeartypeCallHintViolation,
    "error_mean": BeartypeCallHintViolation,
    "error_max": BeartypeCallHintViolation,
    "error_min": BeartypeCallHintViolation,
    "error_any": BeartypeCallHintViolation,
    "error_all": BeartypeCallHintViolation,
    "exception_match": "parameter column",
}
test_grouped_raises_specs["datetime"] = {
    "column_to_aggregate": numpy.array(
        [
            numpy.datetime64("2000"),
            numpy.datetime64("2001"),
            numpy.datetime64("2002"),
            numpy.datetime64("2003"),
            numpy.datetime64("2004"),
        ],
    ),
    "group_id": numpy.array([0, 0, 1, 1, 1]),
    "error_sum": BeartypeCallHintViolation,
    "error_mean": BeartypeCallHintViolation,
    "error_any": BeartypeCallHintViolation,
    "error_all": BeartypeCallHintViolation,
    "exception_match": "parameter column",
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
    column_to_aggregate,
    group_id,
    error_sum,
    exception_match,
    backend,
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
    column_to_aggregate,
    group_id,
    error_mean,
    exception_match,
    backend,
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
    column_to_aggregate,
    group_id,
    error_max,
    exception_match,
    backend,
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
    column_to_aggregate,
    group_id,
    error_min,
    exception_match,
    backend,
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
    column_to_aggregate,
    group_id,
    error_any,
    exception_match,
    backend,
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
    column_to_aggregate,
    group_id,
    error_all,
    exception_match,
    backend,
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


def test_grouped_sum_single_element(backend, xnp):
    """Test grouped_sum with a single-element array."""
    column = xnp.array([42.0])
    group_id = xnp.array([0])

    result = grouped_sum(
        column=column,
        group_id=group_id,
        num_segments=1,
        backend=backend,
    )

    numpy.testing.assert_array_equal(result, xnp.array([42.0]))


def test_grouped_max_all_same_values(backend, xnp):
    """Test grouped_max when all values in groups are identical."""
    column = xnp.array([5.0, 5.0, 5.0, 10.0, 10.0])
    group_id = xnp.array([0, 0, 0, 1, 1])

    result = grouped_max(
        column=column,
        group_id=group_id,
        num_segments=5,
        backend=backend,
    )

    expected = xnp.array([5.0, 5.0, 5.0, 10.0, 10.0])
    numpy.testing.assert_array_equal(result, expected)


def test_grouped_min_all_same_values(backend, xnp):
    """Test grouped_min when all values in groups are identical."""
    column = xnp.array([5.0, 5.0, 5.0, 10.0, 10.0])
    group_id = xnp.array([0, 0, 0, 1, 1])

    result = grouped_min(
        column=column,
        group_id=group_id,
        num_segments=5,
        backend=backend,
    )

    expected = xnp.array([5.0, 5.0, 5.0, 10.0, 10.0])
    numpy.testing.assert_array_equal(result, expected)


def test_sum_by_p_id_all_missing(backend, xnp):
    """Test sum_by_p_id when all p_ids are missing (-1)."""
    column = xnp.array([10.0, 20.0, 30.0])
    p_id_to_aggregate_by = xnp.array([-1, -1, -1])
    p_id_to_store_by = xnp.array([0, 1, 2])

    result = sum_by_p_id(
        column=column,
        p_id_to_aggregate_by=p_id_to_aggregate_by,
        p_id_to_store_by=p_id_to_store_by,
        num_segments=3,
        backend=backend,
    )

    # All missing, so result should be zeros
    numpy.testing.assert_array_equal(result, xnp.array([0.0, 0.0, 0.0]))


def test_sum_by_p_id_bool_column(backend, xnp):
    """Test sum_by_p_id with boolean column."""
    column = xnp.array([True, False, True, True, False])
    p_id_to_aggregate_by = xnp.array([0, 0, 1, 1, 1])
    p_id_to_store_by = xnp.array([0, 1, 2, 3, 4])

    result = sum_by_p_id(
        column=column,
        p_id_to_aggregate_by=p_id_to_aggregate_by,
        p_id_to_store_by=p_id_to_store_by,
        num_segments=5,
        backend=backend,
    )

    # p_id 0: True + False = 1, p_id 1: True + True + False = 2
    expected = xnp.array([1, 2, 0, 0, 0])
    numpy.testing.assert_array_equal(result, expected)


def test_grouped_count_with_many_groups(backend, xnp):
    """Test grouped_count with many distinct groups."""
    # Each element in its own group
    group_id = xnp.array([0, 1, 2, 3, 4])

    result = grouped_count(
        group_id=group_id,
        num_segments=5,
        backend=backend,
    )

    # Each group has exactly 1 element
    expected = xnp.array([1, 1, 1, 1, 1])
    numpy.testing.assert_array_equal(result, expected)


def test_grouped_mean_with_negative_values(backend, xnp):
    """Test grouped_mean with negative values."""
    column = xnp.array([-10.0, 10.0, -5.0, 5.0])
    group_id = xnp.array([0, 0, 1, 1])

    result = grouped_mean(
        column=column,
        group_id=group_id,
        num_segments=4,
        backend=backend,
    )

    # Group 0: (-10 + 10) / 2 = 0, Group 1: (-5 + 5) / 2 = 0
    expected = xnp.array([0.0, 0.0, 0.0, 0.0])
    numpy.testing.assert_array_almost_equal(result, expected)


def test_grouped_sum_large_group_ids_with_gaps(backend, xnp):
    """Test grouped_sum with large group_id values that have gaps."""
    column = xnp.array([1.0, 2.0, 3.0])
    group_id = xnp.array([0, 100, 100])

    result = grouped_sum(
        column=column,
        group_id=group_id,
        num_segments=101,
        backend=backend,
    )

    # Group 0: 1.0, Group 100: 2.0 + 3.0 = 5.0
    expected = xnp.array([1.0, 5.0, 5.0])
    numpy.testing.assert_array_equal(result, expected)


def test_count_by_p_id(backend, xnp):
    """Counts the number of source rows whose `p_id_to_aggregate_by` matches each
    destination `p_id_to_store_by`. Negative `p_id_to_aggregate_by` entries do
    not contribute; empty destinations get count 0.
    """
    p_id_to_aggregate_by = xnp.array([-1, -1, 8, 8, 10])
    p_id_to_store_by = xnp.array([7, 8, 9, 10, 11])

    result = count_by_p_id(
        p_id_to_aggregate_by=p_id_to_aggregate_by,
        p_id_to_store_by=p_id_to_store_by,
        num_segments=5,
        backend=backend,
    )

    expected = xnp.array([0, 2, 0, 1, 0])
    numpy.testing.assert_array_equal(result, expected)


def test_mean_by_p_id(backend, xnp):
    """Mean of source values keyed by `p_id_to_aggregate_by`, scattered to
    `p_id_to_store_by`. Empty destinations are 0.
    """
    column = xnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
    p_id_to_aggregate_by = xnp.array([-1, -1, 8, 8, 10])
    p_id_to_store_by = xnp.array([7, 8, 9, 10, 11])

    result = mean_by_p_id(
        column=column,
        p_id_to_aggregate_by=p_id_to_aggregate_by,
        p_id_to_store_by=p_id_to_store_by,
        num_segments=5,
        backend=backend,
    )

    expected = xnp.array([0.0, 35.0, 0.0, 50.0, 0.0])
    numpy.testing.assert_array_almost_equal(result, expected)


def test_max_by_p_id(backend, xnp):
    """Maximum of source values keyed by `p_id_to_aggregate_by`. Empty
    destinations are 0.
    """
    column = xnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
    p_id_to_aggregate_by = xnp.array([-1, -1, 8, 8, 10])
    p_id_to_store_by = xnp.array([7, 8, 9, 10, 11])

    result = max_by_p_id(
        column=column,
        p_id_to_aggregate_by=p_id_to_aggregate_by,
        p_id_to_store_by=p_id_to_store_by,
        num_segments=5,
        backend=backend,
    )

    expected = xnp.array([0.0, 40.0, 0.0, 50.0, 0.0])
    numpy.testing.assert_array_equal(result, expected)


def test_min_by_p_id(backend, xnp):
    """Minimum of source values keyed by `p_id_to_aggregate_by`. Empty
    destinations are 0.
    """
    column = xnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
    p_id_to_aggregate_by = xnp.array([-1, -1, 8, 8, 10])
    p_id_to_store_by = xnp.array([7, 8, 9, 10, 11])

    result = min_by_p_id(
        column=column,
        p_id_to_aggregate_by=p_id_to_aggregate_by,
        p_id_to_store_by=p_id_to_store_by,
        num_segments=5,
        backend=backend,
    )

    expected = xnp.array([0.0, 30.0, 0.0, 50.0, 0.0])
    numpy.testing.assert_array_equal(result, expected)


def test_any_by_p_id(backend, xnp):
    """Logical OR of source bools keyed by `p_id_to_aggregate_by`. Empty
    destinations are False (empty disjunction).
    """
    column = xnp.array([True, False, True, False, True])
    p_id_to_aggregate_by = xnp.array([-1, -1, 8, 8, 10])
    p_id_to_store_by = xnp.array([7, 8, 9, 10, 11])

    result = any_by_p_id(
        column=column,
        p_id_to_aggregate_by=p_id_to_aggregate_by,
        p_id_to_store_by=p_id_to_store_by,
        num_segments=5,
        backend=backend,
    )

    expected = xnp.array([False, True, False, True, False])
    numpy.testing.assert_array_equal(result, expected)


def test_all_by_p_id(backend, xnp):
    """Logical AND of source bools keyed by `p_id_to_aggregate_by`. Empty
    destinations are True (empty conjunction).
    """
    column = xnp.array([True, False, True, False, True])
    p_id_to_aggregate_by = xnp.array([-1, -1, 8, 8, 10])
    p_id_to_store_by = xnp.array([7, 8, 9, 10, 11])

    result = all_by_p_id(
        column=column,
        p_id_to_aggregate_by=p_id_to_aggregate_by,
        p_id_to_store_by=p_id_to_store_by,
        num_segments=5,
        backend=backend,
    )

    expected = xnp.array([True, False, True, True, True])
    numpy.testing.assert_array_equal(result, expected)


def test_count_by_p_id_ignores_negative_sentinels(backend, xnp):
    """`-1` entries in `p_id_to_aggregate_by` never contribute to the count."""
    p_id_to_aggregate_by = xnp.array([-1, -1, -1])
    p_id_to_store_by = xnp.array([1, 2, 3])

    result = count_by_p_id(
        p_id_to_aggregate_by=p_id_to_aggregate_by,
        p_id_to_store_by=p_id_to_store_by,
        num_segments=3,
        backend=backend,
    )

    expected = xnp.array([0, 0, 0])
    numpy.testing.assert_array_equal(result, expected)


def test_mean_by_p_id_ignores_negative_sentinels(backend, xnp):
    """`-1` entries in `p_id_to_aggregate_by` never enter the mean."""
    column = xnp.array([100.0, 200.0, 10.0, 20.0])
    p_id_to_aggregate_by = xnp.array([-1, -1, 1, 1])
    p_id_to_store_by = xnp.array([0, 1, 2])

    result = mean_by_p_id(
        column=column,
        p_id_to_aggregate_by=p_id_to_aggregate_by,
        p_id_to_store_by=p_id_to_store_by,
        num_segments=3,
        backend=backend,
    )

    expected = xnp.array([0.0, 15.0, 0.0])
    numpy.testing.assert_array_almost_equal(result, expected)


def test_max_by_p_id_ignores_negative_sentinels(backend, xnp):
    """`-1` entries never affect the max; an enormous value at a masked
    position must not leak into the result.
    """
    column = xnp.array([1e10, -1e10, 5.0, 7.0])
    p_id_to_aggregate_by = xnp.array([-1, -1, 1, 1])
    p_id_to_store_by = xnp.array([0, 1, 2])

    result = max_by_p_id(
        column=column,
        p_id_to_aggregate_by=p_id_to_aggregate_by,
        p_id_to_store_by=p_id_to_store_by,
        num_segments=3,
        backend=backend,
    )

    expected = xnp.array([0.0, 7.0, 0.0])
    numpy.testing.assert_array_equal(result, expected)


def test_min_by_p_id_ignores_negative_sentinels(backend, xnp):
    """`-1` entries never affect the min; a tiny value at a masked position
    must not leak into the result.
    """
    column = xnp.array([1e10, -1e10, 5.0, 7.0])
    p_id_to_aggregate_by = xnp.array([-1, -1, 1, 1])
    p_id_to_store_by = xnp.array([0, 1, 2])

    result = min_by_p_id(
        column=column,
        p_id_to_aggregate_by=p_id_to_aggregate_by,
        p_id_to_store_by=p_id_to_store_by,
        num_segments=3,
        backend=backend,
    )

    expected = xnp.array([0.0, 5.0, 0.0])
    numpy.testing.assert_array_equal(result, expected)


def test_any_by_p_id_ignores_negative_sentinels(backend, xnp):
    """`-1` entries never set the disjunction; a True at a masked position
    must not flip the result.
    """
    column = xnp.array([True, True, False, False])
    p_id_to_aggregate_by = xnp.array([-1, -1, 1, 1])
    p_id_to_store_by = xnp.array([0, 1, 2])

    result = any_by_p_id(
        column=column,
        p_id_to_aggregate_by=p_id_to_aggregate_by,
        p_id_to_store_by=p_id_to_store_by,
        num_segments=3,
        backend=backend,
    )

    expected = xnp.array([False, False, False])
    numpy.testing.assert_array_equal(result, expected)


def test_all_by_p_id_ignores_negative_sentinels(backend, xnp):
    """`-1` entries never enter the conjunction; a False at a masked position
    must not collapse the result to False.
    """
    column = xnp.array([False, False, True, True])
    p_id_to_aggregate_by = xnp.array([-1, -1, 1, 1])
    p_id_to_store_by = xnp.array([0, 1, 2])

    result = all_by_p_id(
        column=column,
        p_id_to_aggregate_by=p_id_to_aggregate_by,
        p_id_to_store_by=p_id_to_store_by,
        num_segments=3,
        backend=backend,
    )

    expected = xnp.array([True, True, True])
    numpy.testing.assert_array_equal(result, expected)


def test_sum_by_p_id_ignores_non_negative_keys_absent_from_store(backend, xnp):
    """A non-negative `p_id_to_aggregate_by` entry whose value is not present
    in `p_id_to_store_by` does not silently scatter into the clipped bucket
    `searchsorted` returns. On JAX, the absent-key contribution would
    otherwise corrupt the destination at the clamped index.
    """
    column = xnp.array([100.0, 200.0, 300.0])
    p_id_to_aggregate_by = xnp.array([1, 999, 2])
    p_id_to_store_by = xnp.array([1, 2, 3])

    result = sum_by_p_id(
        column=column,
        p_id_to_aggregate_by=p_id_to_aggregate_by,
        p_id_to_store_by=p_id_to_store_by,
        num_segments=3,
        backend=backend,
    )

    expected = xnp.array([100.0, 300.0, 0.0])
    numpy.testing.assert_array_equal(result, expected)


def test_max_by_p_id_keeps_real_sentinel_valued_data(backend, xnp):
    """A legitimate column value equal to the per-dtype `min` sentinel must
    survive — the empty-bucket replacement must not rewrite real data via
    sentinel equality. Uses int32 so the dtype's min is representable
    identically on numpy and JAX (where float64 would silently downcast).
    """
    column = xnp.array([numpy.iinfo(numpy.int32).min, 5], dtype=numpy.int32)
    p_id_to_aggregate_by = xnp.array([1, 2])
    p_id_to_store_by = xnp.array([1, 2, 3])

    result = max_by_p_id(
        column=column,
        p_id_to_aggregate_by=p_id_to_aggregate_by,
        p_id_to_store_by=p_id_to_store_by,
        num_segments=3,
        backend=backend,
    )

    expected = xnp.array([numpy.iinfo(numpy.int32).min, 5, 0], dtype=numpy.int32)
    numpy.testing.assert_array_equal(result, expected)


def test_min_by_p_id_keeps_real_sentinel_valued_data(backend, xnp):
    """A legitimate column value equal to the per-dtype `max` sentinel must
    survive — the empty-bucket replacement must not rewrite real data via
    sentinel equality. Uses int32 so the dtype's max is representable
    identically on numpy and JAX (where float64 would silently downcast).
    """
    column = xnp.array([numpy.iinfo(numpy.int32).max, 5], dtype=numpy.int32)
    p_id_to_aggregate_by = xnp.array([1, 2])
    p_id_to_store_by = xnp.array([1, 2, 3])

    result = min_by_p_id(
        column=column,
        p_id_to_aggregate_by=p_id_to_aggregate_by,
        p_id_to_store_by=p_id_to_store_by,
        num_segments=3,
        backend=backend,
    )

    expected = xnp.array([numpy.iinfo(numpy.int32).max, 5, 0], dtype=numpy.int32)
    numpy.testing.assert_array_equal(result, expected)
