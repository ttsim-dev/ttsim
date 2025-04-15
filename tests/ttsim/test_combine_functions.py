import datetime
import inspect

import pandas as pd
import pytest

from ttsim.aggregation import AggregateByGroupSpec, AggregateByPIDSpec, AggType
from ttsim.combine_functions import (
    _annotate_aggregation_functions,
    _create_aggregate_by_group_functions,
    _create_aggregation_functions,
    _create_one_aggregation_function,
    _fail_if_targets_not_in_functions,
    _get_name_of_aggregation_source,
)
from ttsim.compute_taxes_and_transfers import compute_taxes_and_transfers
from ttsim.policy_environment import PolicyEnvironment
from ttsim.ttsim_objects import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    DerivedAggregationFunction,
    policy_function,
    policy_input,
)


@pytest.fixture
@policy_function(leaf_name="foo")
def function_with_bool_return(x: bool) -> bool:
    return x


@policy_input()
def x() -> int:
    pass


@policy_input()
def x_f() -> float:
    pass


@policy_input()
def x_b() -> bool:
    pass


@policy_input()
def p_id() -> int:
    pass


@policy_input()
def hh_id() -> int:
    pass


@pytest.fixture
@policy_function(leaf_name="bar")
def function_with_int_return(x: int) -> int:
    return x


@pytest.fixture
@policy_function(leaf_name="baz")
def function_with_float_return(x: int) -> float:
    return x


def identity(x):
    return x


@pytest.mark.parametrize(
    (
        "objects_tree",
        "targets_tree",
        "data_tree",
        "aggregations_specs_from_env",
    ),
    [
        (
            # Aggregations derived from simple function arguments
            {
                "hh_id": hh_id,
                "p_id": p_id,
                "namespace1": {
                    "f": policy_function(leaf_name="f")(identity),
                    "x": x,
                },
            },
            {"namespace1": {"f": None}},
            {
                "namespace1": {"x": pd.Series([1, 1, 1])},
                "hh_id": pd.Series([0, 0, 0]),
                "p_id": pd.Series([0, 1, 2]),
            },
            {},
        ),
        (
            # Aggregations derived from namespaced function arguments
            {
                "hh_id": hh_id,
                "p_id": p_id,
                "namespace1": {"f": policy_function(leaf_name="f")(identity)},
                "inputs": {"x": x},
            },
            {"namespace1": {"f": None}},
            {
                "inputs": {"x": pd.Series([1, 1, 1])},
                "hh_id": pd.Series([0, 0, 0]),
                "p_id": pd.Series([0, 1, 2]),
            },
            {},
        ),
        (
            # Aggregations derived from target
            {
                "hh_id": hh_id,
                "p_id": p_id,
                "namespace1": {
                    "f": policy_function(leaf_name="f")(identity),
                    "x": x,
                },
            },
            {"namespace1": {"f_hh": None}},
            {
                "namespace1": {"x": pd.Series([1, 1, 1])},
                "hh_id": pd.Series([0, 0, 0]),
                "p_id": pd.Series([0, 1, 2]),
            },
            {},
        ),
        (
            # Aggregations derived from simple environment specification
            {
                "hh_id": hh_id,
                "p_id": p_id,
                "namespace1": {
                    "f": policy_function(leaf_name="f")(identity),
                    "x": x,
                },
            },
            {"namespace1": {"f": None}},
            {
                "namespace1": {"x": pd.Series([1, 1, 1])},
                "hh_id": pd.Series([0, 0, 0]),
                "p_id": pd.Series([0, 1, 2]),
            },
            {
                "namespace1": {
                    "y_hh": AggregateByGroupSpec(
                        target="y_hh",
                        source="x",
                        agg=AggType.SUM,
                    )
                },
            },
        ),
        (
            # Aggregations derived from namespaced environment specification
            {
                "hh_id": hh_id,
                "p_id": p_id,
                "namespace1": {"f": policy_function(leaf_name="f")(identity)},
                "inputs": {"x": x},
            },
            {"namespace1": {"f": None}},
            {
                "inputs": {"x": pd.Series([1, 1, 1])},
                "hh_id": pd.Series([0, 0, 0]),
                "p_id": pd.Series([0, 1, 2]),
            },
            {
                "namespace1": {
                    "y_hh": AggregateByGroupSpec(
                        target="y_hh",
                        source="inputs__x",
                        agg=AggType.SUM,
                    )
                },
            },
        ),
    ],
)
def test_create_aggregate_by_group_functions(
    objects_tree,
    targets_tree,
    data_tree,
    aggregations_specs_from_env,
):
    environment = PolicyEnvironment(raw_objects_tree=objects_tree)
    compute_taxes_and_transfers(
        environment=environment,
        data_tree=data_tree,
        targets_tree=targets_tree,
        groupings=("hh",),
    )


START_DATE = datetime.date.fromisoformat("1900-01-01")
END_DATE = datetime.date.fromisoformat("2100-12-31")


@pytest.mark.parametrize(
    (
        "functions",
        "inputs",
        "aggregation_functions",
        "expected_return_type",
    ),
    [
        (
            {},
            {},
            {
                "foo": DerivedAggregationFunction(
                    leaf_name="foo",
                    function=identity,
                    source="x",
                    aggregation_method="count",
                    start_date=START_DATE,
                    end_date=END_DATE,
                    vectorization_strategy="not_required",
                )
            },
            int,
        ),
        (
            {},
            {"x": x},
            {
                "foo": DerivedAggregationFunction(
                    leaf_name="foo",
                    function=identity,
                    source="x",
                    aggregation_method="sum",
                    start_date=START_DATE,
                    end_date=END_DATE,
                    vectorization_strategy="not_required",
                )
            },
            int,
        ),
        (
            {},
            {"x": x_f},
            {
                "foo": DerivedAggregationFunction(
                    leaf_name="foo",
                    function=identity,
                    source="x",
                    aggregation_method="sum",
                    start_date=START_DATE,
                    end_date=END_DATE,
                    vectorization_strategy="not_required",
                )
            },
            float,
        ),
        (
            {},
            {"x": x_b},
            {
                "foo": DerivedAggregationFunction(
                    leaf_name="foo",
                    function=identity,
                    source="x",
                    aggregation_method="sum",
                    start_date=START_DATE,
                    end_date=END_DATE,
                    vectorization_strategy="not_required",
                )
            },
            int,
        ),
        (
            {"n1__foo": function_with_bool_return},
            {},
            {
                "n1__foo_hh": DerivedAggregationFunction(
                    leaf_name="foo_hh",
                    function=function_with_bool_return,
                    source="n1__foo",
                    aggregation_method="sum",
                    start_date=START_DATE,
                    end_date=END_DATE,
                    vectorization_strategy="not_required",
                )
            },
            int,
        ),
        (
            {"n1__foo": function_with_float_return},
            {},
            {
                "n1__foo_hh": DerivedAggregationFunction(
                    leaf_name="foo_hh",
                    function=function_with_float_return,
                    source="n1__foo",
                    aggregation_method="sum",
                    start_date=START_DATE,
                    end_date=END_DATE,
                    vectorization_strategy="not_required",
                )
            },
            float,
        ),
        (
            {"n1__foo": function_with_int_return},
            {},
            {
                "n1__foo_hh": DerivedAggregationFunction(
                    leaf_name="foo_hh",
                    function=function_with_int_return,
                    source="n1__foo",
                    aggregation_method="sum",
                    start_date=START_DATE,
                    end_date=END_DATE,
                    vectorization_strategy="not_required",
                )
            },
            int,
        ),
    ],
)
def test_annotations_for_aggregation(
    functions,
    inputs,
    aggregation_functions,
    expected_return_type,
):
    name_of_aggregation_function = next(iter(aggregation_functions.keys()))
    annotation_of_aggregation_function = _annotate_aggregation_functions(
        functions=functions,
        inputs=inputs,
        aggregation_functions=aggregation_functions,
    )[name_of_aggregation_function].__annotations__["return"]
    assert annotation_of_aggregation_function == expected_return_type


@pytest.mark.parametrize(
    "functions, targets, expected_error_match",
    [
        ({"foo": identity}, {"bar": None}, "('bar',)"),
        ({"foo__baz": identity}, {"foo__bar": None}, "('foo', 'bar')"),
    ],
)
def test_fail_if_targets_are_not_among_functions(
    functions, targets, expected_error_match
):
    with pytest.raises(ValueError) as e:
        _fail_if_targets_not_in_functions(functions, targets)
    assert expected_error_match in str(e.value)


@pytest.mark.parametrize(
    (
        "functions",
        "inputs",
        "aggregations",
        "aggregation_type",
        "top_level_namespace",
        "expected_annotations",
    ),
    [
        (
            {"foo": function_with_bool_return},
            {},
            {
                "foo_hh": AggregateByGroupSpec(
                    target="foo_hh", source="foo", agg=AggType.SUM
                ),
            },
            "group",
            ["foo"],
            {"foo": bool, "return": int},
        ),
        (
            {"foo": function_with_float_return},
            {},
            {
                "foo_hh": AggregateByGroupSpec(
                    target="foo_hh", source="foo", agg=AggType.SUM
                ),
            },
            "group",
            ["foo"],
            {"foo": float, "return": float},
        ),
        (
            {"foo": function_with_int_return},
            {},
            {
                "foo_hh": AggregateByPIDSpec(
                    target="foo_hh",
                    p_id_to_aggregate_by="foreign_id_col",
                    source="foo",
                    agg=AggType.SUM,
                )
            },
            "p_id",
            ["foo"],
            {"foo": int, "return": int},
        ),
    ],
)
def test_annotations_are_applied_to_derived_functions(
    functions,
    inputs,
    aggregations,
    aggregation_type,
    top_level_namespace,
    expected_annotations,
):
    """Test that the annotations are applied to the derived functions."""
    result_func = next(
        iter(
            _create_aggregation_functions(
                functions=functions,
                inputs=inputs,
                aggregation_functions_to_create=aggregations,
                aggregation_type=aggregation_type,
                top_level_namespace=top_level_namespace,
                groupings=("hh",),
            ).values()
        )
    )
    assert result_func.__annotations__ == expected_annotations


@pytest.mark.parametrize(
    (
        "functions",
        "inputs",
        "targets",
        "data",
        "aggregations_from_environment",
        "top_level_namespace",
        "expected",
    ),
    [
        (
            {"foo": policy_function(leaf_name="foo")(identity)},
            {"x": x},
            {},
            {"x": pd.Series([1])},
            {},
            ["foo", "x"],
            ("x_hh"),
        ),
        (
            {"n1__foo": policy_function(leaf_name="foo")(identity)},
            {"hh_id": hh_id, "n2__x": x},
            {},
            {"n2": {"x": pd.Series([1])}},
            {},
            ["n1", "n2"],
            ("n2__x_hh"),
        ),
        (
            {},
            {"x": x},
            {"x_hh": None},
            {"x": pd.Series([1])},
            {},
            ["x"],
            ("x_hh"),
        ),
        (
            {"foo": policy_function(leaf_name="foo")(identity)},
            {"hh_id": hh_id, "x": x},
            {},
            {"x": pd.Series([1])},
            {
                "n1__foo_hh": AggregateByGroupSpec(
                    target="foo_hh",
                    source="foo",
                    agg=AggType.SUM,
                )
            },
            ["x", "foo", "n1"],
            ("n1__foo_hh"),
        ),
    ],
)
def test_derived_aggregation_functions_are_in_correct_namespace(
    functions,
    inputs,
    targets,
    data,
    aggregations_from_environment,
    top_level_namespace,
    expected,
):
    """Test that the derived aggregation functions are in the correct namespace.

    The namespace of the derived aggregation functions should be the same as the
    namespace of the function that is being aggregated.
    """
    result = _create_aggregate_by_group_functions(
        functions=functions,
        inputs=inputs,
        targets=targets,
        data=data,
        top_level_namespace=top_level_namespace,
        groupings=("hh",),
    )
    assert expected in result


@pytest.mark.parametrize(
    (
        "aggregation_target",
        "aggregation_spec",
        "group_by_id",
        "functions",
        "inputs",
        "top_level_namespace",
        "expected_start_date",
        "expected_end_date",
    ),
    [
        (
            "x_hh",
            AggregateByGroupSpec(target="x_hh", source="x", agg=AggType.SUM),
            "hh_id",
            {},
            {"x": x},
            ["x", "x_hh", "hh_id"],
            DEFAULT_START_DATE,
            DEFAULT_END_DATE,
        ),
        (
            "x_hh",
            AggregateByGroupSpec(target="x_hh", source="x", agg=AggType.SUM),
            "hh_id",
            {"x": policy_function(leaf_name="x")(lambda x: x)},
            {},
            ["x", "x_hh", "hh_id"],
            DEFAULT_START_DATE,
            DEFAULT_END_DATE,
        ),
        (
            "x_hh",
            AggregateByGroupSpec(target="x_hh", source="x", agg=AggType.SUM),
            "hh_id",
            {
                "x": policy_function(
                    leaf_name="x", start_date="2025-01-01", end_date="2025-12-31"
                )(lambda x: x)
            },
            {},
            ["x", "x_hh", "hh_id"],
            datetime.date.fromisoformat("2025-01-01"),
            datetime.date.fromisoformat("2025-12-31"),
        ),
    ],
)
def test_aggregate_by_group_function_start_and_end_date(
    aggregation_target,
    aggregation_spec,
    group_by_id,
    functions,
    inputs,
    top_level_namespace,
    expected_start_date,
    expected_end_date,
):
    result = _create_one_aggregation_function(
        aggregation_target=aggregation_target,
        aggregation_spec=aggregation_spec,
        aggregation_type="group",
        group_by_id=group_by_id,
        functions=functions,
        inputs=inputs,
        top_level_namespace=top_level_namespace,
    )
    assert result.start_date == expected_start_date
    assert result.end_date == expected_end_date


@pytest.mark.parametrize(
    (
        "aggregation_target",
        "aggregation_spec",
        "functions",
        "inputs",
        "top_level_namespace",
        "expected_start_date",
        "expected_end_date",
    ),
    [
        (
            "bar",
            AggregateByPIDSpec(
                target="bar_hh",
                source="x",
                agg=AggType.SUM,
                p_id_to_aggregate_by="foreign_id_col",
            ),
            {"x": policy_function(leaf_name="x")(lambda x: x)},
            {},
            ["x", "bar", "foreign_id_col"],
            DEFAULT_START_DATE,
            DEFAULT_END_DATE,
        ),
        (
            "bar",
            AggregateByPIDSpec(
                target="bar_hh",
                source="x",
                agg=AggType.SUM,
                p_id_to_aggregate_by="foreign_id_col",
            ),
            {},
            {"x": x},
            ["x", "bar", "foreign_id_col"],
            DEFAULT_START_DATE,
            DEFAULT_END_DATE,
        ),
        (
            "bar",
            AggregateByPIDSpec(
                target="bar_hh",
                source="x",
                agg=AggType.SUM,
                p_id_to_aggregate_by="foreign_id_col",
            ),
            {
                "x": policy_function(
                    leaf_name="x", start_date="2025-01-01", end_date="2025-12-31"
                )(lambda x: x)
            },
            {},
            ["x", "bar", "foreign_id_col"],
            datetime.date.fromisoformat("2025-01-01"),
            datetime.date.fromisoformat("2025-12-31"),
        ),
    ],
)
def test_aggregate_by_p_id_function_start_and_end_date(
    aggregation_target,
    aggregation_spec,
    functions,
    inputs,
    top_level_namespace,
    expected_start_date,
    expected_end_date,
):
    result = _create_one_aggregation_function(
        aggregation_target=aggregation_target,
        aggregation_spec=aggregation_spec,
        aggregation_type="p_id",
        group_by_id=None,
        functions=functions,
        inputs=inputs,
        top_level_namespace=top_level_namespace,
    )
    assert result.start_date == expected_start_date
    assert result.end_date == expected_end_date


@pytest.mark.parametrize(
    (
        "aggregation_target",
        "aggregation_spec",
        "functions",
        "inputs",
        "group_by_id",
        "top_level_namespace",
        "expected_arg_names",
    ),
    [
        (
            "foo_hh",
            AggregateByGroupSpec(target="foo_hh", source=None, agg=AggType.COUNT),
            {"foo": policy_function(leaf_name="foo")(lambda x: x)},
            {},
            "hh_id",
            ["foo", "hh_id"],
            ["hh_id"],
        ),
        (
            "foo_hh",
            AggregateByGroupSpec(target="foo_hh", source="foo", agg=AggType.SUM),
            {"foo": policy_function(leaf_name="foo")(lambda x: x)},
            {},
            "hh_id",
            ["foo", "hh_id"],
            ["hh_id", "foo"],
        ),
        (
            "foo__bar_hh",
            AggregateByGroupSpec(target="foo__bar_hh", source="bar", agg=AggType.SUM),
            {"foo__bar": policy_function(leaf_name="bar")(lambda x: x)},
            {},
            "hh_id",
            ["foo", "hh_id"],
            ["hh_id", "foo__bar"],
        ),
    ],
)
def test_function_arguments_are_namespaced_for_derived_group_funcs(
    aggregation_target,
    aggregation_spec,
    functions,
    inputs,
    group_by_id,
    top_level_namespace,
    expected_arg_names,
):
    result = _create_one_aggregation_function(
        aggregation_target=aggregation_target,
        aggregation_spec=aggregation_spec,
        aggregation_type="group",
        group_by_id=group_by_id,
        functions=functions,
        inputs=inputs,
        top_level_namespace=top_level_namespace,
    )
    assert all(
        arg_name in inspect.signature(result).parameters
        for arg_name in expected_arg_names
    )


@pytest.mark.parametrize(
    (
        "aggregation_target",
        "aggregation_spec",
        "functions",
        "inputs",
        "group_by_id",
        "top_level_namespace",
        "expected_arg_names",
    ),
    [
        (
            "foo",
            AggregateByPIDSpec(
                target="foo_hh",
                agg=AggType.SUM,
                source="bar",
                p_id_to_aggregate_by="foreign_id_col",
            ),
            {"bar": policy_function(leaf_name="bar")(lambda x: x)},
            {},
            "foreign_id_col",
            ["foo", "foreign_id_col", "bar"],
            ["foreign_id_col", "bar"],
        ),
        (
            "foo__fünc",
            AggregateByPIDSpec(
                target="foo_hh",
                agg=AggType.SUM,
                source="bär",
                p_id_to_aggregate_by="foreign_id_col",
            ),
            {"foo__bär": policy_function(leaf_name="bär")(lambda x: x)},
            {},
            "foreign_id_col",
            ["foo", "foreign_id_col"],
            ["foreign_id_col", "foo__bär"],
        ),
        (
            "foo",
            AggregateByPIDSpec(
                target="foo_hh",
                agg=AggType.SUM,
                source="x",
                p_id_to_aggregate_by="foreign_id_col",
            ),
            {},
            {"x": x},
            "foreign_id_col",
            ["foo", "foreign_id_col", "x"],
            ["foreign_id_col", "x"],
        ),
    ],
)
def test_function_arguments_are_namespaced_for_derived_p_id_funcs(
    aggregation_target,
    aggregation_spec,
    functions,
    inputs,
    group_by_id,
    top_level_namespace,
    expected_arg_names,
):
    result = _create_one_aggregation_function(
        aggregation_target=aggregation_target,
        aggregation_spec=aggregation_spec,
        aggregation_type="p_id",
        group_by_id=group_by_id,
        functions=functions,
        inputs=inputs,
        top_level_namespace=top_level_namespace,
    )
    assert all(
        arg_name in inspect.signature(result).parameters
        for arg_name in expected_arg_names
    )


@pytest.mark.parametrize(
    (
        "aggregation_target",
        "aggregation_spec",
        "functions",
        "inputs",
        "group_by_id",
        "top_level_namespace",
        "source_col_name",
    ),
    [
        (
            "foo_hh",
            AggregateByGroupSpec(target="foo_hh", agg=AggType.SUM, source="foo"),
            {},
            {"foo": policy_function(leaf_name="foo")(lambda x: x)},
            "hh_id",
            ["foo", "hh_id"],
            "foo",
        ),
        (
            "foo__bar_hh",
            AggregateByGroupSpec(target="bar_hh", agg=AggType.SUM, source="bar"),
            {},
            {"foo__bar": policy_function(leaf_name="bar")(lambda x: x)},
            "hh_id",
            ["foo", "hh_id"],
            "foo__bar",
        ),
    ],
)
def test_source_column_name_of_aggregate_by_group_func_is_qualified(
    aggregation_target,
    aggregation_spec,
    functions,
    inputs,
    group_by_id,
    top_level_namespace,
    source_col_name,
):
    result = _create_one_aggregation_function(
        aggregation_target=aggregation_target,
        aggregation_spec=aggregation_spec,
        aggregation_type="group",
        group_by_id=group_by_id,
        functions=functions,
        inputs=inputs,
        top_level_namespace=top_level_namespace,
    )
    assert result.source == source_col_name


@pytest.mark.parametrize(
    (
        "aggregation_target",
        "aggregation_spec",
        "functions",
        "inputs",
        "top_level_namespace",
        "source_col_name",
    ),
    [
        (
            "foo",
            AggregateByPIDSpec(
                target="foo_hh",
                agg=AggType.SUM,
                source="bar",
                p_id_to_aggregate_by="foreign_id_col",
            ),
            {},
            {"bar": policy_function(leaf_name="bar")(lambda x: x)},
            ["foo", "foreign_id_col", "bar"],
            "bar",
        ),
        (
            "foo__fünc",
            AggregateByPIDSpec(
                target="foo_hh",
                agg=AggType.SUM,
                source="bär",
                p_id_to_aggregate_by="foreign_id_col",
            ),
            {},
            {"foo__bär": policy_function(leaf_name="bär")(lambda x: x)},
            ["foo", "foreign_id_col"],
            "foo__bär",
        ),
    ],
)
def test_source_column_name_of_aggregate_by_p_id_func_is_qualified(
    aggregation_target,
    aggregation_spec,
    functions,
    inputs,
    top_level_namespace,
    source_col_name,
):
    result = _create_one_aggregation_function(
        aggregation_target=aggregation_target,
        aggregation_spec=aggregation_spec,
        aggregation_type="p_id",
        group_by_id=None,
        functions=functions,
        inputs=inputs,
        top_level_namespace=top_level_namespace,
    )
    assert result.source == source_col_name


@pytest.mark.parametrize(
    (
        "target_name",
        "groupings",
        "top_level_namespace",
        "expected",
    ),
    [
        (
            "arbeitslosengeld_2__vermögen_bg",
            ("bg",),
            {"vermögen", "arbeitslosengeld_2"},
            "vermögen",
        ),
        (
            "arbeitslosengeld_2__vermögen_bg",
            ("bg",),
            {"arbeitslosengeld_2"},
            "arbeitslosengeld_2__vermögen",
        ),
    ],
)
def test_get_name_of_aggregation_source(
    target_name, groupings, top_level_namespace, expected
):
    assert (
        _get_name_of_aggregation_source(
            target_name=target_name,
            top_level_namespace=top_level_namespace,
            groupings=groupings,
        )
        == expected
    )
