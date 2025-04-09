import datetime
import inspect

import pandas as pd
import pytest

from ttsim.aggregation import AggregateByGroupSpec, AggregateByPIDSpec, AggregationType
from ttsim.combine_functions import (
    _annotate_aggregation_functions,
    _create_aggregate_by_group_functions,
    _create_aggregation_functions,
    _create_one_aggregate_by_group_func,
    _create_one_aggregate_by_p_id_func,
    _fail_if_targets_not_in_functions,
    _get_name_of_aggregation_source,
)
from ttsim.compute_taxes_and_transfers import compute_taxes_and_transfers
from ttsim.function_types import (
    DerivedAggregationFunction,
    group_by_function,
    policy_function,
    policy_input,
)
from ttsim.policy_environment import PolicyEnvironment


@pytest.fixture
@policy_function(leaf_name="foo")
def function_with_bool_return(x: bool) -> bool:
    return x


@policy_input()
def x() -> bool:
    pass


@pytest.fixture
@policy_function(leaf_name="bar")
def function_with_int_return(x: int) -> int:
    return x


@pytest.fixture
@policy_function(leaf_name="baz")
def function_with_float_return(x: int) -> float:
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
                "namespace1": {
                    "f": policy_function(leaf_name="f")(lambda x_hh: x_hh),
                    "x": x,
                }
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
                "namespace1": {
                    "f": policy_function(leaf_name="f")(
                        lambda inputs__x_hh: inputs__x_hh
                    )
                }
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
            {"namespace1": {"f": policy_function(leaf_name="f")(lambda x: x)}},
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
            {"namespace1": {"f": policy_function(leaf_name="f")(lambda y_hh: y_hh)}},
            {"namespace1": {"f": None}},
            {
                "namespace1": {"x": pd.Series([1, 1, 1])},
                "hh_id": pd.Series([0, 0, 0]),
                "p_id": pd.Series([0, 1, 2]),
            },
            {
                "namespace1": {
                    "y_hh": AggregateByGroupSpec(
                        source="x",
                        aggr=AggregationType.SUM,
                    ),
                },
            },
        ),
        (
            # Aggregations derived from namespaced environment specification
            {"namespace1": {"f": policy_function(leaf_name="f")(lambda y_hh: y_hh)}},
            {"namespace1": {"f": None}},
            {
                "inputs": {"x": pd.Series([1, 1, 1])},
                "hh_id": pd.Series([0, 0, 0]),
                "p_id": pd.Series([0, 1, 2]),
            },
            {
                "namespace1": {
                    "y_hh": AggregateByGroupSpec(
                        source="inputs__x",
                        aggr=AggregationType.SUM,
                    ),
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
    environment = PolicyEnvironment(
        raw_objects_tree=objects_tree,
        aggregation_specs_tree=aggregations_specs_from_env,
    )
    compute_taxes_and_transfers(
        environment=environment,
        data_tree=data_tree,
        targets_tree=targets_tree,
    )


START_DATE = datetime.date.fromisoformat("1900-01-01")
END_DATE = datetime.date.fromisoformat("2100-12-31")


@pytest.mark.parametrize(
    (
        "objects",
        "aggregation_functions",
        "expected_return_type",
    ),
    [
        (
            {},
            {
                "foo": DerivedAggregationFunction(
                    leaf_name="foo",
                    function=lambda x: x,
                    source="x",
                    aggregation_method="count",
                    start_date=START_DATE,
                    end_date=END_DATE,
                )
            },
            int,
        ),
        (
            {},
            {
                "foo": DerivedAggregationFunction(
                    leaf_name="foo",
                    function=lambda x: x,
                    source="x",
                    aggregation_method="sum",
                    start_date=START_DATE,
                    end_date=END_DATE,
                ),
                "x": int,
            },
            int,
        ),
        (
            {},
            {
                "foo": DerivedAggregationFunction(
                    leaf_name="foo",
                    function=lambda x: x,
                    source="x",
                    aggregation_method="sum",
                    start_date=START_DATE,
                    end_date=END_DATE,
                ),
                "x": float,
            },
            float,
        ),
        (
            {},
            {
                "foo": DerivedAggregationFunction(
                    leaf_name="foo",
                    function=lambda x: x,
                    source="x",
                    aggregation_method="sum",
                    start_date=START_DATE,
                    end_date=END_DATE,
                ),
                "x": bool,
            },
            int,
        ),
        (
            {"n1__foo": function_with_bool_return},
            {
                "n1__foo_hh": DerivedAggregationFunction(
                    leaf_name="foo_hh",
                    function=function_with_bool_return,
                    source="n1__foo",
                    aggregation_method="sum",
                    start_date=START_DATE,
                    end_date=END_DATE,
                )
            },
            int,
        ),
        (
            {"n1__foo": function_with_float_return},
            {
                "n1__foo_hh": DerivedAggregationFunction(
                    leaf_name="foo_hh",
                    function=function_with_float_return,
                    source="n1__foo",
                    aggregation_method="sum",
                    start_date=START_DATE,
                    end_date=END_DATE,
                )
            },
            float,
        ),
        (
            {"n1__foo": function_with_int_return},
            {
                "n1__foo_hh": DerivedAggregationFunction(
                    leaf_name="foo_hh",
                    function=function_with_int_return,
                    source="n1__foo",
                    aggregation_method="sum",
                    start_date=START_DATE,
                    end_date=END_DATE,
                )
            },
            int,
        ),
    ],
)
def test_annotations_for_aggregation(
    objects,
    aggregation_functions,
    expected_return_type,
):
    name_of_aggregation_function = next(iter(aggregation_functions.keys()))
    annotation_of_aggregation_function = _annotate_aggregation_functions(
        ttsim_objects=objects,
        aggregation_functions=aggregation_functions,
    )[name_of_aggregation_function].__annotations__["return"]
    assert annotation_of_aggregation_function == expected_return_type


@pytest.mark.parametrize(
    "functions, targets, expected_error_match",
    [
        ({"foo": lambda x: x}, {"bar": None}, "('bar',)"),
        ({"foo__baz": lambda x: x}, {"foo__bar": None}, "('foo', 'bar')"),
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
        "aggregations",
        "aggregation_type",
        "top_level_namespace",
        "expected_annotations",
    ),
    [
        (
            {"foo": function_with_bool_return},
            {"foo_hh": AggregateByGroupSpec(source="foo", aggr=AggregationType.SUM)},
            "group",
            ["foo"],
            {"foo": bool, "return": int},
        ),
        (
            {"foo": function_with_float_return},
            {"foo_hh": AggregateByGroupSpec(source="foo", aggr=AggregationType.SUM)},
            "group",
            ["foo"],
            {"foo": float, "return": float},
        ),
        (
            {"foo": function_with_int_return},
            {
                "foo_hh": AggregateByPIDSpec(
                    p_id_to_aggregate_by="foreign_id_col",
                    source="foo",
                    aggr=AggregationType.SUM,
                )
            },
            "p_id",
            ["foo"],
            {"foo": int, "return": int},
        ),
    ],
)
def test_annotations_are_applied_to_derived_functions(
    functions, aggregations, aggregation_type, top_level_namespace, expected_annotations
):
    """Test that the annotations are applied to the derived functions."""
    result_func = next(
        iter(
            _create_aggregation_functions(
                ttsim_objects=functions,
                aggregation_functions_to_create=aggregations,
                aggregation_type=aggregation_type,
                top_level_namespace=top_level_namespace,
            ).values()
        )
    )
    assert result_func.__annotations__ == expected_annotations


@pytest.mark.parametrize(
    (
        "functions",
        "targets",
        "data",
        "aggregations_from_environment",
        "top_level_namespace",
        "expected",
    ),
    [
        (
            {"foo": policy_function(leaf_name="foo")(lambda x_hh: x_hh)},
            {},
            {"x": pd.Series([1])},
            {},
            ["foo", "x"],
            ("x_hh"),
        ),
        (
            {"n1__foo": policy_function(leaf_name="foo")(lambda n2__x_hh: n2__x_hh)},
            {},
            {"n2": {"x": pd.Series([1])}},
            {},
            ["n1", "n2"],
            ("n2__x_hh"),
        ),
        (
            {},
            {"x_hh": None},
            {"x": pd.Series([1])},
            {},
            ["x"],
            ("x_hh"),
        ),
        (
            {"foo": policy_function(leaf_name="foo")(lambda x: x)},
            {},
            {"x": pd.Series([1])},
            {
                "n1__foo_hh": AggregateByGroupSpec(
                    source="foo", aggr=AggregationType.SUM
                )
            },
            ["x", "foo", "n1"],
            ("n1__foo_hh"),
        ),
    ],
)
def test_derived_aggregation_functions_are_in_correct_namespace(
    functions,
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
        targets=targets,
        data=data,
        aggregations_from_environment=aggregations_from_environment,
        top_level_namespace=top_level_namespace,
    )
    assert expected in result


def test_create_aggregation_with_derived_soure_column():
    aggregation_spec_dict = {
        "foo_hh": AggregateByGroupSpec(
            source="bar_bg",
            aggr=AggregationType.SUM,
        )
    }
    result = _create_aggregate_by_group_functions(
        functions={"bg_id": group_by_function()(lambda x: x)},
        targets={},
        data={"bar": pd.Series([1])},
        aggregations_from_environment=aggregation_spec_dict,
        top_level_namespace=["foo", "bar", "bg_id"],
    )
    assert "foo_hh" in result
    assert "bar_bg" in inspect.signature(result["foo_hh"]).parameters


@pytest.mark.parametrize(
    (
        "aggregation_target",
        "aggregation_spec",
        "group_by_id",
        "top_level_namespace",
        "expected_arg_names",
    ),
    [
        (
            "foo_hh",
            AggregateByGroupSpec(aggr=AggregationType.COUNT),
            "hh_id",
            ["foo", "hh_id"],
            ["hh_id"],
        ),
        (
            "foo_hh",
            AggregateByGroupSpec(aggr=AggregationType.SUM, source="foo"),
            "hh_id",
            ["foo", "hh_id"],
            ["hh_id", "foo"],
        ),
        (
            "foo__bar_hh",
            AggregateByGroupSpec(aggr=AggregationType.SUM, source="bar"),
            "hh_id",
            ["foo", "hh_id"],
            ["hh_id", "foo__bar"],
        ),
    ],
)
def test_function_arguments_are_namespaced_for_derived_group_funcs(
    aggregation_target,
    aggregation_spec,
    group_by_id,
    top_level_namespace,
    expected_arg_names,
):
    result = _create_one_aggregate_by_group_func(
        aggregation_target=aggregation_target,
        aggregation_spec=aggregation_spec,
        group_by_id=group_by_id,
        ttsim_objects={},
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
        "top_level_namespace",
        "expected_arg_names",
    ),
    [
        (
            "foo",
            AggregateByPIDSpec(
                aggr=AggregationType.SUM,
                source="bar",
                p_id_to_aggregate_by="foreign_id_col",
            ),
            ["foo", "foreign_id_col", "bar"],
            ["foreign_id_col", "bar"],
        ),
        (
            "foo__fünc",
            AggregateByPIDSpec(
                aggr=AggregationType.SUM,
                source="bär",
                p_id_to_aggregate_by="foreign_id_col",
            ),
            ["foo", "foreign_id_col"],
            ["foreign_id_col", "foo__bär"],
        ),
    ],
)
def test_function_arguments_are_namespaced_for_derived_p_id_funcs(
    aggregation_target,
    aggregation_spec,
    top_level_namespace,
    expected_arg_names,
):
    result = _create_one_aggregate_by_p_id_func(
        aggregation_target=aggregation_target,
        aggregation_spec=aggregation_spec,
        functions={},
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
        "group_by_id",
        "top_level_namespace",
        "source_col_name",
    ),
    [
        (
            "foo_hh",
            AggregateByGroupSpec(aggr=AggregationType.SUM, source="foo"),
            "hh_id",
            ["foo", "hh_id"],
            "foo",
        ),
        (
            "foo__bar_hh",
            AggregateByGroupSpec(aggr=AggregationType.SUM, source="bar"),
            "hh_id",
            ["foo", "hh_id"],
            "foo__bar",
        ),
    ],
)
def test_source_column_name_of_aggregate_by_group_func_is_qualified(
    aggregation_target,
    aggregation_spec,
    group_by_id,
    top_level_namespace,
    source_col_name,
):
    result = _create_one_aggregate_by_group_func(
        aggregation_target=aggregation_target,
        aggregation_spec=aggregation_spec,
        group_by_id=group_by_id,
        ttsim_objects={},
        top_level_namespace=top_level_namespace,
    )
    assert result.source == source_col_name


@pytest.mark.parametrize(
    (
        "aggregation_target",
        "aggregation_spec",
        "top_level_namespace",
        "source_col_name",
    ),
    [
        (
            "foo",
            AggregateByPIDSpec(
                aggr=AggregationType.SUM,
                source="bar",
                p_id_to_aggregate_by="foreign_id_col",
            ),
            ["foo", "foreign_id_col", "bar"],
            "bar",
        ),
        (
            "foo__fünc",
            AggregateByPIDSpec(
                aggr=AggregationType.SUM,
                source="bär",
                p_id_to_aggregate_by="foreign_id_col",
            ),
            ["foo", "foreign_id_col"],
            "foo__bär",
        ),
    ],
)
def test_source_column_name_of_aggregate_by_p_id_func_is_qualified(
    aggregation_target,
    aggregation_spec,
    top_level_namespace,
    source_col_name,
):
    result = _create_one_aggregate_by_p_id_func(
        aggregation_target=aggregation_target,
        aggregation_spec=aggregation_spec,
        functions={},
        top_level_namespace=top_level_namespace,
    )
    assert result.source == source_col_name


@pytest.mark.parametrize(
    (
        "target_name",
        "top_level_namespace",
        "expected",
    ),
    [
        (
            "arbeitslosengeld_2__vermögen_bg",
            {"vermögen", "arbeitslosengeld_2"},
            "vermögen",
        ),
        (
            "arbeitslosengeld_2__vermögen_bg",
            {"arbeitslosengeld_2"},
            "arbeitslosengeld_2__vermögen",
        ),
    ],
)
def test_get_name_of_aggregation_source(target_name, top_level_namespace, expected):
    assert (
        _get_name_of_aggregation_source(
            target_name=target_name,
            top_level_namespace=top_level_namespace,
        )
        == expected
    )
