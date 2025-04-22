import pandas as pd
import pytest

from ttsim.aggregation import AggType
from ttsim.automatically_added_functions import create_agg_by_group_functions
from ttsim.combine_functions import _fail_if_targets_not_in_functions
from ttsim.compute_taxes_and_transfers import compute_taxes_and_transfers
from ttsim.config import IS_JAX_INSTALLED
from ttsim.policy_environment import PolicyEnvironment
from ttsim.ttsim_objects import (
    agg_by_group_function,
    policy_function,
    policy_input,
)

if IS_JAX_INSTALLED:
    jit = True
else:
    jit = False


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


@agg_by_group_function(leaf_name="y_hh", agg_type=AggType.SUM)
def y_hh(hh_id: int, x: int) -> int:
    pass


@agg_by_group_function(leaf_name="y_hh", agg_type=AggType.SUM)
def y_hh_namespaced_input(hh_id: int, inputs__x: int) -> int:
    pass


@pytest.fixture
@policy_function(leaf_name="bar")
def function_with_int_return(x: int) -> int:
    return x


@pytest.fixture
@policy_function(leaf_name="baz")
def function_with_float_return(x: int) -> float:
    return x


def some_x(x):
    return x


def return_x_hh(x_hh: int) -> int:
    return x_hh


def return_y_hh(y_hh: int) -> int:
    return y_hh


def return_n1__x_hh(n1__x_hh: int) -> int:
    return n1__x_hh


@pytest.mark.parametrize(
    (
        "objects_tree",
        "targets_tree",
        "data_tree",
    ),
    [
        (
            # Aggregations derived from simple function arguments
            {
                "hh_id": hh_id,
                "p_id": p_id,
                "n1": {
                    "f": policy_function(leaf_name="f")(return_n1__x_hh),
                    "x": x,
                },
            },
            {"n1": {"f": None}},
            {
                "n1": {"x": pd.Series([1, 1, 1])},
                "hh_id": pd.Series([0, 0, 0]),
                "p_id": pd.Series([0, 1, 2]),
            },
        ),
        (
            # Aggregations derived from namespaced function arguments
            {
                "hh_id": hh_id,
                "p_id": p_id,
                "n1": {"f": policy_function(leaf_name="f")(return_x_hh), "x": x},
            },
            {"n1": {"f": None}},
            {
                "n1": {"x": pd.Series([1, 1, 1])},
                "hh_id": pd.Series([0, 0, 0]),
                "p_id": pd.Series([0, 1, 2]),
            },
        ),
        (
            # Aggregations derived from target
            {
                "hh_id": hh_id,
                "p_id": p_id,
                "n1": {
                    "f": policy_function(leaf_name="f")(some_x),
                    "x": x,
                },
            },
            {"n1": {"f_hh": None}},
            {
                "n1": {"x": pd.Series([1, 1, 1])},
                "hh_id": pd.Series([0, 0, 0]),
                "p_id": pd.Series([0, 1, 2]),
            },
        ),
        (
            # Explicit aggregation via objects tree with leaf name input
            {
                "hh_id": hh_id,
                "p_id": p_id,
                "n1": {
                    "f": policy_function(leaf_name="f")(some_x),
                    "x": x,
                },
                "y_hh": y_hh,
            },
            {"n1": {"f": None}},
            {
                "n1": {"x": pd.Series([1, 1, 1])},
                "hh_id": pd.Series([0, 0, 0]),
                "p_id": pd.Series([0, 1, 2]),
            },
        ),
        (
            # Explicit aggregation via objects tree with namespaced input
            {
                "hh_id": hh_id,
                "p_id": p_id,
                "n1": {
                    "f": policy_function(leaf_name="f")(return_y_hh),
                    "y_hh": y_hh_namespaced_input,
                },
                "inputs": {"x": x},
            },
            {"n1": {"f": None}},
            {
                "inputs": {"x": pd.Series([1, 1, 1])},
                "hh_id": pd.Series([0, 0, 0]),
                "p_id": pd.Series([0, 1, 2]),
            },
        ),
    ],
)
def test_create_agg_by_group_functions(
    objects_tree,
    targets_tree,
    data_tree,
):
    environment = PolicyEnvironment(raw_objects_tree=objects_tree)
    compute_taxes_and_transfers(
        environment=environment,
        data_tree=data_tree,
        targets_tree=targets_tree,
        groupings=("hh",),
        jit=jit,
    )


@pytest.mark.parametrize(
    "functions, targets, expected_error_match",
    [
        ({"foo": some_x}, {"bar": None}, "('bar',)"),
        ({"foo__baz": some_x}, {"foo__bar": None}, "('foo', 'bar')"),
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
        "targets",
        "data",
        "expected",
    ),
    [
        (
            {"foo": policy_function(leaf_name="foo")(return_x_hh)},
            {},
            {"x": pd.Series([1])},
            ("x_hh"),
        ),
        (
            {"n2__foo": policy_function(leaf_name="foo")(return_n1__x_hh)},
            {},
            {"n1__x": pd.Series([1])},
            ("n1__x_hh"),
        ),
        (
            {},
            {"x_hh": None},
            {"x": pd.Series([1])},
            ("x_hh"),
        ),
    ],
)
def test_derived_aggregation_functions_are_in_correct_namespace(
    functions,
    targets,
    data,
    expected,
):
    """Test that the derived aggregation functions are in the correct namespace.

    The namespace of the derived aggregation functions should be the same as the
    namespace of the function that is being aggregated.
    """
    result = create_agg_by_group_functions(
        ttsim_functions_with_time_conversions=functions,
        data=data,
        targets=targets,
        groupings=("hh",),
    )
    assert expected in result


# @pytest.mark.parametrize(
#     (
#         "aggregation_target",
#         "aggregation_spec",
#         "functions",
#         "policy_inputs",
#         "group_by_id",
#         "top_level_namespace",
#         "expected_arg_names",
#     ),
#     [
#         (
#             "foo",
#             AggregateByPIDSpec(
#                 target="foo_hh",
#                 agg=AggType.SUM,
#                 source="bar",
#                 p_id_to_aggregate_by="foreign_id_col",
#             ),
#             {"bar": policy_function(leaf_name="bar")(lambda x: x)},
#             {},
#             "foreign_id_col",
#             ["foo", "foreign_id_col", "bar"],
#             ["foreign_id_col", "bar"],
#         ),
#         (
#             "foo__fünc",
#             AggregateByPIDSpec(
#                 target="foo_hh",
#                 agg=AggType.SUM,
#                 source="bär",
#                 p_id_to_aggregate_by="foreign_id_col",
#             ),
#             {"foo__bär": policy_function(leaf_name="bär")(lambda x: x)},
#             {},
#             "foreign_id_col",
#             ["foo", "foreign_id_col"],
#             ["foreign_id_col", "foo__bär"],
#         ),
#         (
#             "foo",
#             AggregateByPIDSpec(
#                 target="foo_hh",
#                 agg=AggType.SUM,
#                 source="x",
#                 p_id_to_aggregate_by="foreign_id_col",
#             ),
#             {},
#             {"x": x},
#             "foreign_id_col",
#             ["foo", "foreign_id_col", "x"],
#             ["foreign_id_col", "x"],
#         ),
#     ],
# )
# def test_function_arguments_are_namespaced_for_derived_p_id_funcs(
#     aggregation_target,
#     aggregation_spec,
#     functions,
#     policy_inputs,
#     group_by_id,
#     top_level_namespace,
#     expected_arg_names,
# ):
#     result = _create_one_aggregation_function(
#         aggregation_target=aggregation_target,
#         aggregation_spec=aggregation_spec,
#         aggregation_type="p_id",
#         group_by_id=group_by_id,
#         functions=functions,
#         policy_inputs=policy_inputs,
#         top_level_namespace=top_level_namespace,
#     )
#     assert all(
#         arg_name in inspect.signature(result).parameters
#         for arg_name in expected_arg_names
#     )


# @pytest.mark.parametrize(
#     (
#         "aggregation_target",
#         "aggregation_spec",
#         "functions",
#         "policy_inputs",
#         "group_by_id",
#         "top_level_namespace",
#         "source_col_name",
#     ),
#     [
#         (
#             "foo_hh",
#             AggregateByGroupSpec(target="foo_hh", agg=AggType.SUM, source="foo"),
#             {},
#             {"foo": policy_function(leaf_name="foo")(lambda x: x)},
#             "hh_id",
#             ["foo", "hh_id"],
#             "foo",
#         ),
#         (
#             "foo__bar_hh",
#             AggregateByGroupSpec(target="bar_hh", agg=AggType.SUM, source="bar"),
#             {},
#             {"foo__bar": policy_function(leaf_name="bar")(lambda x: x)},
#             "hh_id",
#             ["foo", "hh_id"],
#             "foo__bar",
#         ),
#     ],
# )
# def test_source_column_name_of_aggregate_by_group_func_is_qualified(
#     aggregation_target,
#     aggregation_spec,
#     functions,
#     policy_inputs,
#     group_by_id,
#     top_level_namespace,
#     source_col_name,
# ):
#     result = _create_one_aggregation_function(
#         aggregation_target=aggregation_target,
#         aggregation_spec=aggregation_spec,
#         aggregation_type="group",
#         group_by_id=group_by_id,
#         functions=functions,
#         policy_inputs=policy_inputs,
#         top_level_namespace=top_level_namespace,
#     )
#     assert result.source == source_col_name


# @pytest.mark.parametrize(
#     (
#         "aggregation_target",
#         "aggregation_spec",
#         "functions",
#         "policy_inputs",
#         "top_level_namespace",
#         "source_col_name",
#     ),
#     [
#         (
#             "foo",
#             AggregateByPIDSpec(
#                 target="foo_hh",
#                 agg=AggType.SUM,
#                 source="bar",
#                 p_id_to_aggregate_by="foreign_id_col",
#             ),
#             {},
#             {"bar": policy_function(leaf_name="bar")(lambda x: x)},
#             ["foo", "foreign_id_col", "bar"],
#             "bar",
#         ),
#         (
#             "foo__fünc",
#             AggregateByPIDSpec(
#                 target="foo_hh",
#                 agg=AggType.SUM,
#                 source="bär",
#                 p_id_to_aggregate_by="foreign_id_col",
#             ),
#             {},
#             {"foo__bär": policy_function(leaf_name="bär")(lambda x: x)},
#             ["foo", "foreign_id_col"],
#             "foo__bär",
#         ),
#     ],
# )
# def test_source_column_name_of_aggregate_by_p_id_func_is_qualified(
#     aggregation_target,
#     aggregation_spec,
#     functions,
#     policy_inputs,
#     top_level_namespace,
#     source_col_name,
# ):
#     result = _create_one_aggregation_function(
#         aggregation_target=aggregation_target,
#         aggregation_spec=aggregation_spec,
#         aggregation_type="p_id",
#         group_by_id=None,
#         functions=functions,
#         policy_inputs=policy_inputs,
#         top_level_namespace=top_level_namespace,
#     )
#     assert result.source == source_col_name
