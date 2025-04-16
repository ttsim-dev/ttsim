import copy
import re
import warnings

import numpy
import pandas as pd
import pytest
from mettsim.config import SUPPORTED_GROUPINGS

from ttsim.aggregation import AggregateByGroupSpec, AggType
from ttsim.automatically_added_functions import TIME_UNITS
from ttsim.compute_taxes_and_transfers import (
    FunctionsAndColumnsOverlapWarning,
    _fail_if_foreign_keys_are_invalid_in_data,
    _fail_if_group_variables_not_constant_within_groups,
    _fail_if_p_id_is_non_unique,
    _get_top_level_namespace,
    _partial_parameters_to_functions,
    compute_taxes_and_transfers,
)
from ttsim.config import numpy_or_jax as np
from ttsim.policy_environment import PolicyEnvironment, set_up_policy_environment
from ttsim.shared import assert_valid_ttsim_pytree, merge_trees
from ttsim.ttsim_objects import (
    agg_by_group_function,
    agg_by_p_id_function,
    group_creation_function,
    policy_function,
    policy_input,
)


@policy_input()
def p_id() -> int:
    pass


@policy_input()
def p_id_someone_else() -> int:
    pass


@policy_input()
def hh_id() -> int:
    pass


@policy_input()
def betrag_m() -> float:
    pass


@pytest.fixture(scope="module")
def minimal_input_data():
    n_individuals = 5
    out = {
        "p_id": pd.Series(numpy.arange(n_individuals), name="p_id"),
        "hh_id": pd.Series(numpy.arange(n_individuals), name="hh_id"),
    }
    return out


@pytest.fixture(scope="module")
def minimal_input_data_shared_hh():
    n_individuals = 3
    out = {
        "p_id": pd.Series(numpy.arange(n_individuals), name="p_id"),
        "hh_id": pd.Series([0, 0, 1], name="hh_id"),
        "p_id_someone_else": pd.Series([1, 0, -1], name="p_id_someone_else"),
    }
    return out


@agg_by_group_function(agg_type=AggType.SUM)
def foo_hh(foo: int, hh_id: int) -> int:
    pass


@pytest.fixture(scope="module")
def mettsim_environment():
    return set_up_policy_environment(
        policy_inputs=mettsim_policy_inputs,
        groupings=SUPPORTED_GROUPINGS,
        supported_time_conversions=TIME_UNITS,
    )


# Create a function which is used by some tests below
@policy_function()
def func_before_partial(arg_1, payroll_tax_params):
    return arg_1 + payroll_tax_params["test_param_1"]


func_after_partial = _partial_parameters_to_functions(
    {"test_func": func_before_partial},
    {"payroll_tax": {"test_param_1": 1}},
)["test_func"]


def test_output_as_tree(minimal_input_data):
    environment = PolicyEnvironment(
        {
            "p_id": p_id,
            "module": {
                "test_func": policy_function(leaf_name="test_func")(lambda p_id: p_id)
            },
        }
    )

    out = compute_taxes_and_transfers(
        data_tree=minimal_input_data,
        environment=environment,
        targets_tree={"module": {"test_func": None}},
        groupings=("hh",),
    )

    assert isinstance(out, dict)
    assert "test_func" in out["module"]
    assert isinstance(out["module"]["test_func"], np.ndarray)


def test_warn_if_functions_and_columns_overlap():
    environment = PolicyEnvironment(
        {
            "dupl": policy_function(leaf_name="dupl")(lambda x: x),
            "some_target": policy_function(leaf_name="some_target")(lambda dupl: dupl),
        }
    )
    with pytest.warns(FunctionsAndColumnsOverlapWarning):
        compute_taxes_and_transfers(
            data_tree={
                "p_id": pd.Series([0]),
                "dupl": pd.Series([1]),
            },
            environment=environment,
            targets_tree={"some_target": None},
            groupings=("hh",),
        )


def test_dont_warn_if_functions_and_columns_dont_overlap():
    environment = PolicyEnvironment(
        {"some_func": policy_function(leaf_name="some_func")(lambda x: x)}
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=FunctionsAndColumnsOverlapWarning)
        compute_taxes_and_transfers(
            data_tree={
                "p_id": pd.Series([0]),
                "x": pd.Series([1]),
            },
            environment=environment,
            targets_tree={"some_func": None},
            groupings=("hh",),
        )


def test_recipe_to_ignore_warning_if_functions_and_columns_overlap():
    environment = PolicyEnvironment(
        {
            "dupl": policy_function(leaf_name="dupl")(lambda x: x),
            "unique": policy_function(leaf_name="unique")(lambda x: x**2),
        }
    )
    with warnings.catch_warnings(
        category=FunctionsAndColumnsOverlapWarning, record=True
    ) as warning_list:
        warnings.filterwarnings("ignore", category=FunctionsAndColumnsOverlapWarning)
        compute_taxes_and_transfers(
            data_tree={
                "p_id": pd.Series([0]),
                "dupl": pd.Series([1]),
                "x": pd.Series([1]),
            },
            environment=environment,
            targets_tree={"unique": None},
            groupings=("hh",),
        )

    assert len(warning_list) == 0


def test_fail_if_p_id_does_not_exist():
    data = {"hh_id": pd.Series(data=numpy.arange(8), name="hh_id")}

    with pytest.raises(ValueError):
        _fail_if_p_id_is_non_unique(data)


def test_fail_if_p_id_is_non_unique():
    data = {"p_id": pd.Series(data=numpy.arange(4).repeat(2), name="p_id")}

    with pytest.raises(ValueError):
        _fail_if_p_id_is_non_unique(data)


@pytest.mark.skip
def test_fail_if_foreign_key_points_to_non_existing_p_id(mettsim_environment):
    policy_inputs = mettsim_environment.policy_inputs
    data = {
        "p_id": pd.Series([1, 2, 3]),
        "p_id_spouse": pd.Series([0, 1, 2]),
    }

    with pytest.raises(ValueError, match=r"not a valid p_id in the\sinput data"):
        _fail_if_foreign_keys_are_invalid_in_data(
            data=data, policy_inputs=policy_inputs
        )


@pytest.mark.skip
def test_allow_minus_one_as_foreign_key(mettsim_environment):
    policy_inputs = mettsim_environment.policy_inputs
    data = {
        "p_id": pd.Series([1, 2, 3]),
        "p_id_spouse": pd.Series([-1, 1, 2]),
    }

    _fail_if_foreign_keys_are_invalid_in_data(data=data, policy_inputs=policy_inputs)


@pytest.mark.skip
def test_fail_if_foreign_key_points_to_same_row_if_not_allowed(mettsim_environment):
    policy_inputs = mettsim_environment.policy_inputs
    data = {
        "p_id": pd.Series([1, 2, 3]),
        "child_tax_credit__p_id_recipient": pd.Series([1, 3, 3]),
    }

    _fail_if_foreign_keys_are_invalid_in_data(data=data, policy_inputs=policy_inputs)


@pytest.mark.skip
def test_fail_if_foreign_key_points_to_same_row_if_allowed(mettsim_environment):
    policy_inputs = mettsim_environment.policy_inputs
    data = {
        "p_id": pd.Series([1, 2, 3]),
        "p_id_child_": pd.Series([1, 3, 3]),
    }

    _fail_if_foreign_keys_are_invalid_in_data(data=data, policy_inputs=policy_inputs)


@pytest.mark.parametrize(
    "data, functions",
    [
        (
            {
                "foo_hh": pd.Series([1, 2, 2], name="foo_hh"),
                "hh_id": pd.Series([1, 1, 2], name="hh_id"),
            },
            {},
        ),
        (
            {
                "foo_fam": pd.Series([1, 2, 2], name="foo_fam"),
                "fam_id": pd.Series([1, 1, 2], name="fam_id"),
            },
            {
                "fam_id": group_creation_function()(lambda x: x),
            },
        ),
    ],
)
def test_fail_if_group_variables_not_constant_within_groups(data, functions):
    with pytest.raises(ValueError):
        _fail_if_group_variables_not_constant_within_groups(
            data=data,
            functions=functions,
            groupings=SUPPORTED_GROUPINGS,
        )


def test_missing_root_nodes_raises_error(minimal_input_data):
    def b(a):
        return a

    def c(b):
        return b

    environment = PolicyEnvironment(
        {
            "b": policy_function(leaf_name="b")(b),
            "c": policy_function(leaf_name="c")(c),
        }
    )

    with pytest.raises(
        ValueError,
        match="The following data columns are missing",
    ):
        compute_taxes_and_transfers(
            data_tree=minimal_input_data,
            environment=environment,
            targets_tree={"c": None},
            groupings=("hh",),
        )


def test_function_without_data_dependency_is_not_mistaken_for_data(minimal_input_data):
    @policy_function(leaf_name="a")
    def a():
        return pd.Series(range(minimal_input_data["p_id"].size))

    @policy_function(leaf_name="b")
    def b(a):
        return a

    environment = PolicyEnvironment({"a": a, "b": b})
    compute_taxes_and_transfers(
        data_tree=minimal_input_data,
        environment=environment,
        targets_tree={"b": None},
        groupings=("hh",),
    )


def test_fail_if_targets_are_not_in_functions_or_in_columns_overriding_functions(
    minimal_input_data,
):
    environment = PolicyEnvironment({})

    with pytest.raises(
        ValueError,
        match="The following targets have no corresponding function",
    ):
        compute_taxes_and_transfers(
            data_tree=minimal_input_data,
            environment=environment,
            targets_tree={"unknown_target": None},
            groupings=("hh",),
        )


def test_fail_if_missing_p_id():
    data = {"hh_id": pd.Series([1, 2, 3], name="hh_id")}
    with pytest.raises(
        ValueError,
        match="The input data must contain the p_id",
    ):
        compute_taxes_and_transfers(
            data_tree=data,
            environment=PolicyEnvironment({}),
            targets_tree={},
            groupings=("hh",),
        )


def test_fail_if_non_unique_p_id(minimal_input_data):
    data = copy.deepcopy(minimal_input_data)
    data["p_id"][:] = 1

    with pytest.raises(
        ValueError,
        match="The following p_ids are non-unique",
    ):
        compute_taxes_and_transfers(
            data_tree=data,
            environment=PolicyEnvironment({}),
            targets_tree={},
            groupings=("hh",),
        )


def test_partial_parameters_to_functions():
    # Partial function produces correct result
    assert func_after_partial(2) == 3


def test_partial_parameters_to_functions_removes_argument():
    # Fails if params is added to partial function
    with pytest.raises(
        TypeError,
        match=("got multiple values for argument "),
    ):
        func_after_partial(2, {"test_param_1": 1})

    # No error for original function
    func_before_partial(2, {"test_param_1": 1})


def test_user_provided_aggregate_by_group_specs():
    data = {
        "p_id": pd.Series([1, 2, 3], name="p_id"),
        "hh_id": pd.Series([1, 1, 2], name="hh_id"),
        "module_name": {
            "betrag_m": pd.Series([100, 100, 100], name="betrag_m"),
        },
    }

    inputs = {
        "p_id": p_id,
        "hh_id": hh_id,
        "module_name": {
            "betrag_m": betrag_m,
        },
    }

    aggregation_specs_tree = {
        "module_name": (
            AggregateByGroupSpec(
                target="betrag_m_hh",
                source="betrag_m",
                agg=AggType.SUM,
            ),
        )
    }
    expected_res = pd.Series([200, 200, 100])

    out = compute_taxes_and_transfers(
        data_tree=data,
        environment=PolicyEnvironment(raw_objects_tree=inputs),
        targets_tree={"module_name": {"betrag_m_hh": None}},
        groupings=("hh",),
    )

    numpy.testing.assert_array_almost_equal(
        out["module_name"]["betrag_m_hh"], expected_res
    )


def test_user_provided_aggregation():
    data = {
        "p_id": pd.Series([1, 2, 3], name="p_id"),
        "hh_id": pd.Series([1, 1, 2], name="hh_id"),
        "module_name": {
            "betrag_m": pd.Series([200, 100, 100], name="betrag_m"),
        },
    }
    # Double up, then take max hh_id
    expected = pd.Series([400, 400, 200])

    @policy_function()
    def betrag_m_double(betrag_m):
        return 2 * betrag_m

    @agg_by_group_function(agg_type=AggType.MAX)
    def betrag_m_double_hh(betrag_m_double, hh_id) -> float:
        pass

    environment = PolicyEnvironment(
        {
            "p_id": p_id,
            "hh_id": hh_id,
            "module_name": {
                "betrag_m_double": betrag_m_double,
                "betrag_m_double_hh": betrag_m_double_hh,
            },
        }
    )

    actual = compute_taxes_and_transfers(
        data_tree=data,
        environment=environment,
        targets_tree={"module_name": {"betrag_m_double_hh": None}},
        groupings=("hh",),
        debug=True,
    )

    numpy.testing.assert_array_almost_equal(
        actual["module_name"]["betrag_m_double_hh"], expected
    )


def test_user_provided_aggregation_with_time_conversion():
    data = {
        "p_id": pd.Series([1, 2, 3], name="p_id"),
        "hh_id": pd.Series([1, 1, 2], name="hh_id"),
        "module_name": {
            "betrag_m": pd.Series([200, 100, 100], name="betrag_m"),
        },
    }
    # Double up, convert to quarter, then take max hh_id
    expected = pd.Series([400 * 3, 400 * 3, 200 * 3])

    @policy_function()
    def betrag_double_m(betrag_m):
        return 2 * betrag_m

    @agg_by_group_function(agg_type=AggType.MAX)
    def max_betrag_double_m_hh(betrag_double_m, hh_id) -> float:
        pass

    environment = PolicyEnvironment(
        {
            "p_id": p_id,
            "hh_id": hh_id,
            "module_name": {
                "betrag_double_m": betrag_double_m,
                "max_betrag_double_m_hh": max_betrag_double_m_hh,
            },
        }
    )

    actual = compute_taxes_and_transfers(
        data_tree=data,
        environment=environment,
        targets_tree={"module_name": {"betrag_double_q_hh": None}},
        groupings=("hh",),
        debug=True,
    )

    numpy.testing.assert_array_almost_equal(
        actual["module_name"]["max_betrag_double_q_hh"], expected
    )


def test_aggregate_by_group_specs_agg_not_impl():
    with pytest.raises(
        TypeError,
        match="agg must be of type AggType, not <class 'str'>",
    ):
        AggregateByGroupSpec(
            target="betrag_agg_m",
            source="betrag_m",
            agg="sum",
        )


@agg_by_p_id_function(agg_type=AggType.SUM)
def sum_source_by_p_id_someone_else(
    source: int, p_id: int, p_id_someone_else: int
) -> int:
    pass


@agg_by_p_id_function(agg_type=AggType.SUM)
def sum_source_m_by_p_id_someone_else(
    source_m: int, p_id: int, p_id_someone_else: int
) -> int:
    pass


@pytest.mark.parametrize(
    ("agg_functions, leaf_name, target_tree, expected"),
    [
        (
            {
                "module": {
                    "sum_source_by_p_id_someone_else": sum_source_by_p_id_someone_else
                }
            },
            "source",
            {"module": {"sum_source_by_p_id_someone_else": None}},
            pd.Series([200, 100, 0]),
        ),
        (
            {
                "module": {
                    "sum_source_m_by_p_id_someone_else": sum_source_m_by_p_id_someone_else
                }
            },
            "source_m",
            {"module": {"sum_source_m_by_p_id_someone_else": None}},
            pd.Series([200, 100, 0]),
        ),
    ],
)
def test_user_provided_aggregate_by_p_id_specs(
    agg_functions,
    leaf_name,
    target_tree,
    expected,
    minimal_input_data_shared_hh,
):
    # TODO(@MImmesberger): Remove fake dependency.
    # https://github.com/iza-institute-of-labor-economics/gettsim/issues/666
    @policy_function(leaf_name=leaf_name, vectorization_strategy="not_required")
    def source(p_id: int) -> int:  # noqa: ARG001
        return np.array([100, 200, 300])

    raw_objects_tree = merge_trees(
        agg_functions,
        {
            "module": {leaf_name: source},
            "p_id": p_id,
            "p_id_someone_else": p_id_someone_else,
        },
    )

    environment = PolicyEnvironment(raw_objects_tree=raw_objects_tree)
    out = compute_taxes_and_transfers(
        minimal_input_data_shared_hh,
        environment,
        targets_tree=target_tree,
        groupings=("hh",),
    )["module"][next(iter(target_tree["module"].keys()))]

    numpy.testing.assert_array_almost_equal(out, expected)


# @pytest.mark.parametrize(
#     "input_data, expected_type, expected_output_data",
#     [
#         (pd.Series([0, 1, 0]), bool, pd.Series([False, True, False])),
#         (pd.Series([1.0, 0.0, 1]), bool, pd.Series([True, False, True])),
#         (pd.Series([200, 550, 237]), float, pd.Series([200.0, 550.0, 237.0])),
#         (pd.Series([1.0, 4.0, 10.0]), int, pd.Series([1, 4, 10])),
#         (pd.Series([200.0, 567.0]), int, pd.Series([200, 567])),
#         (pd.Series([1.0, 0.0]), bool, pd.Series([True, False])),
#     ],
# )
# def test_convert_series_to_internal_types(
#     input_data, expected_type, expected_output_data
# ):
#     adjusted_input = convert_series_to_internal_type(input_data, expected_type)
#     pd.testing.assert_series_equal(adjusted_input, expected_output_data)


# @pytest.mark.parametrize(
#     "input_data, expected_type, error_match",
#     [
#         (
#             pd.Series(["Hallo", 200, 325]),
#             float,
#             "Conversion from input type object to float failed.",
#         ),
#         (
#             pd.Series([True, False]),
#             float,
#             "Conversion from input type bool to float failed.",
#         ),
#         (
#             pd.Series(["a", "b", "c"]).astype("category"),
#             float,
#             "Conversion from input type category to float failed.",
#         ),
#         (
#             pd.Series(["2.0", "3.0"]),
#             int,
#             "Conversion from input type object to int failed.",
#         ),
#         (
#             pd.Series([1.5, 1.0, 2.9]),
#             int,
#             "Conversion from input type float64 to int failed.",
#         ),
#         (
#             pd.Series(["a", "b", "c"]).astype("category"),
#             int,
#             "Conversion from input type category to int failed.",
#         ),
#         (
#             pd.Series([5, 2, 3]),
#             bool,
#             "Conversion from input type int64 to bool failed.",
#         ),
#         (
#             pd.Series([1.5, 1.0, 35.0]),
#             bool,
#             "Conversion from input type float64 to bool failed.",
#         ),
#         (
#             pd.Series(["a", "b", "c"]).astype("category"),
#             bool,
#             "Conversion from input type category to bool failed.",
#         ),
#         (
#             pd.Series(["richtig"]),
#             bool,
#             "Conversion from input type object to bool failed.",
#         ),
#         (
#             pd.Series(["True", "False", ""]),
#             bool,
#             "Conversion from input type object to bool failed.",
#         ),
#         (
#             pd.Series(["true"]),
#             bool,
#             "Conversion from input type object to bool failed.",
#         ),
#         (
#             pd.Series(["zweitausendzwanzig"]),
#             numpy.datetime64,
#             "Conversion from input type object to datetime64 failed.",
#         ),
#         (
#             pd.Series([True, True]),
#             numpy.datetime64,
#             "Conversion from input type bool to datetime64 failed.",
#         ),
#         (
#             pd.Series([2020]),
#             str,
#             "The internal type <class 'str'> is not yet supported.",
#         ),
#     ],
# )
# def test_fail_if_cannot_be_converted_to_internal_type(
#     input_data, expected_type, error_match
# ):
#     with pytest.raises(ValueError, match=error_match):
#         convert_series_to_internal_type(input_data, expected_type)


# @pytest.mark.skip
# @pytest.mark.parametrize(
#     "data, functions_overridden",
#     [
#         (
#             {"sp_id": pd.Series([1, 2, 3])},
#             {"sp_id": sp_id},
#         ),
#         (
#             {"fam_id": pd.Series([1, 2, 3])},
#             {"fam_id": fam_id},
#         ),
#     ],
# )
# def test_provide_endogenous_groupings(data, functions_overridden):
#     """Test whether TTSIM handles user-provided grouping IDs, which would otherwise be
#     set endogenously."""
#     _convert_data_to_correct_types(data, functions_overridden)


# @pytest.mark.skip
# @pytest.mark.parametrize(
#     "data, functions_overridden, error_match",
#     [
#         (
#             {"hh_id": pd.Series([1, 1.1, 2])},
#             {},
#             "- hh_id: Conversion from input type float64 to int",
#         ),
#         (
#             {"gondorian": pd.Series([1.1, 0.0, 1.0])},
#             {},
#             "- gondorian: Conversion from input type float64 to bool",
#         ),
#         (
#             {
#                 "hh_id": pd.Series([1.0, 2.0, 3.0]),
#                 "gondorian": pd.Series([2, 0, 1]),
#             },
#             {},
#             "- gondorian: Conversion from input type int64 to bool",
#         ),
#         (
#             {"gondorian": pd.Series(["True", "False"])},
#             {},
#             "- gondorian: Conversion from input type object to bool",
#         ),
#         (
#             {
#                 "hh_id": pd.Series([1, "1", 2]),
#                 "payroll_tax__amount": pd.Series(["2000", 3000, 4000]),
#             },
#             {},
#             "- hh_id: Conversion from input type object to int failed.",
#         ),
#     ],
# )
# def test_fail_if_cannot_be_converted_to_correct_type(
#     data, functions_overridden, error_match
# ):
#     with pytest.raises(ValueError, match=error_match):
#         _convert_data_to_correct_types(data, functions_overridden)


@pytest.mark.parametrize(
    ("tree", "leaf_checker", "err_substr"),
    [
        (
            {"a": 1, "b": 2},
            lambda leaf: leaf is None,
            "Leaf at tree[a] is invalid: got 1 of type <class 'int'>.",
        ),
        (
            {"a": None, "b": {"c": None, "d": 1}},
            lambda leaf: leaf is None,
            "Leaf at tree[b][d] is invalid: got 1 of type <class 'int'>.",
        ),
        (
            [1, 2, 3],
            lambda leaf: leaf is None,
            "tree must be a dict, got <class 'list'>.",
        ),
        (
            {1: 2},
            lambda leaf: leaf is None,
            "Key 1 in tree must be a string but got <class 'int'>.",
        ),
    ],
)
def test_assert_valid_ttsim_pytree(tree, leaf_checker, err_substr):
    with pytest.raises(TypeError, match=re.escape(err_substr)):
        assert_valid_ttsim_pytree(tree, leaf_checker, "tree")


@pytest.mark.parametrize(
    (
        "environment",
        "supported_time_conversions",
        "groupings",
        "expected",
    ),
    [
        (
            PolicyEnvironment(
                raw_objects_tree={
                    "foo_m": policy_function(leaf_name="foo_m")(lambda x: x)
                }
            ),
            ["m", "y"],
            ["hh"],
            {"foo_m", "foo_y", "foo_m_hh", "foo_y_hh"},
        ),
        (
            PolicyEnvironment(
                raw_objects_tree={"foo": policy_function(leaf_name="foo")(lambda x: x)}
            ),
            ["m", "y"],
            ["hh"],
            {"foo", "foo_hh"},
        ),
        (
            PolicyEnvironment(raw_objects_tree={"foo_hh": foo_hh}),
            ["m", "y"],
            ["hh"],
            {"foo", "foo_hh"},
        ),
    ],
)
def test_get_top_level_namespace(
    environment, supported_time_conversions, groupings, expected
):
    result = _get_top_level_namespace(
        environment=environment,
        supported_time_conversions=supported_time_conversions,
        groupings=groupings,
    )
    assert result == expected
