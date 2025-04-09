import copy
import re
import warnings

import dags.tree as dt
import numpy
import pandas as pd
import pytest
from mettsim.config import FOREIGN_KEYS
from mettsim.functions.payroll_tax.group_by_ids import fam_id, sp_id

from gettsim import FunctionsAndColumnsOverlapWarning
from ttsim.aggregation import AggregateByGroupSpec, AggregateByPIDSpec, AggregationType
from ttsim.compute_taxes_and_transfers import (
    _convert_data_to_correct_types,
    _fail_if_foreign_keys_are_invalid,
    _fail_if_group_variables_not_constant_within_groups,
    _fail_if_pid_is_non_unique,
    _partial_parameters_to_functions,
    compute_taxes_and_transfers,
)
from ttsim.config import numpy_or_jax as np
from ttsim.function_types import group_by_function, policy_function
from ttsim.policy_environment import PolicyEnvironment
from ttsim.shared import assert_valid_ttsim_pytree
from ttsim.typing import convert_series_to_internal_type


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
    }
    return out


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
            "module": {
                "test_func": policy_function(leaf_name="test_func")(lambda p_id: p_id)
            }
        }
    )

    out = compute_taxes_and_transfers(
        data_tree=minimal_input_data,
        environment=environment,
        targets_tree={"module": {"test_func": None}},
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
        )

    assert len(warning_list) == 0


def test_fail_if_pid_does_not_exist():
    data = {"hh_id": pd.Series(data=numpy.arange(8), name="hh_id")}

    with pytest.raises(ValueError):
        _fail_if_pid_is_non_unique(data)


def test_fail_if_pid_is_non_unique():
    data = {"p_id": pd.Series(data=numpy.arange(4).repeat(2), name="p_id")}

    with pytest.raises(ValueError):
        _fail_if_pid_is_non_unique(data)


@pytest.mark.parametrize("foreign_key_path", FOREIGN_KEYS)
def test_fail_if_foreign_key_points_to_non_existing_pid(foreign_key_path):
    foreign_key_name = dt.qual_name_from_tree_path(foreign_key_path)
    data = {
        foreign_key_name: pd.Series([0, 1, 4]),
        "p_id": pd.Series([1, 2, 3]),
    }

    with pytest.raises(ValueError, match="not a valid p_id in the\ninput data"):
        _fail_if_foreign_keys_are_invalid(data, p_id=data["p_id"])


@pytest.mark.parametrize("foreign_key_path", FOREIGN_KEYS)
def test_allow_minus_one_as_foreign_key(foreign_key_path):
    foreign_key_name = dt.qual_name_from_tree_path(foreign_key_path)
    data = {
        foreign_key_name: pd.Series([-1, 1, 2]),
        "p_id": pd.Series([1, 2, 3]),
    }

    _fail_if_foreign_keys_are_invalid(data, p_id=data["p_id"])


@pytest.mark.parametrize("foreign_key_path", FOREIGN_KEYS)
def test_fail_if_foreign_key_points_to_pid_of_same_row(foreign_key_path):
    foreign_key_name = dt.qual_name_from_tree_path(foreign_key_path)
    data = {
        foreign_key_name: pd.Series([1, 3, 3]),
        "p_id": pd.Series([1, 2, 3]),
    }

    with pytest.raises(ValueError, match="are equal to the p_id"):
        _fail_if_foreign_keys_are_invalid(data, p_id=data["p_id"])


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
                "fam_id": group_by_function()(lambda x: x),
            },
        ),
    ],
)
def test_fail_if_group_variables_not_constant_within_groups(data, functions):
    with pytest.raises(ValueError):
        _fail_if_group_variables_not_constant_within_groups(
            data=data,
            functions=functions,
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
            minimal_input_data, environment, targets_tree={"c": None}
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
        minimal_input_data, environment, targets_tree={"b": None}
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
            minimal_input_data, environment, targets_tree={"unknown_target": None}
        )


def test_fail_if_missing_pid():
    data = {"hh_id": pd.Series([1, 2, 3], name="hh_id")}
    with pytest.raises(
        ValueError,
        match="The input data must contain the p_id",
    ):
        compute_taxes_and_transfers(data, PolicyEnvironment({}), targets_tree={})


def test_fail_if_non_unique_pid(minimal_input_data):
    data = copy.deepcopy(minimal_input_data)
    data["p_id"][:] = 1

    with pytest.raises(
        ValueError,
        match="The following p_ids are non-unique",
    ):
        compute_taxes_and_transfers(data, PolicyEnvironment({}), targets_tree={})


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

    aggregation_specs_tree = {
        "module_name": {
            "betrag_m_hh": AggregateByGroupSpec(
                source="betrag_m",
                aggr=AggregationType.SUM,
            )
        }
    }
    expected_res = pd.Series([200, 200, 100])

    out = compute_taxes_and_transfers(
        data,
        PolicyEnvironment({}, aggregation_specs_tree=aggregation_specs_tree),
        targets_tree={"module_name": {"betrag_m_hh": None}},
    )

    numpy.testing.assert_array_almost_equal(
        out["module_name"]["betrag_m_hh"], expected_res
    )


@pytest.mark.parametrize(
    "aggregation_specs_tree",
    [
        {
            "module_name": {
                "betrag_double_m_hh": AggregateByGroupSpec(
                    source="betrag_m_double",
                    aggr=AggregationType.MAX,
                ),
            },
        },
        {
            "module_name": {
                "betrag_double_m_hh": AggregateByGroupSpec(
                    source="module_name__betrag_m_double",
                    aggr=AggregationType.MAX,
                ),
            },
        },
    ],
)
def test_user_provided_aggregate_by_group_specs_function(aggregation_specs_tree):
    data = {
        "p_id": pd.Series([1, 2, 3], name="p_id"),
        "hh_id": pd.Series([1, 1, 2], name="hh_id"),
        "module_name": {
            "betrag_m": pd.Series([200, 100, 100], name="betrag_m"),
        },
    }
    expected_res = pd.Series([400, 400, 200])

    def betrag_m_double(betrag_m):
        return 2 * betrag_m

    environment = PolicyEnvironment(
        {
            "module_name": {
                "betrag_m_double": policy_function(leaf_name="betrag_m_double")(
                    betrag_m_double
                )
            },
        },
        aggregation_specs_tree=aggregation_specs_tree,
    )

    out = compute_taxes_and_transfers(
        data,
        environment,
        targets_tree={"module_name": {"betrag_double_m_hh": None}},
    )

    numpy.testing.assert_array_almost_equal(
        out["module_name"]["betrag_double_m_hh"], expected_res
    )


def test_aggregate_by_group_specs_missing_group_sufix():
    data = {
        "p_id": pd.Series([1, 2, 3], name="p_id"),
        "hh_id": pd.Series([1, 1, 2], name="hh_id"),
        "module_name": {
            "betrag_m": pd.Series([100, 100, 100], name="betrag_m"),
        },
    }
    aggregation_specs_tree = {
        "module_name": {
            "betrag_agg_m": AggregateByGroupSpec(
                source="betrag_m",
                aggr=AggregationType.SUM,
            )
        },
    }
    with pytest.raises(
        ValueError,
        match="Name of aggregated column needs to have a suffix",
    ):
        compute_taxes_and_transfers(
            data,
            PolicyEnvironment({}, aggregation_specs_tree=aggregation_specs_tree),
            targets_tree={"module_name": {"betrag_agg_m": None}},
        )


def test_aggregate_by_group_specs_agg_not_impl():
    with pytest.raises(
        ValueError,
        match="aggr must be of type AggregationType, not <class 'str'>",
    ):
        AggregateByGroupSpec(
            source="betrag_m",
            aggr="sum",
        )


@pytest.mark.parametrize(
    ("aggregation_specs_tree, leaf_name, target_tree, expected"),
    [
        (
            {
                "module": {
                    "target_func": AggregateByPIDSpec(
                        p_id_to_aggregate_by="hh_id",
                        source="source_func",
                        aggr=AggregationType.SUM,
                    )
                }
            },
            "source_func",
            {"module": {"target_func": None}},
            pd.Series([200, 100, 0]),
        ),
        (
            {
                "module": {
                    "target_func_m": AggregateByPIDSpec(
                        p_id_to_aggregate_by="hh_id",
                        source="source_func_m",
                        aggr=AggregationType.SUM,
                    )
                }
            },
            "source_func_m",
            {"module": {"target_func_y": None}},
            pd.Series([2400, 1200, 0]),
        ),
        (
            {
                "module": {
                    "target_func_m": AggregateByPIDSpec(
                        p_id_to_aggregate_by="hh_id",
                        source="source_func_m",
                        aggr=AggregationType.SUM,
                    )
                }
            },
            "source_func_m",
            {"module": {"target_func_y_hh": None}},
            pd.Series([3600, 3600, 0]),
        ),
    ],
)
def test_user_provided_aggregate_by_p_id_specs(
    aggregation_specs_tree,
    leaf_name,
    target_tree,
    expected,
    minimal_input_data_shared_hh,
):
    # TODO(@MImmesberger): Remove fake dependency.
    # https://github.com/iza-institute-of-labor-economics/gettsim/issues/666
    @policy_function(leaf_name=leaf_name)
    def source_func(p_id: int) -> int:  # noqa: ARG001
        return 100

    functions_tree = {"module": {leaf_name: source_func}}

    environment = PolicyEnvironment(
        functions_tree,
        aggregation_specs_tree=aggregation_specs_tree,
    )
    out = compute_taxes_and_transfers(
        minimal_input_data_shared_hh,
        environment,
        targets_tree=target_tree,
    )["module"][next(iter(target_tree["module"].keys()))]

    numpy.testing.assert_array_almost_equal(out, expected)


@pytest.mark.parametrize(
    "input_data, expected_type, expected_output_data",
    [
        (pd.Series([0, 1, 0]), bool, pd.Series([False, True, False])),
        (pd.Series([1.0, 0.0, 1]), bool, pd.Series([True, False, True])),
        (pd.Series([200, 550, 237]), float, pd.Series([200.0, 550.0, 237.0])),
        (pd.Series([1.0, 4.0, 10.0]), int, pd.Series([1, 4, 10])),
        (pd.Series([200.0, 567.0]), int, pd.Series([200, 567])),
        (pd.Series([1.0, 0.0]), bool, pd.Series([True, False])),
    ],
)
def test_convert_series_to_internal_types(
    input_data, expected_type, expected_output_data
):
    adjusted_input = convert_series_to_internal_type(input_data, expected_type)
    pd.testing.assert_series_equal(adjusted_input, expected_output_data)


@pytest.mark.parametrize(
    "input_data, expected_type, error_match",
    [
        (
            pd.Series(["Hallo", 200, 325]),
            float,
            "Conversion from input type object to float failed.",
        ),
        (
            pd.Series([True, False]),
            float,
            "Conversion from input type bool to float failed.",
        ),
        (
            pd.Series(["a", "b", "c"]).astype("category"),
            float,
            "Conversion from input type category to float failed.",
        ),
        (
            pd.Series(["2.0", "3.0"]),
            int,
            "Conversion from input type object to int failed.",
        ),
        (
            pd.Series([1.5, 1.0, 2.9]),
            int,
            "Conversion from input type float64 to int failed.",
        ),
        (
            pd.Series(["a", "b", "c"]).astype("category"),
            int,
            "Conversion from input type category to int failed.",
        ),
        (
            pd.Series([5, 2, 3]),
            bool,
            "Conversion from input type int64 to bool failed.",
        ),
        (
            pd.Series([1.5, 1.0, 35.0]),
            bool,
            "Conversion from input type float64 to bool failed.",
        ),
        (
            pd.Series(["a", "b", "c"]).astype("category"),
            bool,
            "Conversion from input type category to bool failed.",
        ),
        (
            pd.Series(["richtig"]),
            bool,
            "Conversion from input type object to bool failed.",
        ),
        (
            pd.Series(["True", "False", ""]),
            bool,
            "Conversion from input type object to bool failed.",
        ),
        (
            pd.Series(["true"]),
            bool,
            "Conversion from input type object to bool failed.",
        ),
        (
            pd.Series(["zweitausendzwanzig"]),
            numpy.datetime64,
            "Conversion from input type object to datetime64 failed.",
        ),
        (
            pd.Series([True, True]),
            numpy.datetime64,
            "Conversion from input type bool to datetime64 failed.",
        ),
        (
            pd.Series([2020]),
            str,
            "The internal type <class 'str'> is not yet supported.",
        ),
    ],
)
def test_fail_if_cannot_be_converted_to_internal_type(
    input_data, expected_type, error_match
):
    with pytest.raises(ValueError, match=error_match):
        convert_series_to_internal_type(input_data, expected_type)


@pytest.mark.parametrize(
    "data, functions_overridden",
    [
        (
            {"sp_id": pd.Series([1, 2, 3])},
            {"sp_id": sp_id},
        ),
        (
            {"fam_id": pd.Series([1, 2, 3])},
            {"fam_id": fam_id},
        ),
    ],
)
def test_provide_endogenous_groupings(data, functions_overridden):
    """Test whether GETTSIM handles user-provided grouping IDs, which would otherwise be
    set endogenously."""
    _convert_data_to_correct_types(data, functions_overridden)


@pytest.mark.parametrize(
    "data, functions_overridden, error_match",
    [
        (
            {"hh_id": pd.Series([1, 1.1, 2])},
            {},
            "- hh_id: Conversion from input type float64 to int",
        ),
        (
            {"gondorian": pd.Series([1.1, 0.0, 1.0])},
            {},
            "- gondorian: Conversion from input type float64 to bool",
        ),
        (
            {
                "hh_id": pd.Series([1.0, 2.0, 3.0]),
                "gondorian": pd.Series([2, 0, 1]),
            },
            {},
            "- gondorian: Conversion from input type int64 to bool",
        ),
        (
            {"gondorian": pd.Series(["True", "False"])},
            {},
            "- gondorian: Conversion from input type object to bool",
        ),
        (
            {
                "hh_id": pd.Series([1, "1", 2]),
                "payroll_tax__amount": pd.Series(["2000", 3000, 4000]),
            },
            {},
            "- hh_id: Conversion from input type object to int failed.",
        ),
    ],
)
def test_fail_if_cannot_be_converted_to_correct_type(
    data, functions_overridden, error_match
):
    with pytest.raises(ValueError, match=error_match):
        _convert_data_to_correct_types(data, functions_overridden)


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
