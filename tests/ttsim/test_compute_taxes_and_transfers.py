from __future__ import annotations

import copy
import re
import warnings
from dataclasses import dataclass

import dags.tree as dt
import numpy
import pandas as pd
import pytest
from mettsim.config import METTSIM_ROOT

from ttsim import (
    AggType,
    DictParam,
    FunctionsAndColumnsOverlapWarning,
    PiecewisePolynomialParam,
    PiecewisePolynomialParamValue,
    PolicyEnvironment,
    RawParam,
    ScalarParam,
    agg_by_group_function,
    agg_by_p_id_function,
    compute_taxes_and_transfers,
    merge_trees,
    param_function,
    policy_function,
    policy_input,
    set_up_policy_environment,
)
from ttsim.compute_taxes_and_transfers import (
    _fail_if_foreign_keys_are_invalid_in_data,
    _fail_if_function_targets_not_in_functions,
    _fail_if_group_variables_not_constant_within_groups,
    _fail_if_p_id_is_non_unique,
    _get_top_level_namespace,
    _partial_params_to_functions,
    _process_params_tree,
    create_agg_by_group_functions,
)
from ttsim.config import IS_JAX_INSTALLED
from ttsim.config import numpy_or_jax as np
from ttsim.shared import assert_valid_ttsim_pytree
from ttsim.typing import TTSIMArray

if IS_JAX_INSTALLED:
    jit = True
else:
    jit = False


@policy_input()
def p_id() -> int:
    pass


@policy_input()
def p_id_someone_else() -> int:
    pass


@policy_input()
def fam_id() -> int:
    pass


@policy_input()
def betrag_m() -> float:
    pass


@policy_function()
def identity(x: int) -> int:
    return x


@policy_function()
def some_func(p_id: int) -> int:
    return p_id


@policy_function()
def another_func(some_func: int) -> int:
    return some_func


@param_function()
def some_scalar_params_func(some_int_param: int) -> int:
    return some_int_param


@dataclass(frozen=True)
class ConvertedParam:
    some_float_param: float
    some_bool_param: bool


@param_function()
def some_converting_params_func(
    raw_param_spec: RawParam,
) -> ConvertedParam:
    return ConvertedParam(
        some_float_param=raw_param_spec["some_float_param"],
        some_bool_param=raw_param_spec["some_bool_param"],
    )


SOME_RAW_PARAM = RawParam(
    value={
        "some_float_param": 1,
        "some_bool_param": False,
    },
    leaf_name="raw_param_spec",
    start_date="2025-01-01",
    end_date="2025-12-31",
    name="raw_param_spec",
    description="Some raw param spec",
    unit=None,
    reference_period=None,
    note=None,
    reference=None,
)


SOME_INT_PARAM = ScalarParam(
    value=1,
    leaf_name="some_int_param",
    start_date="2025-01-01",
    end_date="2025-12-31",
    name="some_int_param",
    description="Some int param",
    unit=None,
    reference_period=None,
    note=None,
    reference=None,
)


SOME_DICT_PARAM = DictParam(
    value={"a": 1, "b": False},
    leaf_name="some_dict_param",
    start_date="2025-01-01",
    end_date="2025-12-31",
    name="some_dict_param",
    description="Some dict param",
    unit=None,
    reference_period=None,
    note=None,
    reference=None,
)


SOME_PIECEWISE_POLYNOMIAL_PARAM = PiecewisePolynomialParam(
    value=PiecewisePolynomialParamValue(
        thresholds=[1, 2, 3],
        intercepts=[1, 2, 3],
        rates=[1, 2, 3],
    ),
    leaf_name="some_piecewise_polynomial_param",
    start_date="2025-01-01",
    end_date="2025-12-31",
    name="some_piecewise_polynomial_param",
    description="Some piecewise polynomial param",
    unit=None,
    reference_period=None,
    note=None,
    reference=None,
)


@pytest.fixture(scope="module")
def minimal_input_data():
    n_individuals = 5
    out = {
        "p_id": pd.Series(numpy.arange(n_individuals), name="p_id"),
        "fam_id": pd.Series(numpy.arange(n_individuals), name="fam_id"),
    }
    return out


@pytest.fixture(scope="module")
def minimal_input_data_shared_fam():
    n_individuals = 3
    out = {
        "p_id": pd.Series(numpy.arange(n_individuals), name="p_id"),
        "fam_id": pd.Series([0, 0, 1], name="fam_id"),
        "p_id_someone_else": pd.Series([1, 0, -1], name="p_id_someone_else"),
    }
    return out


@agg_by_group_function(agg_type=AggType.SUM)
def foo_fam(foo: int, fam_id: int) -> int:
    pass


@pytest.fixture(scope="module")
def mettsim_environment():
    return set_up_policy_environment(
        root=METTSIM_ROOT,
        date="2025-01-01",
    )


# Create a function which is used by some tests below
@policy_function()
def func_before_partial(arg_1, some_param):
    return arg_1 + some_param


func_after_partial = _partial_params_to_functions(
    {"some_func": func_before_partial},
    {"some_param": SOME_INT_PARAM.value},
)["some_func"]


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
def kin_id() -> int:
    pass


@agg_by_group_function(leaf_name="y_kin", agg_type=AggType.SUM)
def y_kin(kin_id: int, x: int) -> int:
    pass


@agg_by_group_function(leaf_name="y_kin", agg_type=AggType.SUM)
def y_kin_namespaced_input(kin_id: int, inputs__x: int) -> int:
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


def return_x_kin(x_kin: int) -> int:
    return x_kin


def return_y_kin(y_kin: int) -> int:
    return y_kin


def return_n1__x_kin(n1__x_kin: int) -> int:
    return n1__x_kin


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
                "kin_id": kin_id,
                "p_id": p_id,
                "n1": {
                    "f": policy_function(
                        leaf_name="f", vectorization_strategy="vectorize"
                    )(return_n1__x_kin),
                    "x": x,
                },
            },
            {"n1": {"f": None}},
            {
                "n1": {"x": pd.Series([1, 1, 1])},
                "kin_id": pd.Series([0, 0, 0]),
                "p_id": pd.Series([0, 1, 2]),
            },
        ),
        (
            # Aggregations derived from namespaced function arguments
            {
                "kin_id": kin_id,
                "p_id": p_id,
                "n1": {
                    "f": policy_function(leaf_name="f")(return_x_kin),
                    "x": x,
                },
            },
            {"n1": {"f": None}},
            {
                "n1": {"x": pd.Series([1, 1, 1])},
                "kin_id": pd.Series([0, 0, 0]),
                "p_id": pd.Series([0, 1, 2]),
                "num_segments": 1,
            },
        ),
        (
            # Aggregations derived from target
            {
                "kin_id": kin_id,
                "p_id": p_id,
                "n1": {
                    "f": policy_function(
                        leaf_name="f", vectorization_strategy="vectorize"
                    )(some_x),
                    "x": x,
                },
            },
            {"n1": {"f_kin": None}},
            {
                "n1": {"x": pd.Series([1, 1, 1])},
                "kin_id": pd.Series([0, 0, 0]),
                "p_id": pd.Series([0, 1, 2]),
                "num_segments": 1,
            },
        ),
        (
            # Explicit aggregation via objects tree with leaf name input
            {
                "kin_id": kin_id,
                "p_id": p_id,
                "n1": {
                    "f": policy_function(
                        leaf_name="f", vectorization_strategy="vectorize"
                    )(some_x),
                    "x": x,
                },
                "y_kin": y_kin,
            },
            {"n1": {"f": None}},
            {
                "n1": {"x": pd.Series([1, 1, 1])},
                "kin_id": pd.Series([0, 0, 0]),
                "p_id": pd.Series([0, 1, 2]),
                "num_segments": 1,
            },
        ),
        (
            # Explicit aggregation via objects tree with namespaced input
            {
                "kin_id": kin_id,
                "p_id": p_id,
                "n1": {
                    "f": policy_function(
                        leaf_name="f", vectorization_strategy="vectorize"
                    )(return_y_kin),
                    "y_kin": y_kin_namespaced_input,
                },
                "inputs": {"x": x},
            },
            {"n1": {"f": None}},
            {
                "inputs": {"x": pd.Series([1, 1, 1])},
                "kin_id": pd.Series([0, 0, 0]),
                "p_id": pd.Series([0, 1, 2]),
                "num_segments": 1,
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
        jit=jit,
    )


@pytest.mark.parametrize(
    "functions, targets, expected_error_match",
    [
        ({"foo": some_x}, {"bar": None}, "('bar',)"),
        ({"foo__baz": some_x}, {"foo__bar": None}, "('foo', 'bar')"),
    ],
)
def test__fail_if_function_targets_not_in_functions(
    functions, targets, expected_error_match
):
    with pytest.raises(ValueError) as e:
        _fail_if_function_targets_not_in_functions(
            functions=functions,
            targets=targets,
        )
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
            {"foo": policy_function(leaf_name="foo")(return_x_kin)},
            {},
            {"x": pd.Series([1])},
            ("x_kin"),
        ),
        (
            {"n2__foo": policy_function(leaf_name="foo")(return_n1__x_kin)},
            {},
            {"n1__x": pd.Series([1])},
            ("n1__x_kin"),
        ),
        (
            {},
            {"x_kin": None},
            {"x": pd.Series([1])},
            ("x_kin"),
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
        groupings=("kin",),
    )
    assert expected in result


def test_output_is_tree(minimal_input_data):
    environment = PolicyEnvironment(
        {
            "p_id": p_id,
            "module": {"some_func": some_func},
        }
    )

    out = compute_taxes_and_transfers(
        data_tree=minimal_input_data,
        environment=environment,
        targets_tree={"module": {"some_func": None}},
        jit=jit,
    )

    assert isinstance(out, dict)
    assert "some_func" in out["module"]
    assert isinstance(out["module"]["some_func"], TTSIMArray)


def test_params_target_is_allowed(minimal_input_data):
    environment = PolicyEnvironment(
        raw_objects_tree={
            "p_id": p_id,
            "module": {"some_func": some_func},
        },
        params_tree={
            "some_param": ScalarParam(
                value=1,
                leaf_name="some_param",
                start_date="2025-01-01",
                end_date="2025-12-31",
                unit="Euros",
                reference_period="Year",
                name={"de": "Ein Parameter", "en": "Some parameter"},
                description={"de": "Ein Parameter", "en": "Some parameter"},
                note=None,
                reference=None,
            ),
        },
    )

    out = compute_taxes_and_transfers(
        data_tree=minimal_input_data,
        environment=environment,
        targets_tree={"some_param": None, "module": {"some_func": None}},
        jit=jit,
    )

    assert isinstance(out, dict)
    assert "some_param" in out
    assert out["some_param"] == 1


def test_warn_if_functions_and_columns_overlap():
    environment = PolicyEnvironment(
        {
            "some_func": some_func,
            "some_target": another_func,
        }
    )
    with pytest.warns(FunctionsAndColumnsOverlapWarning):
        compute_taxes_and_transfers(
            data_tree={
                "p_id": pd.Series([0]),
                "some_func": pd.Series([1]),
            },
            environment=environment,
            targets_tree={"some_target": None},
            jit=jit,
        )


def test_dont_warn_if_functions_and_columns_dont_overlap():
    environment = PolicyEnvironment({"some_func": some_func})
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=FunctionsAndColumnsOverlapWarning)
        compute_taxes_and_transfers(
            data_tree={
                "p_id": pd.Series([0]),
                "x": pd.Series([1]),
            },
            environment=environment,
            targets_tree={"some_func": None},
            jit=jit,
        )


def test_recipe_to_ignore_warning_if_functions_and_columns_overlap():
    environment = PolicyEnvironment(
        {
            "some_func": some_func,
            "unique": another_func,
        }
    )
    with warnings.catch_warnings(
        category=FunctionsAndColumnsOverlapWarning, record=True
    ) as warning_list:
        warnings.filterwarnings("ignore", category=FunctionsAndColumnsOverlapWarning)
        compute_taxes_and_transfers(
            data_tree={
                "p_id": pd.Series([0]),
                "some_func": pd.Series([1]),
                "x": pd.Series([1]),
            },
            environment=environment,
            targets_tree={"unique": None},
            jit=jit,
        )

    assert len(warning_list) == 0


def test_fail_if_p_id_does_not_exist():
    data = {"fam_id": pd.Series(data=numpy.arange(8), name="fam_id")}

    with pytest.raises(ValueError):
        _fail_if_p_id_is_non_unique(data)


def test_fail_if_p_id_is_non_unique():
    data = {"p_id": pd.Series(data=numpy.arange(4).repeat(2), name="p_id")}

    with pytest.raises(ValueError):
        _fail_if_p_id_is_non_unique(data)


def test_fail_if_foreign_key_points_to_non_existing_p_id(mettsim_environment):
    flat_objects_tree = dt.flatten_to_qual_names(mettsim_environment.raw_objects_tree)
    data = {
        "p_id": pd.Series([1, 2, 3]),
        "p_id_spouse": pd.Series([0, 1, 2]),
    }

    with pytest.raises(ValueError, match=r"not a valid p_id in the\sinput data"):
        _fail_if_foreign_keys_are_invalid_in_data(
            data=data, column_objects_param_functions=flat_objects_tree
        )


def test_allow_minus_one_as_foreign_key(mettsim_environment):
    flat_objects_tree = dt.flatten_to_qual_names(mettsim_environment.raw_objects_tree)
    data = {
        "p_id": pd.Series([1, 2, 3]),
        "p_id_spouse": pd.Series([-1, 1, 2]),
    }

    _fail_if_foreign_keys_are_invalid_in_data(
        data=data, column_objects_param_functions=flat_objects_tree
    )


def test_fail_if_foreign_key_points_to_same_row_if_not_allowed(mettsim_environment):
    flat_objects_tree = dt.flatten_to_qual_names(mettsim_environment.raw_objects_tree)
    data = {
        "p_id": pd.Series([1, 2, 3]),
        "child_tax_credit__p_id_recipient": pd.Series([1, 3, 3]),
    }

    _fail_if_foreign_keys_are_invalid_in_data(
        data=data, column_objects_param_functions=flat_objects_tree
    )


def test_fail_if_foreign_key_points_to_same_row_if_allowed(mettsim_environment):
    flat_objects_tree = dt.flatten_to_qual_names(mettsim_environment.raw_objects_tree)
    data = {
        "p_id": pd.Series([1, 2, 3]),
        "p_id_child_": pd.Series([1, 3, 3]),
    }

    _fail_if_foreign_keys_are_invalid_in_data(
        data=data, column_objects_param_functions=flat_objects_tree
    )


def test_fail_if_group_variables_not_constant_within_groups():
    data = {
        "foo_kin": pd.Series([1, 2, 2], name="foo_kin"),
        "kin_id": pd.Series([1, 1, 2], name="kin_id"),
    }
    with pytest.raises(ValueError):
        _fail_if_group_variables_not_constant_within_groups(
            data=data,
            groupings=("kin",),
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
            jit=jit,
        )


def test_function_without_data_dependency_is_not_mistaken_for_data(minimal_input_data):
    @policy_function(leaf_name="a", vectorization_strategy="not_required")
    def a() -> np.ndarray:
        return np.array(range(minimal_input_data["p_id"].size))

    @policy_function(leaf_name="b")
    def b(a):
        return a

    environment = PolicyEnvironment({"a": a, "b": b})
    compute_taxes_and_transfers(
        data_tree=minimal_input_data,
        environment=environment,
        targets_tree={"b": None},
        jit=jit,
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
            jit=jit,
        )


def test_fail_if_missing_p_id():
    data = {"fam_id": pd.Series([1, 2, 3], name="fam_id")}
    with pytest.raises(
        ValueError,
        match="The input data must contain the p_id",
    ):
        compute_taxes_and_transfers(
            data_tree=data,
            environment=PolicyEnvironment({}),
            targets_tree={},
            jit=jit,
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
            jit=jit,
        )


def test_partial_params_to_functions():
    # Partial function produces correct result
    assert func_after_partial(2) == 3


def test_partial_params_to_functions_removes_argument():
    # Fails if params is added to partial function
    with pytest.raises(
        TypeError,
        match=("got multiple values for argument "),
    ):
        func_after_partial(2, 1)

    # No error for original function
    func_before_partial(2, 1)


def test_user_provided_aggregate_by_group_specs():
    data = {
        "p_id": pd.Series([1, 2, 3], name="p_id"),
        "fam_id": pd.Series([1, 1, 2], name="fam_id"),
        "module_name": {"betrag_m": pd.Series([100, 100, 100], name="betrag_m")},
    }

    inputs = {
        "p_id": p_id,
        "fam_id": fam_id,
        "module_name": {"betrag_m": betrag_m},
    }

    expected_res = pd.Series([200, 200, 100])

    out = compute_taxes_and_transfers(
        data_tree=data,
        environment=PolicyEnvironment(raw_objects_tree=inputs),
        targets_tree={"module_name": {"betrag_m_fam": None}},
        jit=jit,
    )

    numpy.testing.assert_array_almost_equal(
        out["module_name"]["betrag_m_fam"], expected_res
    )


def test_user_provided_aggregation():
    data = {
        "p_id": pd.Series([1, 2, 3], name="p_id"),
        "fam_id": pd.Series([1, 1, 2], name="fam_id"),
        "module_name": {"betrag_m": pd.Series([200, 100, 100], name="betrag_m")},
    }
    data["num_segments"] = len(data["fam_id"].unique())
    # Double up, then take max fam_id
    expected = pd.Series([400, 400, 200])

    @policy_function(vectorization_strategy="vectorize")
    def betrag_m_double(betrag_m):
        return 2 * betrag_m

    @agg_by_group_function(agg_type=AggType.MAX)
    def betrag_m_double_fam(betrag_m_double, fam_id) -> float:
        pass

    environment = PolicyEnvironment(
        {
            "p_id": p_id,
            "fam_id": fam_id,
            "module_name": {
                "betrag_m_double": betrag_m_double,
                "betrag_m_double_fam": betrag_m_double_fam,
            },
        }
    )

    actual = compute_taxes_and_transfers(
        data_tree=data,
        environment=environment,
        targets_tree={"module_name": {"betrag_m_double_fam": None}},
        debug=False,
        jit=jit,
    )

    numpy.testing.assert_array_almost_equal(
        actual["module_name"]["betrag_m_double_fam"], expected
    )


def test_user_provided_aggregation_with_time_conversion():
    data = {
        "p_id": pd.Series([1, 2, 3], name="p_id"),
        "fam_id": pd.Series([1, 1, 2], name="fam_id"),
        "module_name": {
            "betrag_m": pd.Series([200, 100, 100], name="betrag_m"),
        },
    }

    # Double up, convert to quarter, then take max fam_id
    expected = pd.Series([400 * 12, 400 * 12, 200 * 12])

    @policy_function(vectorization_strategy="vectorize")
    def betrag_double_m(betrag_m):
        return 2 * betrag_m

    @agg_by_group_function(agg_type=AggType.MAX)
    def max_betrag_double_m_fam(betrag_double_m, fam_id) -> float:
        pass

    environment = PolicyEnvironment(
        {
            "p_id": p_id,
            "fam_id": fam_id,
            "module_name": {
                "betrag_double_m": betrag_double_m,
                "max_betrag_double_m_fam": max_betrag_double_m_fam,
            },
        }
    )

    actual = compute_taxes_and_transfers(
        data_tree=data,
        environment=environment,
        targets_tree={"module_name": {"max_betrag_double_y_fam": None}},
        debug=False,
        jit=jit,
    )

    numpy.testing.assert_array_almost_equal(
        actual["module_name"]["max_betrag_double_y_fam"], expected
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
                    "sum_source_m_by_p_id_someone_else": sum_source_m_by_p_id_someone_else  # noqa: E501
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
    minimal_input_data_shared_fam,
):
    @policy_function(leaf_name=leaf_name, vectorization_strategy="not_required")
    def source() -> int:
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
        minimal_input_data_shared_fam,
        environment,
        targets_tree=target_tree,
        jit=jit,
    )["module"][next(iter(target_tree["module"].keys()))]

    numpy.testing.assert_array_almost_equal(out, expected)


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
        "time_units",
        "expected",
    ),
    [
        (
            PolicyEnvironment(
                raw_objects_tree={
                    "foo_m": policy_function(leaf_name="foo_m")(identity),
                    "fam_id": fam_id,
                }
            ),
            ["m", "y"],
            {"foo_m", "foo_y", "foo_m_fam", "foo_y_fam"},
        ),
        (
            PolicyEnvironment(
                raw_objects_tree={
                    "foo": policy_function(leaf_name="foo")(identity),
                    "fam_id": fam_id,
                }
            ),
            ["m", "y"],
            {"foo", "foo_fam"},
        ),
    ],
)
def test_get_top_level_namespace(environment, time_units, expected):
    result = _get_top_level_namespace(
        environment=environment,
        time_units=time_units,
    )
    assert all(name in result for name in expected)


def test_params_tree_is_processed():
    params_tree = {
        "raw_param_spec": SOME_RAW_PARAM,
        "some_int_param": SOME_INT_PARAM,
        "some_dict_param": SOME_DICT_PARAM,
        "some_piecewise_polynomial_param": SOME_PIECEWISE_POLYNOMIAL_PARAM,
    }
    param_functions = {
        "some_scalar_params_func": some_scalar_params_func,
        "some_converting_params_func": some_converting_params_func,
    }
    processed_params_tree = _process_params_tree(
        params_tree=params_tree,
        param_functions=param_functions,
    )
    expected = {
        "some_converting_params_func": ConvertedParam(
            some_float_param=1,
            some_bool_param=False,
        ),
        "some_scalar_params_func": 1,
        "some_int_param": SOME_INT_PARAM.value,
        "some_dict_param": SOME_DICT_PARAM.value,
        "some_piecewise_polynomial_param": SOME_PIECEWISE_POLYNOMIAL_PARAM.value,
    }
    assert processed_params_tree == expected
