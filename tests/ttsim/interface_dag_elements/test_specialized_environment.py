from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import dags.tree as dt
import numpy
import pandas as pd
import pytest

from ttsim import main, merge_trees
from ttsim.interface_dag_elements.specialized_environment import (
    with_partialled_params_and_scalars,
    with_processed_params_and_scalars,
)
from ttsim.tt_dag_elements import (
    AggType,
    DictParam,
    PiecewisePolynomialParam,
    PiecewisePolynomialParamValue,
    RawParam,
    ScalarParam,
    agg_by_group_function,
    agg_by_p_id_function,
    param_function,
    policy_function,
    policy_input,
)

if TYPE_CHECKING:
    from ttsim.interface_dag_elements.typing import IntColumn, NestedPolicyEnvironment


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


@policy_function(vectorization_strategy="vectorize")
def identity(x: int) -> int:
    return x


@policy_function(vectorization_strategy="vectorize")
def identity_plus_one(identity: int) -> int:
    return identity + 1


@policy_function(vectorization_strategy="vectorize")
def some_func(p_id: int) -> int:
    return p_id


@policy_function(vectorization_strategy="vectorize")
def another_func(some_func: int) -> int:
    return some_func


@param_function()
def some_scalar_params_func(some_int_param: int) -> int:
    return some_int_param


@policy_function(vectorization_strategy="vectorize")
def some_policy_func_taking_scalar_params_func(
    some_scalar_params_func: int,
) -> int:
    return some_scalar_params_func


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


@param_function()
def some_param_function_taking_scalar(
    some_int_scalar: int,
    some_float_scalar: float,
    some_bool_scalar: bool,
) -> float:
    return some_int_scalar + some_float_scalar + int(some_bool_scalar)


@policy_function(vectorization_strategy="vectorize")
def some_policy_function_taking_int_param(some_int_param: int) -> float:
    return some_int_param


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
    return {
        "p_id": numpy.arange(n_individuals),
        "fam_id": numpy.arange(n_individuals),
    }


@pytest.fixture(scope="module")
def minimal_input_data_shared_fam():
    n_individuals = 3
    return {
        "p_id": numpy.arange(n_individuals),
        "fam_id": numpy.array([0, 0, 1]),
        "p_id_someone_else": numpy.array([1, 0, -1]),
    }


@agg_by_group_function(agg_type=AggType.SUM)
def foo_fam(foo: int, fam_id: int) -> int:
    pass


@pytest.fixture(scope="module")
def mettsim_environment() -> NestedPolicyEnvironment:
    return main(
        inputs={
            "orig_policy_objects__root": Path(__file__).parent.parent / "mettsim",
            "date": datetime.date(2025, 1, 1),
        },
        output_names=["policy_environment"],
    )["policy_environment"]


# Create a function which is used by some tests below
@policy_function(vectorization_strategy="vectorize")
def func_before_partial(arg_1, some_param):
    return arg_1 + some_param


@pytest.fixture
@policy_function(leaf_name="foo", vectorization_strategy="vectorize")
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
@policy_function(leaf_name="bar", vectorization_strategy="vectorize")
def function_with_int_return(x: int) -> int:
    return x


@pytest.fixture
@policy_function(leaf_name="baz", vectorization_strategy="vectorize")
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
        "policy_environment",
        "targets__tree",
        "input_data__tree",
    ),
    [
        (
            # Aggregations derived from simple function arguments
            {
                "kin_id": kin_id,
                "p_id": p_id,
                "n1": {
                    "f": policy_function(vectorization_strategy="vectorize")(
                        return_n1__x_kin
                    ),
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
                    "f": policy_function(vectorization_strategy="loop")(return_x_kin),
                    "x": x,
                },
            },
            {"n1": {"f": None}},
            {
                "n1": {"x": pd.Series([1, 1, 1])},
                "kin_id": pd.Series([0, 0, 0]),
                "p_id": pd.Series([0, 1, 2]),
                "num_segments": 3,
            },
        ),
        (
            # Aggregations derived from target
            {
                "kin_id": kin_id,
                "p_id": p_id,
                "n1": {
                    "f": policy_function(vectorization_strategy="loop")(some_x),
                    "x": x,
                },
            },
            {"n1": {"f_kin": None}},
            {
                "n1": {"x": pd.Series([1, 1, 1])},
                "kin_id": pd.Series([0, 0, 0]),
                "p_id": pd.Series([0, 1, 2]),
                "num_segments": 3,
            },
        ),
        (
            # Explicit aggregation via objects tree with leaf name input
            {
                "kin_id": kin_id,
                "p_id": p_id,
                "n1": {
                    "f": policy_function(vectorization_strategy="loop")(some_x),
                    "x": x,
                },
                "y_kin": y_kin,
            },
            {"n1": {"f": None}},
            {
                "n1": {"x": pd.Series([1, 1, 1])},
                "kin_id": pd.Series([0, 0, 0]),
                "p_id": pd.Series([0, 1, 2]),
                "num_segments": 3,
            },
        ),
        (
            # Explicit aggregation via objects tree with namespaced input
            {
                "kin_id": kin_id,
                "p_id": p_id,
                "n1": {
                    "f": policy_function(
                        leaf_name="f",
                        vectorization_strategy="loop",
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
                "num_segments": 3,
            },
        ),
    ],
)
def test_create_agg_by_group_functions(
    policy_environment,
    targets__tree,
    input_data__tree,
    backend,
):
    policy_environment["backend"] = backend
    policy_environment["num_segments"] = len(input_data__tree["p_id"])
    main(
        inputs={
            "policy_environment": policy_environment,
            "input_data__tree": input_data__tree,
            "targets__tree": targets__tree,
            "rounding": False,
            "backend": backend,
        },
        output_names=["results__tree"],
    )["results__tree"]


def test_output_is_tree(minimal_input_data, backend, xnp):
    policy_environment = {
        "p_id": p_id,
        "module": {"some_func": some_func},
    }

    out = main(
        inputs={
            "input_data__tree": minimal_input_data,
            "policy_environment": policy_environment,
            "targets__tree": {"module": {"some_func": None}},
            "rounding": False,
            "backend": backend,
        },
        output_names=["results__tree"],
    )["results__tree"]

    assert isinstance(out, dict)
    assert "some_func" in out["module"]
    assert isinstance(out["module"]["some_func"], xnp.ndarray)


def test_params_target_is_allowed(minimal_input_data):
    policy_environment = {
        "p_id": p_id,
        "module": {"some_func": some_func},
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
    }

    out = main(
        inputs={
            "input_data__tree": minimal_input_data,
            "policy_environment": policy_environment,
            "targets__tree": {"some_param": None, "module": {"some_func": None}},
            "rounding": False,
            "backend": "numpy",
        },
        output_names=["results__tree"],
    )["results__tree"]

    assert isinstance(out, dict)
    assert "some_param" in out
    assert out["some_param"] == 1


def test_function_without_data_dependency_is_not_mistaken_for_data(
    minimal_input_data,
    backend,
    xnp,
):
    @policy_function(leaf_name="a", vectorization_strategy="not_required")
    def a() -> IntColumn:
        return xnp.array(minimal_input_data["p_id"])

    @policy_function(leaf_name="b")
    def b(a: int) -> int:
        return a

    policy_environment = {
        "a": a,
        "b": b,
    }
    results__tree = main(
        inputs={
            "input_data__tree": minimal_input_data,
            "policy_environment": policy_environment,
            "targets__tree": {"b": None},
            "rounding": False,
            "backend": backend,
        },
        output_names=["results__tree"],
    )["results__tree"]
    numpy.testing.assert_array_almost_equal(
        results__tree["b"],
        xnp.array(minimal_input_data["p_id"]),
    )


def test_partial_params_to_functions(xnp, backend):
    # Partial function produces correct result
    func_after_partial = with_partialled_params_and_scalars(
        with_processed_params_and_scalars={
            "some_func": func_before_partial,
            "some_param": SOME_INT_PARAM.value,
        },
        rounding=False,
        xnp=xnp,
        backend=backend,
    )["some_func"]

    assert func_after_partial(2) == 3


def test_partial_params_to_functions_removes_argument(xnp, backend):
    func_after_partial = with_partialled_params_and_scalars(
        with_processed_params_and_scalars={
            "some_func": func_before_partial,
            "some_param": SOME_INT_PARAM.value,
        },
        rounding=False,
        xnp=xnp,
        backend=backend,
    )["some_func"]

    # Fails if params is added to partial function
    with pytest.raises(
        TypeError,
        match=("got multiple values for argument "),
    ):
        func_after_partial(2, 1)

    # No error for original function
    func_before_partial(2, 1)


def test_user_provided_aggregate_by_group_specs(backend):
    data = {
        "p_id": pd.Series([1, 2, 3], name="p_id"),
        "fam_id": pd.Series([1, 1, 2], name="fam_id"),
        "module_name": {"betrag_m": pd.Series([100, 100, 100], name="betrag_m")},
    }

    policy_environment = {
        "p_id": p_id,
        "fam_id": fam_id,
        "module_name": {"betrag_m": betrag_m},
        "backend": backend,
    }

    expected = pd.Series([200, 200, 100], index=pd.Index(data["p_id"], name="p_id"))

    actual = main(
        inputs={
            "input_data__tree": data,
            "policy_environment": policy_environment,
            "targets__tree": {"module_name": {"betrag_m_fam": None}},
            "rounding": False,
            "backend": backend,
        },
        output_names=["results__df_with_nested_columns"],
    )["results__df_with_nested_columns"]

    pd.testing.assert_series_equal(
        actual[("module_name", "betrag_m_fam")],
        expected,
        check_names=False,
        check_dtype=False,
    )


def test_user_provided_aggregation(backend):
    data = {
        "p_id": pd.Series([1, 2, 3], name="p_id"),
        "fam_id": pd.Series([1, 1, 2], name="fam_id"),
        "module_name": {"betrag_m": pd.Series([200, 100, 100], name="betrag_m")},
    }
    # Double up, then take max fam_id
    expected = pd.Series([400, 400, 200], index=pd.Index(data["p_id"], name="p_id"))

    @policy_function(vectorization_strategy="loop")
    def betrag_m_double(betrag_m):
        return 2 * betrag_m

    @agg_by_group_function(agg_type=AggType.MAX)
    def betrag_m_double_fam(betrag_m_double, fam_id) -> float:
        pass

    policy_environment = {
        "p_id": p_id,
        "fam_id": fam_id,
        "module_name": {
            "betrag_m_double": betrag_m_double,
            "betrag_m_double_fam": betrag_m_double_fam,
        },
        "backend": backend,
        "num_segments": len(data["p_id"]),
    }

    actual = main(
        inputs={
            "input_data__tree": data,
            "policy_environment": policy_environment,
            "targets__tree": {"module_name": {"betrag_m_double_fam": None}},
            "rounding": False,
            "backend": backend,
        },
        output_names=["results__df_with_nested_columns"],
    )["results__df_with_nested_columns"]

    pd.testing.assert_series_equal(
        actual[("module_name", "betrag_m_double_fam")],
        expected,
        check_names=False,
        check_dtype=False,
    )


def test_user_provided_aggregation_with_time_conversion(backend):
    data = {
        "p_id": pd.Series([1, 2, 3], name="p_id"),
        "fam_id": pd.Series([1, 1, 2], name="fam_id"),
        "module_name": {
            "betrag_m": pd.Series([200, 100, 100], name="betrag_m"),
        },
    }

    # Double up, convert to quarter, then take max fam_id
    expected = pd.Series(
        [400 * 12, 400 * 12, 200 * 12],
        index=pd.Index(data["p_id"], name="p_id"),
    )

    @policy_function(vectorization_strategy="loop")
    def betrag_double_m(betrag_m):
        return 2 * betrag_m

    @agg_by_group_function(agg_type=AggType.MAX)
    def max_betrag_double_m_fam(betrag_double_m, fam_id) -> float:
        pass

    policy_environment = {
        "p_id": p_id,
        "fam_id": fam_id,
        "module_name": {
            "betrag_double_m": betrag_double_m,
            "max_betrag_double_m_fam": max_betrag_double_m_fam,
        },
        "backend": backend,
        "num_segments": len(data["p_id"]),
    }

    actual = main(
        inputs={
            "input_data__tree": data,
            "policy_environment": policy_environment,
            "targets__tree": {"module_name": {"max_betrag_double_y_fam": None}},
            "rounding": False,
            "backend": backend,
        },
        output_names=["results__df_with_nested_columns"],
    )["results__df_with_nested_columns"]

    pd.testing.assert_series_equal(
        actual[("module_name", "max_betrag_double_y_fam")],
        expected,
        check_names=False,
        check_dtype=False,
    )


@agg_by_p_id_function(agg_type=AggType.SUM)
def sum_source_by_p_id_someone_else(
    source: int,
    p_id: int,
    p_id_someone_else: int,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> int:
    pass


@agg_by_p_id_function(agg_type=AggType.SUM)
def sum_source_m_by_p_id_someone_else(
    source_m: int,
    p_id: int,
    p_id_someone_else: int,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> int:
    pass


@pytest.mark.parametrize(
    ("agg_functions, leaf_name, target_tree, expected"),
    [
        (
            {
                "module": {
                    "sum_source_by_p_id_someone_else": sum_source_by_p_id_someone_else,
                },
            },
            "source",
            {"module": {"sum_source_by_p_id_someone_else": None}},
            pd.Series([200, 100, 0], index=pd.Index([0, 1, 2], name="p_id")),
        ),
        (
            {
                "module": {
                    "sum_source_m_by_p_id_someone_else": sum_source_m_by_p_id_someone_else,  # noqa: E501
                },
            },
            "source_m",
            {"module": {"sum_source_m_by_p_id_someone_else": None}},
            pd.Series([200, 100, 0], index=pd.Index([0, 1, 2], name="p_id")),
        ),
    ],
)
def test_user_provided_aggregate_by_p_id_specs(
    agg_functions,
    leaf_name,
    target_tree,
    expected,
    minimal_input_data_shared_fam,
    backend,
    xnp,
):
    @policy_function(leaf_name=leaf_name, vectorization_strategy="not_required")
    def source() -> int:
        return xnp.array([100, 200, 300])

    policy_environment = merge_trees(
        agg_functions,
        {
            "module": {leaf_name: source},
            "p_id": p_id,
            "p_id_someone_else": p_id_someone_else,
            "backend": backend,
            "num_segments": len(minimal_input_data_shared_fam["p_id"]),
        },
    )

    actual = main(
        inputs={
            "input_data__tree": minimal_input_data_shared_fam,
            "policy_environment": policy_environment,
            "targets__tree": target_tree,
            "rounding": False,
            "backend": backend,
        },
        output_names=["results__df_with_nested_columns"],
    )["results__df_with_nested_columns"]

    pd.testing.assert_series_equal(
        actual[("module", next(iter(target_tree["module"].keys())))],
        expected,
        check_names=False,
        check_dtype=False,
    )


def test_policy_environment_with_params_and_scalars_is_processed(backend, xnp):
    policy_environment = {
        "raw_param_spec": SOME_RAW_PARAM,
        "some_int_param": SOME_INT_PARAM,
        "some_dict_param": SOME_DICT_PARAM,
        "some_piecewise_polynomial_param": SOME_PIECEWISE_POLYNOMIAL_PARAM,
        "some_int_scalar": 1,
        "some_float_scalar": 2.0,
        "some_bool_scalar": True,
        "some_scalar_params_func": some_scalar_params_func,
        "some_converting_params_func": some_converting_params_func,
        "some_param_function_taking_scalar": some_param_function_taking_scalar,
        "backend": backend,
    }
    actual = with_processed_params_and_scalars(
        without_tree_logic_and_with_derived_functions=policy_environment,
        processed_data={"x": xnp.array([1, 2, 3])},
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
        "some_int_scalar": 1,
        "some_float_scalar": 2.0,
        "some_bool_scalar": True,
        "some_param_function_taking_scalar": 4.0,
        "num_segments": 3,
        "backend": backend,
    }
    assert actual == expected


@pytest.mark.parametrize(
    (
        "nested_policy_environment",
        "overriding_data",
        "targets__tree",
        "expected_output",
    ),
    [
        # Overwriting policy function
        (
            {
                "identity": identity,
                "identity_plus_one": identity_plus_one,
            },
            {
                "identity": numpy.array([1, 2, 3, 4, 5]),
            },
            {"identity_plus_one": None},
            {"identity_plus_one": numpy.array([2, 3, 4, 5, 6])},
        ),
        # Overwriting parameter
        (
            {
                "some_int_param": SOME_INT_PARAM,
                "some_policy_function_taking_int_param": some_policy_function_taking_int_param,  # noqa: E501
            },
            {
                "some_int_param": numpy.array([1, 2, 3, 4, 5]),
            },
            {"some_policy_function_taking_int_param": None},
            {"some_policy_function_taking_int_param": numpy.array([1, 2, 3, 4, 5])},
        ),
        # Overwriting parameter function
        (
            {
                "some_int_param": SOME_INT_PARAM,
                "some_scalar_params_func": some_policy_function_taking_int_param,
                "some_policy_func_taking_scalar_params_func": some_policy_func_taking_scalar_params_func,  # noqa: E501
            },
            {
                "some_scalar_params_func": numpy.array([1, 2, 3, 4, 5]),
            },
            {"some_policy_func_taking_scalar_params_func": None},
            {
                "some_policy_func_taking_scalar_params_func": numpy.array(
                    [1, 2, 3, 4, 5],
                ),
            },
        ),
    ],
)
def test_can_override_ttsim_objects_with_data(
    nested_policy_environment,
    overriding_data,
    targets__tree,
    expected_output,
    minimal_input_data,
    backend,
):
    actual = main(
        inputs={
            "input_data__tree": {
                **minimal_input_data,
                **overriding_data,
            },
            "policy_environment": nested_policy_environment,
            "targets__tree": targets__tree,
            "rounding": False,
            "backend": backend,
        },
        output_names=["results__tree"],
    )["results__tree"]

    flat_actual = dt.flatten_to_tree_paths(actual)
    flat_expected = dt.flatten_to_tree_paths(expected_output)

    assert flat_actual.keys() == flat_expected.keys()
    for key in flat_expected:
        numpy.testing.assert_array_almost_equal(flat_actual[key], flat_expected[key])
