from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import dags.tree as dt
import numpy
import pandas as pd
import pytest

from tests.test_unit_system import TEST_UNIT_SYSTEM
from ttsim import main, merge_trees
from ttsim.interface_dag_elements.specialized_environment import (
    with_partialled_params_and_scalars,
    with_processed_params_and_scalars,
)
from ttsim.main_args import InputData, TTTargets
from ttsim.main_target import MainTarget
from ttsim.tt import (
    AggType,
    DictParam,
    PiecewisePolynomialParam,
    PiecewisePolynomialParamValue,
    RawParam,
    ScalarParam,
    TTSIMUnit,
    agg_by_group_function,
    agg_by_p_id_function,
    param_function,
    policy_function,
    policy_input,
)
from ttsim.tt.units import ttsim_unit_from_yaml_value

if TYPE_CHECKING:
    from ttsim.typing import IntColumn, RawParamValue


@policy_input(unit=TTSIMUnit.DIMENSIONLESS)
def p_id() -> int:
    pass


@policy_input(unit=TTSIMUnit.DIMENSIONLESS)
def p_id_someone_else() -> int:
    pass


@policy_input(unit=TTSIMUnit.DIMENSIONLESS)
def fam_id() -> int:
    pass


@policy_input(unit=TTSIMUnit.CURRENCY.PER_MONTH)
def betrag_m() -> float:
    pass


@policy_input(unit=TTSIMUnit.CURRENCY.PER_MONTH)
def income_m() -> float:
    pass


@policy_function(vectorization_strategy="vectorize", unit=TTSIMUnit.CURRENCY.PER_MONTH)
def benefit_m(income_m: float) -> float:
    return income_m * 0.5


@policy_function(vectorization_strategy="vectorize", unit=TTSIMUnit.DIMENSIONLESS)
def identity(x: int) -> int:
    return x


@policy_function(vectorization_strategy="vectorize", unit=TTSIMUnit.DIMENSIONLESS)
def identity_plus_one(identity: int) -> int:
    return identity + 1


@policy_function(vectorization_strategy="vectorize", unit=TTSIMUnit.DIMENSIONLESS)
def some_func(p_id: int) -> int:
    return p_id


@policy_function(vectorization_strategy="vectorize", unit=TTSIMUnit.DIMENSIONLESS)
def another_func(some_func: int) -> int:
    return some_func


@param_function(unit=TTSIMUnit.DIMENSIONLESS)
def some_scalar_params_func(some_int_param: int) -> int:
    return some_int_param


@policy_function(vectorization_strategy="vectorize", unit=TTSIMUnit.DIMENSIONLESS)
def some_policy_func_taking_scalar_params_func(
    some_scalar_params_func: int,
) -> int:
    return some_scalar_params_func


@dataclass(frozen=True)
class ConvertedParam:
    some_float_param: float
    some_bool_param: bool


@param_function(unit=TTSIMUnit.DIMENSIONLESS)
def some_converting_params_func(raw_param_spec: RawParamValue) -> ConvertedParam:
    return ConvertedParam(
        some_float_param=raw_param_spec["some_float_param"],
        some_bool_param=raw_param_spec["some_bool_param"],
    )


@param_function(unit=TTSIMUnit.DIMENSIONLESS)
def some_param_function_taking_scalar(
    some_int_scalar: int,
    some_float_scalar: float,
    some_bool_scalar: bool,
) -> float:
    return some_int_scalar + some_float_scalar + int(some_bool_scalar)


@policy_function(vectorization_strategy="vectorize", unit=TTSIMUnit.DIMENSIONLESS)
def some_policy_function_taking_int_param(some_int_param: int) -> float:
    return some_int_param


SOME_RAW_PARAM = RawParam(
    value={
        "some_float_param": 1,
        "some_bool_param": False,
    },
    start_date=datetime.date(2025, 1, 1),
    end_date=datetime.date(2025, 12, 31),
    name={"de": "Ein raw param spec", "en": "Some raw param spec"},
    description={"de": "Ein raw param spec", "en": "Some raw param spec"},
    unit=TTSIMUnit.DIMENSIONLESS,
    note=None,
    reference=None,
)


SOME_INT_PARAM = ScalarParam(
    value=1,
    start_date=datetime.date(2025, 1, 1),
    end_date=datetime.date(2025, 12, 31),
    name={"de": "Ein int param", "en": "Some int param"},
    description={"de": "Ein int param", "en": "Some int param"},
    unit=TTSIMUnit.DIMENSIONLESS,
    note=None,
    reference=None,
)


SOME_DICT_PARAM = DictParam(
    value={"a": 1, "b": False},
    start_date=datetime.date(2025, 1, 1),
    end_date=datetime.date(2025, 12, 31),
    name={"de": "Ein dict param", "en": "Some dict param"},
    description={"de": "Ein dict param", "en": "Some dict param"},
    unit=TTSIMUnit.DIMENSIONLESS,
    note=None,
    reference=None,
)


@pytest.fixture
def some_piecewise_polynomial_param(xnp):
    return PiecewisePolynomialParam(
        value=PiecewisePolynomialParamValue(
            thresholds=xnp.array([1, 2, 3]),
            intercepts=xnp.array([1, 2, 3]),
            coefficients=xnp.array([[1], [2], [3]]),
        ),
        start_date=datetime.date(2025, 1, 1),
        end_date=datetime.date(2025, 12, 31),
        name={
            "de": "Ein piecewise polynomial param",
            "en": "Some piecewise polynomial param",
        },
        description={
            "de": "Ein piecewise polynomial param",
            "en": "Some piecewise polynomial param",
        },
        input_unit=TTSIMUnit.DIMENSIONLESS,
        output_unit=TTSIMUnit.DIMENSIONLESS,
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


@agg_by_group_function(agg_type=AggType.SUM, unit=TTSIMUnit.DIMENSIONLESS)
def foo_fam(foo: int, fam_id: int) -> int:
    pass


# Create a function which is used by some tests below
@policy_function(vectorization_strategy="vectorize", unit=TTSIMUnit.DIMENSIONLESS)
def func_before_partial(arg_1: int, some_param: int) -> int:
    return arg_1 + some_param


@pytest.fixture
@policy_function(
    leaf_name="foo", vectorization_strategy="vectorize", unit=TTSIMUnit.DIMENSIONLESS
)
def function_with_bool_return(x: bool) -> bool:
    return x


@policy_input(unit=TTSIMUnit.DIMENSIONLESS)
def x() -> int:
    pass


@policy_input(unit=TTSIMUnit.DIMENSIONLESS)
def x_f() -> float:
    pass


@policy_input(unit=TTSIMUnit.DIMENSIONLESS)
def x_b() -> bool:
    pass


@policy_input(unit=TTSIMUnit.DIMENSIONLESS)
def kin_id() -> int:
    pass


@agg_by_group_function(
    leaf_name="y_kin",
    agg_type=AggType.SUM,
    unit=TTSIMUnit.DIMENSIONLESS.PER_LEVEL("kin"),
)
def y_kin(kin_id: int, x: int) -> int:
    pass


# A SUM is the group's whatever the source's base (GEP 10) — even a dimensionless
# share acquires the target level, so the declared unit spells it.
@agg_by_group_function(
    leaf_name="y_kin",
    agg_type=AggType.SUM,
    unit=TTSIMUnit.DIMENSIONLESS.PER_LEVEL("kin"),
)
def y_kin_namespaced_input(kin_id: int, inputs__x: int) -> int:
    pass


@pytest.fixture
@policy_function(
    leaf_name="bar", vectorization_strategy="vectorize", unit=TTSIMUnit.DIMENSIONLESS
)
def function_with_int_return(x: int) -> int:
    return x


@pytest.fixture
@policy_function(
    leaf_name="baz", vectorization_strategy="vectorize", unit=TTSIMUnit.DIMENSIONLESS
)
def function_with_float_return(x: int) -> float:
    return x


def some_x(x: int) -> int:
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
        "tt_targets__tree",
        "input_data__tree",
    ),
    [
        (
            # Aggregations derived from simple function arguments
            {
                "kin_id": kin_id,
                "p_id": p_id,
                "n1": {
                    "f": policy_function(
                        leaf_name="f", unit=TTSIMUnit.DIMENSIONLESS, verify_units=False
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
                    "f": policy_function(
                        leaf_name="f", unit=TTSIMUnit.DIMENSIONLESS, verify_units=False
                    )(return_x_kin),
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
            # Aggregations derived from target
            {
                "kin_id": kin_id,
                "p_id": p_id,
                "n1": {
                    "f": policy_function(leaf_name="f", unit=TTSIMUnit.DIMENSIONLESS)(
                        some_x
                    ),
                    "x": x,
                },
            },
            {"n1": {"f_kin": None}},
            {
                "n1": {"x": pd.Series([1, 1, 1])},
                "kin_id": pd.Series([0, 0, 0]),
                "p_id": pd.Series([0, 1, 2]),
            },
        ),
        (
            # Explicit aggregation via objects tree with leaf name input
            {
                "kin_id": kin_id,
                "p_id": p_id,
                "n1": {
                    "f": policy_function(leaf_name="f", unit=TTSIMUnit.DIMENSIONLESS)(
                        some_x
                    ),
                    "x": x,
                },
                "y_kin": y_kin,
            },
            {"n1": {"f": None}},
            {
                "n1": {"x": pd.Series([1, 1, 1])},
                "kin_id": pd.Series([0, 0, 0]),
                "p_id": pd.Series([0, 1, 2]),
            },
        ),
        (
            # Explicit aggregation via objects tree with namespaced input
            {
                "kin_id": kin_id,
                "p_id": p_id,
                "n1": {
                    "f": policy_function(
                        leaf_name="f", unit=TTSIMUnit.DIMENSIONLESS, verify_units=False
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
            },
        ),
    ],
)
def test_create_agg_by_group_functions(
    policy_environment,
    tt_targets__tree,
    input_data__tree,
    backend,
):
    main(
        main_target="results__tree",
        policy_environment=policy_environment,
        policy_date=datetime.date(2024, 1, 1),
        input_data=InputData.tree(tree=input_data__tree),
        evaluation_date=datetime.date(2024, 1, 1),
        tt_targets=TTTargets.tree(tt_targets__tree),
        rounding=False,
        backend=backend,
        unit_system=TEST_UNIT_SYSTEM,
    )


def test_output_is_tree(minimal_input_data, backend, xnp):
    policy_environment = {
        "p_id": p_id,
        "module": {"some_func": some_func},
    }

    out = main(
        main_target="results__tree",
        policy_environment=policy_environment,
        policy_date=datetime.date(2024, 1, 1),
        input_data=InputData.tree(tree=minimal_input_data),
        evaluation_date=datetime.date(2024, 1, 1),
        tt_targets=TTTargets.tree({"module": {"some_func": None}}),
        rounding=False,
        backend=backend,
        unit_system=TEST_UNIT_SYSTEM,
    )

    assert isinstance(out, dict)
    assert "some_func" in out["module"]
    assert isinstance(out["module"]["some_func"], xnp.ndarray)


def test_params_target_is_allowed(minimal_input_data):
    policy_environment = {
        "p_id": p_id,
        "module": {"some_func": some_func},
        "some_param": ScalarParam(
            value=1,
            start_date=datetime.date(2025, 1, 1),
            end_date=datetime.date(2025, 12, 31),
            # Parameters pin down the concrete currency (GEP 10); a complete
            # currency stock takes no period.
            unit=ttsim_unit_from_yaml_value(value="CASTAR", where="test setup"),
            name={"de": "Ein Parameter", "en": "Some parameter"},
            description={"de": "Ein Parameter", "en": "Some parameter"},
            note=None,
            reference=None,
        ),
    }

    out = main(
        main_target="results__tree",
        policy_environment=policy_environment,
        policy_date=datetime.date(2024, 1, 1),
        input_data=InputData.tree(tree=minimal_input_data),
        evaluation_date=datetime.date(2024, 1, 1),
        tt_targets=TTTargets.tree({"some_param": None, "module": {"some_func": None}}),
        rounding=False,
        backend="numpy",
        unit_system=TEST_UNIT_SYSTEM,
    )

    assert isinstance(out, dict)
    assert "some_param" in out
    assert out["some_param"] == 1


def test_function_without_data_dependency_is_not_mistaken_for_data(
    minimal_input_data,
    backend,
    xnp,
):
    @policy_function(
        leaf_name="a",
        vectorization_strategy="not_required",
        unit=TTSIMUnit.DIMENSIONLESS,
        verify_units=False,
    )
    def a() -> IntColumn:
        return xnp.array(minimal_input_data["p_id"])

    @policy_function(leaf_name="b", unit=TTSIMUnit.DIMENSIONLESS)
    def b(a: int) -> int:
        return a

    policy_environment = {
        "a": a,
        "b": b,
    }
    results__tree = main(
        main_target="results__tree",
        policy_environment=policy_environment,
        policy_date=datetime.date(2024, 1, 1),
        input_data=InputData.tree(tree=minimal_input_data),
        evaluation_date=datetime.date(2024, 1, 1),
        tt_targets=TTTargets.tree({"b": None}),
        rounding=False,
        backend=backend,
        unit_system=TEST_UNIT_SYSTEM,
    )
    numpy.testing.assert_array_almost_equal(
        results__tree["b"],
        xnp.array(minimal_input_data["p_id"]),
    )


def test_partial_params_to_functions(xnp, dnp, backend):
    # Partial function produces correct result
    func_after_partial = with_partialled_params_and_scalars(
        with_processed_params_and_scalars={
            "some_func": func_before_partial,
            "some_param": SOME_INT_PARAM.value,
        },
        len_p_id=1,
        dnp=dnp,
        rounding=False,
        xnp=xnp,
        backend=backend,
    )["some_func"]

    assert func_after_partial(2) == 3


def test_partial_params_to_functions_removes_argument(xnp, dnp, backend):
    func_after_partial = with_partialled_params_and_scalars(
        with_processed_params_and_scalars={
            "some_func": func_before_partial,
            "some_param": SOME_INT_PARAM.value,
        },
        len_p_id=1,
        rounding=False,
        xnp=xnp,
        dnp=dnp,
        backend=backend,
    )["some_func"]

    # Fails if params is added to partial function
    with pytest.raises(
        TypeError,
        match=("got multiple values for argument "),
    ):
        func_after_partial(2, 1)

    # No error for original function
    func_before_partial(arg_1=2, some_param=1)


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
    }

    expected = pd.Series([200, 200, 100], index=pd.Index(data["p_id"], name="p_id"))

    actual = main(
        main_target="results__df_with_nested_columns",
        policy_environment=policy_environment,
        policy_date=datetime.date(2024, 1, 1),
        input_data=InputData.tree(tree=data),
        evaluation_date=datetime.date(2024, 1, 1),
        tt_targets=TTTargets.tree({"module_name": {"betrag_m_fam": None}}),
        rounding=False,
        backend=backend,
        unit_system=TEST_UNIT_SYSTEM,
    )

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

    @policy_function(
        vectorization_strategy="vectorize", unit=TTSIMUnit.CURRENCY.PER_MONTH
    )
    def betrag_double_m(betrag_m: float) -> float:
        return 2 * betrag_m

    @agg_by_group_function(
        agg_type=AggType.MAX, unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_LEVEL("fam")
    )
    def betrag_double_m_fam(betrag_double_m: float, fam_id: int) -> float:
        pass

    policy_environment = {
        "p_id": p_id,
        "fam_id": fam_id,
        "module_name": {
            "betrag_double_m": betrag_double_m,
            "betrag_double_m_fam": betrag_double_m_fam,
        },
    }

    actual = main(
        main_target="results__df_with_nested_columns",
        policy_environment=policy_environment,
        policy_date=datetime.date(2024, 1, 1),
        input_data=InputData.tree(tree=data),
        evaluation_date=datetime.date(2024, 1, 1),
        tt_targets=TTTargets.tree({"module_name": {"betrag_double_m_fam": None}}),
        rounding=False,
        backend=backend,
        unit_system=TEST_UNIT_SYSTEM,
    )

    pd.testing.assert_series_equal(
        actual[("module_name", "betrag_double_m_fam")],
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

    @policy_function(
        vectorization_strategy="vectorize", unit=TTSIMUnit.CURRENCY.PER_MONTH
    )
    def betrag_double_m(betrag_m: float) -> float:
        return 2 * betrag_m

    @agg_by_group_function(
        agg_type=AggType.MAX, unit=TTSIMUnit.CURRENCY.PER_MONTH.PER_LEVEL("fam")
    )
    def max_betrag_double_m_fam(betrag_double_m: float, fam_id: int) -> float:
        pass

    policy_environment = {
        "p_id": p_id,
        "fam_id": fam_id,
        "module_name": {
            "betrag_double_m": betrag_double_m,
            "max_betrag_double_m_fam": max_betrag_double_m_fam,
        },
    }

    actual = main(
        main_target="results__df_with_nested_columns",
        policy_environment=policy_environment,
        policy_date=datetime.date(2024, 1, 1),
        input_data=InputData.tree(tree=data),
        evaluation_date=datetime.date(2024, 1, 1),
        tt_targets=TTTargets.tree({"module_name": {"max_betrag_double_y_fam": None}}),
        rounding=False,
        backend=backend,
        unit_system=TEST_UNIT_SYSTEM,
    )

    pd.testing.assert_series_equal(
        actual[("module_name", "max_betrag_double_y_fam")],
        expected,
        check_names=False,
        check_dtype=False,
    )


@agg_by_p_id_function(agg_type=AggType.SUM, unit=TTSIMUnit.DIMENSIONLESS)
def sum_source_by_p_id_someone_else(
    source: int,
    p_id: int,
    p_id_someone_else: int,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> int:
    pass


@agg_by_p_id_function(agg_type=AggType.SUM, unit=TTSIMUnit.CURRENCY.PER_MONTH)
def sum_source_m_by_p_id_someone_else(
    source_m: int,
    p_id: int,
    p_id_someone_else: int,
    num_segments: int,
    backend: Literal["numpy", "jax"],
) -> int:
    pass


@pytest.mark.parametrize(
    ("agg_functions", "leaf_name", "source_unit", "target_tree", "expected"),
    [
        (
            {
                "module": {
                    "sum_source_by_p_id_someone_else": sum_source_by_p_id_someone_else,
                },
            },
            "source",
            TTSIMUnit.DIMENSIONLESS,
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
            # The _m suffix denotes a flow, so a flow token is required.
            TTSIMUnit.CURRENCY.PER_MONTH,
            {"module": {"sum_source_m_by_p_id_someone_else": None}},
            pd.Series([200, 100, 0], index=pd.Index([0, 1, 2], name="p_id")),
        ),
    ],
)
def test_user_provided_aggregate_by_p_id_specs(
    agg_functions,
    leaf_name,
    source_unit,
    target_tree,
    expected,
    minimal_input_data_shared_fam,
    backend,
    xnp,
):
    @policy_function(
        leaf_name=leaf_name,
        vectorization_strategy="not_required",
        unit=source_unit,
        verify_units=False,
    )
    def source() -> IntColumn:
        return xnp.array([100, 200, 300])

    policy_environment = merge_trees(
        agg_functions,
        {
            "module": {leaf_name: source},
            "p_id": p_id,
            "p_id_someone_else": p_id_someone_else,
        },
    )

    actual = main(
        main_target="results__df_with_nested_columns",
        input_data=InputData.tree(tree=minimal_input_data_shared_fam),
        policy_environment=policy_environment,
        policy_date=datetime.date(2024, 1, 1),
        evaluation_date=datetime.date(2024, 1, 1),
        tt_targets=TTTargets.tree(target_tree),
        rounding=False,
        backend=backend,
        unit_system=TEST_UNIT_SYSTEM,
    )

    pd.testing.assert_series_equal(
        actual[("module", next(iter(target_tree["module"].keys())))],
        expected,
        check_names=False,
        check_dtype=False,
    )


def test_policy_environment_with_params_and_scalars_is_processed(
    xnp, dnp, backend, some_piecewise_polynomial_param
):
    policy_environment = {
        "raw_param_spec": SOME_RAW_PARAM,
        "some_int_param": SOME_INT_PARAM,
        "some_dict_param": SOME_DICT_PARAM,
        "some_piecewise_polynomial_param": some_piecewise_polynomial_param,
        "some_int_scalar": 1,
        "some_float_scalar": 2.0,
        "some_bool_scalar": True,
        "some_scalar_params_func": some_scalar_params_func,
        "some_converting_params_func": some_converting_params_func,
        "some_param_function_taking_scalar": some_param_function_taking_scalar,
    }
    actual = with_processed_params_and_scalars(
        without_tree_logic_and_with_derived_functions=policy_environment,
        processed_data={"x": xnp.array([1, 2, 3])},
        backend=backend,
        xnp=xnp,
        dnp=dnp,
        evaluation_date=datetime.date(2024, 1, 1),
    )
    expected = {
        "some_converting_params_func": ConvertedParam(
            some_float_param=1,
            some_bool_param=False,
        ),
        "some_scalar_params_func": 1,
        "some_int_param": SOME_INT_PARAM.value,
        "some_dict_param": SOME_DICT_PARAM.value,
        "some_piecewise_polynomial_param": some_piecewise_polynomial_param.value,
        "some_int_scalar": 1,
        "some_float_scalar": 2.0,
        "some_bool_scalar": True,
        "some_param_function_taking_scalar": 4.0,
        "evaluation_year": 2024,
        "evaluation_month": 1,
        "evaluation_day": 1,
    }
    assert actual == expected


@pytest.mark.parametrize(
    (
        "nested_policy_environment",
        "overriding_data",
        "tt_targets__tree",
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
    tt_targets__tree,
    expected_output,
    minimal_input_data,
    backend,
):
    actual = main(
        main_target="results__tree",
        input_data=InputData.tree(tree={**minimal_input_data, **overriding_data}),
        policy_environment=nested_policy_environment,
        policy_date=datetime.date(2024, 1, 1),
        evaluation_date=datetime.date(2024, 1, 1),
        tt_targets=TTTargets.tree(tt_targets__tree),
        include_warn_nodes=False,
        include_fail_nodes=False,
        rounding=False,
        backend=backend,
        unit_system=TEST_UNIT_SYSTEM,
    )

    flat_actual = dt.flatten_to_tree_paths(actual)
    flat_expected = dt.flatten_to_tree_paths(expected_output)

    assert flat_actual.keys() == flat_expected.keys()
    for key in flat_expected:
        numpy.testing.assert_array_almost_equal(flat_actual[key], flat_expected[key])


def test_scalars_in_input_data_become_part_of_specialized_environment(xnp, backend):
    policy_environment = {
        "identity": identity,
        "identity_plus_one": identity_plus_one,
    }
    input_data = {
        "p_id": xnp.array([1, 2, 3]),
        "identity": 1,
    }
    root_nodes = main(
        main_target=MainTarget.labels.root_nodes,
        policy_environment=policy_environment,
        policy_date=datetime.date(2024, 1, 1),
        input_data=InputData.tree(input_data),
        tt_targets=TTTargets.tree({"identity_plus_one": None}),
        evaluation_date_str="2024-01-01",
        backend=backend,
        include_warn_nodes=False,
        unit_system=TEST_UNIT_SYSTEM,
    )
    assert root_nodes == set()


def test_derived_time_converted_scalar_drives_derived_consumer(xnp, backend):
    """A scalar input whose qname is the source unit of a derived time-conversion
    function (`income_y` for `income_m` here) reaches the derived consumer's
    body: requesting `benefit_m` succeeds, returns `income_y / 12 * 0.5` for
    every row, and the scalar does not surface as an unbound root node.
    """
    policy_environment = {
        "p_id": p_id,
        "income_m": income_m,
        "benefit_m": benefit_m,
    }
    input_data = {
        "p_id": xnp.array([1, 2, 3]),
        "income_y": 12000,
    }
    common_kwargs = {
        "policy_environment": policy_environment,
        "input_data": InputData.tree(input_data),
        "tt_targets": TTTargets.tree({"benefit_m": None}),
        "policy_date_str": "2024-01-01",
        "evaluation_date_str": "2024-01-01",
        "backend": backend,
        "include_warn_nodes": False,
        "include_fail_nodes": False,
    }
    root_nodes = main(
        main_target=MainTarget.labels.root_nodes,
        **common_kwargs,  # ty: ignore[invalid-argument-type]
        unit_system=TEST_UNIT_SYSTEM,
    )
    assert root_nodes == set()
    result = main(
        main_target=MainTarget.results.tree,
        **common_kwargs,  # ty: ignore[invalid-argument-type]
        unit_system=TEST_UNIT_SYSTEM,
    )
    numpy.testing.assert_allclose(result["benefit_m"], numpy.full(3, 500.0))
