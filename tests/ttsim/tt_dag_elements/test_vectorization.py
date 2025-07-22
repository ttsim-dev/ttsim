from __future__ import annotations

import datetime
import functools
import inspect
import string
from pathlib import Path
from typing import TYPE_CHECKING

import dags.tree as dt
import numpy
import pytest
from numpy.testing import assert_array_equal

from ttsim.interface_dag_elements.orig_policy_objects import (
    column_objects_and_param_functions,
)
from ttsim.interface_dag_elements.policy_environment import (
    _active_column_objects_and_param_functions,
)
from ttsim.tt_dag_elements import (
    GroupCreationFunction,
    PolicyInput,
    policy_function,
)
from ttsim.tt_dag_elements.column_objects_param_function import (
    AggByGroupFunction,
    AggByPIDFunction,
)
from ttsim.tt_dag_elements.vectorization import (
    TranslateToVectorizableError,
    _is_lambda_function,
    _make_vectorizable,
    make_vectorizable_source,
    vectorize_function,
)

if TYPE_CHECKING:
    from types import ModuleType

    from ttsim.interface_dag_elements.typing import IntColumn

# ======================================================================================
# String comparison
# ======================================================================================


def string_equal(s1, s2):
    remove = string.punctuation + string.whitespace
    mapping = {ord(c): None for c in remove}
    return s1.translate(mapping) == s2.translate(mapping)


def test_compare_str():
    assert string_equal("This ! is a     test.", "This is a test")
    assert not string_equal("This is a test", "This is not a test")


# ======================================================================================
# Test functions (no error)
# ======================================================================================


def f1(x):
    if x < 0:
        return 0
    else:
        return 1


def f1_exp(x):
    return numpy.where(x < 0, 0, 1)


def f2(x):
    if x < 0:
        out = 0
    else:
        out = 1
    return out


def f2_exp(x):
    out = numpy.where(x < 0, 0, 1)
    return out


def f3(x):
    return 0 if x < 0 else 1


def f3_exp(x):
    return numpy.where(x < 0, 0, 1)


def f4(x):
    out = 1
    if x < 0:
        out = 0
    return out


def f4_exp(x):
    out = 1
    out = numpy.where(x < 0, 0, out)
    return out


def f5(x):
    if x < 0:
        out = -1
    elif x > 0:
        out = 1
    else:
        out = 0
    return out


def f5_exp(x):
    out = numpy.where(x < 0, -1, numpy.where(x > 0, 1, 0))
    return out


def f6(flag, another_flag):
    if flag and not another_flag:
        out = 1
    else:
        out = 0
    return out


def f6_exp(flag, another_flag):
    out = numpy.where(numpy.logical_and(flag, numpy.logical_not(another_flag)), 1, 0)
    return out


def f7(x):
    out = 0 if x < 0 else 1
    return out


def f7_exp(x):
    out = numpy.where(x < 0, 0, 1)
    return out


def f8(x):
    return -1 if x < 0 else (1 if x > 0 else 0)


def f8_exp(x):
    return numpy.where(x < 0, -1, numpy.where(x > 0, 1, 0))


# expect no change since there is no if-clause and no [and|or] statement.
def f9(x):
    y = numpy.sum(x)
    z = numpy.prod(x)
    return y * z


def f10(x):
    flag = (x < 0) and (x > -1)
    another_flag = (x < 0) or (x > -1)
    return flag and not another_flag


def f10_exp(x):
    flag = numpy.logical_and(x < 0, x > -1)
    another_flag = numpy.logical_or(x < 0, x > -1)
    return numpy.logical_and(flag, numpy.logical_not(another_flag))


def f11(x):
    if x < 0:
        out = -1
    else:
        out = 1 if x > 0 else 0
    return out


def f11_exp(x):
    out = numpy.where(x < 0, -1, numpy.where(x > 0, 1, 0))
    return out


def f12(x):
    out = 0
    if x < 1:
        out += 1
    return out


def f12_exp(x):
    out = 0
    out += numpy.where(x < 1, 1, out)
    return out


def f13(x):
    a = x < 0
    b = x > 0
    c = x != 0
    d = True
    return ((a and b) or c) and d


def f13_exp(x):
    a = x < 0
    b = x > 0
    c = x != 0
    d = True
    return numpy.logical_and(numpy.logical_or(numpy.logical_and(a, b), c), d)


def f14(x):
    a = x < 0
    b = x > 0
    c = x != 0
    d = True
    return (a and b and c) or d


def f14_exp(x):
    a = x < 0
    b = x > 0
    c = x != 0
    d = True
    return numpy.logical_or(numpy.logical_and(numpy.logical_and(a, b), c), d)


def f15(x):
    return min(x, 0)


def f15_exp(x):
    return numpy.minimum(x, 0)


def f16(x):
    return float(sum(x))


def f16_exp(x):
    return float(numpy.sum(x))


def f17(x):
    a = x < 0
    b = x // 2
    return any((a, b))


def f17_exp(x):
    a = x < 0
    b = x // 2
    return numpy.any((a, b))


def f18(x):
    return int(any(x)) + 1


def f18_exp(x):
    return int(numpy.any(x)) + 1


x = numpy.arange(-10, 10)
rng = numpy.random.default_rng(seed=0)
flag = rng.binomial(1, 0.25, size=100)
another_flag = rng.binomial(1, 0.75, size=100)


TEST_CASES = [
    (f1, f1_exp, (x,)),
    (f2, f2_exp, (x,)),
    (f3, f3_exp, (x,)),
    (f4, f4_exp, (x,)),
    (f5, f5_exp, (x,)),
    (f6, f6_exp, (flag, another_flag)),
    (f7, f7_exp, (x,)),
    (f8, f8_exp, (x,)),
    (f9, f9, (x,)),
    (f10, f10_exp, (x,)),
    (f11, f11_exp, (x,)),
    (f12, f12_exp, (x,)),
    (f13, f13_exp, (x,)),
    (f14, f14_exp, (x,)),
    (f15, f15_exp, (x,)),
    (f16, f16_exp, (x,)),
    (f17, f17_exp, (x,)),
    (f18, f18_exp, (x,)),
]


# ======================================================================================
# Tests (no error)
# ======================================================================================


@pytest.mark.parametrize(("func", "expected", "args"), TEST_CASES)
def test_change_if_to_where_source(func, expected, args):  # noqa: ARG001
    exp = inspect.getsource(expected)
    exp = exp.replace("_exp", "")
    got = make_vectorizable_source(func, backend="numpy", xnp=numpy)
    assert string_equal(exp, got)


@pytest.mark.parametrize(("func", "expected", "args"), TEST_CASES)
def test_change_if_to_where_wrapper(func, expected, args):
    got_func = _make_vectorizable(func, backend="numpy", xnp=numpy)
    got = got_func(*args)
    exp = expected(*args)
    assert_array_equal(got, exp)


# ======================================================================================
# Test correct error raising
# ======================================================================================


def g1(x):
    # function with multiple operations in the if-clause
    a = 0
    b = 1
    if x < 0:
        a = 1
        b = 0
    return a + b


def g2(x):
    # function with illegal operations in the if-clause
    if x < 0:
        print(x)  # noqa: T201
    else:
        print(not x)  # noqa: T201


def g3(x):
    # return statement in if-body but no else clause
    if x < 0:
        return 0
    return 1


def g4(x):
    # max with three arguments
    return max(x, 0, 1)


def test_notimplemented_error():
    with pytest.raises(NotImplementedError):
        _make_vectorizable(f1, backend="dask", xnp=numpy)


@pytest.mark.parametrize("func", [g1, g2, g3, g4])
def test_disallowed_operation_source(func):
    with pytest.raises(TranslateToVectorizableError):
        make_vectorizable_source(func, backend="numpy", xnp=numpy)


@pytest.mark.parametrize("func", [g1, g2, g3, g4])
def test_disallowed_operation_wrapper(func):
    with pytest.raises(TranslateToVectorizableError):
        _make_vectorizable(func, backend="numpy", xnp=numpy)


# ======================================================================================
# Test that functions defined in METTSIM can be made vectorizable
# ======================================================================================


for year in range(1990, 2023):

    @pytest.mark.parametrize(
        ("funcname", "func"),
        (
            (funcname, pf.function)
            for funcname, pf in dt.flatten_to_tree_paths(
                _active_column_objects_and_param_functions(
                    orig=column_objects_and_param_functions(
                        root=Path(__file__).parent.parent / "mettsim"
                    ),
                    policy_date=datetime.date(year=year, month=1, day=1),
                ),
            ).items()
            if not isinstance(
                pf,
                GroupCreationFunction
                | AggByGroupFunction
                | AggByPIDFunction
                | PolicyInput,
            )
        ),
    )
    def test_convertible(funcname, func, backend, xnp):  # noqa: ARG001
        # Leave funcname for debugging purposes.
        _make_vectorizable(func, backend=backend, xnp=xnp)


# ======================================================================================
# Test that vectorized functions defined in METTSIM can be called with array input
# ======================================================================================


def test_housing_benefits_amount_m_fam(backend, xnp):
    """Test housing benefits amount function with conditional logic."""
    # Test original function on scalar input
    # ==============================================================================
    eligibility__requirement_fulfilled_fam = True
    income__amount_m_fam = 1000.0
    assistance_rate = 0.8

    from tests.ttsim.mettsim.housing_benefits.amount import (  # noqa: PLC0415
        amount_m_fam,
    )

    exp = amount_m_fam.function(
        eligibility__requirement_fulfilled_fam=eligibility__requirement_fulfilled_fam,
        income__amount_m_fam=income__amount_m_fam,
        assistance_rate=assistance_rate,
    )
    assert exp == 800.0

    exp_false = amount_m_fam.function(
        eligibility__requirement_fulfilled_fam=False,
        income__amount_m_fam=income__amount_m_fam,
        assistance_rate=assistance_rate,
    )
    assert exp_false == 0.0

    # Create array inputs and assert that original function raises error
    # ==============================================================================
    shape = (10, 2)
    eligibility__requirement_fulfilled_fam = xnp.full(shape, True)  # noqa: FBT003
    income__amount_m_fam = xnp.full(shape, income__amount_m_fam)
    assistance_rate = xnp.full(shape, assistance_rate)

    with pytest.raises(ValueError, match="truth value of an array with more than"):
        amount_m_fam.function(
            eligibility__requirement_fulfilled_fam=eligibility__requirement_fulfilled_fam,
            income__amount_m_fam=income__amount_m_fam,
            assistance_rate=assistance_rate,
        )

    # Call converted function on array input and test result
    # ==============================================================================
    converted = _make_vectorizable(
        amount_m_fam.function,
        backend=backend,
        xnp=xnp,
    )
    got = converted(
        eligibility__requirement_fulfilled_fam=eligibility__requirement_fulfilled_fam,
        income__amount_m_fam=income__amount_m_fam,
        assistance_rate=assistance_rate,
    )
    assert_array_equal(got, xnp.full(shape, exp))

    # Test mixed eligibility
    eligibility__requirement_fulfilled_fam = xnp.array([[True, False], [False, True]])
    income__amount_m_fam = xnp.array([[1000.0, 1000.0], [1000.0, 1000.0]])
    assistance_rate = xnp.array([[0.8, 0.8], [0.8, 0.8]])

    got_mixed = converted(
        eligibility__requirement_fulfilled_fam=eligibility__requirement_fulfilled_fam,
        income__amount_m_fam=income__amount_m_fam,
        assistance_rate=assistance_rate,
    )
    expected_mixed = xnp.array([[800.0, 0.0], [0.0, 800.0]])
    assert_array_equal(got_mixed, expected_mixed)


def test_payroll_tax_amount_y(backend, xnp):
    """Test payroll tax amount function with multiple conditional logic."""
    # Test original function on scalar input
    # ==============================================================================
    amount_standard_y = 1000.0
    amount_reduced_y = 800.0
    parent_is_noble_fam = False
    wealth_fam = 30000.0  # Below threshold
    wealth_threshold_for_reduced_tax_rate = 40000.0

    from tests.ttsim.mettsim.payroll_tax.amount import amount_y  # noqa: PLC0415

    exp_standard = amount_y.function(
        amount_standard_y=amount_standard_y,
        amount_reduced_y=amount_reduced_y,
        parent_is_noble_fam=parent_is_noble_fam,
        wealth_fam=wealth_fam,
        wealth_threshold_for_reduced_tax_rate=wealth_threshold_for_reduced_tax_rate,
    )
    assert exp_standard == 1000.0

    exp_reduced = amount_y.function(
        amount_standard_y=amount_standard_y,
        amount_reduced_y=amount_reduced_y,
        parent_is_noble_fam=parent_is_noble_fam,
        wealth_fam=60000.0,
        wealth_threshold_for_reduced_tax_rate=wealth_threshold_for_reduced_tax_rate,
    )
    assert exp_reduced == 800.0

    exp_noble = amount_y.function(
        amount_standard_y=amount_standard_y,
        amount_reduced_y=amount_reduced_y,
        parent_is_noble_fam=True,
        wealth_fam=wealth_fam,
        wealth_threshold_for_reduced_tax_rate=wealth_threshold_for_reduced_tax_rate,
    )
    assert exp_noble == 0.0

    # Create array inputs and assert that original function raises error
    # ==============================================================================
    shape = (10, 2)
    amount_standard_y = xnp.full(shape, amount_standard_y)
    amount_reduced_y = xnp.full(shape, amount_reduced_y)
    parent_is_noble_fam = xnp.full(shape, parent_is_noble_fam)
    wealth_fam = xnp.full(shape, wealth_fam)
    wealth_threshold_for_reduced_tax_rate = xnp.full(
        shape, wealth_threshold_for_reduced_tax_rate
    )

    with pytest.raises(ValueError, match="truth value of an array with more than"):
        amount_y.function(
            amount_standard_y=amount_standard_y,
            amount_reduced_y=amount_reduced_y,
            parent_is_noble_fam=parent_is_noble_fam,
            wealth_fam=wealth_fam,
            wealth_threshold_for_reduced_tax_rate=wealth_threshold_for_reduced_tax_rate,
        )

    # Call converted function on array input and test result
    # ==============================================================================
    converted = _make_vectorizable(
        amount_y.function,
        backend=backend,
        xnp=xnp,
    )
    got = converted(
        amount_standard_y=amount_standard_y,
        amount_reduced_y=amount_reduced_y,
        parent_is_noble_fam=parent_is_noble_fam,
        wealth_fam=wealth_fam,
        wealth_threshold_for_reduced_tax_rate=wealth_threshold_for_reduced_tax_rate,
    )
    assert_array_equal(got, xnp.full(shape, exp_standard))

    # Test mixed conditions
    parent_is_noble_fam = xnp.array([[True, False], [False, True]])
    wealth_fam = xnp.array([[30000.0, 60000.0], [30000.0, 70000.0]])
    amount_standard_y = xnp.array([[1000.0, 1000.0], [1000.0, 1000.0]])
    amount_reduced_y = xnp.array([[800.0, 800.0], [800.0, 800.0]])
    wealth_threshold_for_reduced_tax_rate = xnp.array(
        [[40000.0, 40000.0], [40000.0, 40000.0]]
    )

    got_mixed = converted(
        amount_standard_y=amount_standard_y,
        amount_reduced_y=amount_reduced_y,
        parent_is_noble_fam=parent_is_noble_fam,
        wealth_fam=wealth_fam,
        wealth_threshold_for_reduced_tax_rate=wealth_threshold_for_reduced_tax_rate,
    )
    # Expected: noble=0, reduced=800, standard=1000, noble=0
    expected_mixed = xnp.array([[0.0, 800.0], [1000.0, 0.0]])
    assert_array_equal(got_mixed, expected_mixed)


def test_orc_hunting_bounty_amount(backend, xnp):
    """Test orc hunting bounty function with conditional logic."""
    # Test original function on scalar input
    # ==============================================================================
    small_orcs_hunted = 5
    large_orcs_hunted = 2
    parent_is_noble = True
    bounty_per_orc = type(
        "BountyPerOrc",
        (),
        {
            "small_orc": 10,
            "large_orc": type(
                "BountyPerLargeOrc", (), {"noble_hunter": 50, "peasant_hunter": 30}
            )(),
        },
    )()

    from tests.ttsim.mettsim.orc_hunting_bounty.orc_hunting_bounty import (  # noqa: PLC0415
        amount,
    )

    exp_noble = amount.function(
        small_orcs_hunted=small_orcs_hunted,
        large_orcs_hunted=large_orcs_hunted,
        parent_is_noble=parent_is_noble,
        bounty_per_orc=bounty_per_orc,
    )
    assert exp_noble == 150.0  # 5*10 + 2*50

    exp_peasant = amount.function(
        small_orcs_hunted=small_orcs_hunted,
        large_orcs_hunted=large_orcs_hunted,
        parent_is_noble=False,
        bounty_per_orc=bounty_per_orc,
    )
    assert exp_peasant == 110.0  # 5*10 + 2*30

    # Create array inputs and assert that original function raises error
    # ==============================================================================
    shape = (10, 2)
    small_orcs_hunted = xnp.full(shape, small_orcs_hunted)
    large_orcs_hunted = xnp.full(shape, large_orcs_hunted)
    parent_is_noble = xnp.full(shape, parent_is_noble)

    with pytest.raises(ValueError, match="truth value of an array with more than"):
        amount.function(
            small_orcs_hunted=small_orcs_hunted,
            large_orcs_hunted=large_orcs_hunted,
            parent_is_noble=parent_is_noble,
            bounty_per_orc=bounty_per_orc,
        )

    # Call converted function on array input and test result
    # ==============================================================================
    converted = _make_vectorizable(
        amount.function,
        backend=backend,
        xnp=xnp,
    )
    got = converted(
        small_orcs_hunted=small_orcs_hunted,
        large_orcs_hunted=large_orcs_hunted,
        parent_is_noble=parent_is_noble,
        bounty_per_orc=bounty_per_orc,
    )
    assert_array_equal(got, xnp.full(shape, exp_noble))

    # Test mixed noble/peasant conditions
    parent_is_noble = xnp.array([[True, False], [False, True]])
    small_orcs_hunted = xnp.array([[5, 5], [5, 5]])
    large_orcs_hunted = xnp.array([[2, 2], [2, 2]])

    got_mixed = converted(
        small_orcs_hunted=small_orcs_hunted,
        large_orcs_hunted=large_orcs_hunted,
        parent_is_noble=parent_is_noble,
        bounty_per_orc=bounty_per_orc,
    )
    # Expected: noble=150, peasant=110, peasant=110, noble=150
    expected_mixed = xnp.array([[150.0, 110.0], [110.0, 150.0]])
    assert_array_equal(got_mixed, expected_mixed)


# ======================================================================================
# Lambda functions
# ======================================================================================


def test_is_lambda_function_true():
    assert _is_lambda_function(lambda x: x)


def test_is_lambda_function_wrapped():
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    assert _is_lambda_function(decorator(lambda x: x))


def test_is_lambda_function_false():
    def f(x):
        return x

    assert not _is_lambda_function(f)


def test_is_lambda_function_non_function_input():
    assert not _is_lambda_function(42)
    assert not _is_lambda_function("not a function")
    assert not _is_lambda_function([1, 2, 3])
    assert not _is_lambda_function({1: "a", 2: "b"})
    assert not _is_lambda_function(None)


def test_lambda_functions_disallowed_make_vectorizable(xnp):
    with pytest.raises(TranslateToVectorizableError, match="Lambda functions are not"):
        _make_vectorizable(lambda x: x, backend="numpy", xnp=xnp)


def test_lambda_functions_disallowed_make_vectorizable_source(xnp):
    with pytest.raises(TranslateToVectorizableError, match="Lambda functions are not"):
        make_vectorizable_source(lambda x: x, backend="numpy", xnp=xnp)


# ======================================================================================
# Policy functions
# ======================================================================================


def test_make_vectorizable_policy_func(backend, xnp):
    @policy_function()
    def alter_bis_24(alter: int) -> bool:
        return alter <= 24

    vectorized = alter_bis_24.vectorize(backend=backend, xnp=xnp)

    got = vectorized(xnp.array([20, 25, 30]))
    exp = xnp.array([True, False, False])
    assert_array_equal(got, exp)


# ======================================================================================
# Dags functions
# ======================================================================================


def test_make_vectorizable_nested_func():
    def f_a(x: int) -> int:
        return x

    def f_b(a: int) -> int:
        return a + 2

    def f_manual(x: int) -> int:
        return f_b(f_a(x))

    vectorized = _make_vectorizable(f_manual, backend="numpy", xnp=numpy)
    got = vectorized(numpy.array([1, 2, 3]))
    exp = numpy.array([3, 4, 5])
    assert_array_equal(got, exp)


@policy_function()
def scalar_func(x: int) -> int:
    if x < 0:
        return 0
    else:
        return x * 2


@policy_function(vectorization_strategy="not_required")
def already_vectorized_func(x: IntColumn, xnp: ModuleType) -> IntColumn:
    return xnp.where(x < 0, 0, x * 2)


def test_loop_vectorize_scalar_func(backend, xnp):
    fun = vectorize_function(
        scalar_func.function,
        vectorization_strategy="loop",
        backend=backend,
        xnp=numpy,
    )
    assert numpy.array_equal(fun(xnp.array([-1, 0, 2, 3])), xnp.array([0, 0, 4, 6]))


def test_vectorize_scalar_func(backend, xnp):
    fun = vectorize_function(
        scalar_func.function,
        vectorization_strategy="vectorize",
        backend=backend,
        xnp=numpy,
    )
    assert numpy.array_equal(fun(xnp.array([-1, 0, 2, 3])), xnp.array([0, 0, 4, 6]))


def test_already_vectorized_func(xnp):
    assert numpy.array_equal(
        already_vectorized_func(xnp.array([-1, 0, 2, 3]), xnp),
        xnp.array([0, 0, 4, 6]),
    )


def test_vectorize_function_annotations(backend, xnp):
    def f(a, x: int, y: float, z: bool, p1: str, p2: dict[str, float]) -> float:  # noqa: ARG001
        return 1.0

    vectorized = vectorize_function(
        f,
        vectorization_strategy="vectorize",
        backend=backend,
        xnp=xnp,
    )

    expected_annotations = {
        "a": "IntColumn | FloatColumn | BoolColumn",
        "x": "IntColumn",
        "y": "FloatColumn",
        "z": "BoolColumn",
        "p1": "str",
        "p2": "dict[str, float]",
        "return": "FloatColumn",
    }
    assert inspect.get_annotations(vectorized) == expected_annotations


# ======================================================================================
# Test forbidden type conversions and augmented assignments
# ======================================================================================


def forbidden_type_conversion_float(x):
    return float(x)


def forbidden_type_conversion_int(x):
    return int(x)


def forbidden_type_conversion_bool(x):
    return bool(x)


def forbidden_type_conversion_complex(x):
    return complex(x)


def forbidden_type_conversion_str(x):
    return str(x)


def forbidden_augassign_add(x):
    y = x
    y += 1
    return y


def forbidden_augassign_sub(x):
    y = x
    y -= 1
    return y


def forbidden_augassign_mult(x):
    y = x
    y *= 2
    return y


def forbidden_augassign_div(x):
    y = x
    y /= 2
    return y


@pytest.mark.parametrize(
    "func",
    [
        forbidden_type_conversion_float,
        forbidden_type_conversion_int,
        forbidden_type_conversion_bool,
        forbidden_type_conversion_complex,
        forbidden_type_conversion_str,
    ],
)
def test_forbidden_type_conversions_raise(func, xnp):
    """Test that forbidden type conversions raise the correct error."""
    with pytest.raises(TranslateToVectorizableError, match="Forbidden type conversion"):
        _make_vectorizable(func, backend="numpy", xnp=xnp)
    with pytest.raises(TranslateToVectorizableError, match="Forbidden type conversion"):
        make_vectorizable_source(func, backend="numpy", xnp=xnp)


@pytest.mark.parametrize(
    "func",
    [
        forbidden_augassign_add,
        forbidden_augassign_sub,
        forbidden_augassign_mult,
        forbidden_augassign_div,
    ],
)
def test_forbidden_augassign_raise(func, xnp):
    """Test that forbidden augmented assignments raise the correct error."""
    with pytest.raises(
        TranslateToVectorizableError, match="Forbidden augmented assignment"
    ):
        _make_vectorizable(func, backend="numpy", xnp=xnp)
    with pytest.raises(
        TranslateToVectorizableError, match="Forbidden augmented assignment"
    ):
        make_vectorizable_source(func, backend="numpy", xnp=xnp)
