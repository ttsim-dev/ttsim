"""Canonicalization of pandas-nullable and pyarrow input dtypes.

ttsim accepts inputs in many dtypes (numpy, pandas-nullable, pyarrow-backed)
and canonicalizes them to plain numpy arrays the TT DAG can operate on:

- Float-nullable / float-pyarrow → ``float64`` with ``pd.NA`` mapped to ``NaN``.
- Int/UInt-nullable / int-pyarrow / uint-pyarrow without NA → ``int64``.
- Bool-nullable / bool-pyarrow without NA → numpy ``bool_``.
- Int/UInt/Bool columns with NA → fail with a precise per-column error so
  the user can decide whether to fill the missing values or convert the
  column to a float type that supports NaN.
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from tests.test_unit_system import TEST_UNIT_SYSTEM
from ttsim import InputData, MainTarget, TTTargets, main
from ttsim.interface_dag_elements.processed_data import (
    _canonicalize_input_dtype,
)

if TYPE_CHECKING:
    from types import ModuleType
    from typing import Literal


_DATE = datetime.date(2025, 1, 1)


def _pyarrow_or_skip(dtype: str) -> str:
    pytest.importorskip("pyarrow")
    return dtype


def test_float_nullable_input_with_na_round_trips_as_nan(
    backend: Literal["numpy", "jax"],
):
    """A `Float64`-nullable input column with a `pd.NA` survives as `NaN`
    when requested back as an input-data target.
    """
    df = pd.DataFrame(
        {
            ("p_id",): [1, 2, 3],
            ("wage_m",): pd.array([100.0, pd.NA, 300.0], dtype="Float64"),
        },
    )
    result = main(
        main_target=MainTarget.results.df_with_nested_columns,
        input_data=InputData.df_with_nested_columns(df),
        tt_targets=TTTargets.tree({"wage_m": None}),
        policy_environment={},
        policy_date=datetime.date(2025, 1, 1),
        evaluation_date=datetime.date(2025, 1, 1),
        backend=backend,
        unit_system=TEST_UNIT_SYSTEM,
    )

    expected = pd.DataFrame(
        {("wage_m",): [100.0, np.nan, 300.0]},
        index=pd.Index([1, 2, 3], name="p_id"),
    )
    pd.testing.assert_frame_equal(
        expected, result, check_dtype=False, check_index_type=False
    )


@pytest.mark.parametrize(
    ("series_factory", "expected_dtype_kind"),
    [
        (lambda: pd.Series([1, 2, 3], dtype="Int8"), "i"),
        (lambda: pd.Series([1, 2, 3], dtype="Int16"), "i"),
        (lambda: pd.Series([1, 2, 3], dtype="Int32"), "i"),
        (lambda: pd.Series([1, 2, 3], dtype="Int64"), "i"),
        (lambda: pd.Series([1, 2, 3], dtype="UInt8"), "i"),
        (lambda: pd.Series([1, 2, 3], dtype="UInt16"), "i"),
        (lambda: pd.Series([1, 2, 3], dtype="UInt32"), "i"),
        (lambda: pd.Series([1, 2, 3], dtype="UInt64"), "i"),
        (lambda: pd.Series([1.0, 2.0, 3.0], dtype="Float32"), "f"),
        (lambda: pd.Series([1.0, 2.0, 3.0], dtype="Float64"), "f"),
        (lambda: pd.Series([True, False, True], dtype="boolean"), "b"),
        (
            lambda: pd.Series([1, 2, 3], dtype=_pyarrow_or_skip("int32[pyarrow]")),
            "i",
        ),
        (
            lambda: pd.Series([1, 2, 3], dtype=_pyarrow_or_skip("uint16[pyarrow]")),
            "i",
        ),
        (
            lambda: pd.Series(
                [1.0, 2.0, 3.0], dtype=_pyarrow_or_skip("float64[pyarrow]")
            ),
            "f",
        ),
        (
            lambda: pd.Series(
                [True, False, True], dtype=_pyarrow_or_skip("bool[pyarrow]")
            ),
            "b",
        ),
    ],
)
def test_canonicalize_input_dtype_normalises_extension_dtypes(
    series_factory, expected_dtype_kind, xnp: ModuleType
):
    """Each nullable / pyarrow column without NAs ends up as a numpy array
    with a backend-native dtype kind (``i`` for integers, ``f`` for floats,
    ``b`` for booleans). JAX may choose narrower widths than int64 when its
    x64 mode is off — the test asserts kind, not exact width.
    """
    result = _canonicalize_input_dtype(arr=series_factory(), xnp=xnp)
    assert result.dtype.kind == expected_dtype_kind


def test_canonicalize_input_dtype_floats_map_na_to_nan(xnp: ModuleType):
    series = pd.Series([1.0, pd.NA, 3.0], dtype="Float64")
    result = _canonicalize_input_dtype(arr=series, xnp=xnp)
    assert result.dtype.kind == "f"
    assert bool(xnp.isnan(result[1]))


def test_int_input_with_na_fails_with_actionable_message(
    backend: Literal["numpy", "jax"],
):
    """Integer columns with `pd.NA` cannot be represented in numpy `int64`
    — surface the offending qname and row position so the user knows what
    to fix.
    """
    df = pd.DataFrame(
        {
            ("p_id",): [1, 2, 3],
            ("age",): pd.array([25, pd.NA, 45], dtype="Int64"),
        },
    )
    with pytest.raises(ValueError, match=r"(?s)age.*first\s+NA\s+at\s+row\s+1"):
        main(
            main_target=MainTarget.results.df_with_nested_columns,
            input_data=InputData.df_with_nested_columns(df),
            tt_targets=TTTargets.tree({"age": None}),
            policy_environment={},
            policy_date=datetime.date(2025, 1, 1),
            evaluation_date=_DATE,
            backend=backend,
            unit_system=TEST_UNIT_SYSTEM,
        )


def test_bool_input_with_na_fails_with_actionable_message(
    backend: Literal["numpy", "jax"],
):
    """Boolean columns with `pd.NA` are also rejected — numpy has no
    nullable bool sentinel."""
    df = pd.DataFrame(
        {
            ("p_id",): [1, 2, 3],
            ("is_eligible",): pd.array([True, pd.NA, False], dtype="boolean"),
        },
    )
    with pytest.raises(ValueError, match=r"(?s)is_eligible.*first\s+NA\s+at\s+row\s+1"):
        main(
            main_target=MainTarget.results.df_with_nested_columns,
            input_data=InputData.df_with_nested_columns(df),
            tt_targets=TTTargets.tree({"is_eligible": None}),
            policy_environment={},
            policy_date=datetime.date(2025, 1, 1),
            evaluation_date=_DATE,
            backend=backend,
            unit_system=TEST_UNIT_SYSTEM,
        )


def test_multiple_int_or_bool_columns_with_na_all_reported(
    backend: Literal["numpy", "jax"],
):
    """When several int / bool columns carry NAs, all of them are surfaced
    in the single error message — no need to fix-then-rerun in a loop.
    """
    df = pd.DataFrame(
        {
            ("p_id",): [1, 2, 3],
            ("age",): pd.array([25, pd.NA, 45], dtype="Int64"),
            ("is_eligible",): pd.array([pd.NA, True, False], dtype="boolean"),
        },
    )
    with pytest.raises(
        ValueError,
        match=(
            r"(?s)age.*first\s+NA\s+at\s+row\s+1"
            r".*is_eligible.*first\s+NA\s+at\s+row\s+0"
        ),
    ):
        main(
            main_target=MainTarget.results.df_with_nested_columns,
            input_data=InputData.df_with_nested_columns(df),
            tt_targets=TTTargets.tree({"age": None, "is_eligible": None}),
            policy_environment={},
            policy_date=datetime.date(2025, 1, 1),
            evaluation_date=_DATE,
            backend=backend,
            unit_system=TEST_UNIT_SYSTEM,
        )


def test_pyarrow_int_input_round_trips_as_int(backend: Literal["numpy", "jax"]):
    """A pyarrow-backed integer column without NAs round-trips through
    ttsim and lands back as an integer-typed result column.
    """
    pytest.importorskip("pyarrow")
    df = pd.DataFrame(
        {
            ("p_id",): [1, 2, 3],
            ("age",): pd.array([25, 35, 45], dtype="int32[pyarrow]"),
        },
    )
    result = main(
        main_target=MainTarget.results.df_with_nested_columns,
        input_data=InputData.df_with_nested_columns(df),
        tt_targets=TTTargets.tree({"age": None}),
        policy_environment={},
        policy_date=datetime.date(2025, 1, 1),
        evaluation_date=_DATE,
        backend=backend,
        unit_system=TEST_UNIT_SYSTEM,
    )
    assert result[("age",)].tolist() == [25, 35, 45]


def test_uint64_overflow_fails_with_actionable_message(
    backend: Literal["numpy", "jax"],
):
    """A raw ``numpy.uint64`` array containing a value above ``int64.max``
    cannot be coerced to int64 safely. The user gets a ``ValueError`` naming
    the offending qname and value rather than silent wrap-around in
    downstream signed arithmetic.
    """
    int64_max = np.iinfo(np.int64).max
    overflowing = np.array([1, 2, np.uint64(int64_max) + 1], dtype=np.uint64)
    flat = {
        ("p_id",): np.array([1, 2, 3]),
        ("balance",): overflowing,
    }
    with pytest.raises(ValueError, match=r"(?s)int64\s+max.*balance"):
        main(
            main_target=MainTarget.results.df_with_nested_columns,
            input_data=InputData.flat(flat),  # ty: ignore[invalid-argument-type]
            tt_targets=TTTargets.tree({"balance": None}),
            policy_environment={},
            policy_date=datetime.date(2025, 1, 1),
            evaluation_date=_DATE,
            backend=backend,
            unit_system=TEST_UNIT_SYSTEM,
        )


def test_uint64_overflow_from_dataframe_fails_with_actionable_message(
    backend: Literal["numpy", "jax"],
):
    """A pandas-nullable ``UInt64`` column with a value above ``int64.max``
    raises the same ``ValueError`` as the raw-numpy path; without the
    canonicalize-time check, the value would silently wrap when cast to
    int64 inside ``_canonicalize_series``.
    """
    int64_max = np.iinfo(np.int64).max
    df = pd.DataFrame(
        {
            ("p_id",): [1, 2, 3],
            ("balance",): pd.array([1, 2, int64_max + 1], dtype="UInt64"),
        },
    )
    with pytest.raises(ValueError, match=r"(?s)balance.*int64\s+max"):
        main(
            main_target=MainTarget.results.df_with_nested_columns,
            input_data=InputData.df_with_nested_columns(df),
            tt_targets=TTTargets.tree({"balance": None}),
            policy_environment={},
            policy_date=datetime.date(2025, 1, 1),
            evaluation_date=_DATE,
            backend=backend,
            unit_system=TEST_UNIT_SYSTEM,
        )
