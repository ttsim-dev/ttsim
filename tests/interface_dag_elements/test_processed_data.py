from __future__ import annotations

import numpy
import pandas as pd
import pytest

from ttsim.exceptions import UnitDefinitionError
from ttsim.interface_dag_elements.input_data import sort_indices
from ttsim.interface_dag_elements.processed_data import (
    _canonicalize_input_dtype,
    processed_data,
)
from ttsim.tt import Unit, policy_input
from ttsim.tt.units import UNIT_REGISTRY


@pytest.fixture
def input_data__flat():
    return {
        ("p_id",): numpy.array([5, 333, 7, 2]),
        ("hh_id",): numpy.array([55555, 7, 3, 55555]),
        ("n0", "p_id_whatever"): numpy.array([-1, 333, 5, -1]),
    }


def test_processed_data(input_data__flat, xnp):
    expected = {
        "p_id": xnp.array([0, 1, 2, 3]),
        "hh_id": xnp.array([2, 2, 0, 1]),
        "n0__p_id_whatever": xnp.array([-1, -1, 1, 3]),
    }
    pd.testing.assert_frame_equal(
        pd.DataFrame(
            processed_data(
                input_data__flat=input_data__flat,
                input_data__sort_indices=sort_indices(
                    input_data__flat=input_data__flat, xnp=xnp
                ),
                xnp=xnp,
                policy_environment={},
                labels__grouping_levels=(),
                data_currency="CASTAR",
                computation_currency="CASTAR",
            )
        ),
        pd.DataFrame(expected),
    )


def test_processed_data_foreign_key_out_of_bounds(xnp):
    # Add out-of-bounds numbers (-5, 999), in foreign key. Should be unchanged, error
    # will be raised in `fail_if.foreign_keys_are_invalid_in_data`.
    input_data__flat = {
        ("p_id",): numpy.array([2, 5, 7, 333]),
        ("hh_id",): numpy.array([55555, 55555, 3, 7]),
        ("n0", "p_id_whatever"): numpy.array([999, -1, -5, 333]),
    }
    input_data__sort_indices = sort_indices(input_data__flat=input_data__flat, xnp=xnp)

    expected = {
        "p_id": xnp.array([0, 1, 2, 3]),
        "hh_id": xnp.array([2, 2, 0, 1]),
        "n0__p_id_whatever": xnp.array([999, -1, -5, 3]),
    }
    pd.testing.assert_frame_equal(
        pd.DataFrame(
            processed_data(
                input_data__flat=input_data__flat,
                input_data__sort_indices=input_data__sort_indices,
                xnp=xnp,
                policy_environment={},
                labels__grouping_levels=(),
                data_currency="CASTAR",
                computation_currency="CASTAR",
            )
        ),
        pd.DataFrame(expected),
    )


def test_processed_data_foreign_key_inside_bounds(xnp):
    # Add non-existent foreign key (22). Should be unchanged, error will be raised in
    # `fail_if.foreign_keys_are_invalid_in_data`.
    input_data__flat = {
        ("p_id",): numpy.array([2, 5, 7, 333]),
        ("hh_id",): numpy.array([55555, 55555, 4444, 7]),
        ("n0", "p_id_whatever"): numpy.array([-1, -1, 3, 333]),
    }
    input_data__sort_indices = sort_indices(input_data__flat=input_data__flat, xnp=xnp)

    expected = {
        "p_id": xnp.array([0, 1, 2, 3]),
        "hh_id": xnp.array([2, 2, 1, 0]),
        "n0__p_id_whatever": xnp.array([-1, -1, 3, 3]),
    }
    pd.testing.assert_frame_equal(
        pd.DataFrame(
            processed_data(
                input_data__flat=input_data__flat,
                input_data__sort_indices=input_data__sort_indices,
                xnp=xnp,
                policy_environment={},
                labels__grouping_levels=(),
                data_currency="CASTAR",
                computation_currency="CASTAR",
            )
        ),
        pd.DataFrame(expected),
    )


def test_processed_data_single_column(xnp):
    """Test processed_data with a single column (p_id only)."""
    input_data__flat = {
        ("p_id",): numpy.array([3, 1, 2]),
    }
    input_data__sort_indices = sort_indices(input_data__flat=input_data__flat, xnp=xnp)

    expected = {
        "p_id": xnp.array([0, 1, 2]),
    }

    pd.testing.assert_frame_equal(
        pd.DataFrame(
            processed_data(
                input_data__flat=input_data__flat,
                input_data__sort_indices=input_data__sort_indices,
                xnp=xnp,
                policy_environment={},
                labels__grouping_levels=(),
                data_currency="CASTAR",
                computation_currency="CASTAR",
            )
        ),
        pd.DataFrame(expected),
    )


def _pyarrow_uint32_series():
    pytest.importorskip("pyarrow")
    return pd.Series([0, 100], dtype="uint32[pyarrow]")


@pytest.mark.parametrize(
    "uint_input_factory",
    [
        lambda: numpy.array([0, 100], dtype=numpy.uint8),
        lambda: numpy.array([0, 100], dtype=numpy.uint16),
        lambda: numpy.array([0, 100], dtype=numpy.uint32),
        lambda: numpy.array([0, 100], dtype=numpy.uint64),
        lambda: pd.Series([0, 100], dtype=numpy.uint32),
        lambda: pd.Series([0, 100], dtype="UInt32"),
        _pyarrow_uint32_series,
    ],
    ids=[
        "numpy_uint8",
        "numpy_uint16",
        "numpy_uint32",
        "numpy_uint64",
        "pd_series_numpy_uint32",
        "pd_series_UInt32_nullable",
        "pd_series_uint32_pyarrow",
    ],
)
def test_canonicalize_input_dtype_returns_signed_array(uint_input_factory, xnp):
    """Coerced output is a signed integer (exact width depends on the backend:
    int64 on numpy, int32 on jax with the default x64-disabled config).
    """
    uint_input = uint_input_factory()
    result = _canonicalize_input_dtype(arr=uint_input, xnp=xnp)
    assert result.dtype.kind == "i"
    assert int(result[0]) == 0
    assert int(result[1]) == 100
    # Subtracting a larger value stays signed instead of wrapping into a huge
    # positive uint value.
    assert int(result[0] - xnp.asarray(1230, dtype=result.dtype)) == -1230


def test_canonicalize_input_dtype_passes_non_uint_through(xnp):
    arr = numpy.array([-5, 0, 5], dtype=numpy.int32)
    result = _canonicalize_input_dtype(arr=arr, xnp=xnp)
    assert result.dtype == xnp.int32
    assert int(result[0]) == -5


def test_processed_data_coerces_uint_columns_to_signed(xnp):
    input_data__flat = {
        ("p_id",): numpy.array([5, 7], dtype=numpy.uint32),
        ("wage",): numpy.array([0, 100], dtype=numpy.uint32),
    }
    result = processed_data(
        input_data__flat=input_data__flat,
        input_data__sort_indices=sort_indices(
            input_data__flat=input_data__flat, xnp=xnp
        ),
        xnp=xnp,
        policy_environment={},
        labels__grouping_levels=(),
        data_currency="CASTAR",
        computation_currency="CASTAR",
    )
    assert result["wage"].dtype.kind == "i"
    # Subtraction stays signed instead of underflowing into uint wraparound.
    diff = result["wage"] - xnp.asarray([1230, 50], dtype=result["wage"].dtype)
    assert int(diff[0]) == -1230
    assert int(diff[1]) == 50


def test_processed_data_single_row(xnp):
    """Test processed_data with a single row."""
    input_data__flat = {
        ("p_id",): numpy.array([42]),
        ("hh_id",): numpy.array([100]),
    }
    input_data__sort_indices = sort_indices(input_data__flat=input_data__flat, xnp=xnp)

    expected = {
        "p_id": xnp.array([0]),
        "hh_id": xnp.array([0]),
    }

    pd.testing.assert_frame_equal(
        pd.DataFrame(
            processed_data(
                input_data__flat=input_data__flat,
                input_data__sort_indices=input_data__sort_indices,
                xnp=xnp,
                policy_environment={},
                labels__grouping_levels=(),
                data_currency="CASTAR",
                computation_currency="CASTAR",
            )
        ),
        pd.DataFrame(expected),
    )


def test_processed_data_converts_pint_tagged_currency_input(xnp):
    """Layer-2 boundary (GEP 10): a tagged input is converted to the data currency.

    End-to-end through the ``processed_data`` interface node: a wealth column
    handed in as silver pennies rides along castar data and is rescaled at the
    boundary (4 silver pennies = 1 castar).
    """
    from mettsim import middle_earth  # noqa: F401, PLC0415 (registers the currencies)

    input_data__flat = {
        ("p_id",): numpy.array([0, 1]),
        ("wealth",): UNIT_REGISTRY.Quantity(numpy.array([4.0, 8.0]), "SILVER_PENNY"),
    }
    out = processed_data(
        input_data__flat=input_data__flat,
        input_data__sort_indices=sort_indices(
            input_data__flat=input_data__flat, xnp=xnp
        ),
        xnp=xnp,
        policy_environment={},
        labels__grouping_levels=(),
        data_currency="CASTAR",
        computation_currency="CASTAR",
    )
    assert not isinstance(out["wealth"], UNIT_REGISTRY.Quantity)
    assert list(numpy.asarray(out["wealth"])) == pytest.approx([1.0, 2.0])


def test_processed_data_converts_untagged_currency_input_by_declared_unit(xnp):
    """The column boundary on the way in (GEP 10): untagged data converts, too.

    Data arrives in the data currency; every column whose *declared* unit
    carries a currency component is converted to the computation currency.
    Non-currency columns are untouched.
    """
    from mettsim import middle_earth  # noqa: F401, PLC0415 (registers the currencies)

    @policy_input(unit=Unit.CURRENCY)
    def wealth() -> float:
        pass

    @policy_input(unit=Unit.YEARS)
    def age() -> int:
        pass

    input_data__flat = {
        ("p_id",): numpy.array([0, 1]),
        ("wealth",): numpy.array([4.0, 8.0]),
        ("age",): numpy.array([30, 40]),
    }
    out = processed_data(
        input_data__flat=input_data__flat,
        input_data__sort_indices=sort_indices(
            input_data__flat=input_data__flat, xnp=xnp
        ),
        xnp=xnp,
        policy_environment={"wealth": wealth, "age": age},
        labels__grouping_levels=("hh",),
        data_currency="SILVER_PENNY",
        computation_currency="CASTAR",
    )
    assert list(numpy.asarray(out["wealth"])) == pytest.approx([1.0, 2.0])
    assert list(numpy.asarray(out["age"])) == [30, 40]


def test_derived_input_ignores_siblings_a_derivation_cannot_produce(xnp):
    """`income_hh` aggregates the declared non-currency `income`; the currency
    flow `income_m` shares the base name but no derivation produces `income_hh`
    from it (a time suffix is only ever rebased, never dropped), so the column
    must not convert."""
    from mettsim import middle_earth  # noqa: F401, PLC0415 (registers the currencies)

    @policy_input(unit=Unit.CURRENCY.PER_MONTH)
    def income_m() -> float:
        pass

    @policy_input(unit=Unit.YEARS)
    def income() -> int:
        pass

    input_data__flat = {
        ("p_id",): numpy.array([0, 1]),
        ("income_hh",): numpy.array([4.0, 4.0]),
    }
    out = processed_data(
        input_data__flat=input_data__flat,
        input_data__sort_indices=sort_indices(
            input_data__flat=input_data__flat, xnp=xnp
        ),
        xnp=xnp,
        policy_environment={"income_m": income_m, "income": income},
        labels__grouping_levels=("hh",),
        data_currency="SILVER_PENNY",
        computation_currency="CASTAR",
    )
    assert list(numpy.asarray(out["income_hh"])) == pytest.approx([4.0, 4.0])


def test_derived_input_with_disagreeing_sources_is_rejected(xnp):
    """Two declared flows share a base name but disagree on carrying a currency;
    a derived name that could come from either is a loud error, not a guess."""
    from mettsim import middle_earth  # noqa: F401, PLC0415 (registers the currencies)

    @policy_input(unit=Unit.CURRENCY.PER_MONTH)
    def income_m() -> float:
        pass

    @policy_input(unit=Unit.HOURS.PER_WEEK)
    def income_w() -> float:
        pass

    input_data__flat = {
        ("p_id",): numpy.array([0, 1]),
        ("income_y_hh",): numpy.array([4.0, 4.0]),
    }
    with pytest.raises(UnitDefinitionError, match="disagree"):
        processed_data(
            input_data__flat=input_data__flat,
            input_data__sort_indices=sort_indices(
                input_data__flat=input_data__flat, xnp=xnp
            ),
            xnp=xnp,
            policy_environment={"income_m": income_m, "income_w": income_w},
            labels__grouping_levels=("hh",),
            data_currency="SILVER_PENNY",
            computation_currency="CASTAR",
        )


def test_processed_data_converts_derived_input_via_its_family(xnp):
    """A derived name provided as data inherits its family's currency presence.

    ``wage_m_hh`` has no declaration of its own; the declared ``wage_m`` in the
    same namespace makes it a currency column (aggregation never adds or
    removes the currency component), so it converts at the boundary.
    """
    from mettsim import middle_earth  # noqa: F401, PLC0415 (registers the currencies)

    @policy_input(unit=Unit.CURRENCY.PER_MONTH)
    def wage_m() -> float:
        pass

    input_data__flat = {
        ("p_id",): numpy.array([0, 1]),
        ("wage_m_hh",): numpy.array([4.0, 4.0]),
    }
    out = processed_data(
        input_data__flat=input_data__flat,
        input_data__sort_indices=sort_indices(
            input_data__flat=input_data__flat, xnp=xnp
        ),
        xnp=xnp,
        policy_environment={"wage_m": wage_m},
        labels__grouping_levels=("hh",),
        data_currency="SILVER_PENNY",
        computation_currency="CASTAR",
    )
    assert list(numpy.asarray(out["wage_m_hh"])) == pytest.approx([1.0, 1.0])
