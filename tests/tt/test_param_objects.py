import dataclasses

import numpy
import pytest

from ttsim.tt.param_objects import (
    ConsecutiveIntLookupTableParamValue,
    DictParam,
    RawParam,
    ScalarParam,
    _year_fraction,
    convert_sparse_to_consecutive_int_lookup_table,
    get_consecutive_int_lookup_table_param_value,
    get_month_based_phase_inout_of_age_thresholds_param_value,
    get_year_based_phase_inout_of_age_thresholds_param_value,
)

# =============================================================================
# convert_sparse_to_consecutive_int_lookup_table tests
# =============================================================================


@pytest.mark.parametrize(
    (
        "raw",
        "expected_result",
    ),
    [
        (
            {1: 1, 3: 3, "min_int_in_table": 0, "max_int_in_table": 5},
            {0: 1, 1: 1, 2: 1, 3: 3, 4: 3, 5: 3},
        ),
        (
            {1: 1, 3: 0, "min_int_in_table": 0, "max_int_in_table": 5},
            {0: 1, 1: 1, 2: 1, 3: 0, 4: 0, 5: 0},
        ),
    ],
)
def test_convert_sparse_to_consecutive_int_lookup_table(raw, expected_result, xnp):
    result = convert_sparse_to_consecutive_int_lookup_table(raw, xnp)
    for k, v in expected_result.items():
        assert result.look_up(k) == v


def test_convert_sparse_not_dict_raises(xnp):
    """Test that non-dict input raises TypeError."""
    with pytest.raises(TypeError, match="must be a dictionary"):
        convert_sparse_to_consecutive_int_lookup_table([1, 2, 3], xnp)


def test_convert_sparse_missing_min_max_raises(xnp):
    """Test that missing min_int_in_table or max_int_in_table raises TypeError."""
    with pytest.raises(TypeError, match=r"min_int_in_table.*max_int_in_table"):
        convert_sparse_to_consecutive_int_lookup_table({1: 1, 2: 2}, xnp)

    with pytest.raises(TypeError, match=r"min_int_in_table.*max_int_in_table"):
        convert_sparse_to_consecutive_int_lookup_table(
            {1: 1, "min_int_in_table": 0}, xnp
        )


def test_convert_sparse_non_int_min_max_raises(xnp):
    """Test that non-integer min_int_in_table or max_int_in_table raises TypeError."""
    with pytest.raises(TypeError, match="must be integers"):
        convert_sparse_to_consecutive_int_lookup_table(
            {1: 1, "min_int_in_table": 0.5, "max_int_in_table": 5}, xnp
        )

    with pytest.raises(TypeError, match="must be integers"):
        convert_sparse_to_consecutive_int_lookup_table(
            {1: 1, "min_int_in_table": 0, "max_int_in_table": "5"}, xnp
        )


def test_convert_sparse_non_int_keys_raises(xnp):
    """Test that non-integer keys in raw dict raises TypeError."""
    with pytest.raises(TypeError, match="int keys"):
        convert_sparse_to_consecutive_int_lookup_table(
            {"a": 1, "min_int_in_table": 0, "max_int_in_table": 5}, xnp
        )


def test_convert_sparse_min_larger_than_smallest_key_raises(xnp):
    """Test that min_int_in_table > smallest key raises ValueError."""
    with pytest.raises(ValueError, match=r"smallest integer.*must not be larger"):
        convert_sparse_to_consecutive_int_lookup_table(
            {0: 1, 3: 3, "min_int_in_table": 2, "max_int_in_table": 5}, xnp
        )


def test_convert_sparse_max_smaller_than_largest_key_raises(xnp):
    """Test that max_int_in_table < largest key raises ValueError."""
    with pytest.raises(ValueError, match=r"largest integer.*must not be smaller"):
        convert_sparse_to_consecutive_int_lookup_table(
            {1: 1, 4: 4, "min_int_in_table": 0, "max_int_in_table": 4}, xnp
        )


# =============================================================================
# ConsecutiveIntLookupTableParamValue tests
# =============================================================================


def test_lookup_table_init_basic(xnp):
    """Test basic initialization of ConsecutiveIntLookupTableParamValue."""
    values = xnp.array([10.0, 20.0, 30.0])
    bases = xnp.array([0])

    lut = ConsecutiveIntLookupTableParamValue(
        xnp=xnp, values_to_look_up=values, bases_to_subtract=bases
    )

    assert lut.xnp is xnp
    numpy.testing.assert_array_equal(lut.values_to_look_up, values.flatten())


def test_lookup_table_init_2d(xnp):
    """Test initialization with 2D values."""
    values = xnp.array([[10.0, 20.0], [30.0, 40.0]])
    bases = xnp.array([0, 0])

    lut = ConsecutiveIntLookupTableParamValue(
        xnp=xnp, values_to_look_up=values, bases_to_subtract=bases
    )

    # Values should be flattened
    numpy.testing.assert_array_equal(
        lut.values_to_look_up, xnp.array([10.0, 20.0, 30.0, 40.0])
    )


def test_lookup_table_look_up_single_dim(xnp):
    """Test lookup with single-dimensional table."""
    values = xnp.array([10.0, 20.0, 30.0])
    bases = xnp.array([0])

    lut = ConsecutiveIntLookupTableParamValue(
        xnp=xnp, values_to_look_up=values, bases_to_subtract=bases
    )

    numpy.testing.assert_almost_equal(lut.look_up(0), 10.0)
    numpy.testing.assert_almost_equal(lut.look_up(1), 20.0)
    numpy.testing.assert_almost_equal(lut.look_up(2), 30.0)


def test_lookup_table_look_up_with_nonzero_base(xnp):
    """Test lookup with non-zero base subtraction."""
    values = xnp.array([10.0, 20.0, 30.0])
    bases = xnp.array([5])  # Subtract 5 from indices

    lut = ConsecutiveIntLookupTableParamValue(
        xnp=xnp, values_to_look_up=values, bases_to_subtract=bases
    )

    # Index 5 maps to values[0], etc.
    numpy.testing.assert_almost_equal(lut.look_up(5), 10.0)
    numpy.testing.assert_almost_equal(lut.look_up(6), 20.0)
    numpy.testing.assert_almost_equal(lut.look_up(7), 30.0)


# =============================================================================
# get_consecutive_int_lookup_table_param_value tests
# =============================================================================


def test_get_consecutive_int_lookup_table_1d(xnp):
    """Test 1D lookup table creation."""
    raw = {0: 10.0, 1: 20.0, 2: 30.0}
    result = get_consecutive_int_lookup_table_param_value(raw, xnp)

    assert result.look_up(0) == 10.0
    assert result.look_up(1) == 20.0
    assert result.look_up(2) == 30.0


def test_get_consecutive_int_lookup_table_with_nonzero_min(xnp):
    """Test lookup table with non-zero minimum key."""
    raw = {5: 100.0, 6: 200.0, 7: 300.0}
    result = get_consecutive_int_lookup_table_param_value(raw, xnp)

    assert result.look_up(5) == 100.0
    assert result.look_up(6) == 200.0
    assert result.look_up(7) == 300.0


# =============================================================================
# get_month_based_phase_inout_of_age_thresholds_param_value tests
# =============================================================================


def test_month_based_phase_inout_basic(xnp):
    """Test basic month-based phase-in/out parameter creation."""
    raw = {
        "first_year_to_consider": 2020,
        "last_year_to_consider": 2022,
        2020: {1: {"years": 65, "months": 0}},
        2021: {1: {"years": 65, "months": 6}},
        2022: {1: {"years": 66, "months": 0}},
    }
    result = get_month_based_phase_inout_of_age_thresholds_param_value(raw.copy(), xnp)

    # Month since AD for 2020-01 = 2020 * 12 + 0 = 24240
    m_2020_01 = 2020 * 12 + 0
    m_2021_01 = 2021 * 12 + 0
    m_2022_01 = 2022 * 12 + 0

    # 65 years 0 months = 65.0
    numpy.testing.assert_almost_equal(result.look_up(m_2020_01), 65.0)
    # 65 years 6 months = 65.5
    numpy.testing.assert_almost_equal(result.look_up(m_2021_01), 65.5)
    # 66 years 0 months = 66.0
    numpy.testing.assert_almost_equal(result.look_up(m_2022_01), 66.0)


def test_month_based_phase_inout_fill_gaps(xnp):
    """Test that gaps in months are filled with previous value."""
    raw = {
        "first_year_to_consider": 2020,
        "last_year_to_consider": 2020,
        2020: {
            1: {"years": 65, "months": 0},
            6: {"years": 65, "months": 6},
        },
    }
    result = get_month_based_phase_inout_of_age_thresholds_param_value(raw.copy(), xnp)

    m_2020_01 = 2020 * 12 + 0
    m_2020_02 = 2020 * 12 + 1
    m_2020_05 = 2020 * 12 + 4
    m_2020_06 = 2020 * 12 + 5

    # Months 2-5 should have same value as month 1
    numpy.testing.assert_almost_equal(result.look_up(m_2020_01), 65.0)
    numpy.testing.assert_almost_equal(result.look_up(m_2020_02), 65.0)
    numpy.testing.assert_almost_equal(result.look_up(m_2020_05), 65.0)
    numpy.testing.assert_almost_equal(result.look_up(m_2020_06), 65.5)


def test_month_based_phase_inout_first_year_error(xnp):
    """Test error when first_year_to_consider > first year in data."""
    raw = {
        "first_year_to_consider": 2021,  # After 2020
        "last_year_to_consider": 2022,
        2020: {1: {"years": 65, "months": 0}},
        2022: {1: {"years": 66, "months": 0}},
    }
    with pytest.raises(ValueError, match="first_m_since_ad_to_consider"):
        get_month_based_phase_inout_of_age_thresholds_param_value(raw.copy(), xnp)


def test_month_based_phase_inout_last_year_error(xnp):
    """Test error when last_year_to_consider < last year in data."""
    raw = {
        "first_year_to_consider": 2020,
        "last_year_to_consider": 2021,  # Before 2022
        2020: {1: {"years": 65, "months": 0}},
        2022: {1: {"years": 66, "months": 0}},
    }
    with pytest.raises(ValueError, match="last_m_since_ad_to_consider"):
        get_month_based_phase_inout_of_age_thresholds_param_value(raw.copy(), xnp)


def test_month_based_phase_inout_non_int_keys_error(xnp):
    """Test error when year keys are not integers."""
    raw = {
        "first_year_to_consider": 2020,
        "last_year_to_consider": 2022,
        "2020": {1: {"years": 65, "months": 0}},  # String key instead of int
    }
    with pytest.raises(ValueError, match="All keys must be integers"):
        get_month_based_phase_inout_of_age_thresholds_param_value(raw.copy(), xnp)


# =============================================================================
# get_year_based_phase_inout_of_age_thresholds_param_value tests
# =============================================================================


def test_year_based_phase_inout_basic(xnp):
    """Test basic year-based phase-in/out parameter creation."""
    raw = {
        "first_year_to_consider": 2020,
        "last_year_to_consider": 2022,
        2020: {"years": 65, "months": 0},
        2021: {"years": 65, "months": 6},
        2022: {"years": 66, "months": 0},
    }
    result = get_year_based_phase_inout_of_age_thresholds_param_value(raw.copy(), xnp)

    numpy.testing.assert_almost_equal(result.look_up(2020), 65.0)
    numpy.testing.assert_almost_equal(result.look_up(2021), 65.5)
    numpy.testing.assert_almost_equal(result.look_up(2022), 66.0)


def test_year_based_phase_inout_first_year_error(xnp):
    """Test error when first_year_to_consider > first year in data."""
    raw = {
        "first_year_to_consider": 2021,  # After 2020
        "last_year_to_consider": 2022,
        2020: {"years": 65, "months": 0},
        2022: {"years": 66, "months": 0},
    }
    with pytest.raises(ValueError, match="first_year_to_consider"):
        get_year_based_phase_inout_of_age_thresholds_param_value(raw.copy(), xnp)


def test_year_based_phase_inout_last_year_error(xnp):
    """Test error when last_year_to_consider < last year in data."""
    raw = {
        "first_year_to_consider": 2020,
        "last_year_to_consider": 2021,  # Before 2022
        2020: {"years": 65, "months": 0},
        2022: {"years": 66, "months": 0},
    }
    with pytest.raises(ValueError, match="last_year_to_consider"):
        get_year_based_phase_inout_of_age_thresholds_param_value(raw.copy(), xnp)


def test_year_based_phase_inout_non_int_keys_error(xnp):
    """Test error when year keys are not integers."""
    raw = {
        "first_year_to_consider": 2020,
        "last_year_to_consider": 2022,
        "2020": {"years": 65, "months": 0},  # String key instead of int
    }
    with pytest.raises(ValueError, match="All keys must be integers"):
        get_year_based_phase_inout_of_age_thresholds_param_value(raw.copy(), xnp)


def test_year_based_phase_inout_fills_before_and_after(xnp):
    """Test that values before/after phase are filled correctly."""
    raw = {
        "first_year_to_consider": 2018,
        "last_year_to_consider": 2025,  # Must be > 2022
        2020: {"years": 65, "months": 0},
        2021: {"years": 65, "months": 6},
        2022: {"years": 66, "months": 0},
    }
    result = get_year_based_phase_inout_of_age_thresholds_param_value(raw.copy(), xnp)

    # Years before phase-inout should have first value
    numpy.testing.assert_almost_equal(result.look_up(2018), 65.0)
    numpy.testing.assert_almost_equal(result.look_up(2019), 65.0)
    numpy.testing.assert_almost_equal(result.look_up(2020), 65.0)

    # Years after phase-inout should have last value
    numpy.testing.assert_almost_equal(result.look_up(2023), 66.0)
    numpy.testing.assert_almost_equal(result.look_up(2024), 66.0)
    numpy.testing.assert_almost_equal(result.look_up(2025), 66.0)


# =============================================================================
# ScalarParam tests
# =============================================================================


def test_scalar_param_with_bool_value():
    """Test ScalarParam accepts bool value."""
    param = ScalarParam(value=True)
    assert param.value is True


def test_scalar_param_with_int_value():
    """Test ScalarParam accepts int value."""
    param = ScalarParam(value=42)
    assert param.value == 42


def test_scalar_param_with_float_value():
    """Test ScalarParam accepts float value."""
    param = ScalarParam(value=3.14)
    assert param.value == 3.14


def test_scalar_param_with_note_and_reference():
    """Test ScalarParam accepts optional note and reference fields."""
    param = ScalarParam(value=100, note="A note", reference="GEP-5")
    assert param.value == 100
    assert param.note == "A note"
    assert param.reference == "GEP-5"


def test_scalar_param_is_frozen():
    """Test ScalarParam is immutable (frozen dataclass)."""
    param = ScalarParam(value=10)
    with pytest.raises(dataclasses.FrozenInstanceError):
        param.value = 20  # ty: ignore[frozen-instance]


# =============================================================================
# DictParam tests
# =============================================================================


def test_dict_param_with_str_int_values():
    """Test DictParam with dict[str, int] value."""
    param = DictParam(value={"a": 1, "b": 2})
    assert param.value == {"a": 1, "b": 2}


def test_dict_param_rejects_note_key():
    """Test DictParam raises ValueError when 'note' is a key in value."""
    with pytest.raises(ValueError, match="'note' and 'reference' cannot be keys"):
        DictParam(value={"note": 1, "other": 2})


def test_dict_param_rejects_reference_key():
    """Test DictParam raises ValueError when 'reference' is a key in value."""
    with pytest.raises(ValueError, match="'note' and 'reference' cannot be keys"):
        DictParam(value={"reference": "some_ref", "other": 2})


# =============================================================================
# RawParam tests
# =============================================================================


def test_raw_param_with_nested_dict():
    """Test RawParam supports nested dict values."""
    nested = {"level1": {"level2": {"level3": 42}}}
    param = RawParam(value=nested)
    assert param.value == nested
    assert param.value["level1"]["level2"]["level3"] == 42


def test_raw_param_rejects_note_key():
    """Test RawParam raises ValueError when 'note' is a key in value."""
    with pytest.raises(ValueError, match="'note' and 'reference' cannot be keys"):
        RawParam(value={"note": "forbidden", "other": 1})


def test_raw_param_rejects_reference_key():
    """Test RawParam raises ValueError when 'reference' is a key in value."""
    with pytest.raises(ValueError, match="'note' and 'reference' cannot be keys"):
        RawParam(value={"reference": "forbidden", "other": 1})


# =============================================================================
# _year_fraction tests
# =============================================================================


def test_year_fraction_years_only():
    """Test _year_fraction with years only (months=0)."""
    result = _year_fraction({"years": 65, "months": 0})
    assert result == 65.0


def test_year_fraction_with_months():
    """Test _year_fraction with years and months."""
    result = _year_fraction({"years": 65, "months": 6})
    assert result == 65.5


def test_year_fraction_zero_years():
    """Test _year_fraction with zero years."""
    result = _year_fraction({"years": 0, "months": 6})
    assert result == 0.5


def test_year_fraction_full_year_months():
    """Test _year_fraction with 12 months equals 1 year."""
    result = _year_fraction({"years": 0, "months": 12})
    assert result == 1.0
