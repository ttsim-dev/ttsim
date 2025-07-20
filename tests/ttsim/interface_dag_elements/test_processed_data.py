from __future__ import annotations

import numpy
import pandas as pd
import pytest

from ttsim.interface_dag_elements.processed_data import processed_data


@pytest.fixture
def input_data__flat():
    return {
        ("p_id",): numpy.array([5, 333, 7, 2]),
        ("hh_id",): numpy.array([55555, 7, 3, 55555]),
        ("n0", "p_id_whatever"): numpy.array([-1, 333, 5, -1]),
    }


def test_processed_data(input_data__flat, xnp):
    expected = {
        "p_id": xnp.array([1, 3, 2, 0]),
        "hh_id": xnp.array([2, 1, 0, 2]),
        "n0__p_id_whatever": xnp.array([-1, 3, 1, -1]),
    }
    pd.testing.assert_frame_equal(
        pd.DataFrame(processed_data(input_data__flat=input_data__flat, xnp=xnp)),
        pd.DataFrame(expected),
    )


def test_processed_data_foreign_key_out_of_bounds(xnp):
    # Add invalid numbers (-5, too large), in foreign key. Should be unchanged, error
    # will be raised in `fail_if.foreign_keys_are_invalid_in_data` if the
    input_data__flat = {
        ("p_id",): numpy.array([5, 333, 7, 2]),
        ("hh_id",): numpy.array([55555, 7, 3, 55555]),
        ("n0", "p_id_whatever"): numpy.array([-1, 333, -5, 999]),
    }
    expected = {
        "p_id": xnp.array([1, 3, 2, 0]),
        "hh_id": xnp.array([2, 1, 0, 2]),
        "n0__p_id_whatever": xnp.array([-1, 3, -5, 999]),  # 999 preserved unchanged
    }
    pd.testing.assert_frame_equal(
        pd.DataFrame(processed_data(input_data__flat=input_data__flat, xnp=xnp)),
        pd.DataFrame(expected),
    )
