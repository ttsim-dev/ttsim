from __future__ import annotations

import numpy
import pandas as pd
import pytest

from ttsim.interface_dag_elements.input_data import sort_indices
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
            )
        ),
        pd.DataFrame(expected),
    )
