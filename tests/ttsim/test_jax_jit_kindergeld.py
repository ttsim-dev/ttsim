from pathlib import Path

import dags.tree as dt
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from _gettsim.kindergeld.kindergeld import betrag_ohne_staffelung_m as betrag_m
from _gettsim_tests.utils import (
    cached_set_up_policy_environment,
    load_policy_test_data,
)
from ttsim import compute_taxes_and_transfers
from ttsim.config import IS_JAX_INSTALLED

if IS_JAX_INSTALLED:
    import jax

SRC = Path().parent.parent / "src"
TEST_DATA = SRC / "_gettsim_tests" / "test_data"


# ======================================================================================
# Unit tests for each policy function
# ======================================================================================


@pytest.mark.skipif(not IS_JAX_INSTALLED, reason="JAX is not installed")
def test_kindergeld_policy_func():
    policy_func = betrag_m
    policy_func_jitted = jax.jit(policy_func)

    inputs = {
        "anzahl_ansprüche": jax.numpy.array([1, 2, 3]),
        # params are not vectorized over
        "kindergeld_params": {"kindergeld": 250},
    }
    policy_func_jitted(**inputs)


# ======================================================================================
# End-to-end tests (for compute_taxes_and_transfers)
# ======================================================================================


@pytest.fixture
def kindergeld_policy_test():
    name = "alleinerz_2_children_low_unterhalt.yaml"
    kindergeld_2024 = load_policy_test_data("kindergeld/2024")
    single_test = [
        test_data for test_data in kindergeld_2024 if test_data.path.name == name
    ]
    return single_test[1]  # index=1 -> betrag_m


@pytest.mark.skipif(not IS_JAX_INSTALLED, reason="JAX is not installed")
def test_compute_taxes_and_transfers_kindergeld(kindergeld_policy_test):
    test = kindergeld_policy_test

    environment = cached_set_up_policy_environment(date=test.date)

    result = compute_taxes_and_transfers(
        data_tree=test.input_tree,
        environment=environment,
        targets_tree=test.target_structure,
        jit=True,
    )

    flat_result = dt.flatten_to_qual_names(result)
    flat_expected_output_tree = dt.flatten_to_qual_names(test.expected_output_tree)

    if flat_expected_output_tree:
        result_dataframe = pd.DataFrame(flat_result)
        expected_dataframe = pd.DataFrame(flat_expected_output_tree)
        assert_frame_equal(
            result_dataframe,
            expected_dataframe,
            atol=test.info["precision"],
            check_dtype=False,
        )
