from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pytest
from mettsim.config import METTSIM_ROOT

from ttsim.config import IS_JAX_INSTALLED
from ttsim.testing_utils import (
    PolicyTest,
    execute_test,
    load_policy_test_data,
)

TEST_DIR = Path(__file__).parent

POLICY_TEST_IDS_AND_CASES = load_policy_test_data(test_dir=TEST_DIR, policy_name="")


@pytest.mark.parametrize(
    "test",
    POLICY_TEST_IDS_AND_CASES.values(),
    ids=POLICY_TEST_IDS_AND_CASES.keys(),
)
def test_mettsim(test: PolicyTest):
    if IS_JAX_INSTALLED:
        execute_test(test, root=METTSIM_ROOT, jit=True)
    else:
        execute_test(test, root=METTSIM_ROOT, jit=False)


@pytest.mark.xfail(
    reason="Testing infrastructure cannot handle single-element expected output"
)
def test_mettsim_single_element_expected_output():
    test = PolicyTest(
        info={"precision_atol": 0.01},
        input_tree={
            "p_id": np.array([0]),
            "property_tax": {"acre_size_in_hectares": np.array([20])},
        },
        expected_output_tree={"property_tax": {"amount_y": np.array([1000.0])}},
        path=None,
        date=datetime.date(2020, 1, 1),
        test_dir=TEST_DIR,
    )
    execute_test(test, root=METTSIM_ROOT, jit=False)
