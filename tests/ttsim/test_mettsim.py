from __future__ import annotations

import numpy as np
import pytest
from utils import (
    PolicyTest,
    execute_test,
    get_policy_test_ids_and_cases,
)

from ttsim.config import IS_JAX_INSTALLED

policy_test_ids_and_cases = get_policy_test_ids_and_cases()


@pytest.mark.parametrize(
    "test",
    policy_test_ids_and_cases.values(),
    ids=policy_test_ids_and_cases.keys(),
)
def test_mettsim(test: PolicyTest):
    if IS_JAX_INSTALLED:
        execute_test(test, jit=True)
    else:
        execute_test(test, jit=False)


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
        date="2020-01-01",
    )
    execute_test(test, jit=False)
