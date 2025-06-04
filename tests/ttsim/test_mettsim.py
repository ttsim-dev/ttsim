from __future__ import annotations

from pathlib import Path

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
