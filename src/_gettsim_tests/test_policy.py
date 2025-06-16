from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy
import pytest

from _gettsim.config import GETTSIM_ROOT
from ttsim import main
from ttsim.testing_utils import (
    PolicyTest,
    check_env_completeness,
    execute_test,
    load_policy_test_data,
)

TEST_DIR = Path(__file__).parent

POLICY_TEST_IDS_AND_CASES = load_policy_test_data(
    test_dir=TEST_DIR,
    policy_name="",
    xnp=numpy,
)


@pytest.fixture
def orig_gettsim_objects():
    return main(
        inputs={
            "orig_policy_objects__root": GETTSIM_ROOT,
        },
        targets=[
            "orig_policy_objects__column_objects_and_param_functions",
            "orig_policy_objects__param_specs",
        ],
    )


@pytest.mark.parametrize(
    "test",
    POLICY_TEST_IDS_AND_CASES.values(),
    ids=POLICY_TEST_IDS_AND_CASES.keys(),
)
def test_policy(test: PolicyTest, backend: Literal["numpy", "jax"]):
    execute_test(test=test, root=GETTSIM_ROOT, backend=backend)


@pytest.mark.parametrize("date", [f"{year}-01-01" for year in range(2015, 2025)])
def test_gettsim_policy_environment_is_complete(orig_gettsim_objects, date):
    """Test that GETTSIM's policy environment contains all root nodes of its DAG."""
    check_env_completeness(
        name="GETTSIM",
        date_str=date,
        orig_policy_objects=orig_gettsim_objects,
    )
