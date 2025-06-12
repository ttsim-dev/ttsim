from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy
import pytest
from mettsim.config import METTSIM_ROOT

from ttsim.plot_dag import plot_tt_dag
from ttsim.testing_utils import (
    PolicyTest,
    execute_test,
    load_policy_test_data,
)

TEST_DIR = Path(__file__).parent

POLICY_TEST_IDS_AND_CASES = load_policy_test_data(
    test_dir=TEST_DIR, policy_name="", xnp=numpy
)


@pytest.mark.parametrize(
    "test",
    POLICY_TEST_IDS_AND_CASES.values(),
    ids=POLICY_TEST_IDS_AND_CASES.keys(),
)
def test_mettsim(test: PolicyTest, backend: Literal["numpy", "jax"]):
    execute_test(test=test, root=METTSIM_ROOT, backend=backend)


def test_mettsim_policy_environment_dag_with_params():
    plot_tt_dag(
        with_params=True,
        inputs_for_main={
            "date_str": "2020-01-01",
            "orig_policy_objects__root": METTSIM_ROOT,
        },
        title="METTSIM Policy Environment DAG with parameters",
        output_path=Path("mettsim_dag_with_params.html"),
    )


def test_mettsim_policy_environment_dag_without_params():
    plot_tt_dag(
        with_params=False,
        inputs_for_main={
            "date_str": "2020-01-01",
            "orig_policy_objects__root": METTSIM_ROOT,
        },
        title="METTSIM Policy Environment DAG without parameters",
        output_path=Path("mettsim_dag_without_params.html"),
    )
