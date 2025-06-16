from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy
import pytest
from mettsim.config import METTSIM_ROOT

from ttsim import main
from ttsim.plot_dag import (
    plot_tt_dag,
)
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
def orig_mettsim_objects():
    return main(
        inputs={
            "orig_policy_objects__root": METTSIM_ROOT,
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
def test_mettsim(test: PolicyTest, backend: Literal["numpy", "jax"]):
    execute_test(test=test, root=METTSIM_ROOT, backend=backend)


def test_mettsim_policy_environment_dag_with_params():
    plot_tt_dag(
        date_str="2020-01-01",
        root=METTSIM_ROOT,
        include_params=True,
        title="METTSIM Policy Environment DAG with parameters",
        show_node_description=True,
    ).write_html(Path("mettsim_dag_with_params.html"))


def test_mettsim_policy_environment_dag_without_params():
    plot_tt_dag(
        date_str="2020-01-01",
        root=METTSIM_ROOT,
        include_params=False,
        title="METTSIM Policy Environment DAG without parameters",
        show_node_description=True,
    ).write_html(Path("mettsim_dag_without_params.html"))


@pytest.mark.parametrize("date_str", ["2019-01-01", "2021-01-01"])
def test_mettsim_policy_environment_is_complete(date_str, orig_mettsim_objects):
    """Test that METTSIM's policy environment contains all root nodes of its DAG."""
    check_env_completeness(
        name="METTSIM",
        date_str=date_str,
        orig_policy_objects=orig_mettsim_objects,
    )
