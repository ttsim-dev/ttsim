from __future__ import annotations

import inspect
from pathlib import Path
from typing import Literal

import dags
import numpy
import pytest
from mettsim.config import METTSIM_ROOT

from ttsim import main
from ttsim.interface_dag_elements.fail_if import format_list_linewise
from ttsim.plot_dag import (
    all_targets_from_namespace,
    dummy_callable,
    plot_tt_dag,
    specialized_environment_for_plotting,
)
from ttsim.testing_utils import (
    PolicyTest,
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
    )["orig_policy_objects__column_objects_and_param_functions"]


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
        include_param_functions=True,
        title="METTSIM Policy Environment DAG with parameters",
        show_node_description=True,
    ).write_html(Path("mettsim_dag_with_params.html"))


def test_mettsim_policy_environment_dag_without_params():
    plot_tt_dag(
        date_str="2020-01-01",
        root=METTSIM_ROOT,
        include_param_functions=False,
        title="METTSIM Policy Environment DAG without parameters",
        show_node_description=True,
    ).write_html(Path("mettsim_dag_without_params.html"))


@pytest.mark.parametrize("date", ["2019-01-01", "2021-01-01"])
def test_mettsim_policy_environment_is_complete(orig_mettsim_objects, date):
    """Test that METTSIM's policy environment contains all root nodes of its DAG."""
    inputs_for_main = {
        "date_str": date,
        "orig_policy_objects__root": METTSIM_ROOT,
        "targets__include_param_functions": True,
        "targets__namespace": "all",
        "orig_policy_objects__column_objects_and_param_functions": orig_mettsim_objects,
    }
    all_targets = all_targets_from_namespace(inputs_for_main)
    specialized_environment = specialized_environment_for_plotting(inputs_for_main)
    functions = {
        qn: dummy_callable(n) if not callable(n) else n
        for qn, n in specialized_environment.items()
    }
    f = dags.concatenate_functions(
        functions=functions,
        targets=all_targets,
        return_type="dict",
        enforce_signature=False,
        set_annotations=False,
    )
    args = inspect.signature(f).parameters
    if args:
        raise ValueError(
            "METTSIM's full DAG should include all root nodes but the following inputs "
            "are missing in the specialized policy environment:"
            f"\n\n{format_list_linewise(args.keys())}"
        )
