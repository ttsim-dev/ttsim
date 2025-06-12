from __future__ import annotations

import inspect
from pathlib import Path

import dags
import pytest

from _gettsim.config import GETTSIM_ROOT
from ttsim.config import IS_JAX_INSTALLED
from ttsim.interface_dag import main
from ttsim.interface_dag_elements.fail_if import format_list_linewise
from ttsim.plot_dag import (
    all_targets_from_namespace,
    plot_tt_dag,
    specialized_environment_for_targets,
)
from ttsim.testing_utils import (
    PolicyTest,
    execute_test,
    load_policy_test_data,
)
from ttsim.tt_dag_elements import (
    ParamObject,
    PolicyInput,
)

TEST_DIR = Path(__file__).parent

POLICY_TEST_IDS_AND_CASES = load_policy_test_data(test_dir=TEST_DIR, policy_name="")


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
    )["orig_policy_objects__column_objects_and_param_functions"]


@pytest.mark.parametrize(
    "test",
    POLICY_TEST_IDS_AND_CASES.values(),
    ids=POLICY_TEST_IDS_AND_CASES.keys(),
)
def test_policy(test: PolicyTest):
    if IS_JAX_INSTALLED:
        execute_test(test, root=GETTSIM_ROOT, jit=True)
    else:
        execute_test(test, root=GETTSIM_ROOT, jit=False)


def test_gettsim_policy_environment_dag_with_params():
    plot_tt_dag(
        date_str="2020-01-01",
        root=GETTSIM_ROOT,
        include_param_functions=True,
        title="GETTSIM Policy Environment DAG with parameters",
        output_path=Path("gettsim_dag_with_params.html"),
    )


def test_gettsim_policy_environment_dag_without_params():
    plot_tt_dag(
        date_str="2020-01-01",
        root=GETTSIM_ROOT,
        include_param_functions=False,
        title="GETTSIM Policy Environment DAG without parameters",
        output_path=Path("gettsim_dag_without_params.html"),
    )


@pytest.mark.parametrize("date", [f"{year}-01-01" for year in range(2015, 2025)])
def test_gettsim_policy_environment_is_complete(orig_gettsim_objects, date):
    """Test that GETTSIM's policy environment contains all root nodes of its DAG."""
    inputs_for_main = {
        "date_str": date,
        "orig_policy_objects__root": GETTSIM_ROOT,
        "targets__include_param_functions": True,
        "targets__namespace": "all",
        "orig_policy_objects__column_objects_and_param_functions": orig_gettsim_objects,
    }
    all_targets = all_targets_from_namespace(inputs_for_main)
    specialized_environment = specialized_environment_for_targets(inputs_for_main)
    functions = {
        qn: n.dummy_callable() if isinstance(n, PolicyInput | ParamObject) else n
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
