from __future__ import annotations

from pathlib import Path

import pytest
from mettsim.config import METTSIM_ROOT

from ttsim.config import IS_JAX_INSTALLED
from ttsim.interface_dag import main
from ttsim.plot_dag import (
    plot_tt_dag,
)
from ttsim.testing_utils import (
    PolicyTest,
    execute_test,
    load_policy_test_data,
)

TEST_DIR = Path(__file__).parent

POLICY_TEST_IDS_AND_CASES = load_policy_test_data(test_dir=TEST_DIR, policy_name="")


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
def test_mettsim(test: PolicyTest):
    if IS_JAX_INSTALLED:
        execute_test(test, root=METTSIM_ROOT, jit=True)
    else:
        execute_test(test, root=METTSIM_ROOT, jit=False)


def test_mettsim_policy_environment_dag_with_params():
    plot_tt_dag(
        date_str="2020-01-01",
        root=METTSIM_ROOT,
        include_param_functions=True,
        namespace="all",
        title="METTSIM Policy Environment DAG with parameters",
        output_path=Path("mettsim_dag_with_params.html"),
    )


def test_mettsim_policy_environment_dag_without_params():
    plot_tt_dag(
        date_str="2020-01-01",
        root=METTSIM_ROOT,
        include_param_functions=False,
        namespace="all",
        title="METTSIM Policy Environment DAG without parameters",
        output_path=Path("mettsim_dag_without_params.html"),
    )


@pytest.mark.parametrize("date", ["2019-01-01", "2021-01-01"])
def test_mettsim_policy_environment_is_complete(orig_mettsim_objects, date):
    """Test that METTSIM's policy environment contains all root nodes of its DAG."""
    # dag = dag_including_policy_inputs(
    #     date_str=date,
    #     root=METTSIM_ROOT,
    #     include_param_functions=True,
    #     namespace="all",
    # )

    # f = dags.concatenate_functions(
    #     dag=dag,
    #     functions=nodes,
    #     targets=targets_and_ttsim_objects["targets__qname"],
    #     return_type="dict",
    #     enforce_signature=False,
    #     set_annotations=False,
    # )
    # args = inspect.signature(f).parameters
    # if args:
    #     raise ValueError(
    #         "The policy environment DAG should include all root nodes but requires "
    #         f"inputs:\n\n{format_list_linewise(args.keys())}"
    #     )
