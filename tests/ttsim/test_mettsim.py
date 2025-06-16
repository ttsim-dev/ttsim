from __future__ import annotations

import inspect
from pathlib import Path
from typing import Literal

import dags
import dags.tree as dt
import numpy
import pytest
from mettsim.config import METTSIM_ROOT

from ttsim import main
from ttsim.interface_dag_elements.fail_if import format_list_linewise
from ttsim.plot_dag import (
    dummy_callable,
    plot_tt_dag,
)
from ttsim.testing_utils import (
    PolicyTest,
    execute_test,
    load_policy_test_data,
)
from ttsim.tt_dag_elements.column_objects_param_function import PolicyInput

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


@pytest.mark.parametrize("date", ["2019-01-01", "2021-01-01"])
def test_mettsim_policy_environment_is_complete(date, orig_mettsim_objects):
    """Test that METTSIM's policy environment contains all root nodes of its DAG."""
    inputs_for_main = {
        "date_str": date,
        **orig_mettsim_objects,
    }
    environment = main(
        inputs=inputs_for_main,
        targets=["policy_environment"],
    )["policy_environment"]
    qname_environment = dt.flatten_to_qnames(environment)
    qnames_policy_inputs = [
        k for k, v in qname_environment.items() if isinstance(v, PolicyInput)
    ]
    tgt = "specialized_environment__without_tree_logic_and_with_derived_functions"
    env = main(
        inputs={
            "policy_environment": environment,
            "labels__processed_data_columns": qnames_policy_inputs,
            "targets__qname": list(qname_environment),
        },
        targets=[tgt],
    )[tgt]
    all_nodes = {
        qn: dummy_callable(n) if not callable(n) else n for qn, n in env.items()
    }
    f = dags.concatenate_functions(
        functions=all_nodes,
        targets=qnames_policy_inputs,
        return_type="dict",
        enforce_signature=False,
        set_annotations=False,
    )
    args = inspect.signature(f).parameters
    if args:
        raise ValueError(
            "METTSIM's full DAG should include all root nodes but the following inputs "
            "are missing in the specialized policy environment:"
            f"\n\n{format_list_linewise(args.keys())}\n\n"
            "Please add corresponding elements. Typically, these will be "
            "`@policy_input()`s or parameters in the yaml files."
        )
