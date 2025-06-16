from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal

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

if TYPE_CHECKING:
    import datetime

    from ttsim.interface_dag_elements.typing import (
        FlatColumnObjectsParamFunctions,
        FlatOrigParamSpecs,
    )

TEST_DIR = Path(__file__).parent

POLICY_TEST_IDS_AND_CASES = load_policy_test_data(
    test_dir=TEST_DIR,
    policy_name="",
    xnp=numpy,
)


def get_orig_mettsim_objects() -> dict[
    str, FlatColumnObjectsParamFunctions | FlatOrigParamSpecs
]:
    return main(
        inputs={
            "orig_policy_objects__root": METTSIM_ROOT,
        },
        targets=[
            "orig_policy_objects__column_objects_and_param_functions",
            "orig_policy_objects__param_specs",
        ],
    )


def dates_in_orig_mettsim_objects() -> list[datetime.date]:
    objects = get_orig_mettsim_objects()
    start_dates = {
        v.start_date
        for v in objects[
            "orig_policy_objects__column_objects_and_param_functions"
        ].values()
    }
    end_dates = {
        v.end_date + timedelta(days=1)
        for v in objects[
            "orig_policy_objects__column_objects_and_param_functions"
        ].values()
    }
    return sorted(start_dates | end_dates)


@pytest.fixture
def orig_mettsim_objects():
    return get_orig_mettsim_objects()


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
    )


def test_mettsim_policy_environment_dag_without_params():
    plot_tt_dag(
        date_str="2020-01-01",
        root=METTSIM_ROOT,
        include_params=False,
        title="METTSIM Policy Environment DAG without parameters",
        show_node_description=True,
    )


@pytest.mark.parametrize(
    "date",
    dates_in_orig_mettsim_objects(),
    ids=lambda x: x.isoformat(),
)
def test_mettsim_policy_environment_is_complete(orig_mettsim_objects, date):
    """Test that METTSIM's policy environment contains all root nodes of its DAG."""
    check_env_completeness(
        name="METTSIM",
        date_str=date.isoformat(),
        orig_policy_objects=orig_mettsim_objects,
    )
