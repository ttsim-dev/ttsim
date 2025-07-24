from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy
import pytest

from ttsim import main
from ttsim.main_args import InputData
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

    from ttsim.typing import (
        FlatColumnObjectsParamFunctions,
        FlatOrigParamSpecs,
    )

METTSIM_ROOT = Path(__file__).parent.parent / "mettsim"


POLICY_TEST_IDS_AND_CASES = load_policy_test_data(
    test_dir=Path(__file__).parent.parent / "mettsim_tests",
    policy_name="",
    xnp=numpy,
)


def get_orig_mettsim_objects() -> dict[
    str, FlatColumnObjectsParamFunctions | FlatOrigParamSpecs
]:
    return main(
        main_targets=[
            "orig_policy_objects__column_objects_and_param_functions",
            "orig_policy_objects__param_specs",
        ],
        orig_policy_objects={"root": METTSIM_ROOT},
    )["orig_policy_objects"]


def dates_in_orig_mettsim_objects() -> list[datetime.date]:
    orig_objects = get_orig_mettsim_objects()
    start_dates = {
        v.start_date
        for v in orig_objects["column_objects_and_param_functions"].values()
    }
    end_dates = {
        v.end_date + timedelta(days=1)
        for v in orig_objects["column_objects_and_param_functions"].values()
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
        policy_date_str="2020-01-01",
        root=METTSIM_ROOT,
        include_params=True,
        title="METTSIM Policy Environment DAG with parameters",
        show_node_description=True,
    )


def test_mettsim_policy_environment_dag_without_params():
    plot_tt_dag(
        policy_date_str="2020-01-01",
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
        policy_date=date,
        orig_policy_objects=orig_mettsim_objects,
    )


def test_fail_functions_are_executed_with_priority(backend: Literal["numpy", "jax"]):
    data = {("p_id",): numpy.array([0, 1, 2, 3])}
    with pytest.raises(
        ValueError,
        match=r"The following data columns are missing.",
    ):
        main(
            main_target="results__tree",
            policy_date_str="2020-01-01",
            input_data=InputData.flat(data),
            orig_policy_objects={"root": METTSIM_ROOT},
            tt_targets={"tree": {"property_tax": {"amount_y": None}}},
            backend=backend,
        )
