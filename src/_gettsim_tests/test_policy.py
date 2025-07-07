from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import dags.tree as dt
import numpy
import pytest

from ttsim import main
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

GETTSIM_ROOT = Path(__file__).parent.parent / "_gettsim"
TEST_DIR = Path(__file__).parent

POLICY_TEST_IDS_AND_CASES = load_policy_test_data(
    test_dir=TEST_DIR,
    policy_name="",
    xnp=numpy,
)


def get_orig_gettsim_objects() -> dict[
    str, FlatColumnObjectsParamFunctions | FlatOrigParamSpecs
]:
    out = main(
        orig_policy_objects={"root": GETTSIM_ROOT},
        main_targets=[
            "orig_policy_objects__column_objects_and_param_functions",
            "orig_policy_objects__param_specs",
        ],
    )
    return {k.replace("orig_policy_objects__", ""): v for k, v in out.items()}


def dates_in_orig_gettsim_objects() -> list[datetime.date]:
    orig_objects = get_orig_gettsim_objects()
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
def orig_gettsim_objects():
    return get_orig_gettsim_objects()


@pytest.mark.parametrize(
    "test",
    POLICY_TEST_IDS_AND_CASES.values(),
    ids=POLICY_TEST_IDS_AND_CASES.keys(),
)
def test_policy(test: PolicyTest, backend: Literal["numpy", "jax"]):
    execute_test(test=test, root=GETTSIM_ROOT, backend=backend)


@pytest.mark.parametrize(
    "date",
    dates_in_orig_gettsim_objects(),
    ids=lambda x: x.isoformat(),
)
def test_gettsim_policy_environment_is_complete(orig_gettsim_objects, date):
    """Test that GETTSIM's policy environment contains all root nodes of its DAG."""
    if date.year < 2015:
        pytest.skip(
            "Policy environment for dates before 2015 are not complete. See issue #962."
        )

    check_env_completeness(
        name="GETTSIM",
        date=date,
        orig_policy_objects=orig_gettsim_objects,
    )


@pytest.mark.parametrize(
    "date",
    dates_in_orig_gettsim_objects(),
    ids=lambda x: x.isoformat(),
)
def test_top_level_elements_not_repeated_in_paths(
    date, backend: Literal["numpy", "jax"]
):
    try:
        gettsim_objects = main(
            orig_policy_objects={"root": GETTSIM_ROOT},
            backend=backend,
            date=date,
            rounding=False,
            main_targets=[
                "specialized_environment__with_partialled_params_and_scalars",
                "labels__top_level_namespace",
            ],
        )
    except Exception:  # noqa: BLE001
        msg = (
            "Skipped because environment cannot be created for date "
            f"{date.isoformat()}."
        )
        pytest.skip(msg)

    dt.fail_if_top_level_elements_repeated_in_paths(
        all_tree_paths=dt.flatten_to_tree_paths(
            dt.unflatten_from_qnames(
                gettsim_objects[
                    "specialized_environment__with_partialled_params_and_scalars"
                ]
            )
        ),
        top_level_namespace=gettsim_objects["labels__top_level_namespace"],
    )
