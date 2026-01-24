from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Literal

import numpy
import pytest
from mettsim import middle_earth

from ttsim import OrigPolicyObjects, TTTargets, main, plot
from ttsim.main_args import InputData
from ttsim.testing_utils import (
    PolicyTest,
    check_env_completeness,
    execute_test,
    load_policy_cases,
)

if TYPE_CHECKING:
    import datetime

    from ttsim.typing import (
        FlatColumnObjectsParamFunctions,
        FlatOrigParamSpecs,
    )


POLICY_TEST_IDS_AND_CASES = load_policy_cases(
    policy_cases_root=(
        middle_earth.ROOT_PATH.parent / "tests_middle_earth" / "policy_cases"
    ),
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
        orig_policy_objects=OrigPolicyObjects.root(middle_earth.ROOT_PATH),
    )["orig_policy_objects"]


def dates_in_orig_mettsim_objects() -> list[datetime.date]:
    orig_objects = get_orig_mettsim_objects()
    start_dates = {
        v.start_date  # ty: ignore[possibly-missing-attribute]
        for v in orig_objects["column_objects_and_param_functions"].values()
    }
    end_dates = {
        v.end_date + timedelta(days=1)  # ty: ignore[possibly-missing-attribute]
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
def test_policy_cases(test: PolicyTest, backend: Literal["numpy", "jax"]):
    execute_test(test=test, root=middle_earth.ROOT_PATH, backend=backend)


def test_python314_annotation_extraction_bug(backend: Literal["numpy", "jax"]):
    """Check Python 3.14 annotation extraction bug (fixed in dags>=0.4.2)."""

    policy_cases_root = (
        middle_earth.ROOT_PATH.parent.parent / "tests_middle_earth" / "policy_cases"
    )

    cases = load_policy_cases(
        policy_cases_root=policy_cases_root,
        policy_name="",
        xnp=numpy,
    )

    test_file = (
        policy_cases_root
        / "payroll_tax"
        / "2025-01-01"
        / "annotation_bug_reproducer.yaml"
    )
    for test in cases.values():
        if str(test.path) == str(test_file):
            # In Python 3.14, this will raise AnnotationMismatchError (test fails)
            # In Python 3.13, this will succeed (test passes)
            execute_test(test=test, root=middle_earth.ROOT_PATH, backend=backend)
            break
    else:
        pytest.fail(f"Could not find test case: {test_file}")


def test_mettsim_policy_environment_dag_with_params():
    plot.dag.tt(
        policy_date_str="2020-01-01",
        root=middle_earth.ROOT_PATH,
        include_params=True,
        title="METTSIM Policy Environment DAG with parameters",
        show_node_description=True,
    )


def test_mettsim_policy_environment_dag_without_params():
    plot.dag.tt(
        policy_date_str="2020-01-01",
        root=middle_earth.ROOT_PATH,
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
            orig_policy_objects=OrigPolicyObjects.root(middle_earth.ROOT_PATH),
            tt_targets=TTTargets.tree({"property_tax": {"amount_y": None}}),
            backend=backend,
        )


def test_tt_function_cloudpickle(backend: Literal["numpy", "jax"]):
    """tt_function should be cloudpicklable after policy modules are loaded."""
    import cloudpickle

    data = {
        ("age",): numpy.array([30, 30]),
        ("kin_id",): numpy.array([0, 0]),
        ("p_id",): numpy.array([0, 1]),
        ("p_id_parent_1",): numpy.array([-1, -1]),
        ("p_id_parent_2",): numpy.array([-1, -1]),
        ("p_id_spouse",): numpy.array([1, 0]),
        ("parent_is_noble",): numpy.array([False, False]),
        ("payroll_tax", "child_tax_credit", "p_id_recipient"): numpy.array([-1, -1]),
        ("payroll_tax", "income", "gross_wage_y"): numpy.array([10000.0, 0.0]),
        ("wealth",): numpy.array([0.0, 0.0]),
    }

    tt_func = main(
        main_target="tt_function",
        policy_date_str="2025-01-01",
        input_data=InputData.flat(data),
        orig_policy_objects=OrigPolicyObjects.root(middle_earth.ROOT_PATH),
        tt_targets=TTTargets.tree({"payroll_tax": {"amount_y": None}}),
        backend=backend,
    )
    processed_data = main(
        main_target="processed_data",
        policy_date_str="2025-01-01",
        input_data=InputData.flat(data),
        orig_policy_objects=OrigPolicyObjects.root(middle_earth.ROOT_PATH),
        tt_targets=TTTargets.tree({"payroll_tax": {"amount_y": None}}),
        backend=backend,
    )

    # Pickle roundtrip
    pickled = cloudpickle.dumps(tt_func)
    unpickled_func = cloudpickle.loads(pickled)

    original_result = tt_func(processed_data)
    restored_result = unpickled_func(processed_data)

    for key in original_result:
        assert numpy.allclose(original_result[key], restored_result[key]), (
            f"Mismatch for {key}"
        )
