from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy
import pytest
from mettsim import middle_earth

from ttsim import OrigPolicyObjects, TTTargets, main, plot
from ttsim.exceptions import UnitConsistencyError
from ttsim.main_args import InputData
from ttsim.testing_utils import (
    PolicyTest,
    check_env_completeness,
    check_policy_environment_units,
    execute_test,
    get_policy_date_partition,
    load_policy_cases,
)
from ttsim.tt.units import TTSIMUnit, UnitAnnotatedColumn

if TYPE_CHECKING:
    import datetime

    from ttsim.typing import (
        FlatColumnObjectsParamFunctions,
        FlatOrigParamSpecs,
    )


POLICY_TEST_IDS_AND_CASES = load_policy_cases(
    policy_cases_root=(
        middle_earth.ROOT_PATH.parent.parent / "tests_middle_earth" / "policy_cases"
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
        unit_system=middle_earth.UNIT_SYSTEM,
    )["orig_policy_objects"]


def dates_in_orig_mettsim_objects() -> list[datetime.date]:
    """One date per structural regime of METTSIM, parameter-only changes included."""
    return get_policy_date_partition(
        orig_policy_objects=get_orig_mettsim_objects(),
        unit_system=middle_earth.UNIT_SYSTEM,
    )


@pytest.fixture
def orig_mettsim_objects():
    return get_orig_mettsim_objects()


@pytest.mark.parametrize(
    "test",
    POLICY_TEST_IDS_AND_CASES.values(),
    ids=POLICY_TEST_IDS_AND_CASES.keys(),
)
def test_policy_cases(test: PolicyTest, backend: Literal["numpy", "jax"]):
    execute_test(
        test=test,
        root=middle_earth.ROOT_PATH,
        backend=backend,
        unit_system=middle_earth.UNIT_SYSTEM,
        default_data_currency="SILVER_PENNY",
    )


def test_enough_policy_cases_are_collected():
    """Guard against silently skipping all policy cases via a broken glob root."""
    assert len(POLICY_TEST_IDS_AND_CASES) >= 20


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
            execute_test(
                test=test,
                root=middle_earth.ROOT_PATH,
                backend=backend,
                unit_system=middle_earth.UNIT_SYSTEM,
                default_data_currency="SILVER_PENNY",
            )
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
        unit_system=middle_earth.UNIT_SYSTEM,
    )


def test_mettsim_policy_environment_dag_without_params():
    plot.dag.tt(
        policy_date_str="2020-01-01",
        root=middle_earth.ROOT_PATH,
        include_params=False,
        title="METTSIM Policy Environment DAG without parameters",
        show_node_description=True,
        unit_system=middle_earth.UNIT_SYSTEM,
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
        unit_system=middle_earth.UNIT_SYSTEM,
    )


@pytest.mark.parametrize(
    "date",
    dates_in_orig_mettsim_objects(),
    ids=lambda x: x.isoformat(),
)
def test_mettsim_units_are_complete_and_consistent(orig_mettsim_objects, date):
    """GEP 10 Layer-1 check over all policy dates (ttsim #121).

    Every active node must declare (or auto-receive) a unit, and every
    function body the check can evaluate must infer a unit consistent with
    its declaration.
    """
    check_policy_environment_units(
        policy_date=date,
        orig_policy_objects=orig_mettsim_objects,
        unit_system=middle_earth.UNIT_SYSTEM,
    )


def test_fail_functions_are_executed_with_priority(backend: Literal["numpy", "jax"]):
    data: dict[tuple[str, ...], Any] = {("p_id",): numpy.array([0, 1, 2, 3])}
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
            unit_system=middle_earth.UNIT_SYSTEM,
        )


@pytest.mark.parametrize(
    "policy_date_str",
    # One policy date per statutory era: 2025 computes in castar, 2000 in
    # silver pennies. The boundary relation must hold in both.
    ["2025-01-01", "2000-01-01"],
)
def test_data_currency_converts_at_the_column_boundary(
    policy_date_str: str, backend: Literal["numpy", "jax"]
):
    """The same household in both data currencies (GEP 10).

    Castar amounts times four (1 castar = 4 silver pennies) must give the
    silver-penny amounts — for the inputs by construction, for the outputs
    because both runs perform the identical computation in the policy date's
    statutory currency and only the column boundary converts: inputs from the
    data currency on the way in, currency-denominated results back on the way
    out. Parameters are never touched.
    """
    results = {}
    for data_currency, factor in (("CASTAR", 1.0), ("SILVER_PENNY", 4.0)):
        input_tree = {
            "p_id": numpy.array([0, 1]),
            "kin_id": numpy.array([0, 0]),
            "p_id_spouse": numpy.array([1, 0]),
            "p_id_parent_1": numpy.array([-1, -1]),
            "p_id_parent_2": numpy.array([-1, -1]),
            "birth_year": numpy.array([1995, 1995]),  # age 30 in 2025
            "parent_is_noble": numpy.array([False, False]),
            "wealth": numpy.array([0.0, 0.0]) * factor,
            "payroll_tax": {
                "child_tax_credit": {"p_id_recipient": numpy.array([-1, -1])},
                "income": {"gross_wage_y": numpy.array([10000.0, 0.0]) * factor},
            },
        }
        results[data_currency] = main(
            main_target="results__tree",
            policy_date_str=policy_date_str,
            input_data=InputData.tree(input_tree),
            orig_policy_objects=OrigPolicyObjects.root(middle_earth.ROOT_PATH),
            tt_targets=TTTargets.tree({"payroll_tax": {"amount_y": None}}),
            rounding=False,
            data_currency=data_currency,
            backend=backend,
            unit_system=middle_earth.UNIT_SYSTEM,
        )
    castar = results["CASTAR"]["payroll_tax"]["amount_y"]
    silver = results["SILVER_PENNY"]["payroll_tax"]["amount_y"]
    numpy.testing.assert_allclose(numpy.asarray(silver), numpy.asarray(castar) * 4.0)
    assert float(numpy.asarray(castar)[0]) > 0.0


def _annotated_payroll_tree(currency: str, factor: float = 1.0) -> dict[str, Any]:
    """A two-person payroll-tax input tree, every leaf a `UnitAnnotatedColumn`.

    Ids and the person-level boolean are ``TTSIMUnit.DIMENSIONLESS`` (the person leaf
    is implied, never spelled); the birth year is a calendar year; wealth is a
    concrete-currency stock; the wage a concrete-currency flow per year. The
    concrete currency is reached off the `TTSIMUnit` builder (``TTSIMUnit.CASTAR``,
    ``TTSIMUnit.SILVER_PENNY``), injected when mettsim registered it.
    """
    money = getattr(TTSIMUnit, currency)

    def col(values: Any, unit: Any) -> UnitAnnotatedColumn:
        return UnitAnnotatedColumn(values=values, unit=unit)

    return {
        "p_id": col(values=numpy.array([0, 1]), unit=TTSIMUnit.DIMENSIONLESS),
        "kin_id": col(values=numpy.array([0, 0]), unit=TTSIMUnit.DIMENSIONLESS),
        "p_id_spouse": col(values=numpy.array([1, 0]), unit=TTSIMUnit.DIMENSIONLESS),
        "p_id_parent_1": col(
            values=numpy.array([-1, -1]), unit=TTSIMUnit.DIMENSIONLESS
        ),
        "p_id_parent_2": col(
            values=numpy.array([-1, -1]), unit=TTSIMUnit.DIMENSIONLESS
        ),
        "birth_year": col(
            values=numpy.array([1995, 1995]), unit=TTSIMUnit.CALENDAR_YEAR
        ),  # age 30
        "parent_is_noble": col(
            values=numpy.array([False, False]), unit=TTSIMUnit.DIMENSIONLESS
        ),
        "wealth": col(values=numpy.array([100.0, 200.0]) * factor, unit=money),
        "payroll_tax": {
            "child_tax_credit": {
                "p_id_recipient": col(
                    values=numpy.array([-1, -1]), unit=TTSIMUnit.DIMENSIONLESS
                )
            },
            "income": {
                "gross_wage_y": col(
                    values=numpy.array([10000.0, 0.0]) * factor, unit=money.PER_YEAR
                )
            },
        },
    }


def test_unit_annotated_input_rejects_wrong_dimension(
    backend: Literal["numpy", "jax"],
):
    """A currency tag on the CALENDAR_YEAR birth-year column is rejected (GEP 10)."""
    tree = _annotated_payroll_tree("CASTAR")
    tree["birth_year"] = UnitAnnotatedColumn(
        values=numpy.array([1995, 1995]), unit=TTSIMUnit.CASTAR
    )
    with pytest.raises(UnitConsistencyError, match="inconsistent with the DAG"):
        main(
            main_target="results__tree",
            policy_date_str="2025-01-01",
            input_data=InputData.tree_with_unit_annotations(tree),
            orig_policy_objects=OrigPolicyObjects.root(middle_earth.ROOT_PATH),
            tt_targets=TTTargets.tree({"payroll_tax": {"amount_y": None}}),
            rounding=False,
            data_currency="CASTAR",
            backend=backend,
            unit_system=middle_earth.UNIT_SYSTEM,
        )


def test_unit_annotated_input_rejects_agnostic_currency(
    backend: Literal["numpy", "jax"],
):
    """A currency column tagged with the agnostic CURRENCY is rejected (GEP 10).

    Input data, like a parameter, must name the concrete currency its numbers are
    written in.
    """
    tree = _annotated_payroll_tree("CASTAR")
    tree["wealth"] = UnitAnnotatedColumn(
        values=numpy.array([100.0, 200.0]), unit=TTSIMUnit.CURRENCY
    )
    with pytest.raises(UnitConsistencyError, match="concrete currency"):
        main(
            main_target="results__tree",
            policy_date_str="2025-01-01",
            input_data=InputData.tree_with_unit_annotations(tree),
            orig_policy_objects=OrigPolicyObjects.root(middle_earth.ROOT_PATH),
            tt_targets=TTTargets.tree({"payroll_tax": {"amount_y": None}}),
            rounding=False,
            data_currency="CASTAR",
            backend=backend,
            unit_system=middle_earth.UNIT_SYSTEM,
        )


def test_unit_annotated_input_rejects_wrong_grouping_level(
    backend: Literal["numpy", "jax"],
):
    """A person-level wage column tagged with a group level is rejected (GEP 10)."""
    tree = _annotated_payroll_tree("CASTAR")
    tree["payroll_tax"]["income"]["gross_wage_y"] = UnitAnnotatedColumn(
        values=numpy.array([10000.0, 0.0]),
        unit=TTSIMUnit.CASTAR.PER_YEAR.PER_LEVEL("fam"),
    )
    with pytest.raises(UnitConsistencyError, match="the column is at"):
        main(
            main_target="results__tree",
            policy_date_str="2025-01-01",
            input_data=InputData.tree_with_unit_annotations(tree),
            orig_policy_objects=OrigPolicyObjects.root(middle_earth.ROOT_PATH),
            tt_targets=TTTargets.tree({"payroll_tax": {"amount_y": None}}),
            rounding=False,
            data_currency="CASTAR",
            backend=backend,
            unit_system=middle_earth.UNIT_SYSTEM,
        )
