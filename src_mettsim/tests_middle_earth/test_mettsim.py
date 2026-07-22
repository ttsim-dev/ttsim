from __future__ import annotations

from datetime import timedelta
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
    check_env_units,
    execute_test,
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
    orig_objects = get_orig_mettsim_objects()
    start_dates = {
        v.start_date  # ty: ignore[unresolved-attribute]
        for v in orig_objects["column_objects_and_param_functions"].values()
    }
    end_dates = {
        v.end_date + timedelta(days=1)  # ty: ignore[unresolved-attribute]
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
    check_env_units(
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


def test_quadratic_schedule_is_exact_under_boundary_conversion(
    backend: Literal["numpy", "jax"],
):
    """Boundary conversion is exact for a money-quadratic schedule (GEP 10).

    The king's levy schedule carries a quadratic coefficient (a
    Progressionsfaktor) of units ``1/currency`` — the formula shape for which
    scaling parameters breaks run-currency covariance. Under boundary-only
    conversion the relation is exact by construction: both runs compute the
    identical schedule in the statutory currency, and the silver-penny levy is
    four times the castar levy purely because the boundary converts inputs in
    and results out.
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
            # Wealth lands in the schedule's quadratic bracket in both currencies.
            "wealth": numpy.array([100.0, 200.0]) * factor,
            "payroll_tax": {
                "child_tax_credit": {"p_id_recipient": numpy.array([-1, -1])},
                "income": {"gross_wage_y": numpy.array([0.0, 0.0]) * factor},
            },
        }
        results[data_currency] = main(
            main_target="results__tree",
            policy_date_str="2025-01-01",
            input_data=InputData.tree(input_tree),
            orig_policy_objects=OrigPolicyObjects.root(middle_earth.ROOT_PATH),
            tt_targets=TTTargets.tree({"kings_levy": {"amount_y": None}}),
            rounding=False,
            data_currency=data_currency,
            backend=backend,
            unit_system=middle_earth.UNIT_SYSTEM,
        )
    castar = numpy.asarray(results["CASTAR"]["kings_levy"]["amount_y"])
    silver = numpy.asarray(results["SILVER_PENNY"]["kings_levy"]["amount_y"])
    numpy.testing.assert_allclose(silver, castar * 4.0)
    assert float(castar[1]) > 0.0


def test_parameters_are_never_scaled(backend: Literal["numpy", "jax"]):
    """Parameters keep their statutory values in every era (GEP 10).

    The policy environment is built without reference to any data currency:
    the silver-penny-era orc bounties — including the per-leaf-declared
    currency leaves at nesting depth two — carry exactly the numbers the
    statute states, and so does the age bound.
    """
    env = main(
        main_target="policy_environment",
        policy_date_str="1900-01-01",
        orig_policy_objects=OrigPolicyObjects.root(middle_earth.ROOT_PATH),
        backend=backend,
        unit_system=middle_earth.UNIT_SYSTEM,
    )
    raw = env["orc_hunting_bounty"]["raw_bounties_per_orc"].value
    assert raw["small_orc"] == 100
    assert raw["large_orc"]["peasant_hunter"] == 200
    assert raw["large_orc"]["noble_hunter"] == 100
    assert raw["minimum_hunter_age"] == 20


def _bare_payroll_tree(factor: float = 1.0) -> dict[str, Any]:
    """A household for the 2025 payroll-tax run, currency amounts scaled by `factor`."""
    return {
        "p_id": numpy.array([0, 1]),
        "kin_id": numpy.array([0, 0]),
        "p_id_spouse": numpy.array([1, 0]),
        "p_id_parent_1": numpy.array([-1, -1]),
        "p_id_parent_2": numpy.array([-1, -1]),
        "birth_year": numpy.array([1995, 1995]),  # age 30 in 2025
        "parent_is_noble": numpy.array([False, False]),
        "wealth": numpy.array([100.0, 200.0]) * factor,
        "payroll_tax": {
            "child_tax_credit": {"p_id_recipient": numpy.array([-1, -1])},
            "income": {"gross_wage_y": numpy.array([10000.0, 0.0]) * factor},
        },
    }


def _annotated_payroll_tree(currency: str, factor: float = 1.0) -> dict[str, Any]:
    """`_bare_payroll_tree` with every leaf a `UnitAnnotatedColumn` (GEP 10).

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


def _run_payroll(
    input_data: InputData,
    data_currency: str,
    backend: Literal["numpy", "jax"],
    main_target: str,
):
    return main(
        main_target=main_target,
        policy_date_str="2025-01-01",
        input_data=input_data,
        orig_policy_objects=OrigPolicyObjects.root(middle_earth.ROOT_PATH),
        tt_targets=TTTargets.tree({"payroll_tax": {"amount_y": None}}),
        rounding=False,
        data_currency=data_currency,
        backend=backend,
        unit_system=middle_earth.UNIT_SYSTEM,
    )


def test_unit_annotated_input_matches_bare_run(backend: Literal["numpy", "jax"]):
    """Data-currency-tagged annotated input gives the same result as bare (GEP 10)."""
    bare = _run_payroll(
        input_data=InputData.tree(_bare_payroll_tree()),
        data_currency="CASTAR",
        backend=backend,
        main_target="results__tree",
    )
    annotated = _run_payroll(
        input_data=InputData.tree_with_unit_annotations(
            _annotated_payroll_tree("CASTAR")
        ),
        data_currency="CASTAR",
        backend=backend,
        main_target="results__tree",
    )
    numpy.testing.assert_allclose(
        numpy.asarray(annotated["payroll_tax"]["amount_y"]),
        numpy.asarray(bare["payroll_tax"]["amount_y"]),
    )


def test_unit_annotated_input_converts_currency_at_boundary(
    backend: Literal["numpy", "jax"],
):
    """Silver-penny-tagged input (x4) with castar data currency converts at the
    boundary."""
    bare = _run_payroll(
        input_data=InputData.tree(_bare_payroll_tree()),
        data_currency="CASTAR",
        backend=backend,
        main_target="results__tree",
    )
    converted = _run_payroll(
        input_data=InputData.tree_with_unit_annotations(
            _annotated_payroll_tree(currency="SILVER_PENNY", factor=4.0)
        ),
        data_currency="CASTAR",
        backend=backend,
        main_target="results__tree",
    )
    numpy.testing.assert_allclose(
        numpy.asarray(converted["payroll_tax"]["amount_y"]),
        numpy.asarray(bare["payroll_tax"]["amount_y"]),
    )
    assert float(numpy.asarray(bare["payroll_tax"]["amount_y"])[0]) > 0.0


def test_results_tree_with_unit_annotations_are_precise(
    backend: Literal["numpy", "jax"],
):
    """Annotated results carry precise data-currency units (GEP 10)."""
    tagged = _run_payroll(
        input_data=InputData.tree(_bare_payroll_tree()),
        data_currency="CASTAR",
        backend=backend,
        main_target="results__tree_with_unit_annotations",
    )
    bare = _run_payroll(
        input_data=InputData.tree(_bare_payroll_tree()),
        data_currency="CASTAR",
        backend=backend,
        main_target="results__tree",
    )
    amount = tagged["payroll_tax"]["amount_y"]
    assert isinstance(amount, UnitAnnotatedColumn)
    assert amount.unit == TTSIMUnit.CASTAR.PER_YEAR
    assert str(amount.unit) == "CASTAR_PER_YEAR"
    numpy.testing.assert_allclose(
        numpy.asarray(amount.values),
        numpy.asarray(bare["payroll_tax"]["amount_y"]),
    )


def test_unit_annotated_input_and_output_round_trip(
    backend: Literal["numpy", "jax"],
):
    """A pint-tagged input tree in and a pint-tagged output tree out, end to end.

    Feeds silver-penny-tagged input (x4) into a run with castar data currency
    and requests the annotated results tree: the boundary converts the input
    and the output leaf carries the precise data-currency unit, with a
    magnitude equal to the bare castar run.
    """
    bare = _run_payroll(
        input_data=InputData.tree(_bare_payroll_tree()),
        data_currency="CASTAR",
        backend=backend,
        main_target="results__tree",
    )
    tagged = _run_payroll(
        input_data=InputData.tree_with_unit_annotations(
            _annotated_payroll_tree(currency="SILVER_PENNY", factor=4.0)
        ),
        data_currency="CASTAR",
        backend=backend,
        main_target="results__tree_with_unit_annotations",
    )
    amount = tagged["payroll_tax"]["amount_y"]
    assert isinstance(amount, UnitAnnotatedColumn)
    assert amount.unit == TTSIMUnit.CASTAR.PER_YEAR
    assert str(amount.unit) == "CASTAR_PER_YEAR"
    numpy.testing.assert_allclose(
        numpy.asarray(amount.values),
        numpy.asarray(bare["payroll_tax"]["amount_y"]),
    )


def test_unit_annotated_input_rejects_wrong_dimension(
    backend: Literal["numpy", "jax"],
):
    """A currency tag on the CALENDAR_YEAR birth-year column is rejected (GEP 10)."""
    tree = _annotated_payroll_tree("CASTAR")
    tree["birth_year"] = UnitAnnotatedColumn(
        values=numpy.array([1995, 1995]), unit=TTSIMUnit.CASTAR
    )
    with pytest.raises(UnitConsistencyError, match="inconsistent with the DAG"):
        _run_payroll(
            input_data=InputData.tree_with_unit_annotations(tree),
            data_currency="CASTAR",
            backend=backend,
            main_target="results__tree",
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
        _run_payroll(
            input_data=InputData.tree_with_unit_annotations(tree),
            data_currency="CASTAR",
            backend=backend,
            main_target="results__tree",
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
        _run_payroll(
            input_data=InputData.tree_with_unit_annotations(tree),
            data_currency="CASTAR",
            backend=backend,
            main_target="results__tree",
        )


def test_age_is_computed_from_the_birth_year(backend: Literal["numpy", "jax"]):
    """The calendar-point worked example (GEP 10, S1).

    Age is computed as ``policy_year - birth_year`` (a duration in years from
    two calendar years), and the birthday check reads the calendar-month
    framework node as a cyclic ordinal against the birth month.
    """
    results = main(
        main_target="results__tree",
        policy_date_str="2025-01-01",
        input_data=InputData.tree(
            {
                "p_id": numpy.array([0, 1, 2]),
                "birth_year": numpy.array([1995, 2015, 1914]),
                "birth_month": numpy.array([1, 6, 12]),
            }
        ),
        orig_policy_objects=OrigPolicyObjects.root(middle_earth.ROOT_PATH),
        tt_targets=TTTargets.tree({"age": None, "had_birthday_this_year": None}),
        rounding=False,
        backend=backend,
        unit_system=middle_earth.UNIT_SYSTEM,
    )
    numpy.testing.assert_array_equal(
        numpy.asarray(results["age"]), numpy.array([30, 10, 111])
    )
    # policy_month is January (1): only a person born in January has already had
    # their birthday by 2025-01-01.
    numpy.testing.assert_array_equal(
        numpy.asarray(results["had_birthday_this_year"]),
        numpy.array([True, False, False]),
    )
