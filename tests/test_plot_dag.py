from __future__ import annotations

import datetime
from types import ModuleType
from typing import Any

import pytest

from mettsim import middle_earth
from ttsim.main import main
from ttsim.main_args import InputData, TTTargets
from ttsim.main_target import MainTarget
from ttsim.plot.dag.interface import interface
from ttsim.plot.dag.tt import (
    _get_tt_dag_with_node_metadata,
    _QNameNodeSelector,
)
from ttsim.tt import (
    ScalarParam,
    param_function,
    policy_function,
)
from ttsim.tt.column_objects_param_function import PolicyInput


def get_required_policy_env_objects(policy_date: datetime.date) -> dict[str, Any]:
    return {
        "policy_year": ScalarParam(
            value=policy_date.year,
            start_date=policy_date,
            end_date=policy_date,
        ),
        "policy_month": ScalarParam(
            value=policy_date.month,
            start_date=policy_date,
            end_date=policy_date,
        ),
        "policy_day": ScalarParam(
            value=policy_date.day,
            start_date=policy_date,
            end_date=policy_date,
        ),
        "evaluation_year": PolicyInput(
            leaf_name="evaluation_year",
            data_type=int,
            start_date=policy_date,
            end_date=policy_date,
            description="The evaluation year, will typically be set via `main`.",
        ),
        "evaluation_month": PolicyInput(
            leaf_name="evaluation_month",
            data_type=int,
            start_date=policy_date,
            end_date=policy_date,
            description="The evaluation month, will typically be set via `main`.",
        ),
        "evaluation_day": PolicyInput(
            leaf_name="evaluation_day",
            data_type=int,
            start_date=policy_date,
            end_date=policy_date,
            description="The evaluation day, will typically be set via `main`.",
        ),
    }


SOME_PARAM_OBJECT = ScalarParam(
    value=111,
    start_date="2025-01-01",
    end_date="2025-12-31",
    unit=None,
    reference_period="",
    name="",
    description="",
)


@param_function(
    start_date="2025-01-01",
    end_date="2025-12-31",
)
def some_param_function():
    return 1


@policy_function(
    start_date="2025-01-01",
    end_date="2025-12-31",
)
def some_policy_function():
    return 1


@pytest.mark.parametrize(
    "include_fail_and_warn_nodes",
    [
        True,
        False,
    ],
)
def test_plot_full_interface_dag(include_fail_and_warn_nodes):
    interface(include_fail_and_warn_nodes=include_fail_and_warn_nodes)


@pytest.mark.parametrize(
    (
        "node_selector",
        "expected_nodes",
    ),
    [
        (
            _QNameNodeSelector(
                qnames={"payroll_tax__amount_y"},
                type="ancestors",
                order=1,
            ),
            [
                "payroll_tax__amount_y",
                "payroll_tax__amount_standard_y",
                "payroll_tax__amount_reduced_y",
                "parent_is_noble_fam",
                "wealth_fam",
                "payroll_tax__wealth_threshold_for_reduced_tax_rate",
            ],
        ),
        (
            _QNameNodeSelector(
                qnames={"payroll_tax__amount_m"},
                type="ancestors",
                order=1,
            ),
            [
                "payroll_tax__amount_m",
                "payroll_tax__amount_y",
            ],
        ),
        (
            _QNameNodeSelector(
                qnames={"payroll_tax__amount_m"},
                type="ancestors",
                order=2,
            ),
            [
                "payroll_tax__amount_m",
                "payroll_tax__amount_y",
                "payroll_tax__amount_standard_y",
                "payroll_tax__amount_reduced_y",
                "parent_is_noble_fam",
                "wealth_fam",
                "payroll_tax__wealth_threshold_for_reduced_tax_rate",
            ],
        ),
        (
            _QNameNodeSelector(
                qnames={"payroll_tax__amount_m", "property_tax__amount_m"},
                type="ancestors",
                order=1,
            ),
            [
                "payroll_tax__amount_m",
                "payroll_tax__amount_y",
                "property_tax__amount_m",
                "property_tax__amount_y",
            ],
        ),
        (
            _QNameNodeSelector(
                qnames={"property_tax__amount_m"},
                type="ancestors",
            ),
            [
                "evaluation_year",
                "property_tax__acre_size_in_hectares",
                "property_tax__acre_size_in_hectares_after_cap",
                "evaluation_year",
                "property_tax__tax_schedule",
                "property_tax__year_from_which_cap_is_applied",
                "property_tax__cap_in_hectares",
                "property_tax__amount_y",
                "property_tax__amount_m",
            ],
        ),
        (
            _QNameNodeSelector(
                qnames={"payroll_tax__amount_y"},
                type="neighbors",
                order=1,
            ),
            [
                "payroll_tax__amount_m",
                "payroll_tax__amount_y",
                "payroll_tax__amount_standard_y",
                "payroll_tax__amount_reduced_y",
                "parent_is_noble_fam",
                "wealth_fam",
                "payroll_tax__wealth_threshold_for_reduced_tax_rate",
            ],
        ),
        (
            _QNameNodeSelector(
                qnames={"payroll_tax__amount_m"},
                type="neighbors",
                order=1,
            ),
            [
                "housing_benefits__income__amount_m",
                "payroll_tax__amount_m",
                "payroll_tax__amount_y",
            ],
        ),
        (
            _QNameNodeSelector(
                qnames={"payroll_tax__amount_m"},
                type="neighbors",
                order=2,
            ),
            [
                "housing_benefits__income__amount_m_fam",
                "housing_benefits__income__amount_m",
                "payroll_tax__amount_m",
                "payroll_tax__amount_y",
                "payroll_tax__amount_standard_y",
                "payroll_tax__amount_reduced_y",
                "parent_is_noble_fam",
                "wealth_fam",
                "payroll_tax__wealth_threshold_for_reduced_tax_rate",
            ],
        ),
        (
            _QNameNodeSelector(
                qnames={"payroll_tax__amount_m", "property_tax__amount_m"},
                type="neighbors",
                order=1,
            ),
            [
                "housing_benefits__income__amount_m",
                "payroll_tax__amount_m",
                "payroll_tax__amount_y",
                "property_tax__amount_m",
                "property_tax__amount_y",
            ],
        ),
        (
            _QNameNodeSelector(
                qnames={"payroll_tax__amount_y"},
                type="descendants",
                order=1,
            ),
            [
                "payroll_tax__amount_m",
                "payroll_tax__amount_y",
            ],
        ),
        (
            _QNameNodeSelector(
                qnames={"payroll_tax__amount_m"},
                type="descendants",
                order=1,
            ),
            [
                "housing_benefits__income__amount_m",
                "payroll_tax__amount_m",
            ],
        ),
        (
            _QNameNodeSelector(
                qnames={"payroll_tax__amount_m"},
                type="descendants",
                order=2,
            ),
            [
                "housing_benefits__income__amount_m_fam",
                "housing_benefits__income__amount_m",
                "payroll_tax__amount_m",
            ],
        ),
        (
            _QNameNodeSelector(
                qnames={"payroll_tax__amount_m", "property_tax__amount_m"},
                type="descendants",
                order=1,
            ),
            [
                "housing_benefits__income__amount_m",
                "payroll_tax__amount_m",
                "property_tax__amount_m",
            ],
        ),
        (
            _QNameNodeSelector(
                qnames={"housing_benefits__income__amount_m"},
                type="descendants",
            ),
            [
                "housing_benefits__amount_m_fam",
                "housing_benefits__eligibility__requirement_fulfilled_fam",
                "housing_benefits__income__amount_m_fam",
                "housing_benefits__income__amount_m",
            ],
        ),
        (
            _QNameNodeSelector(
                qnames={"payroll_tax__amount_m", "property_tax__amount_m"},
                type="nodes",
            ),
            [
                "payroll_tax__amount_m",
                "property_tax__amount_m",
            ],
        ),
    ],
)
def test_node_selector(node_selector, expected_nodes):
    dag = _get_tt_dag_with_node_metadata(
        root=middle_earth.ROOT_PATH,
        policy_date_str="2025-01-01",
        node_selector=node_selector,
        include_params=True,
    )
    assert set(dag.nodes()) == set(expected_nodes)


@pytest.mark.parametrize(
    (
        "include_params",
        "expected_nodes",
    ),
    [
        (
            True,
            [
                "some_param",
                "some_param_function",
                "some_policy_function",
            ],
        ),
        (
            False,
            [
                "some_policy_function",
            ],
        ),
    ],
)
def test_params_are_removed_from_dag(include_params, expected_nodes):
    environment = {
        "some_param": SOME_PARAM_OBJECT,
        "some_param_function": some_param_function,
        "some_policy_function": some_policy_function,
        **get_required_policy_env_objects(policy_date=datetime.date(2025, 1, 1)),
    }
    dag = _get_tt_dag_with_node_metadata(
        root=middle_earth.ROOT_PATH,
        policy_date_str="2025-01-01",
        policy_environment=environment,
        include_params=include_params,
    )
    assert set(dag.nodes()) == set(expected_nodes)


def test_orphaned_dates_are_removed_from_dag():
    dag = _get_tt_dag_with_node_metadata(
        root=middle_earth.ROOT_PATH,
        policy_date_str="2025-01-01",
        include_params=True,
    )
    assert "evaluation_day" not in dag.nodes()
    assert "policy_day" not in dag.nodes()


def test_input_data_overrides_plotting_dag(xnp: ModuleType):
    dag = main(
        main_target=MainTarget.specialized_environment_from_policy_inputs.complete_dag,
        policy_date_str="2025-01-01",
        orig_policy_objects={"root": middle_earth.ROOT_PATH},
        tt_targets=TTTargets(qname=["payroll_tax__amount_y"]),
        input_data=InputData.tree(
            {
                "p_id": xnp.array([100]),
                "payroll_tax": {
                    "income": {
                        "amount_y": xnp.array([100]),
                    }
                },
            }
        ),
    )
    assert "payroll_tax__income__amount_y" in dag.nodes()
    assert "payroll_tax__income__gross_wage_y" not in dag.nodes()
    assert "payroll_tax__income__deductions_y" not in dag.nodes()
    assert "payroll_tax__amount_y" in dag.nodes()
