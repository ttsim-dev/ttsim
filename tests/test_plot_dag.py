from __future__ import annotations

from pathlib import Path

import pytest

from ttsim.main import main
from ttsim.plot_dag import (
    _get_tt_dag_with_node_metadata,
    _QNameNodeSelector,
    plot_interface_dag,
)
from ttsim.tt import (
    ScalarParam,
    param_function,
    policy_function,
)

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
    plot_interface_dag(include_fail_and_warn_nodes=include_fail_and_warn_nodes)


@pytest.mark.parametrize(
    (
        "node_selector",
        "expected_nodes",
    ),
    [
        (
            _QNameNodeSelector(
                qnames=["payroll_tax__amount_y"],
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
                qnames=["payroll_tax__amount_m"],
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
                qnames=["payroll_tax__amount_m"],
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
                qnames=["payroll_tax__amount_m", "property_tax__amount_m"],
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
                qnames=["property_tax__amount_m"],
                type="ancestors",
            ),
            [
                "xnp",  # not part of the plot, will be removed by a downstream func
                "property_tax__acre_size_in_hectares_after_cap",
                "property_tax__tax_schedule",
                "property_tax__year_from_which_cap_is_applied",
                "property_tax__cap_in_hectares",
                "property_tax__amount_y",
                "property_tax__amount_m",
            ],
        ),
        (
            _QNameNodeSelector(
                qnames=["payroll_tax__amount_y"],
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
                qnames=["payroll_tax__amount_m"],
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
                qnames=["payroll_tax__amount_m"],
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
                qnames=["payroll_tax__amount_m", "property_tax__amount_m"],
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
                qnames=["payroll_tax__amount_y"],
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
                qnames=["payroll_tax__amount_m"],
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
                qnames=["payroll_tax__amount_m"],
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
                qnames=["payroll_tax__amount_m", "property_tax__amount_m"],
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
                qnames=["housing_benefits__income__amount_m"],
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
                qnames=["payroll_tax__amount_m", "property_tax__amount_m"],
                type="nodes",
            ),
            [
                "payroll_tax__amount_m",
                "property_tax__amount_m",
            ],
        ),
    ],
)
def test_node_selector(node_selector, expected_nodes, backend):
    environment = main(
        main_target="policy_environment",
        policy_date_str="2025-01-01",
        orig_policy_objects={"root": Path(__file__).parent / "mettsim"},
        backend=backend,
    )
    dag = _get_tt_dag_with_node_metadata(
        environment=environment,
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
    }
    dag = _get_tt_dag_with_node_metadata(
        environment=environment,
        include_params=include_params,
    )
    assert set(dag.nodes()) == set(expected_nodes)
