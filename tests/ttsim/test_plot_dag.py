from __future__ import annotations

from pathlib import Path

import pytest
from mettsim.config import METTSIM_ROOT

from ttsim.interface_dag import main
from ttsim.plot_dag import (
    _get_tt_dag_with_node_metadata,
    _QNameNodeSelector,
    plot_interface_dag,
)


def test_plot_full_interface_dag():
    plot_interface_dag().write_html(Path("full_interface_dag.html"))


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
def test_node_selector(node_selector, expected_nodes):
    environment = main(
        inputs={
            "date_str": "2025-01-01",
            "orig_policy_objects__root": METTSIM_ROOT,
        },
        targets=["policy_environment"],
    )["policy_environment"]
    dag = _get_tt_dag_with_node_metadata(
        environment=environment,
        node_selector=node_selector,
        include_params=True,
    )
    assert set(dag.nodes()) == set(expected_nodes)
