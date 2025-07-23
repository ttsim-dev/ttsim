from __future__ import annotations

from pathlib import Path

from ttsim.plot_dag import plot_tt_dag

GETTSIM_ROOT = Path(__file__).parent.parent / "_gettsim"


def test_gettsim_policy_environment_dag_with_params():
    plot_tt_dag(
        policy_date_str="2025-01-01",
        root=GETTSIM_ROOT,
        include_params=True,
        title="GETTSIM Policy Environment DAG with parameters",
        show_node_description=True,
    )


def test_gettsim_policy_environment_dag_without_params():
    plot_tt_dag(
        policy_date_str="2025-01-01",
        root=GETTSIM_ROOT,
        include_params=False,
        title="GETTSIM Policy Environment DAG without parameters",
        show_node_description=True,
    )
