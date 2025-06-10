from __future__ import annotations

from pathlib import Path

from _gettsim.config import GETTSIM_ROOT
from ttsim.plot_dag import plot_tt_dag


def test_gettsim_policy_environment_dag_with_params():
    plot_tt_dag(
        with_params=True,
        inputs_for_main={
            "date_str": "2025-01-01",
            "orig_policy_objects__root": GETTSIM_ROOT,
        },
        title="GETTSIM Policy Environment DAG with parameters",
        output_path=Path("gettsim_dag_with_params.html"),
    )


def test_gettsim_policy_environment_dag_without_params():
    plot_tt_dag(
        with_params=False,
        inputs_for_main={
            "date_str": "2025-01-01",
            "orig_policy_objects__root": GETTSIM_ROOT,
        },
        title="GETTSIM Policy Environment DAG without parameters",
        output_path=Path("gettsim_dag_without_params.html"),
    )
