from __future__ import annotations

from pathlib import Path

from ttsim.plot_dag import plot_full_interface_dag


def test_plot_full_interface_dag():
    plot_full_interface_dag(output_path=Path("full_interface_dag.html"))
