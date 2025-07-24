from __future__ import annotations

try:
    # Import the version from _version.py which is dynamically created by
    # setuptools-scm upon installing the project with pip.
    # Do not put it under version control!
    from ttsim._version import __version__, __version_tuple__, version, version_tuple
except ImportError:
    __version__ = "unknown"
    __version_tuple__ = ("unknown", "unknown", "unknown")
    version = "unknown"
    version_tuple = ("unknown", "unknown", "unknown")

from ttsim import tt
from ttsim.copy_environment import copy_environment
from ttsim.interface_dag_elements.shared import merge_trees, upsert_tree
from ttsim.main import main
from ttsim.main_args import (
    InputData,
    Labels,
    OrigPolicyObjects,
    RawResults,
    Results,
    SpecializedEnvironment,
    TTTargets,
)
from ttsim.main_target import MainTarget
from ttsim.plot_dag import plot_interface_dag, plot_tt_dag

copy_environment = copy_environment
merge_trees = merge_trees
upsert_tree = upsert_tree
main = main
plot_tt_dag = plot_tt_dag
plot_interface_dag = plot_interface_dag
MainTarget = MainTarget
InputData = InputData
Labels = Labels
OrigPolicyObjects = OrigPolicyObjects
RawResults = RawResults
Results = Results
SpecializedEnvironment = SpecializedEnvironment
TTTargets = TTTargets
tt = tt

__all__ = [
    "InputData",
    "Labels",
    "MainTarget",
    "OrigPolicyObjects",
    "RawResults",
    "Results",
    "SpecializedEnvironment",
    "TTTargets",
    "__version__",
    "__version_tuple__",
    "copy_environment",
    "main",
    "merge_trees",
    "plot_interface_dag",
    "plot_tt_dag",
    "tt",
    "upsert_tree",
    "version",
    "version_tuple",
]
