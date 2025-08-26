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

from ttsim import plot
from ttsim.entry_point import main
from ttsim.interface_dag_elements.shared import (
    copy_environment,
    merge_trees,
    upsert_tree,
)
from ttsim.main_args import (
    InputData,
    Labels,
    OrigPolicyObjects,
    RawResults,
    Results,
    SpecializedEnvironment,
    SpecializedEnvironmentForPlottingAndTemplates,
    TTTargets,
)
from ttsim.main_target import MainTarget

copy_environment = copy_environment
merge_trees = merge_trees
upsert_tree = upsert_tree
main = main
MainTarget = MainTarget
InputData = InputData
Labels = Labels
OrigPolicyObjects = OrigPolicyObjects
RawResults = RawResults
Results = Results
SpecializedEnvironment = SpecializedEnvironment
SpecializedEnvironmentForPlottingAndTemplates = (
    SpecializedEnvironmentForPlottingAndTemplates
)
TTTargets = TTTargets

__all__ = [
    "InputData",
    "Labels",
    "MainTarget",
    "OrigPolicyObjects",
    "RawResults",
    "Results",
    "SpecializedEnvironment",
    "SpecializedEnvironmentForPlottingAndTemplates",
    "TTTargets",
    "__version__",
    "__version_tuple__",
    "copy_environment",
    "main",
    "merge_trees",
    "plot",
    "upsert_tree",
    "version",
    "version_tuple",
]
