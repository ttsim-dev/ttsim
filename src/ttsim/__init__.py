from __future__ import annotations

from ttsim._version import __version__, __version_tuple__, version, version_tuple
from ttsim.copy_environment import copy_environment
from ttsim.interface_dag import main
from ttsim.interface_dag_elements import MainTarget
from ttsim.interface_dag_elements.shared import merge_trees
from ttsim.main_args import (
    InputData,
    Labels,
    OrigPolicyObjects,
    RawResults,
    Results,
    SpecializedEnvironment,
    TTTargets,
)

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
    "version",
    "version_tuple",
]
