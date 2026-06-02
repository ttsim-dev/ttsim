from __future__ import annotations

# Make jaxtyping accept bare `Ellipsis` axis specs (produced by
# `from __future__ import annotations` evaluation) before any
# jaxtyping-subscripted type is created. See the module docstring.
from ttsim import _jaxtyping_patch  # noqa: F401

# Register beartype's package claw before any ttsim submodule imports so
# every ttsim.* module loads with runtime type checks installed via
# INTERNAL_CONF. User-facing constructors stack an explicit
# @beartype(conf=...) decorator that maps violations to the relevant
# project exception (see ttsim._beartype_conf).
#
# On by default (GEP 9, Option A): set `TTSIM_BEARTYPE_CLAW=0` to opt out.
# `RUNTIME_TYPE_CHECKING_ENABLED` reads that env var once and is the single switch —
# the same flag sets every conf's strategy in `ttsim._beartype_conf`, so
# opting out also disables the perimeter `@beartype` decorators and the
# synthesized forwarder, not just this package claw.
from ttsim._beartype_conf import RUNTIME_TYPE_CHECKING_ENABLED

if RUNTIME_TYPE_CHECKING_ENABLED:
    import warnings

    from beartype.claw import beartype_package
    from beartype.roar import BeartypeClawDecorWarning

    from ttsim._beartype_conf import INTERNAL_CONF

    beartype_package("ttsim", conf=INTERNAL_CONF)

    # `@interface_input` produces a non-callable `InterfaceInput` metadata
    # dataclass. The claw cannot `@beartype` a non-callable object, so it
    # warns once per such object on import. There is nothing to type-check
    # on a pure data holder, so suppress exactly that known-harmless case;
    # any other `BeartypeClawDecorWarning` still surfaces.
    warnings.filterwarnings(
        "ignore",
        message=r'Object "InterfaceInput\(',
        category=BeartypeClawDecorWarning,
    )

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

from ttsim import plot, unit_converters
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
unit_converters = unit_converters

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
    "unit_converters",
    "upsert_tree",
    "version",
    "version_tuple",
]
