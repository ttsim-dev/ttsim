"""This module contains the main namespace of gettsim."""

from __future__ import annotations

try:
    # Import the version from _version.py which is dynamically created by
    # setuptools-scm upon installing the project with pip.
    # Do not put it under version control!
    from _gettsim._version import version as __version__
except ImportError:
    __version__ = "unknown"


import itertools
import warnings
from typing import Any

import pytest

from _gettsim_tests import TEST_DIR
from ttsim import (
    FunctionsAndDataOverlapWarning,
    GroupCreationFunction,
    PolicyFunction,
    compute_taxes_and_transfers,
    group_creation_function,
    plot_dag,
    policy_function,
    set_up_policy_environment,
)

COUNTER_TEST_EXECUTIONS = itertools.count()


def test(*args: Any) -> None:
    n_test_executions = next(COUNTER_TEST_EXECUTIONS)

    if n_test_executions == 0:
        pytest.main([str(TEST_DIR), "--noconftest", *args])
    else:
        warnings.warn(
            "Repeated execution of the test suite is not possible. Start a new Python "
            "session or restart the kernel in a Jupyter/IPython notebook to re-run the "
            "tests.",
            stacklevel=2,
        )


__all__ = [
    "FunctionsAndDataOverlapWarning",
    "GroupCreationFunction",
    "PolicyFunction",
    "__version__",
    "compute_taxes_and_transfers",
    "group_creation_function",
    "plot_dag",
    "policy_function",
    "set_up_policy_environment",
]
