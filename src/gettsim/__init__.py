"""This module contains the main namespace of gettsim."""

from __future__ import annotations

try:
    # Import the version from _version.py which is dynamically created by
    # setuptools-scm upon installing the project with pip.
    # Do not put it under version control!
    from _gettsim._version import version as __version__
except ImportError:
    __version__ = "unknown"


from typing import Literal

import pytest

from _gettsim_tests import TEST_DIR


def test(backend: Literal["numpy", "jax"] = "numpy") -> None:
    pytest.main([str(TEST_DIR), "--backend", backend])


__all__ = [
    "__version__",
    "test",
]
