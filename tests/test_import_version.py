from __future__ import annotations

import sys

import ttsim


def test_import():
    assert hasattr(ttsim, "__version__")


def test_python_version():
    assert sys.version_info >= (3, 11)
