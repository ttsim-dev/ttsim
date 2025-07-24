from __future__ import annotations

import sys

import pytest

import ttsim


@pytest.mark.xfail(reason="Requires own package.")
def test_import():
    assert hasattr(ttsim, "__version__")


def test_python_version():
    assert sys.version_info >= (3, 11)
