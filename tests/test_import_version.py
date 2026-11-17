from __future__ import annotations

import ttsim


def test_import():
    assert hasattr(ttsim, "__version__")
