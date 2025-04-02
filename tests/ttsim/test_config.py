import importlib

import pytest

import ttsim


def test_default_backend():
    from ttsim.config import numpy_or_jax

    assert numpy_or_jax.__name__ == "numpy"


def test_set_backend():
    is_jax_installed = importlib.util.find_spec("jax") is not None

    # expect default backend
    from ttsim.config import numpy_or_jax

    assert numpy_or_jax.__name__ == "numpy"

    if is_jax_installed:
        # set jax backend
        ttsim.config.set_array_backend("jax")
        from ttsim.config import numpy_or_jax

        assert numpy_or_jax.__name__ == "jax.numpy"

        from ttsim.config import USE_JAX

        assert USE_JAX
    else:
        with pytest.raises(AssertionError):
            ttsim.config.set_array_backend("jax")


@pytest.mark.parametrize("backend", ["dask", "jax.numpy"])
def test_wrong_backend(backend):
    with pytest.raises(ValueError):
        ttsim.config.set_array_backend(backend)
