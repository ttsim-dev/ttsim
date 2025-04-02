from __future__ import annotations

import importlib

import numpy

# Defaults
USE_JAX = False
numpy_or_jax = numpy


def set_array_backend(backend: str):
    """Set array library backend.

    backend (str): Must be in {'jax', 'numpy'}.

    """
    if backend not in {"jax", "numpy"}:
        raise ValueError(f"Backend must be in {'jax', 'numpy'} but is {backend}.")

    if backend == "jax":
        assert importlib.util.find_spec("jax") is not None, "JAX is not installed."
        global USE_JAX  # noqa: PLW0603
        global numpy_or_jax  # noqa: PLW0603
        import jax

        USE_JAX = True
        numpy_or_jax = jax.numpy
        jax.config.update("jax_platform_name", "cpu")
