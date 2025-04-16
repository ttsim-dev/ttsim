from __future__ import annotations

try:
    import jax
except ImportError:
    IS_JAX_INSTALLED = False
else:
    IS_JAX_INSTALLED = True


if IS_JAX_INSTALLED:
    numpy_or_jax = jax.numpy
else:
    import numpy

    numpy_or_jax = numpy
