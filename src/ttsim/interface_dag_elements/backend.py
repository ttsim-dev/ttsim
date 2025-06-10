from __future__ import annotations

from types import ModuleType

import numpy

from ttsim.interface_dag_elements.interface_node_objects import interface_function


@interface_function(in_top_level_namespace=True)
def xnp(backend: str) -> ModuleType:
    """
    Return the backend for numerical operations (either NumPy or jax).
    """

    if backend == "numpy":
        xnp = numpy
    elif backend == "jax":
        try:
            import jax
        except ImportError:
            raise ImportError(
                "jax is not installed. Please install jax to use the 'jax' backend."
            )
        xnp = jax.numpy
    else:
        raise ValueError(f"Unsupported backend: {backend}. Choose 'numpy' or 'jax'.")
    return xnp


@interface_function(in_top_level_namespace=True)
def dnp(backend: str) -> ModuleType:
    """
    Return the backend for datetime objects (either NumPy or jax-datetime)
    """
    global dnp

    if backend == "numpy":
        dnp = numpy
    elif backend == "jax":
        try:
            import jax_datetime
        except ImportError:
            raise ImportError(
                "jax-datetime is not installed. Please install jax-datetime to use the 'jax' backend."
            )
        dnp = jax_datetime
    return dnp
