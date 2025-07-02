from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from types import ModuleType
import numpy

from ttsim.interface_dag_elements.interface_node_objects import (
    interface_function,
    interface_input,
)


@interface_input(in_top_level_namespace=True)
def backend() -> Literal["numpy", "jax"]:
    """The computing backend to use for the taxes and transfers function."""


@interface_function(in_top_level_namespace=True)
def xnp(backend: Literal["numpy", "jax"]) -> ModuleType:
    """
    Return the backend for numerical operations (either NumPy or jax).
    """
    if backend == "numpy":
        xnp = numpy
    elif backend == "jax":
        import jax

        xnp = jax.numpy
    else:
        raise ValueError(f"Unsupported backend: {backend}. Choose 'numpy' or 'jax'.")
    return xnp


@interface_function(in_top_level_namespace=True)
def dnp(backend: Literal["numpy", "jax"]) -> ModuleType:
    """
    Return the backend for datetime objects (either NumPy or jax-datetime)
    """
    if backend == "numpy":
        dnp = numpy
    elif backend == "jax":
        # import jax_datetime # noqa: ERA001

        dnp = numpy  # jax_datetime
    return dnp
