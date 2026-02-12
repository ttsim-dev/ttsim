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
    """The name of the backend to use for the taxes and transfers function."""


@interface_function(in_top_level_namespace=True)
def xnp(backend: Literal["numpy", "jax"]) -> ModuleType:
    """The actual backend used for numerical operations (numpy or jax.numpy)."""
    if backend == "numpy":
        xnp = numpy
    elif backend == "jax":
        import jax  # noqa: PLC0415

        xnp = jax.numpy
    else:
        raise ValueError(f"Unsupported backend: {backend}. Choose 'numpy' or 'jax'.")
    return xnp


@interface_function(in_top_level_namespace=True)
def dnp(backend: Literal["numpy", "jax"]) -> ModuleType:
    """The backend used for datetime objects (numpy or jax-datetime)."""
    if backend == "numpy":
        dnp = numpy
    elif backend == "jax":
        # import jax_datetime # noqa: ERA001

        dnp = numpy  # jax_datetime
    return dnp
