from __future__ import annotations

from typing import Literal

from ttsim.interface_dag_elements.interface_node_objects import interface_function
from ttsim.tt.units import base_currency


@interface_function(in_top_level_namespace=True)
def currency(
    backend: Literal["numpy", "jax"],  # noqa: ARG001
) -> str | None:
    """The currency of the input data and of the output.

    Defaults to the registered base currency (``None`` if no currency has been
    registered, e.g. a bare ttsim run). Override via ``main(currency=...)`` with
    another registered currency to run the whole system in that currency:
    parameters are converted from their legal source currency to this one at
    build time, and the output is produced in this currency. Functions
    themselves stay currency-agnostic.

    The unused ``backend`` dependency anchors this node in the interface DAG: an
    argument-less interface function would be treated as a missing root input.
    """
    return base_currency()
