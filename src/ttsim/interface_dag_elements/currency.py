from __future__ import annotations

from typing import Literal

from ttsim.interface_dag_elements.interface_node_objects import interface_function
from ttsim.tt.currencies import base_currency


@interface_function(leaf_name="currency", in_top_level_namespace=True)
def currency(
    backend: Literal["numpy", "jax"],  # noqa: ARG001
) -> str:
    """The currency the whole run is denominated in.

    Defaults to the registered base currency (a downstream package registers it
    on import, so users need not pass ``currency=`` themselves; a run with no
    registered currency fails). Override via ``main(currency=...)`` with another
    registered currency to run the system in it: parameters are converted from
    their legal source currency to this one at build time and the output is
    produced in it; functions themselves stay currency-agnostic.
    """
    return base_currency()
