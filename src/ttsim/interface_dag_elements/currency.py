from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ttsim.interface_dag_elements.interface_node_objects import (
    input_dependent_interface_function,
)
from ttsim.tt.units import base_currency

if TYPE_CHECKING:
    from ttsim.typing import FlatOrigParamSpecs


@input_dependent_interface_function(
    include_if_any_input_present=[
        "orig_policy_objects__root",
        "orig_policy_objects__param_specs",
    ],
    leaf_name="currency",
    in_top_level_namespace=True,
)
def currency_from_policy_objects(
    orig_policy_objects__param_specs: FlatOrigParamSpecs,  # noqa: ARG001
) -> str | None:
    """The currency the whole run is denominated in.

    Defaults to the single registered base currency (``None`` if none is
    registered, e.g. a bare ttsim run). With base currencies of more than one
    family registered in the process there is no default — pass ``currency=``
    explicitly. Override via ``main(currency=...)`` with another registered
    currency of the base's family to run the system in it: parameters are
    converted from their legal source currency to this one at build time and the
    output is produced in it; functions themselves stay currency-agnostic.
    """
    # Depends on the policy objects only for ordering: their modules register the
    # currencies, so the base must be read after they are materialised.
    return base_currency()


@input_dependent_interface_function(
    include_if_no_input_present=[
        "orig_policy_objects__root",
        "orig_policy_objects__param_specs",
    ],
    leaf_name="currency",
    in_top_level_namespace=True,
)
def currency_from_registry(
    backend: Literal["numpy", "jax"],  # noqa: ARG001
) -> str | None:
    """The currency the whole run is denominated in.

    Defaults to the single registered base currency (``None`` if none is
    registered, e.g. a bare ttsim run). With base currencies of more than one
    family registered in the process there is no default — pass ``currency=``
    explicitly. Override via ``main(currency=...)`` with another registered
    currency of the base's family to run the system in it: parameters are
    converted from their legal source currency to this one at build time and the
    output is produced in it; functions themselves stay currency-agnostic.
    """
    return base_currency()
