from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, cast

from ttsim.exceptions import UnitDefinitionError
from ttsim.interface_dag_elements.interface_node_objects import (
    input_dependent_interface_function,
)
from ttsim.tt.units import (
    base_currency,
    currency_family_root,
    parse_compositional_unit,
    token_source_currency,
)

if TYPE_CHECKING:
    from ttsim.typing import FlatOrigParamSpecs

#: The spec keys that may declare a unit (GEP 10); their values name the
#: concrete currencies a package's parameters are denominated in.
_UNIT_DECLARATION_KEYS = ("unit", "input_unit", "output_unit")


@input_dependent_interface_function(
    include_if_any_input_present=[
        "orig_policy_objects__root",
        "orig_policy_objects__param_specs",
    ],
    leaf_name="currency",
    in_top_level_namespace=True,
)
def currency_from_policy_objects(
    orig_policy_objects__param_specs: FlatOrigParamSpecs,
) -> str | None:
    """The currency of the input data and of the output.

    Defaults to the base currency of the *family* the policy parameters are
    denominated in — read off the parameters' ``unit:`` declarations, so the
    default follows the policy objects in play even when several packages'
    currency families are registered in one process. Without policy objects to
    read the denomination off (a run on a directly-provided policy
    environment), the default is the registered base currency (``None`` if no
    currency has been registered, e.g. a bare ttsim run). Override via
    ``main(currency=...)`` with another registered currency of that family to
    run the whole system in it: parameters are converted from their legal
    source currency to this one at build time, and the output is produced in
    this currency. Functions themselves stay currency-agnostic.
    """
    roots = sorted(
        {
            currency_family_root(name)
            for spec in orig_policy_objects__param_specs.values()
            # The raw specs are plain dicts read from YAML; the OrigParamSpec
            # protocol exposes no iteration.
            for name in _currencies_in_spec(cast("Mapping[Any, Any]", spec))
        }
    )
    if len(roots) == 1:
        return roots[0]
    if len(roots) > 1:
        raise UnitDefinitionError(
            f"The policy parameters are denominated in currencies of "
            f"{len(roots)} different families ({', '.join(roots)}), so there "
            f"is no default run currency. Pass `currency=...` explicitly "
            f"(GEP 10)."
        )
    # No currency-denominated parameter: fall back to the registry-wide base.
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
    """The currency of the input data and of the output.

    Defaults to the base currency of the *family* the policy parameters are
    denominated in — read off the parameters' ``unit:`` declarations, so the
    default follows the policy objects in play even when several packages'
    currency families are registered in one process. Without policy objects to
    read the denomination off (a run on a directly-provided policy
    environment), the default is the registered base currency (``None`` if no
    currency has been registered, e.g. a bare ttsim run). Override via
    ``main(currency=...)`` with another registered currency of that family to
    run the whole system in it: parameters are converted from their legal
    source currency to this one at build time, and the output is produced in
    this currency. Functions themselves stay currency-agnostic.
    """
    return base_currency()


def _currencies_in_spec(spec: Mapping[Any, Any]) -> set[str]:
    """The concrete currencies a raw parameter spec's declarations name.

    Walks every ``unit:`` / ``input_unit:`` / ``output_unit:`` in the spec —
    top-level or dated — across *all* dates: the date does not matter for the
    family derivation, a package's currencies all chain to one base.
    """
    found: set[str] = set()
    for key, value in spec.items():
        if key in _UNIT_DECLARATION_KEYS:
            found |= _currencies_in_unit_value(value)
        elif isinstance(value, Mapping):
            found |= _currencies_in_spec(value)
    return found


def _currencies_in_unit_value(value: Any) -> set[str]:  # noqa: ANN401
    """The concrete currencies one ``unit:`` value names.

    The value is a single token or a (nested) per-leaf mapping of tokens. A
    spelling the parser rejects is skipped here; the load-time validation
    reports it.
    """
    if isinstance(value, str):
        try:
            token = parse_compositional_unit(value)
        except UnitDefinitionError:
            return set()
        source = token_source_currency(token)
        return set() if source is None else {source}
    if isinstance(value, Mapping):
        return set().union(
            *(_currencies_in_unit_value(leaf) for leaf in value.values())
        )
    return set()
