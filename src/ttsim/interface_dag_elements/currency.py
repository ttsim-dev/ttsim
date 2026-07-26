from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any

import dags.tree as dt
import pint

from ttsim.interface_dag_elements.interface_node_objects import interface_function
from ttsim.tt.units import (
    UNSET_UNIT,
    UnitSystem,
    strip_input_quantity_at_boundary,
    ttsim_unit_has_currency,
)
from ttsim.typing import SpecEnvWithoutTreeLogicAndWithDerivedFunctions

if TYPE_CHECKING:
    from ttsim.typing import FlatData


@interface_function(leaf_name="data_currency", in_top_level_namespace=True)
def data_currency(unit_system: UnitSystem) -> str:
    """The currency the user's data arrives in and results are returned in.

    Defaults to the policy system's base currency. Override via
    ``main(data_currency=...)`` with another of the system's currencies.
    """
    return unit_system.base_currency


@interface_function(leaf_name="computation_currency", in_top_level_namespace=True)
def computation_currency(policy_date: datetime.date, unit_system: UnitSystem) -> str:
    """The currency the computation runs in — the policy date's statutory currency.

    Read off the dated mapping the policy system declares.
    """
    return unit_system.statutory_currency_for_date(policy_date)


@interface_function(in_top_level_namespace=True)
def input_data_in_computation_currency(
    input_data__flat: FlatData,
    specialized_environment__without_tree_logic_and_with_derived_functions: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,  # noqa: E501
    data_currency: str,
    computation_currency: str,
    unit_system: UnitSystem,
) -> FlatData:
    """The input data with every value a bare magnitude in the computation currency."""
    registry: pint.UnitRegistry = unit_system.registry
    stripped = {
        path: (
            strip_input_quantity_at_boundary(
                quantity=value,
                data_currency=data_currency,
                registry=registry,
                column_label=dt.qname_from_tree_path(path),
            )
            if isinstance(value, pint.Quantity)
            else value
        )
        for path, value in input_data__flat.items()
    }
    if data_currency == computation_currency:
        return stripped
    factor = unit_system.currency_conversion_factor(
        source_currency=data_currency,
        target_currency=computation_currency,
    )
    return {
        path: (
            _convert_currency_value(
                value=value,
                factor=factor,
                qname=dt.qname_from_tree_path(path),
                specialized_environment=(
                    specialized_environment__without_tree_logic_and_with_derived_functions
                ),
            )
        )
        for path, value in stripped.items()
    }


def _convert_currency_value(
    value: Any,  # noqa: ANN401 (a column array or an input scalar)
    *,
    factor: float,
    qname: str,
    specialized_environment: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
) -> Any:  # noqa: ANN401
    """Apply a currency factor when the qname declares a currency quantity."""
    declared_unit = getattr(specialized_environment.get(qname), "unit", UNSET_UNIT)
    if not ttsim_unit_has_currency(declared_unit):
        return value
    return value * factor
