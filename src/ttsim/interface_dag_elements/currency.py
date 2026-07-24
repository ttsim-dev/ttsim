from __future__ import annotations

import datetime

from ttsim.interface_dag_elements.interface_node_objects import interface_function
from ttsim.tt.currencies import UnitSystem


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
