from __future__ import annotations

import datetime
from typing import Literal

from ttsim.interface_dag_elements.interface_node_objects import interface_function
from ttsim.tt.currencies import base_currency, statutory_currency


@interface_function(leaf_name="data_currency", in_top_level_namespace=True)
def data_currency(
    backend: Literal["numpy", "jax"],  # noqa: ARG001
) -> str:
    """The currency the user's data arrives in and results are returned in.

    Defaults to the registered base currency (a downstream package registers it
    on import, so users need not pass ``data_currency=`` themselves). Override
    via ``main(data_currency=...)`` with another registered currency. Only the
    column boundary is affected: input columns are converted from this currency
    into the computation currency on the way in, and currency-denominated
    results are converted back on the way out. The computation itself runs in
    the policy date's statutory currency (GEP 10).
    """
    return base_currency()


@interface_function(leaf_name="computation_currency", in_top_level_namespace=True)
def computation_currency(policy_date: datetime.date) -> str:
    """The currency the computation runs in — the policy date's statutory currency.

    Read off the dated mapping a downstream package registers on import
    (``register_statutory_currencies``); not a user knob. Parameters keep their
    statutory values in this currency, and every parameter must be declared in
    it. User data is converted to and from it at the column boundary only
    (GEP 10).
    """
    return statutory_currency(policy_date)
