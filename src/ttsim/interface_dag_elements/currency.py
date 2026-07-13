from __future__ import annotations

import datetime
from typing import Literal

from ttsim.interface_dag_elements.interface_node_objects import interface_function
from ttsim.tt.currencies import base_currency, statutory_currency_for_date


@interface_function(leaf_name="data_currency", in_top_level_namespace=True)
def data_currency(
    backend: Literal["numpy", "jax"],  # noqa: ARG001
) -> str:
    """The currency the user's data arrives in and results are returned in.

    Defaults to the registered base currency. Override via ``main(data_currency=...)``
    with another registered currency. Only input and output data are affected: input
    columns are converted from this currency into the computation currency, and
    currency-denominated results are converted back.
    """
    return base_currency()


@interface_function(leaf_name="computation_currency", in_top_level_namespace=True)
def computation_currency(policy_date: datetime.date) -> str:
    """The currency the computation runs in — the policy date's statutory currency.

    Read off the dated mapping a downstream package registers on import
    (``register_statutory_currencies``).
    """
    return statutory_currency_for_date(policy_date)
