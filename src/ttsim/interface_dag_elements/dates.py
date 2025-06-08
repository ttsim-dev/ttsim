from __future__ import annotations

from typing import TYPE_CHECKING

from ttsim.interface_dag_elements.interface_node_objects import (
    interface_function,
    interface_input,
)
from ttsim.interface_dag_elements.shared import to_datetime

if TYPE_CHECKING:
    import datetime

    from ttsim.interface_dag_elements.typing import DashedISOString


@interface_input(in_top_level_namespace=True)
def date_str() -> DashedISOString:
    """The date to be used as policy date and evaluation date."""


@interface_function(in_top_level_namespace=True)
def date(date_str: DashedISOString) -> datetime.date:
    """The date to be used as policy date and evaluation date."""
    return to_datetime(date_str)
