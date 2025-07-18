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
def policy_date_str() -> DashedISOString:
    """The date to be used as policy date in YYYY-MM-DD format.

    Will also be used as evaluation date in case that is neither provided as an argument
    to `main` nor as part of the input data / parameters.
    """


@interface_input(in_top_level_namespace=True)
def evaluation_date_str() -> DashedISOString:
    """The date to be used as evaluation date in YYYY-MM-DD format.

    Will be overridden by values for year / month / day found in the input data or
    parameters.
    """


@interface_function(in_top_level_namespace=True)
def policy_date(policy_date_str: DashedISOString) -> datetime.date:
    """The date to be used as policy date.

    Will also be used as evaluation date in case that is neither provided as an argument
    to `main` nor as part of the input data / parameters.
    """
    return to_datetime(policy_date_str)


@interface_function(in_top_level_namespace=True)
def evaluation_date(evaluation_date_str: DashedISOString) -> datetime.date:
    """The date to be used as evaluation date in YYYY-MM-DD format.

    Will be overridden by values for year / month / day found in the input data or
    parameters.
    """
    return to_datetime(evaluation_date_str)
