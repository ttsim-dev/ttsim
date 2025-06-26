from __future__ import annotations

from typing import TYPE_CHECKING

from ttsim.interface_dag_elements.interface_node_objects import (
    input_dependent_interface_function,
    interface_function,
    interface_input,
)
from ttsim.interface_dag_elements.shared import to_datetime

if TYPE_CHECKING:
    import datetime

    from ttsim.interface_dag_elements.typing import DashedISOString


@interface_input(in_top_level_namespace=True)
def policy_date_str() -> DashedISOString:
    """The date to be used as policy date."""


@interface_input(in_top_level_namespace=True)
def evaluation_date_str() -> DashedISOString:
    """The date to be used as evaluation date."""


@interface_input(in_top_level_namespace=True)
def date_str() -> DashedISOString:
    """The date to be used as policy date and evaluation date."""


@interface_function(in_top_level_namespace=True)
def date(date_str: DashedISOString) -> datetime.date:
    """The date to be used as policy date and evaluation date."""
    return to_datetime(date_str)


@input_dependent_interface_function(
    include_if_all_inputs_present=["policy_date_str", "evaluation_date_str"],
    leaf_name="policy_date",
    in_top_level_namespace=True,
)
def policy_date_from_policy_date_str(policy_date_str: DashedISOString) -> datetime.date:
    """The date to be used as policy date."""
    return to_datetime(policy_date_str)


@input_dependent_interface_function(
    include_if_any_input_present=["date", "date_str"],
    leaf_name="policy_date",
    in_top_level_namespace=True,
)
def policy_date_from_date(date: datetime.date) -> datetime.date:
    """The date to be used as policy date."""
    return date


@input_dependent_interface_function(
    include_if_all_inputs_present=["policy_date_str", "evaluation_date_str"],
    leaf_name="evaluation_date",
    in_top_level_namespace=True,
)
def evaluation_date_from_evaluation_date_str(
    evaluation_date_str: DashedISOString,
) -> datetime.date:
    """The date to be used as evaluation date."""
    return to_datetime(evaluation_date_str)


@input_dependent_interface_function(
    include_if_any_input_present=["date", "date_str"],
    leaf_name="evaluation_date",
    in_top_level_namespace=True,
)
def evaluation_date_from_date(date: datetime.date) -> datetime.date:
    """The date to be used as evaluation date."""
    return date
