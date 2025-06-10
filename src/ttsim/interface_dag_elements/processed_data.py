from __future__ import annotations

from typing import TYPE_CHECKING

import dags.tree as dt

from ttsim.interface_dag_elements.interface_node_objects import interface_function

if TYPE_CHECKING:
    from types import ModuleType

    from ttsim.interface_dag_elements.typing import NestedData, QNameData


@interface_function(in_top_level_namespace=True)
def processed_data(
    input_data__tree: NestedData,
    xnp: ModuleType,
) -> QNameData:
    """Process the data for use in the taxes and transfers function.

    This is where the conversion of p_ids will happen.

    Args:
        input_data__tree:
            The input data provided by the user.

    Returns:
        A DataFrame.
    """
    return {
        k: xnp.asarray(v) for k, v in dt.flatten_to_qual_names(input_data__tree).items()
    }


def process_input_data(
    input_data__tree: dict,
    xnp: ModuleType,
) -> dict:
    """Process input data."""
    return {
        k: xnp.asarray(v) for k, v in dt.flatten_to_qual_names(input_data__tree).items()
    }
