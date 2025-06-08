from __future__ import annotations

from typing import TYPE_CHECKING

import dags.tree as dt

from ttsim.config import numpy_or_jax as np
from ttsim.interface_dag_elements.interface_node_objects import interface_function

if TYPE_CHECKING:
    from ttsim.interface_dag_elements.typing import NestedData, QNameData


@interface_function(in_top_level_namespace=True)
def processed_data(
    input_data__tree: NestedData,
) -> QNameData:
    """Process the data for use in the taxes and transfers function.

    This is where the conversion of p_ids will happen.

    Args:
        input_data__tree:
            The input data provided by the user.

    Returns:
        A DataFrame.
    """
    flattened_tree = dt.flatten_to_qual_names(input_data__tree)
    processed_input_data = {}
    old_p_ids = flattened_tree["p_id"]
    new_p_ids = _reorder_ids(np.asarray(flattened_tree["p_id"]))
    for k, v in flattened_tree.items():
        if k.endswith("_id"):
            processed_input_data[k] = _reorder_ids(np.asarray(v))
        elif "__p_id_" in k:
            variable_with_new_ids = np.asarray(v)
            for i in range(new_p_ids.shape[0]):
                variable_with_new_ids = np.where(
                    v == old_p_ids[i], new_p_ids[i], variable_with_new_ids
                )
            processed_input_data[k] = variable_with_new_ids
        else:
            processed_input_data[k] = np.asarray(v)
    return processed_input_data


def _reorder_ids(ids: np.ndarray) -> np.ndarray:
    """Make ID's consecutively numbered."""
    sorting = np.argsort(ids)
    ids_sorted = ids[sorting]
    index_after_sort = np.arange(ids.shape[0])[sorting]
    # Look for difference from previous entry in sorted array
    diff_to_prev = np.where(np.diff(ids_sorted) >= 1, 1, 0)
    # Sum up all differences to get new id
    cons_ids = np.concatenate((np.asarray([0]), np.cumsum(diff_to_prev)))
    return cons_ids[np.argsort(index_after_sort)]
