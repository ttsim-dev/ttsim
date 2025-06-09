from __future__ import annotations

from typing import TYPE_CHECKING

import dags.tree as dt

from ttsim.config import numpy_or_jax as np
from ttsim.interface_dag_elements.interface_node_objects import interface_function
from ttsim.interface_dag_elements.shared import reorder_ids

if TYPE_CHECKING:
    from ttsim.interface_dag_elements.typing import FlatData, QNameData


@interface_function(in_top_level_namespace=True)
def processed_data(
    input_data__flat: FlatData,
) -> QNameData:
    """Process the data for use in the taxes and transfers function.

    This is where the conversion of p_ids will happen.

    Args:
        input_data__tree:
            The input data provided by the user.

    Returns:
        A DataFrame.
    """

    processed_input_data = {}
    old_p_ids = np.asarray(input_data__flat[("p_id",)])
    new_p_ids = reorder_ids(old_p_ids)
    for path, data in input_data__flat.items():
        qual_name = dt.qual_name_from_tree_path(path)
        if path[-1].endswith("_id"):
            processed_input_data[qual_name] = reorder_ids(np.asarray(data))
        elif path[-1].startswith("p_id_"):
            variable_with_new_ids = np.asarray(data)
            for i in range(new_p_ids.shape[0]):
                variable_with_new_ids = np.where(
                    data == old_p_ids[i], new_p_ids[i], variable_with_new_ids
                )
            processed_input_data[qual_name] = variable_with_new_ids
        else:
            processed_input_data[qual_name] = np.asarray(data)
    return processed_input_data


@interface_function()
def flat(tree: FlatData) -> FlatData:
    """The input DataFrame as a nested data structure.

    Args:
        tree:
            The input tree.

    Returns:
        Mapping of tree paths to input data.
    """
    return dt.flatten_to_tree_paths(tree)
