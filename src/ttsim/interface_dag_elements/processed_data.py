from __future__ import annotations

from typing import TYPE_CHECKING

import dags.tree as dt

from ttsim.interface_dag_elements.interface_node_objects import interface_function
from ttsim.tt_dag_elements.column_objects_param_function import reorder_ids

if TYPE_CHECKING:
    from types import ModuleType

    from ttsim.interface_dag_elements.typing import FlatData, QNameData


@interface_function(in_top_level_namespace=True)
def processed_data(input_data__flat: FlatData, xnp: ModuleType) -> QNameData:
    """Process the data for use in the taxes and transfers function.

    Replace id's by consecutive integers starting at zero.
    The Jax-based backend will work correctly only with these transformed indices.
    They will be transformed back when converting raw results to results.

    Args:
        input_data__tree:
            The input data provided by the user.

    Returns
    -------
        A DataFrame.
    """
    processed_input_data = {}
    old_p_ids = xnp.asarray(input_data__flat[("p_id",)])
    new_p_ids = reorder_ids(ids=old_p_ids, xnp=xnp)
    for path, data in input_data__flat.items():
        qname = dt.qname_from_tree_path(path)
        if path[-1].endswith("_id"):
            processed_input_data[qname] = reorder_ids(ids=xnp.asarray(data), xnp=xnp)
        elif path[-1].startswith("p_id_"):
            variable_with_new_ids = xnp.asarray(data)
            for i in range(new_p_ids.shape[0]):
                variable_with_new_ids = xnp.where(
                    data == old_p_ids[i],
                    new_p_ids[i],
                    variable_with_new_ids,
                )
            processed_input_data[qname] = variable_with_new_ids
        else:
            processed_input_data[qname] = xnp.asarray(data)
    return processed_input_data
