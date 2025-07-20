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

    orig_p_ids = xnp.asarray(input_data__flat[("p_id",)])
    internal_p_ids = reorder_ids(ids=orig_p_ids, xnp=xnp)
    sort_indices = xnp.argsort(orig_p_ids)
    sorted_orig_ids = orig_p_ids[sort_indices]
    sorted_internal_ids = internal_p_ids[sort_indices]

    processed_input_data = {"p_id": internal_p_ids}
    for path, data in input_data__flat.items():
        qname = dt.qname_from_tree_path(path)
        if path == ("p_id",):
            continue
        if path[-1].endswith("_id"):
            processed_input_data[qname] = reorder_ids(ids=xnp.asarray(data), xnp=xnp)
        elif path[-1].startswith("p_id_"):
            data_array = xnp.asarray(data)
            insert_positions = xnp.searchsorted(sorted_orig_ids, data_array)
            variable_with_new_ids = xnp.where(
                sorted_orig_ids[insert_positions] == data_array,
                sorted_internal_ids[insert_positions],
                data_array,
            )
            processed_input_data[qname] = variable_with_new_ids
        else:
            processed_input_data[qname] = xnp.asarray(data)
    return processed_input_data
