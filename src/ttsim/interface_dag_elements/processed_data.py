from __future__ import annotations

from typing import TYPE_CHECKING

import dags.tree as dt

from ttsim.interface_dag_elements.interface_node_objects import interface_function
from ttsim.tt.column_objects_param_function import reorder_ids

if TYPE_CHECKING:
    from types import ModuleType

    from ttsim.typing import FlatData, IntColumn, QNameData


@interface_function(in_top_level_namespace=True)
def processed_data(
    input_data__flat: FlatData, input_data__sort_indices: IntColumn, xnp: ModuleType
) -> QNameData:
    """Process the data for use in the taxes and transfers function.

    Replace id's by consecutive integers starting at zero.
    The Jax-based backend will work correctly only with these transformed indices.
    They will be transformed back when converting raw results to results.

    As an optimization, user data is sorted by original p_id.

    Args:
        input_data__flat:
            The input data provided by the user.
        input_data__sort_indices:
            Sort indices used for restoring original order of user data.
        xnp:
            The backend module (numpy or jax).

    Returns
    -------
        A processed data dictionary.
    """

    orig_p_ids = xnp.asarray(input_data__flat[("p_id",)])
    sorted_orig_p_ids = orig_p_ids[input_data__sort_indices]
    internal_p_ids = xnp.arange(len(orig_p_ids))

    processed_input_data = {"p_id": internal_p_ids}
    for path, data in input_data__flat.items():
        qname = dt.qname_from_tree_path(path)
        if path == ("p_id",):
            continue

        sorted_data_array = xnp.asarray(data[input_data__sort_indices])

        if path[-1].endswith("_id"):
            processed_input_data[qname] = reorder_ids(ids=sorted_data_array, xnp=xnp)
        elif path[-1].startswith("p_id_"):
            # Second line makes sure out-of-bounds ids don't raise an error. Any garbage
            # that is actually used will be checked inside
            # fail_if.foreign_keys_are_invalid_in_data, so don't worry here.
            insert_positions = xnp.minimum(
                xnp.searchsorted(sorted_orig_p_ids, sorted_data_array),
                len(sorted_orig_p_ids) - 1,
            )
            variable_with_new_ids = xnp.where(
                sorted_orig_p_ids[insert_positions] == sorted_data_array,
                internal_p_ids[insert_positions],
                sorted_data_array,
            )
            processed_input_data[qname] = variable_with_new_ids
        else:
            processed_input_data[qname] = sorted_data_array

    return processed_input_data
