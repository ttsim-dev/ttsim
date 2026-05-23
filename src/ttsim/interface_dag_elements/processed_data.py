from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING

import dags.tree as dt
import numpy as np
import pandas as pd
from jaxtyping import Shaped

from ttsim.interface_dag_elements.interface_node_objects import interface_function
from ttsim.tt.column_objects_param_function import reorder_ids
from ttsim.typing import Array, BoolColumn, FloatColumn, IntColumn

if TYPE_CHECKING:
    from ttsim.typing import FlatData, QNameData


def _coerce_uint_to_int64(
    arr: Shaped[Array | np.ndarray, " n_obs"] | pd.Series,
    xnp: ModuleType,
) -> IntColumn | FloatColumn | BoolColumn:
    """Coerce unsigned-integer arrays to ``int64``; pass others through unchanged.

    Without this, signed arithmetic on uint columns (e.g. ``gross - deductions``)
    silently underflows into a huge positive value.
    """
    if pd.api.types.is_unsigned_integer_dtype(arr):
        return xnp.asarray(arr, dtype=xnp.int64)
    return xnp.asarray(arr)


@interface_function(in_top_level_namespace=True)
def processed_data(
    input_data__flat: FlatData, input_data__sort_indices: IntColumn, xnp: ModuleType
) -> QNameData:
    """The internal processed data for use in the taxes and transfers function.

    We replace identifiers by consecutive integers starting at zero and sort the data
    according to the original `p_id`.

    The transformations will be undone when going from raw results to results.
    """

    orig_p_ids = _coerce_uint_to_int64(input_data__flat[("p_id",)], xnp)
    sorted_orig_p_ids = orig_p_ids[input_data__sort_indices]
    internal_p_ids = xnp.arange(len(orig_p_ids))

    processed_input_data = {"p_id": internal_p_ids}
    for path, data in input_data__flat.items():
        qname = dt.qname_from_tree_path(path)
        if path == ("p_id",):
            continue
        if not hasattr(data, "__len__"):
            # Scalars don't need to be sorted.
            processed_input_data[qname] = data
            continue

        sorted_data = _coerce_uint_to_int64(data[input_data__sort_indices], xnp)

        if path[-1].endswith("_id"):
            processed_input_data[qname] = reorder_ids(ids=sorted_data, xnp=xnp)
        elif path[-1].startswith("p_id_"):
            # Second line makes sure out-of-bounds ids don't raise an error. Any garbage
            # that is actually used will be checked inside
            # fail_if.foreign_keys_are_invalid_in_data, so don't worry here.
            insert_positions = xnp.minimum(
                xnp.searchsorted(sorted_orig_p_ids, sorted_data),
                len(sorted_orig_p_ids) - 1,
            )
            variable_with_new_ids = xnp.where(
                sorted_orig_p_ids[insert_positions] == sorted_data,
                internal_p_ids[insert_positions],
                sorted_data,
            )
            processed_input_data[qname] = variable_with_new_ids
        else:
            processed_input_data[qname] = sorted_data

    return processed_input_data
