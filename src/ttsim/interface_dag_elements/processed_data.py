from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING

import dags.tree as dt
import numpy as np
import pandas as pd
from jaxtyping import Shaped

from ttsim.interface_dag_elements.interface_node_objects import interface_function
from ttsim.tt.column_objects_param_function import reorder_ids
from ttsim.typing import Array, IntColumn

if TYPE_CHECKING:
    from ttsim.typing import FlatData, QNameData


def _canonicalize_input_dtype(
    arr: Shaped[Array | np.ndarray, " n_obs"] | pd.Series,
    xnp: ModuleType,
) -> Shaped[Array | np.ndarray, " n_obs"]:
    """Canonicalize a column to a backend-native dtype the TT DAG can operate on.

    Handles three families of pandas extension dtypes uniformly with plain
    numpy / JAX arrays:

    - **Float** (numpy float, pandas-nullable ``Float64``, ``float[pyarrow]``)
      → ``float64`` with ``pd.NA`` mapped to ``NaN``.
    - **Unsigned integer** (numpy uint, ``UInt*``, ``uint*[pyarrow]``) →
      ``int64`` so signed arithmetic on them does not underflow into a huge
      positive value.
    - **Signed integer / Bool** (nullable / pyarrow variants) → numpy
      ``int64`` / ``bool_`` when the column has no missing values; left as
      an object-dtype array with ``pd.NA`` in place otherwise, for the
      ``input_data_has_int_or_bool_missing_values`` fail-if to surface.
    """
    if isinstance(arr, pd.Series):
        return _canonicalize_series(arr, xnp)
    if pd.api.types.is_unsigned_integer_dtype(arr):
        return xnp.asarray(arr, dtype=xnp.int64)
    return xnp.asarray(arr)


def _canonicalize_series(
    arr: pd.Series,
    xnp: ModuleType,
) -> Shaped[Array | np.ndarray, " n_obs"]:
    """Series-only branch of `_canonicalize_input_dtype`."""
    dtype = arr.dtype
    if pd.api.types.is_float_dtype(dtype):
        return xnp.asarray(arr.astype("float64"))
    if pd.api.types.is_bool_dtype(dtype):
        if arr.isna().any():
            return arr.to_numpy(dtype=object)
        return xnp.asarray(arr.astype("bool"))
    if pd.api.types.is_integer_dtype(dtype):
        if arr.isna().any():
            return arr.to_numpy(dtype=object)
        return xnp.asarray(arr.astype("int64"))
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

    orig_p_ids = _canonicalize_input_dtype(input_data__flat[("p_id",)], xnp)
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

        sorted_data = _canonicalize_input_dtype(data[input_data__sort_indices], xnp)

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
