from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING

import dags.tree as dt
import numpy as np
import pandas as pd
import pint
from jaxtyping import Shaped

from ttsim.interface_dag_elements.interface_node_objects import interface_function
from ttsim.tt.column_objects_param_function import reorder_ids
from ttsim.tt.units import strip_input_quantity_at_boundary
from ttsim.typing import Array, IntColumn

if TYPE_CHECKING:
    from ttsim.typing import FlatData, QNameData


def _canonicalize_input_dtype(
    arr: Shaped[Array | np.ndarray, " n_obs"] | pd.Series | pint.Quantity,
    xnp: ModuleType,
    *,
    column_label: str | None = None,
    run_currency: str | None = None,
) -> Shaped[Array | np.ndarray, " n_obs"]:
    """Canonicalize a column to a backend-native dtype the TT DAG can operate on.

    Handles three families of pandas extension dtypes uniformly with plain
    numpy / JAX arrays:

    - **Float** (numpy float, pandas-nullable ``Float64``, ``float[pyarrow]``)
      → ``float64`` with ``pd.NA`` mapped to ``NaN``.
    - **Unsigned integer** (numpy uint, ``UInt*``, ``uint*[pyarrow]``) →
      ``int64`` so signed arithmetic on them does not underflow into a huge
      positive value. Values above ``int64.max`` raise a ``ValueError``
      naming the offending column.
    - **Signed integer / Bool** (nullable / pyarrow variants) → numpy
      ``int64`` / ``bool_`` when the column has no missing values; left as
      an object-dtype array with ``pd.NA`` in place otherwise, for the
      ``input_data_has_int_or_bool_missing_values`` fail-if to surface.

    Args:
        arr: The column data, as a numpy / JAX array or a pandas Series.
        xnp: Backend numpy module.
        column_label: Qualified-name or other identifier for the column;
            used in error messages when a uint overflow is detected.
    """
    if isinstance(arr, pint.Quantity):
        arr = strip_input_quantity_at_boundary(
            arr, run_currency=run_currency, column_label=column_label
        )
    if isinstance(arr, pd.Series):
        return _canonicalize_series(arr=arr, xnp=xnp, column_label=column_label)
    if pd.api.types.is_unsigned_integer_dtype(arr):
        _fail_if_uint_overflows_int64(arr, column_label=column_label)
        return xnp.asarray(arr, dtype=xnp.int64)
    return xnp.asarray(arr)


def _canonicalize_series(
    arr: pd.Series,
    xnp: ModuleType,
    *,
    column_label: str | None = None,
) -> Shaped[Array | np.ndarray, " n_obs"]:
    """Series-only branch of `_canonicalize_input_dtype`."""
    dtype = arr.dtype
    if pd.api.types.is_float_dtype(dtype):
        return xnp.asarray(arr.astype("float64"))
    if pd.api.types.is_bool_dtype(dtype):
        if arr.isna().any():
            return arr.to_numpy(dtype=object)
        return xnp.asarray(arr.astype("bool"))
    # `is_integer_dtype` matches both signed and unsigned nullable / pyarrow
    # variants, so `UInt*` / ``uint*[pyarrow]`` Series take the same int64
    # cast as their signed counterparts.
    if pd.api.types.is_integer_dtype(dtype):
        if arr.isna().any():
            return arr.to_numpy(dtype=object)
        if pd.api.types.is_unsigned_integer_dtype(dtype):
            _fail_if_uint_overflows_int64(arr, column_label=column_label)
        return xnp.asarray(arr.astype("int64"))
    return xnp.asarray(arr)


def _fail_if_uint_overflows_int64(
    arr: pd.Series | Shaped[Array | np.ndarray, " n_obs"],
    *,
    column_label: str | None,
) -> None:
    """Raise if any value in an unsigned-integer column exceeds ``int64.max``."""
    int64_max = np.iinfo(np.int64).max
    if isinstance(arr, pd.Series):
        over_mask = arr > int64_max
        if not over_mask.any():
            return
        first_over = int(arr[over_mask].iloc[0])
    else:
        np_arr = np.asarray(arr)
        over = np_arr[np_arr > int64_max]
        if over.size == 0:
            return
        first_over = int(over[0])
    label = f" '{column_label}'" if column_label else ""
    msg = (
        f"Unsigned integer input column{label} contains value {first_over} > "
        f"int64 max ({int64_max}); cannot coerce to int64 safely."
    )
    raise ValueError(msg)


@interface_function(in_top_level_namespace=True)
def processed_data(
    input_data__flat: FlatData,
    input_data__sort_indices: IntColumn,
    xnp: ModuleType,
    currency: str | None,
) -> QNameData:
    """The internal processed data for use in the taxes and transfers function.

    We replace identifiers by consecutive integers starting at zero and sort the data
    according to the original `p_id`.

    The transformations will be undone when going from raw results to results.
    """

    orig_p_ids = _canonicalize_input_dtype(
        arr=input_data__flat[("p_id",)],
        xnp=xnp,
        column_label="p_id",
        run_currency=currency,
    )
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

        sorted_data = _canonicalize_input_dtype(
            arr=data[input_data__sort_indices],
            xnp=xnp,
            column_label=qname,
            run_currency=currency,
        )

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
