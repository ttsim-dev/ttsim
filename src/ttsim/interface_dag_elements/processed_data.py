from __future__ import annotations

from collections.abc import Iterable
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

import dags.tree as dt
import numpy as np
import pandas as pd
import pint
from jaxtyping import Shaped

from ttsim.interface_dag_elements.interface_node_objects import interface_function
from ttsim.tt.column_objects_param_function import reorder_ids
from ttsim.tt.currencies import UnitSystem
from ttsim.tt.units import (
    UNSET_UNIT,
    CompositeUnit,
    strip_input_quantity_at_boundary,
    token_declares_a_currency,
)
from ttsim.typing import (
    Array,
    IntColumn,
    SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
)

if TYPE_CHECKING:
    from ttsim.typing import FlatData, QNameData


def _canonicalize_input_dtype(
    arr: Shaped[Array | np.ndarray, " n_obs"] | pd.Series | pint.Quantity,
    xnp: ModuleType,
    *,
    column_label: str | None = None,
    data_currency: str | None = None,
    registry: pint.UnitRegistry | None = None,
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
        # A pint-tagged column only ever reaches here via `processed_data`, which
        # always has a concrete data currency and registry; the currency-less
        # converter callers (`data_converters`, plain `pd.Series` leaves) never
        # pass a `Quantity`.
        arr = strip_input_quantity_at_boundary(
            quantity=arr,
            data_currency=cast("str", data_currency),
            registry=cast("pint.UnitRegistry", registry),
            column_label=column_label,
        )
    if isinstance(arr, pd.Series):
        return _canonicalize_series(arr=arr, xnp=xnp, column_label=column_label)
    if pd.api.types.is_unsigned_integer_dtype(arr):
        _fail_if_uint_overflows_int64(arr=arr, column_label=column_label)
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
            _fail_if_uint_overflows_int64(arr=arr, column_label=column_label)
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


def qnames_with_currency_declarations(
    qnames: Iterable[str],
    specialized_environment: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
) -> set[str]:
    """The subset of ``qnames`` whose declared unit carries a currency component.

    Identifies the input values to convert between the data currency and the
    computation currency (GEP 10). Every value carries its unit in the
    specialized environment — its own declaration, the minted unit of a
    derived function, or a `PolicyInput` stub for data supplied at a derived
    name — so this is a plain lookup. A *parameter's* concrete statutory
    currency counts, too: a data column overriding a parameter is still user
    data, arriving in the data currency. A qname with no unit is not
    converted (the mandatory-units check reports missing declarations).
    """
    out: set[str] = set()
    for qname in qnames:
        token = getattr(specialized_environment.get(qname), "unit", UNSET_UNIT)
        if not isinstance(token, CompositeUnit):
            continue
        if token_declares_a_currency(token):
            out.add(qname)
    return out


def currency_conversion_factor_and_columns(
    qnames: Iterable[str],
    specialized_environment: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
    source_currency: str,
    target_currency: str,
    unit_system: UnitSystem,
) -> tuple[float, set[str]]:
    """The conversion factor and the input or result values it applies to.

    The shared setup of the two conversions (GEP 10): ``processed_data``
    converts input columns and scalar values from the data currency to the
    computation currency,
    ``results`` converts computed columns back. Equal currencies short-circuit
    to a factor of ``1.0`` with no environment walk.
    """
    if source_currency == target_currency:
        return 1.0, set()
    factor = unit_system.currency_conversion_factor(
        source_currency=source_currency, target_currency=target_currency
    )
    currency_qnames = qnames_with_currency_declarations(
        qnames=qnames,
        specialized_environment=specialized_environment,
    )
    return factor, currency_qnames


def value_in_target_currency(
    value: Any,  # noqa: ANN401 (a column array or an input scalar)
    qname: str,
    currency_qnames: set[str],
    factor: float,
) -> Any:  # noqa: ANN401
    """Convert one input or result value into the target currency (GEP 10).

    Multiplies a currency-denominated value by the conversion factor; leaves
    everything else — including an object-dtype column (int/bool data with
    missing values, reported by its own fail-if node) — untouched.
    """
    if qname not in currency_qnames:
        return value
    dtype = getattr(value, "dtype", None)
    if dtype is not None and dtype.kind == "O":
        return value
    return value * factor


@interface_function(in_top_level_namespace=True)
def processed_data(
    input_data__flat: FlatData,
    input_data__sort_indices: IntColumn,
    xnp: ModuleType,
    specialized_environment__without_tree_logic_and_with_derived_functions: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,  # noqa: E501
    data_currency: str,
    computation_currency: str,
    unit_system: UnitSystem,
) -> QNameData:
    """The internal processed data for use in the taxes and transfers function.

    We replace identifiers by consecutive integers starting at zero and sort the data
    according to the original `p_id`. Currency-denominated inputs are converted
    from the data currency to the computation currency (GEP 10).

    The transformations will be undone when going from raw results to results.
    """
    factor, currency_qnames = currency_conversion_factor_and_columns(
        qnames=[
            dt.qname_from_tree_path(path)
            for path in input_data__flat
            if path != ("p_id",)
        ],
        specialized_environment=(
            specialized_environment__without_tree_logic_and_with_derived_functions
        ),
        source_currency=data_currency,
        target_currency=computation_currency,
        unit_system=unit_system,
    )

    orig_p_ids = _canonicalize_input_dtype(
        arr=input_data__flat[("p_id",)],
        xnp=xnp,
        column_label="p_id",
        data_currency=data_currency,
        registry=unit_system.registry,
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
            processed_input_data[qname] = value_in_target_currency(
                value=data,
                qname=qname,
                currency_qnames=currency_qnames,
                factor=factor,
            )
            continue

        sorted_data = _canonicalize_input_dtype(
            arr=data[input_data__sort_indices],
            xnp=xnp,
            column_label=qname,
            data_currency=data_currency,
            registry=unit_system.registry,
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
            processed_input_data[qname] = value_in_target_currency(
                value=sorted_data,
                qname=qname,
                currency_qnames=currency_qnames,
                factor=factor,
            )

    return processed_input_data
