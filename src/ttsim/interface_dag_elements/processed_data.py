from __future__ import annotations

from collections.abc import Iterable
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

import dags.tree as dt
import numpy as np
import pandas as pd
import pint
from jaxtyping import Shaped

from ttsim.exceptions import UnitDefinitionError
from ttsim.interface_dag_elements.interface_node_objects import interface_function
from ttsim.interface_dag_elements.shared import (
    get_re_pattern_for_all_time_units_and_groupings,
)
from ttsim.tt.column_objects_param_function import reorder_ids
from ttsim.tt.currencies import currency_conversion_factor
from ttsim.tt.units import (
    UNSET_UNIT,
    CompositeUnit,
    strip_input_quantity_at_boundary,
    token_is_agnostic_currency,
    token_source_currency,
)
from ttsim.typing import Array, IntColumn, OrderedQNames, PolicyEnvironment
from ttsim.unit_converters import TIME_UNIT_IDS_TO_LABELS

if TYPE_CHECKING:
    from ttsim.typing import FlatData, QNameData


def _canonicalize_input_dtype(
    arr: Shaped[Array | np.ndarray, " n_obs"] | pd.Series | pint.Quantity,
    xnp: ModuleType,
    *,
    column_label: str | None = None,
    data_currency: str | None = None,
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
        # always has a concrete data currency; the currency-less converter callers
        # (`data_converters`, plain `pd.Series` leaves) never pass a `Quantity`.
        arr = strip_input_quantity_at_boundary(
            quantity=arr,
            data_currency=cast("str", data_currency),
            column_label=column_label,
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


def qnames_with_currency_declarations(
    qnames: Iterable[str],
    policy_environment: PolicyEnvironment,
    grouping_levels: OrderedQNames,
) -> set[str]:
    """The subset of ``qnames`` whose declared unit carries a currency component.

    Identifies the columns the boundary converts (GEP 10). A qname's own
    declaration in the policy environment wins. A derived name — an
    auto-aggregation or time-conversion target provided as data or requested as
    a target — inherits the currency presence of the declared nodes it can be
    derived from (same namespace and base name; the grouping suffix stripped or
    kept; a time suffix only ever rebased, never added or removed), because
    those derivations never add or remove the currency component. Derivation
    sources that disagree on currency presence are a loud error rather than a
    guess. A qname with no declaration and no derivation source is not
    converted (the mandatory-units check reports missing declarations).

    A *parameter's* concrete statutory currency counts, too: a data column
    overriding a parameter is still user data, arriving in the data currency.
    Parameter *values* are exempt from conversion not here, but by never
    passing through this boundary — requested parameters flow through
    ``raw_results__params``.
    """
    pattern = get_re_pattern_for_all_time_units_and_groupings(
        time_units=tuple(TIME_UNIT_IDS_TO_LABELS),
        grouping_levels=grouping_levels,
    )

    def parse(qname: str) -> tuple[str, str, str | None, str | None]:
        path = dt.tree_path_from_qname(qname)
        match = pattern.fullmatch(path[-1])
        namespace = dt.qname_from_tree_path(path[:-1]) if len(path) > 1 else ""
        if match is None:
            return (namespace, path[-1], None, None)
        return (
            namespace,
            match.group("base_name"),
            match.group("time_unit"),
            match.group("grouping"),
        )

    declared: dict[str, bool] = {}
    declared_variants: dict[tuple[str, str], list[tuple[str | None, str | None, bool]]]
    declared_variants = {}
    for env_qname, obj in dt.flatten_to_qnames(policy_environment).items():
        token = getattr(obj, "unit", UNSET_UNIT)
        if token is UNSET_UNIT or not isinstance(token, CompositeUnit):
            continue
        has_currency = (
            token_is_agnostic_currency(token)
            or token_source_currency(token) is not None
        )
        declared[env_qname] = has_currency
        namespace, base_name, time_unit, grouping = parse(env_qname)
        declared_variants.setdefault((namespace, base_name), []).append(
            (time_unit, grouping, has_currency)
        )

    out: set[str] = set()
    for qname in qnames:
        if qname in declared:
            if declared[qname]:
                out.add(qname)
        elif _inherited_currency_presence(
            qname=qname, parsed=parse(qname), declared_variants=declared_variants
        ):
            out.add(qname)
    return out


def _inherited_currency_presence(
    qname: str,
    parsed: tuple[str, str, str | None, str | None],
    declared_variants: dict[tuple[str, str], list[tuple[str | None, str | None, bool]]],
) -> bool:
    """Whether a derived qname inherits a currency from its derivation sources.

    Raises:
        UnitDefinitionError: If the possible sources disagree on carrying a
            currency.
    """
    namespace, base_name, time_unit, grouping = parsed
    sources = [
        source_has_currency
        for source_time_unit, source_grouping, source_has_currency in (
            declared_variants.get((namespace, base_name), [])
        )
        if source_grouping in (grouping, None)
        and (time_unit is None) == (source_time_unit is None)
    ]
    if not sources:
        return False
    if all(sources):
        return True
    if any(sources):
        raise UnitDefinitionError(
            f"Cannot decide whether {qname!r} is currency-denominated: the "
            f"declared nodes it could be derived from disagree on carrying "
            f"a currency. Rename the non-currency sibling so the base "
            f"names differ (GEP 10)."
        )
    return False


def boundary_currency_conversion(
    qnames: Iterable[str],
    policy_environment: PolicyEnvironment,
    grouping_levels: OrderedQNames,
    source_currency: str,
    target_currency: str,
) -> tuple[float, set[str]]:
    """The conversion factor and the columns it applies to, for one boundary.

    The shared setup of both boundary crossings (GEP 10): ``processed_data``
    converts inputs from the data currency to the computation currency,
    ``results`` converts computed columns back. Equal currencies short-circuit
    to a factor of ``1.0`` with no environment walk.
    """
    if source_currency == target_currency:
        return 1.0, set()
    factor = currency_conversion_factor(
        source_currency=source_currency, target_currency=target_currency
    )
    currency_qnames = qnames_with_currency_declarations(
        qnames=qnames,
        policy_environment=policy_environment,
        grouping_levels=grouping_levels,
    )
    return factor, currency_qnames


def value_in_target_currency(
    value: Any,  # noqa: ANN401 (a column array or an input scalar)
    qname: str,
    currency_qnames: set[str],
    factor: float,
) -> Any:  # noqa: ANN401
    """Convert one value across the column boundary (GEP 10).

    Multiplies a currency-denominated value by the conversion factor; leaves
    everything else — including an object-dtype column (int/bool data with
    missing values, reported by its own fail-if node) — untouched.
    """
    if factor == 1.0 or qname not in currency_qnames:
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
    policy_environment: PolicyEnvironment,
    labels__grouping_levels: OrderedQNames,
    data_currency: str,
    computation_currency: str,
) -> QNameData:
    """The internal processed data for use in the taxes and transfers function.

    We replace identifiers by consecutive integers starting at zero and sort the data
    according to the original `p_id`. Currency-denominated inputs are converted
    from the data currency to the computation currency (GEP 10).

    The transformations will be undone when going from raw results to results.
    """
    factor, currency_qnames = boundary_currency_conversion(
        qnames=[
            dt.qname_from_tree_path(path)
            for path in input_data__flat
            if path != ("p_id",)
        ],
        policy_environment=policy_environment,
        grouping_levels=labels__grouping_levels,
        source_currency=data_currency,
        target_currency=computation_currency,
    )

    orig_p_ids = _canonicalize_input_dtype(
        arr=input_data__flat[("p_id",)],
        xnp=xnp,
        column_label="p_id",
        data_currency=data_currency,
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
