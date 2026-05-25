from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import dags.tree as dt
import numpy
import pandas as pd

from ttsim.interface_dag_elements.data_converters import (
    df_with_mapped_columns_to_flat_data,
    df_with_nested_columns_to_flat_data,
    df_with_qname_columns_to_flat_data,
)
from ttsim.interface_dag_elements.interface_node_objects import (
    input_dependent_interface_function,
    interface_function,
    interface_input,
)
from ttsim.interface_dag_elements.processed_data import (
    _canonicalize_input_dtype,
)

if TYPE_CHECKING:
    from types import ModuleType

    from ttsim.typing import (
        FlatData,
        IntColumn,
        NestedData,
        NestedInputsMapper,
        QNameData,
    )


@interface_input(leaf_name="df")
def df_and_mapper__df() -> pd.DataFrame:
    """A DataFrame with input data and arbitrary column names."""


@interface_input(leaf_name="mapper")
def df_and_mapper__mapper() -> NestedInputsMapper:
    """
    A dictionary mapping expected tree paths to column names in the input DataFrame.
    """


@interface_input()
def df_with_nested_columns() -> pd.DataFrame:
    """A DataFrame with nested column names corresponding to the expected tree paths."""


@interface_input()
def df_with_qname_columns() -> pd.DataFrame:
    """A DataFrame whose flat column index holds qualified-name strings."""


@interface_input()
def tree() -> NestedData:
    """The input data as a nested dictionary of arrays."""


@interface_input()
def qname() -> QNameData:
    """The input data as a flat dictionary keyed by qualified names."""


@input_dependent_interface_function(
    include_if_all_inputs_present=[
        "input_data__df_and_mapper__df",
        "input_data__df_and_mapper__mapper",
    ],
    leaf_name="flat",
)
def flat_from_df_and_mapper(
    df_and_mapper__df: pd.DataFrame,
    df_and_mapper__mapper: NestedInputsMapper,
    backend: Literal["numpy", "jax"],
    xnp: ModuleType,
) -> FlatData:
    """The input data as a flat dictionary of arrays."""
    return df_with_mapped_columns_to_flat_data(
        df=df_and_mapper__df,
        mapper=df_and_mapper__mapper,
        backend=backend,
        xnp=xnp,
    )


@input_dependent_interface_function(
    include_if_all_inputs_present=["input_data__df_with_nested_columns"],
    leaf_name="flat",
)
def flat_from_df_with_nested_columns(
    df_with_nested_columns: pd.DataFrame,
    backend: Literal["numpy", "jax"],
    xnp: ModuleType,
) -> FlatData:
    """The input data as a flat dictionary of arrays."""
    return df_with_nested_columns_to_flat_data(
        df=df_with_nested_columns,
        backend=backend,
        xnp=xnp,
    )


@input_dependent_interface_function(
    include_if_all_inputs_present=["input_data__df_with_qname_columns"],
    leaf_name="flat",
)
def flat_from_df_with_qname_columns(
    df_with_qname_columns: pd.DataFrame,
    backend: Literal["numpy", "jax"],
    xnp: ModuleType,
) -> FlatData:
    """The input data as a flat dictionary of arrays."""
    return df_with_qname_columns_to_flat_data(
        df=df_with_qname_columns,
        backend=backend,
        xnp=xnp,
    )


@input_dependent_interface_function(
    include_if_all_inputs_present=["input_data__tree"],
    leaf_name="flat",
)
def flat_from_tree(
    tree: NestedData,
    xnp: ModuleType,  # noqa: ARG001
) -> FlatData:
    """The input data as a flat dictionary of arrays."""
    # Broadcast scalar leaves to length-`n_obs` arrays so the tree input
    # path produces the same shape as the df-based paths. Users who want
    # scalars partialled into derived consumers must opt in by supplying
    # their data via `InputData.flat`, which bypasses this conversion. Any
    # `pd.Series` leaves go through `_canonicalize_input_dtype` so
    # nullable / pyarrow dtypes are normalised. The canonicaliser and the
    # broadcast use `numpy` rather than `xnp` so the user's dtype survives
    # this stage intact; the backend conversion happens once in
    # `processed_data._canonicalize_input_dtype` so the JAX int32 downcast
    # under `jax_enable_x64=False` doesn't fire here and shrink the
    # downstream pandas-index dtype.
    flat = dt.flatten_to_tree_paths(tree)
    p_id = flat.get(("p_id",))
    if p_id is None or not hasattr(p_id, "__len__"):
        return {
            path: (
                _canonicalize_input_dtype(
                    value, numpy, column_label=dt.qname_from_tree_path(path)
                )
                if isinstance(value, pd.Series)
                else value
            )
            for path, value in flat.items()
        }
    n_obs = len(p_id)
    out: FlatData = {}
    for path, value in flat.items():
        if not hasattr(value, "__len__"):
            out[path] = numpy.full(n_obs, value)
        elif isinstance(value, pd.Series):
            out[path] = _canonicalize_input_dtype(
                value, numpy, column_label=dt.qname_from_tree_path(path)
            )
        else:
            out[path] = value
    return out


@input_dependent_interface_function(
    include_if_all_inputs_present=["input_data__qname"],
    leaf_name="flat",
)
def flat_from_qname(
    qname: QNameData,
    xnp: ModuleType,  # noqa: ARG001
) -> FlatData:
    """The input data as a flat dictionary of arrays."""
    # Mirror `flat_from_tree`'s scalar-broadcast handling so the two input
    # paths produce shape-compatible outputs. `pd.Series` leaves go through
    # `_canonicalize_input_dtype` directly so nullable / pyarrow dtypes are
    # normalised; backend arrays and plain Python sequences pass through
    # `numpy.asarray`. Scalar leaves are broadcast to length-`n_obs` via
    # `numpy.full` when `p_id` is present, matching the behaviour of
    # `flat_from_tree`. Using `numpy` (not `xnp`) at this stage preserves
    # the user's dtype; backend conversion happens later in
    # `processed_data`.
    flat = {dt.tree_path_from_qname(q): value for q, value in qname.items()}
    p_id = flat.get(("p_id",))
    n_obs = len(p_id) if p_id is not None and hasattr(p_id, "__len__") else None
    out: FlatData = {}
    for path, value in flat.items():
        label = dt.qname_from_tree_path(path)
        if isinstance(value, pd.Series):
            out[path] = _canonicalize_input_dtype(value, numpy, column_label=label)
        elif not hasattr(value, "__len__"):
            out[path] = numpy.full(n_obs, value) if n_obs is not None else value
        else:
            out[path] = _canonicalize_input_dtype(
                numpy.asarray(value), numpy, column_label=label
            )
    return out


@interface_function()
def sort_indices(input_data__flat: FlatData, xnp: ModuleType) -> IntColumn:
    """Sort indices for restoring the original row order."""
    return xnp.argsort(xnp.asarray(input_data__flat[("p_id",)]))
