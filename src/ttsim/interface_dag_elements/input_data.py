from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import dags.tree as dt
import numpy
import pandas as pd
import pint

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
from ttsim.tt.currencies import UnitSystem
from ttsim.tt.units import (
    CompositeUnit,
    input_strip_unit,
    resolve_compositional_unit,
    strip_input_quantity_at_boundary,
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


@interface_input()
def tree_with_unit_annotations() -> NestedData:
    """The input data as a nested dict of :class:`UnitAnnotatedColumn` leaves.

    Like :func:`tree`, but each leaf tags its column with a concrete unit, opting
    into boundary unit validation against the DAG (currency converted, tag checked
    against the declared unit). Use bare :func:`tree` for untagged data.

    Example::

        {
            "wage_m": UnitAnnotatedColumn(values, unit=TTSIMUnit.EUR.PER_MONTH),
            "rent_m_bg": UnitAnnotatedColumn(
                values, unit=TTSIMUnit.EUR.PER_MONTH.PER_BG
            ),
            "p_id": UnitAnnotatedColumn(values, unit=TTSIMUnit.DIMENSIONLESS),
        }
    """


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
    # `pd.Series` leaves go through `_canonicalize_input_dtype` so
    # nullable / pyarrow dtypes are normalised to numpy.
    return {
        path: (
            _canonicalize_input_dtype(
                arr=value, xnp=numpy, column_label=dt.qname_from_tree_path(path)
            )
            if isinstance(value, pd.Series)
            else value
        )
        for path, value in dt.flatten_to_tree_paths(tree).items()
    }


@input_dependent_interface_function(
    include_if_all_inputs_present=["input_data__qname"],
    leaf_name="flat",
)
def flat_from_qname(
    qname: QNameData,
    xnp: ModuleType,  # noqa: ARG001
) -> FlatData:
    """The input data as a flat dictionary of arrays."""
    # `pd.Series` leaves go through `_canonicalize_input_dtype` directly so
    # nullable / pyarrow dtypes are normalised; plain Python lists /
    # sequences first become numpy arrays so the canonicaliser sees the
    # already-narrow input type its claw enforces. Backend arrays
    # (`numpy.ndarray`, JAX `Array`) pass through `numpy.asarray` as well,
    # which is a no-op for numpy arrays and pulls JAX arrays back to numpy
    # so the canonicaliser sees one uniform input type.
    return {
        dt.tree_path_from_qname(q): _canonicalize_input_dtype(
            arr=value if isinstance(value, pd.Series) else numpy.asarray(value),
            xnp=numpy,
        )
        for q, value in qname.items()
    }


@input_dependent_interface_function(
    include_if_all_inputs_present=["input_data__tree_with_unit_annotations"],
    leaf_name="flat",
)
def flat_from_tree_with_unit_annotations(
    tree_with_unit_annotations: NestedData,
    data_currency: str,
    unit_system: UnitSystem,
) -> FlatData:
    """The input data as a flat dictionary of arrays."""
    registry = unit_system.registry
    flat = dt.flatten_to_tree_paths(tree_with_unit_annotations)
    return {
        path: strip_input_quantity_at_boundary(
            quantity=registry.Quantity(
                col.values, input_strip_unit(unit=col.unit, registry=registry)
            ),
            data_currency=data_currency,
            registry=registry,
            column_label=dt.qname_from_tree_path(path),
        )
        for path, col in flat.items()
    }


@input_dependent_interface_function(
    include_if_all_inputs_present=["input_data__tree_with_unit_annotations"],
    leaf_name="units",
)
def units_from_tree_with_unit_annotations(
    tree_with_unit_annotations: NestedData,
    unit_system: UnitSystem,
) -> dict[str, pint.Unit]:
    """Each input column's resolved (agnostic) tag, with its grouping level, by qname.

    Resolved off every :class:`UnitAnnotatedColumn`'s tag so
    ``fail_if__input_units_are_inconsistent`` can compare it against the column's
    declared unit on all three axes — currency presence, grouping level, and the
    residual measurement.
    """
    flat = dt.flatten_to_tree_paths(tree_with_unit_annotations)
    return {
        dt.qname_from_tree_path(path): resolve_compositional_unit(
            unit=col.unit, registry=unit_system.registry, with_level=True
        )
        for path, col in flat.items()
    }


@input_dependent_interface_function(
    include_if_all_inputs_present=["input_data__tree_with_unit_annotations"],
    leaf_name="unit_tokens",
)
def unit_tokens_from_tree_with_unit_annotations(
    tree_with_unit_annotations: NestedData,
) -> dict[str, CompositeUnit]:
    """Each input column's *declared tag* as a compositional token, by qname.

    The pre-resolution :class:`CompositeUnit` the user tagged the column with. The
    input check compares its grouping level against the declared unit's token, so
    a per-person head-count tag (``PERSON_COUNT_PER_PERSON``) is not conflated with
    a plain ``DIMENSIONLESS`` — the two resolve to the same dimensionless pint unit
    (GEP 10).
    """
    flat = dt.flatten_to_tree_paths(tree_with_unit_annotations)
    return {dt.qname_from_tree_path(path): col.unit for path, col in flat.items()}


@interface_function()
def sort_indices(input_data__flat: FlatData, xnp: ModuleType) -> IntColumn:
    """Sort indices for restoring the original row order."""
    return xnp.argsort(xnp.asarray(input_data__flat[("p_id",)]))
