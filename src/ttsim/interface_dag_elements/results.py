from __future__ import annotations

from typing import Any

import dags.tree as dt
import numpy
import pandas as pd
import pint

from ttsim.interface_dag_elements.data_converters import (
    nested_data_to_df_with_mapped_columns,
    nested_data_to_df_with_nested_columns,
    nested_data_to_df_with_qname_columns,
)
from ttsim.interface_dag_elements.interface_node_objects import interface_function
from ttsim.tt.units import (
    UnitAnnotatedColumn,
    composite_from_resolved_unit,
    output_unit_in_run_currency,
)
from ttsim.typing import (
    FlatData,
    FlatResults,
    IntColumn,
    NestedResults,
    NestedStrings,
    QNameData,
    QNameResults,
)


@interface_function()
def tree(
    raw_results__columns_with_original_p_ids: QNameData,
    raw_results__params: QNameResults,
    raw_results__from_input_data: QNameData,
    input_data__sort_indices: IntColumn,
) -> NestedResults:
    """The combined results as a tree with original row order restored."""
    restore_order = numpy.empty(len(input_data__sort_indices), dtype=int)
    restore_order[input_data__sort_indices] = numpy.arange(
        len(input_data__sort_indices)
    )

    def reorder_arrays(v: Any) -> Any:  # noqa: ANN401
        return v[restore_order] if hasattr(v, "shape") and v.ndim > 0 else v

    return dt.unflatten_from_qnames(
        {
            **raw_results__params,
            **raw_results__from_input_data,
            **{
                k: reorder_arrays(v)
                for k, v in raw_results__columns_with_original_p_ids.items()
            },
        }
    )


@interface_function()
def tree_with_unit_annotations(
    tree: NestedResults,
    unit_checks__resolved_units: dict[str, pint.Unit | dict[str | int, Any]],
    currency: str | None,
) -> NestedResults:
    """The combined results as a tree of :class:`UnitAnnotatedColumn` leaves.

    Like :func:`tree`, but every leaf is wrapped in a ``UnitAnnotatedColumn`` —
    the same shape as the unit-annotated input tree (GEP 10) — whose ``unit`` is
    the node's resolved unit restated in the concrete run currency
    (``Unit.EUR.PER_MONTH``, never the agnostic ``CURRENCY``). A node with no
    resolved unit is left bare.
    """
    resolved = unit_checks__resolved_units
    tagged: dict[str, Any] = {}
    for qname, value in dt.flatten_to_qnames(tree).items():
        unit = resolved.get(qname)
        tagged[qname] = (
            UnitAnnotatedColumn(
                values=value,
                unit=composite_from_resolved_unit(
                    output_unit_in_run_currency(units=unit, run_currency=currency)
                ),
            )
            if isinstance(unit, pint.Unit)
            else value
        )
    return dt.unflatten_from_qnames(tagged)


@interface_function()
def df_with_mapper(
    tree: NestedResults,
    input_data__flat: FlatData,
    tt_targets__tree: NestedStrings,
) -> pd.DataFrame:
    """The results DataFrame with mapped column names."""
    return nested_data_to_df_with_mapped_columns(
        nested_data_to_convert=tree,
        nested_outputs_df_column_names=tt_targets__tree,
        data_with_p_id=input_data__flat,
    )


@interface_function()
def df_with_nested_columns(
    tree: NestedResults,
    input_data__flat: FlatData,
) -> pd.DataFrame:
    """The results DataFrame with nested column names corresponding to tree paths."""
    return nested_data_to_df_with_nested_columns(
        nested_data_to_convert=tree,
        index=pd.Index(input_data__flat[("p_id",)], name="p_id"),
    )


@interface_function()
def df_with_qname_columns(
    tree: NestedResults,
    input_data__flat: FlatData,
) -> pd.DataFrame:
    """Results DataFrame with qname-string columns (one flat string per column)."""
    return nested_data_to_df_with_qname_columns(
        nested_data_to_convert=tree,
        index=pd.Index(input_data__flat[("p_id",)], name="p_id"),
    )


@interface_function()
def flat(tree: NestedResults) -> FlatResults:
    """Results as a flat mapping of tree-path tuples to result leaves."""
    return dt.flatten_to_tree_paths(tree)


@interface_function()
def qname(tree: NestedResults) -> QNameResults:
    """Results as a flat mapping of qualified-name strings to result leaves."""
    return dt.flatten_to_qnames(tree)
