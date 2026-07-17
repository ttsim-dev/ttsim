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
from ttsim.interface_dag_elements.processed_data import (
    currency_conversion_factor_and_columns,
    value_in_target_currency,
)
from ttsim.tt.currencies import UnitSystem
from ttsim.tt.units import (
    UnitAnnotatedColumn,
    composite_from_resolved_unit,
    input_target_unit_in_data_currency,
    output_unit_in_data_currency,
    param_unit_in_computation_currency,
)
from ttsim.typing import (
    FlatData,
    FlatResults,
    IntColumn,
    NestedResults,
    NestedStrings,
    QNameData,
    QNameResults,
    SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
)


@interface_function()
def tree(
    raw_results__columns_with_original_p_ids: QNameData,
    raw_results__params: QNameResults,
    raw_results__from_input_data: QNameData,
    input_data__sort_indices: IntColumn,
    specialized_environment__without_tree_logic_and_with_derived_functions: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,  # noqa: E501
    data_currency: str,
    computation_currency: str,
    unit_system: UnitSystem,
) -> NestedResults:
    """The combined results as a tree with original row order restored.

    Currency-denominated computed columns are converted from the computation
    currency to the data currency here (GEP 10). Requested parameters keep
    their statutory values; input columns requested as targets are returned
    exactly as provided, hence already in the data currency.
    """
    factor, currency_qnames = currency_conversion_factor_and_columns(
        qnames=raw_results__columns_with_original_p_ids,
        specialized_environment=(
            specialized_environment__without_tree_logic_and_with_derived_functions
        ),
        source_currency=computation_currency,
        target_currency=data_currency,
        unit_system=unit_system,
    )

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
                k: value_in_target_currency(
                    value=reorder_arrays(v),
                    qname=k,
                    currency_qnames=currency_qnames,
                    factor=factor,
                )
                for k, v in raw_results__columns_with_original_p_ids.items()
            },
        }
    )


@interface_function()
def tree_with_unit_annotations(
    tree: NestedResults,
    raw_results__from_input_data: QNameData,
    raw_results__params: QNameResults,
    unit_checks__resolved_units: dict[str, pint.Unit | dict[str | int, Any]],
    data_currency: str,
    computation_currency: str,
    unit_system: UnitSystem,
) -> NestedResults:
    """The combined results as a tree of :class:`UnitAnnotatedColumn` leaves.

    Like :func:`tree`, but every leaf is wrapped in a ``UnitAnnotatedColumn``
    whose unit is spelled in the currency its value is denominated in:

    - an input column returned as a target and a computed column carry the data
      currency (the value crossed to it at the boundary);
    - a requested parameter keeps its statutory value, so it carries the
      computation currency and is never relabelled to the data currency (GEP 10).

    A leaf with no resolved unit is left bare.
    """
    registry = unit_system.registry
    resolved = unit_checks__resolved_units
    tagged: dict[str, Any] = {}
    for qname, value in dt.flatten_to_qnames(tree).items():
        unit = resolved.get(qname)
        if not isinstance(unit, pint.Unit):
            tagged[qname] = value
            continue
        if qname in raw_results__from_input_data:
            result_unit = input_target_unit_in_data_currency(
                units=unit, data_currency=data_currency, registry=registry
            )
        elif qname in raw_results__params:
            result_unit = param_unit_in_computation_currency(
                units=unit, computation_currency=computation_currency, registry=registry
            )
        else:
            result_unit = output_unit_in_data_currency(
                units=unit, data_currency=data_currency, registry=registry
            )
        tagged[qname] = UnitAnnotatedColumn(
            values=value,
            unit=composite_from_resolved_unit(units=result_unit, registry=registry),
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
