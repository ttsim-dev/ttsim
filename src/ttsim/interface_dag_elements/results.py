from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import dags.tree as dt
import numpy
import pandas as pd
import pint

from ttsim.interface_dag_elements.currency import _convert_currency_value
from ttsim.interface_dag_elements.data_converters import (
    nested_data_to_df_with_mapped_columns,
    nested_data_to_df_with_nested_columns,
    nested_data_to_df_with_qname_columns,
)
from ttsim.interface_dag_elements.interface_node_objects import interface_function
from ttsim.tt.units import (
    UnitAnnotatedColumn,
    UnitSystem,
    pint_unit_with_currency,
    ttsim_unit_from_pint_unit,
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

    Currency-denominated computed columns are converted from the computation currency to
    the data currency here. Requested parameters keep their statutory values; input
    columns requested as targets are returned exactly as provided, hence already in the
    data currency.
    """
    currencies_agree = computation_currency == data_currency
    factor = (
        1.0
        if currencies_agree
        else unit_system.currency_conversion_factor(
            source_currency=computation_currency,
            target_currency=data_currency,
        )
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
                k: reorder_arrays(v)
                if currencies_agree
                else _convert_currency_value(
                    value=reorder_arrays(v),
                    factor=factor,
                    qname=k,
                    specialized_environment=(
                        specialized_environment__without_tree_logic_and_with_derived_functions
                    ),
                )
                for k, v in raw_results__columns_with_original_p_ids.items()
            },
        }
    )


@interface_function()
def tree_with_unit_annotations(
    tree: NestedResults,
    raw_results__params: QNameResults,
    unit_checks__resolved_pint_units: dict[str, pint.Unit | dict[str | int, Any]],
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

    A parameter whose value is a mapping (a ``dict`` parameter) is annotated leaf
    by leaf: its resolved unit is either one unit covering the whole structure or
    a nested mapping mirroring it, and each numeric leaf is wrapped individually.

    A leaf with no resolved unit is left bare.
    """
    registry = unit_system.registry
    resolved = unit_checks__resolved_pint_units
    tagged: dict[str, Any] = {}
    param_leaf_qnames = set(
        dt.flatten_to_qnames(dt.unflatten_from_qnames(dict(raw_results__params)))
    )
    for qname, value in dt.flatten_to_qnames(tree).items():
        if qname in param_leaf_qnames:
            continue
        unit = resolved.get(qname)
        if not isinstance(unit, pint.Unit):
            tagged[qname] = value
            continue
        # An input column returned as a target keeps the values the user handed
        # in, and a computed column crossed into the data currency at the result
        # boundary: both are labelled in the data currency.
        result_unit = pint_unit_with_currency(
            units=unit, currency=data_currency, registry=registry
        )
        label = ttsim_unit_from_pint_unit(units=result_unit, registry=registry)
        tagged[qname] = UnitAnnotatedColumn(values=value, unit=label)
    for qname, value in raw_results__params.items():
        tagged[qname] = _annotated_param(
            value=value,
            resolved_unit=resolved.get(qname),
            computation_currency=computation_currency,
            registry=registry,
        )
    return dt.unflatten_from_qnames(tagged)


def _annotated_param(
    value: Any,  # noqa: ANN401
    resolved_unit: pint.Unit | dict[str | int, Any] | None,
    computation_currency: str,
    registry: pint.UnitRegistry,
) -> Any:  # noqa: ANN401
    """A requested parameter's value with every numeric leaf unit-annotated.

    ``resolved_unit`` mirrors the value: one :class:`pint.Unit` covering the whole
    structure, or a mapping keyed like the value with a unit per leaf. A leaf the
    resolved unit does not cover — and any value the unit check leaves structured,
    such as a schedule or a lookup table — stays bare.
    """
    if isinstance(value, Mapping):
        # Heterogeneous by construction: a leaf's entry is a unit, a nested
        # mapping, or absent.
        leaf_units: Mapping[Any, Any]
        if isinstance(resolved_unit, Mapping):
            leaf_units = resolved_unit
        elif isinstance(resolved_unit, pint.Unit):
            leaf_units = dict.fromkeys(value, resolved_unit)
        else:
            return value
        return {
            key: _annotated_param(
                value=leaf,
                resolved_unit=cast(
                    "pint.Unit | dict[str | int, Any] | None", leaf_units.get(key)
                ),
                computation_currency=computation_currency,
                registry=registry,
            )
            for key, leaf in value.items()
        }
    if not isinstance(resolved_unit, pint.Unit):
        return value
    result_unit = pint_unit_with_currency(
        units=resolved_unit, currency=computation_currency, registry=registry
    )
    return UnitAnnotatedColumn(
        values=value,
        unit=ttsim_unit_from_pint_unit(units=result_unit, registry=registry),
    )


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
