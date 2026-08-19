"""Narrow semantic evidence needed by GEP 10's grouping-level safeguards.

These labels are internal checker metadata, not additional public units. They only
distinguish explicit head counts and yes/no indicators from other quantities where
the restricted group arithmetic needs that fact.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any, TypeAlias, cast

from ttsim.interface_dag_elements.shared import FRAMEWORK_PARTIAL_ARGUMENTS
from ttsim.tt.aggregation import AggType
from ttsim.tt.column_objects_param_function import (
    AggByGroupFunction,
    ColumnObject,
    ParamFunction,
    PolicyInput,
    qname_is_person_pointer,
)
from ttsim.tt.param_objects import ParamObject
from ttsim.tt.type_resolution import (
    BOOL_KINDS,
    ResolvedKind,
    TypeResolutionError,
    resolve_kind_of_annotation,
    resolve_kind_of_column_function,
)
from ttsim.tt.units import CompositeUnit, QuantityKind

QuantityKindTree: TypeAlias = QuantityKind | Mapping[str | int, Any]


def quantity_kind(
    qname: str,
    obj: Any,  # noqa: ANN401
    env: Mapping[str, Any],
) -> QuantityKind:
    """Return the count/indicator evidence independently known for one node."""
    if isinstance(obj, AggByGroupFunction):
        if obj.agg_type is AggType.COUNT:
            return QuantityKind.COUNT
        if obj.agg_type in (AggType.ANY, AggType.ALL):
            return QuantityKind.INDICATOR
        if obj.agg_type is AggType.SUM and _aggregation_source_is_boolean(obj, env):
            return QuantityKind.COUNT
    declaration = getattr(obj, "unit", None)
    if isinstance(declaration, CompositeUnit) and (
        declaration.kind is not QuantityKind.GENERIC
    ):
        return declaration.kind
    kind = _resolved_kind(qname=qname, obj=obj)
    if kind in BOOL_KINDS or _parameter_values_are_booleans(obj):
        return QuantityKind.INDICATOR
    return QuantityKind.GENERIC


def quantity_kind_for_leaf(
    declaration: CompositeUnit,
    value: Any,  # noqa: ANN401
) -> QuantityKind:
    """Return evidence for one parameter leaf, without consulting its siblings."""
    if declaration.kind is not QuantityKind.GENERIC:
        return declaration.kind
    if isinstance(value, bool):
        return QuantityKind.INDICATOR
    return QuantityKind.GENERIC


def quantity_kind_for_scalar_type(
    declaration: CompositeUnit,
    scalar_type: Any,  # noqa: ANN401
) -> QuantityKind:
    """Return evidence for one annotated scalar field."""
    if declaration.kind is not QuantityKind.GENERIC:
        return declaration.kind
    if scalar_type is bool:
        return QuantityKind.INDICATOR
    return QuantityKind.GENERIC


def quantity_kinds_by_qname(
    env: Mapping[str, Any],
) -> dict[str, QuantityKindTree]:
    """Collect the narrow evidence for every environment node."""
    return {
        qname: (
            _quantity_kind_tree(
                declaration=cast("Mapping[str | int, Any]", obj.unit),
                value=getattr(obj, "value", None),
            )
            if isinstance(getattr(obj, "unit", None), Mapping)
            else quantity_kind(qname=qname, obj=obj, env=env)
        )
        for qname, obj in env.items()
    }


def _quantity_kind_tree(
    declaration: Mapping[str | int, Any] | CompositeUnit,
    value: Any,  # noqa: ANN401
) -> QuantityKindTree:
    """Mirror a mapping declaration while retaining each leaf semantic."""
    if isinstance(declaration, Mapping):
        values = value if isinstance(value, Mapping) else {}
        return {
            key: _quantity_kind_tree(
                declaration=token,
                value=values.get(key),
            )
            for key, token in declaration.items()
        }
    return quantity_kind_for_leaf(declaration=declaration, value=value)


def _aggregation_source_is_boolean(
    obj: AggByGroupFunction, env: Mapping[str, Any]
) -> bool:
    sources = [
        name
        for name in inspect.signature(obj.function).parameters
        if not name.endswith("_id")
        and not qname_is_person_pointer(name)
        and name not in FRAMEWORK_PARTIAL_ARGUMENTS
    ]
    if len(sources) != 1 or sources[0] not in env:
        return False
    return _resolved_kind(qname=sources[0], obj=env[sources[0]]) in BOOL_KINDS


def _resolved_kind(qname: str, obj: Any) -> ResolvedKind | None:  # noqa: ANN401
    try:
        if isinstance(obj, PolicyInput):
            return resolve_kind_of_annotation(annotation=obj.data_type, node_name=qname)
        if isinstance(obj, ColumnObject | ParamFunction):
            return resolve_kind_of_column_function(func=obj.function, node_name=qname)
    except TypeResolutionError:
        pass
    return None


def _parameter_values_are_booleans(obj: Any) -> bool:  # noqa: ANN401
    return isinstance(obj, ParamObject) and _all_leaves(
        getattr(obj, "value", None), predicate=lambda value: isinstance(value, bool)
    )


def _all_leaves(value: Any, predicate: Any) -> bool:  # noqa: ANN401
    if isinstance(value, Mapping):
        return bool(value) and all(
            _all_leaves(part, predicate) for part in value.values()
        )
    return predicate(value)
