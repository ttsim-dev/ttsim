"""Narrow semantic evidence needed by GEP 10's grouping-level safeguards.

These labels are internal checker metadata, not additional public units. They only
distinguish explicit head counts and yes/no indicators from other quantities where
the restricted group arithmetic needs that fact.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Mapping
from typing import Any

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
from ttsim.tt.units import QuantityKind

_INTEGER_KINDS = frozenset({ResolvedKind.INT_SCALAR, ResolvedKind.INT_COLUMN})

# A direct integer only qualifies when its documentation actually says it is a
# count. English and German terms cover the policy packages maintained here; the
# integer annotation by itself is deliberately insufficient.
_COUNT_MEANING = re.compile(
    r"(?:\bnumber of\b|\bmaximum number\b|\bcount\b|\bhead count\b|"
    r"\bhousehold size\b|\bfamily size\b|\bgroup size\b|\banzahl\b|"
    r"\bhaushaltsgröße\b|\bzahl der\b|\bkinderzahl\b|"
    r"\bmaximal(?:e|en|er|es)?\s+anzahl\b|\bmaximalzahl\b)",
    flags=re.IGNORECASE,
)

_NOT_A_COUNT = re.compile(
    r"(?:\bnot (?:a )?(?:head )?count\b|\bkein(?:e|en|er|es)?\s+anzahl\b|"
    r"\bcategory\b|\bclassification\b|\bidentifier\b|\brent class\b|"
    r"\bmietstufe\b)",
    flags=re.IGNORECASE,
)


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
    kind = _resolved_kind(qname=qname, obj=obj)
    if kind in BOOL_KINDS or _parameter_values_are_booleans(obj):
        return QuantityKind.INDICATOR
    if (
        kind in _INTEGER_KINDS or _parameter_values_are_integers(obj)
    ) and _documents_count(qname=qname, obj=obj):
        return QuantityKind.COUNT
    return QuantityKind.GENERIC


def quantity_kind_for_leaf(
    qname: str,
    value: Any,  # noqa: ANN401
) -> QuantityKind:
    """Return evidence for one parameter leaf, without consulting its siblings."""
    if isinstance(value, bool):
        return QuantityKind.INDICATOR
    if (
        isinstance(value, int)
        and not isinstance(value, bool)
        and _text_documents_count(qname.replace("_", " "))
    ):
        return QuantityKind.COUNT
    return QuantityKind.GENERIC


def quantity_kind_for_scalar_type(
    qname: str,
    scalar_type: Any,  # noqa: ANN401
) -> QuantityKind:
    """Return evidence for one annotated scalar field."""
    if scalar_type is bool:
        return QuantityKind.INDICATOR
    if scalar_type is int and _text_documents_count(qname.replace("_", " ")):
        return QuantityKind.COUNT
    return QuantityKind.GENERIC


def documented_quantity_kind(qname: str, obj: Any) -> QuantityKind:  # noqa: ANN401
    """Return count evidence stated in this declaration's own documentation."""
    return (
        QuantityKind.COUNT
        if _documents_count(qname=qname, obj=obj)
        else QuantityKind.GENERIC
    )


def quantity_kinds_by_qname(
    env: Mapping[str, Any],
) -> dict[str, QuantityKind]:
    """Collect the narrow evidence for every environment node."""
    return {
        qname: quantity_kind(qname=qname, obj=obj, env=env)
        for qname, obj in env.items()
    }


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


def _documents_count(qname: str, obj: Any) -> bool:  # noqa: ANN401
    pieces = [qname.replace("_", " ")]
    for attribute in ("description", "docstring", "name"):
        value = getattr(obj, attribute, None)
        if isinstance(value, str):
            pieces.append(value)
        elif isinstance(value, Mapping):
            pieces.extend(str(part) for part in value.values() if part is not None)
    return _text_documents_count(" ".join(pieces))


def _text_documents_count(text: str) -> bool:
    """Recognize affirmative count wording and let explicit contrary prose win."""
    return _NOT_A_COUNT.search(text) is None and _COUNT_MEANING.search(text) is not None


def _parameter_values_are_booleans(obj: Any) -> bool:  # noqa: ANN401
    return isinstance(obj, ParamObject) and _all_leaves(
        getattr(obj, "value", None), predicate=lambda value: isinstance(value, bool)
    )


def _parameter_values_are_integers(obj: Any) -> bool:  # noqa: ANN401
    return isinstance(obj, ParamObject) and _all_leaves(
        getattr(obj, "value", None),
        predicate=lambda value: isinstance(value, int) and not isinstance(value, bool),
    )


def _all_leaves(value: Any, predicate: Any) -> bool:  # noqa: ANN401
    if isinstance(value, Mapping):
        return bool(value) and all(
            _all_leaves(part, predicate) for part in value.values()
        )
    return predicate(value)
