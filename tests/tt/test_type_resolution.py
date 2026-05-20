"""Tests for the build-time DAG type-resolution sweep.

The central guard is `test_agg_rule_table_agrees_with_overload_stacks`: it
asserts the hand-written `AGG_RULE_TABLE` reproduces, entry for entry, the
`@overload` stacks on the `grouped_*` primitives in `ttsim.tt.aggregation`.
If a primitive's overloads ever change without the table being updated, the
test hard-fails, so the table cannot silently rot.
"""

import typing

import pytest

from ttsim.exceptions import TTSIMError
from ttsim.tt.aggregation import (
    AggType,
    grouped_all,
    grouped_any,
    grouped_count,
    grouped_max,
    grouped_mean,
    grouped_min,
    grouped_sum,
)
from ttsim.tt.type_resolution import (
    AGG_RULE_TABLE,
    ResolvedKind,
    TypeResolutionError,
    column_kind_to_type_string,
    resolve_agg_output_kind,
    resolve_kind_of_annotation,
    vectorized_column_kind,
)

# Map an `AggType` to the primitive whose `@overload` stack encodes the
# ground-truth input-kind -> output-kind rules.
_AGG_TYPE_TO_PRIMITIVE = {
    AggType.SUM: grouped_sum,
    AggType.MEAN: grouped_mean,
    AggType.MAX: grouped_max,
    AggType.MIN: grouped_min,
    AggType.ANY: grouped_any,
    AggType.ALL: grouped_all,
}

# Map a column-type alias name to its `ResolvedKind`.
_TYPE_STRING_TO_COLUMN_KIND = {
    "FloatColumn": ResolvedKind.FLOAT_COLUMN,
    "IntColumn": ResolvedKind.INT_COLUMN,
    "BoolColumn": ResolvedKind.BOOL_COLUMN,
}


def _overload_rules(primitive: typing.Callable[..., object]) -> dict[
    ResolvedKind, ResolvedKind
]:
    """Extract the input-kind -> output-kind mapping from a primitive's overloads."""
    rules: dict[ResolvedKind, ResolvedKind] = {}
    for overload in typing.get_overloads(primitive):
        annotations = overload.__annotations__
        input_kind = _TYPE_STRING_TO_COLUMN_KIND[annotations["column"]]
        output_kind = _TYPE_STRING_TO_COLUMN_KIND[annotations["return"]]
        rules[input_kind] = output_kind
    return rules


@pytest.mark.parametrize("agg_type", sorted(_AGG_TYPE_TO_PRIMITIVE, key=str))
def test_agg_rule_table_agrees_with_overload_stacks(agg_type: AggType) -> None:
    """`AGG_RULE_TABLE[agg_type]` reproduces the primitive's `@overload` stack."""
    expected = _overload_rules(_AGG_TYPE_TO_PRIMITIVE[agg_type])
    assert AGG_RULE_TABLE[agg_type] == expected


def test_count_primitive_returns_int_column() -> None:
    """`grouped_count` has the single return annotation `IntColumn`."""
    assert grouped_count.__annotations__["return"] == "IntColumn"


def test_resolve_agg_output_kind_count_ignores_input() -> None:
    """`COUNT` resolves to `INT_COLUMN` regardless of the source column kind."""
    resolved = resolve_agg_output_kind(
        AggType.COUNT,
        ResolvedKind.BOOL_COLUMN,
        node_name="x_hh",
    )
    assert resolved == ResolvedKind.INT_COLUMN


def test_resolve_agg_output_kind_sum_of_bool_is_int_column() -> None:
    """Summing a `BoolColumn` yields an `IntColumn` (a bool sum counts trues)."""
    resolved = resolve_agg_output_kind(
        AggType.SUM,
        ResolvedKind.BOOL_COLUMN,
        node_name="anzahl_kinder_hh",
    )
    assert resolved == ResolvedKind.INT_COLUMN


def test_resolve_agg_output_kind_mean_of_int_is_float_column() -> None:
    """Averaging an `IntColumn` yields a `FloatColumn`."""
    resolved = resolve_agg_output_kind(
        AggType.MEAN,
        ResolvedKind.INT_COLUMN,
        node_name="durchschnittsalter_hh",
    )
    assert resolved == ResolvedKind.FLOAT_COLUMN


def test_resolve_agg_output_kind_max_of_bool_raises() -> None:
    """`MAX` of a `BoolColumn` has no rule and raises `TypeResolutionError`."""
    with pytest.raises(TypeResolutionError, match="cannot be"):
        resolve_agg_output_kind(
            AggType.MAX,
            ResolvedKind.BOOL_COLUMN,
            node_name="x_hh",
        )


def test_resolve_kind_of_annotation_int_string() -> None:
    """The string annotation `"IntColumn"` resolves to `INT_COLUMN`."""
    assert (
        resolve_kind_of_annotation("IntColumn", node_name="x")
        == ResolvedKind.INT_COLUMN
    )


def test_resolve_kind_of_annotation_scalar_type_object() -> None:
    """The live type object `int` resolves to `INT_SCALAR`."""
    assert resolve_kind_of_annotation(int, node_name="x") == ResolvedKind.INT_SCALAR


def test_resolve_kind_of_annotation_unknown_is_other() -> None:
    """An annotation denoting neither a numeric column nor scalar is `OTHER`."""
    assert (
        resolve_kind_of_annotation("ConsecutiveIntLookupTableParamValue", node_name="x")
        == ResolvedKind.OTHER
    )


def test_resolve_kind_of_annotation_missing_raises() -> None:
    """A node with no return annotation cannot be resolved and raises."""
    with pytest.raises(TypeResolutionError, match="no return annotation"):
        resolve_kind_of_annotation("", node_name="x")


def test_vectorized_column_kind_promotes_scalar_to_column() -> None:
    """A scalar `FLOAT_SCALAR` return becomes a `FLOAT_COLUMN` after vectorization."""
    assert (
        vectorized_column_kind(ResolvedKind.FLOAT_SCALAR, node_name="x")
        == ResolvedKind.FLOAT_COLUMN
    )


def test_vectorized_column_kind_keeps_column_kind() -> None:
    """An already-column return kind is unchanged by vectorization."""
    assert (
        vectorized_column_kind(ResolvedKind.BOOL_COLUMN, node_name="x")
        == ResolvedKind.BOOL_COLUMN
    )


def test_vectorized_column_kind_other_raises() -> None:
    """A non-numeric `OTHER` kind has no vectorized column kind and raises."""
    with pytest.raises(TypeResolutionError, match="neither"):
        vectorized_column_kind(ResolvedKind.OTHER, node_name="x")


def test_column_kind_to_type_string_round_trips() -> None:
    """`column_kind_to_type_string` maps `INT_COLUMN` to `"IntColumn"`."""
    assert column_kind_to_type_string(ResolvedKind.INT_COLUMN) == "IntColumn"


def test_column_kind_to_type_string_scalar_raises() -> None:
    """A scalar kind has no column-type string and raises."""
    with pytest.raises(TypeResolutionError, match="column kinds"):
        column_kind_to_type_string(ResolvedKind.INT_SCALAR)


def test_type_resolution_error_is_ttsim_error() -> None:
    """`TypeResolutionError` is part of the `TTSIMError` hierarchy."""
    assert issubclass(TypeResolutionError, TTSIMError)
