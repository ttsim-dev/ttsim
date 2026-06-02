"""Tests for the build-time DAG type-resolution sweep.

The central guard is `test_agg_rule_table_agrees_with_overload_stacks`: it
asserts the hand-written `AGG_RULE_TABLE` reproduces, entry for entry, the
`@overload` stacks on the `grouped_*` primitives in `ttsim.tt.aggregation`.
If a primitive's overloads ever change without the table being updated, the
test hard-fails, so the table cannot silently rot.
"""

import os
import typing

import numpy as np
import pytest
from dags import get_annotations

from ttsim.exceptions import TTSIMError
from ttsim.interface_dag_elements.automatically_added_functions import (
    create_agg_by_group_functions,
)
from ttsim.tt import ColumnFunction, policy_function
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
    build_beartype_checkable_wrapper,
    column_kind_to_type_string,
    resolve_agg_output_kind,
    resolve_kind_of_annotation,
    vectorized_column_kind,
)
from ttsim.typing import FloatColumn, IntColumn

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


def _column_kind_of_overload_annotation(annotation: object) -> ResolvedKind:
    """Map an `@overload` column annotation to its `ResolvedKind`.

    The annotation is either the alias-name string (`"IntColumn"`, under
    `from __future__ import annotations`) or the live `jaxtyping` type
    object (when the beartype claw has resolved it). The `jaxtyping`
    object's `repr` names the dtype (`Float`/`Int`/`Bool`), so a substring
    probe classifies both forms.
    """
    text = annotation if isinstance(annotation, str) else repr(annotation)
    if "Float" in text:
        return ResolvedKind.FLOAT_COLUMN
    if "Int" in text:
        return ResolvedKind.INT_COLUMN
    if "Bool" in text:
        return ResolvedKind.BOOL_COLUMN
    msg = f"Unrecognized overload column annotation: {annotation!r}"
    raise AssertionError(msg)


def _overload_rules(
    primitive: typing.Callable[..., object],
) -> dict[ResolvedKind, ResolvedKind]:
    """Extract the input-kind -> output-kind mapping from a primitive's overloads."""
    rules: dict[ResolvedKind, ResolvedKind] = {}
    for overload in typing.get_overloads(primitive):
        annotations = overload.__annotations__
        input_kind = _column_kind_of_overload_annotation(annotations["column"])
        output_kind = _column_kind_of_overload_annotation(annotations["return"])
        rules[input_kind] = output_kind
    return rules


@pytest.mark.parametrize("agg_type", sorted(_AGG_TYPE_TO_PRIMITIVE, key=str))
def test_agg_rule_table_agrees_with_overload_stacks(agg_type: AggType) -> None:
    """`AGG_RULE_TABLE[agg_type]` reproduces the primitive's `@overload` stack."""
    expected = _overload_rules(_AGG_TYPE_TO_PRIMITIVE[agg_type])
    assert AGG_RULE_TABLE[agg_type] == expected


def test_count_primitive_returns_int_column() -> None:
    """`grouped_count`'s return annotation resolves to `INT_COLUMN`.

    The annotation appears either as the string `"IntColumn"` (under
    `from __future__ import annotations`) or as the live `jaxtyping`
    type object (when the beartype claw has resolved it); both must
    resolve to `INT_COLUMN`, which is what `AGG_RULE_TABLE` encodes for
    `COUNT`.
    """
    annotation = grouped_count.__annotations__["return"]
    if isinstance(annotation, str):
        assert annotation == "IntColumn"
    else:
        # Claw-resolved live type object; its repr names the alias kind.
        assert "Int" in repr(annotation)


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


def test_resolve_kind_of_annotation_live_jaxtyping_type() -> None:
    """A claw-resolved live `jaxtyping` column type resolves to its column kind."""
    from ttsim.typing import IntColumn  # noqa: PLC0415

    assert (
        resolve_kind_of_annotation(IntColumn, node_name="x") == ResolvedKind.INT_COLUMN
    )


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


def _auto_agg_wrapper_from_int_source() -> typing.Callable[..., object]:
    """Build the synthesized `x_hh` aggregation wrapper for an int source `x`."""

    @policy_function()
    def x(p_id: int) -> int:
        return p_id

    column_functions: dict[str, ColumnFunction] = {
        "x": x.remove_tree_logic(
            tree_path=("x",),
            top_level_namespace={"x", "p_id"},
        ),
    }
    derived = create_agg_by_group_functions(
        column_functions=column_functions,
        qname_policy_environment={},
        input_columns=set(),
        tt_targets=("x_hh",),
        grouping_levels=("hh",),
    )
    wrapper = derived["x_hh"]
    assert isinstance(wrapper, ColumnFunction)
    return wrapper.function


def test_auto_agg_wrapper_carries_concrete_return_annotation() -> None:
    """The synthesized `x_hh` wrapper for an int source declares `-> IntColumn`.

    Without the build-time resolution sweep the wrapper would advertise the
    imprecise `FloatColumn | IntColumn` union that `grouped_sum`'s runtime
    implementation signature carries.
    """
    annotations = get_annotations(_auto_agg_wrapper_from_int_source())
    assert annotations["return"] == "IntColumn"


@pytest.mark.skipif(
    os.environ.get("TTSIM_BEARTYPE_CLAW", "1") == "0",
    reason="Requires the beartype claw to be active.",
)
def test_auto_agg_wrapper_rejects_misused_source_column() -> None:
    """Calling the synthesized wrapper with a non-column source raises a claw violation.

    The wrapper forwards to the claw-decorated `grouped_sum`, so a misused
    source column surfaces as a `beartype` hint violation.
    """
    from beartype.roar import BeartypeCallHintViolation  # noqa: PLC0415

    wrapper = _auto_agg_wrapper_from_int_source()
    group_ids = np.array([0, 0, 1])
    with pytest.raises(BeartypeCallHintViolation):
        wrapper(
            x="not a column",
            hh_id=group_ids,
            num_segments=2,
            backend="numpy",
        )


def test_typed_wrapper_recognizes_live_column_type_objects() -> None:
    """`build_beartype_checkable_wrapper` treats live `jaxtyping` column types
    as numeric and installs the wide-numeric beartype check at the boundary.

    A user function declared with `vectorization_strategy="not_required"` and
    without `from __future__ import annotations` reaches the wrapper builder
    with live `jaxtyping` objects on its `__signature__`. The classifier must
    recognise those as numeric so the wide-union beartype check fires on
    structural misuse (a string argument) at the user boundary.
    """
    from beartype.roar import BeartypeCallHintViolation  # noqa: PLC0415

    def underlying(x: FloatColumn) -> FloatColumn:
        return x

    wrapper = build_beartype_checkable_wrapper(
        underlying,
        annotations={"x": FloatColumn, "return": FloatColumn},
        node_name="underlying",
    )

    with pytest.raises(BeartypeCallHintViolation):
        wrapper(x="not a column")


def test_typed_wrapper_rejects_non_identifier_node_name() -> None:
    """The forwarder name is interpolated into source compiled by `exec`, so
    `node_name` must be a Python identifier — qualified names (`pkg.mod`) or
    other punctuation are rejected.
    """

    def underlying(x: FloatColumn) -> FloatColumn:
        return x

    with pytest.raises(ValueError, match="Python identifier"):
        build_beartype_checkable_wrapper(
            underlying,
            annotations={"x": "FloatColumn", "return": "FloatColumn"},
            node_name="pkg.mod.underlying",
        )


def test_typed_wrapper_skips_non_numeric_param_annotation() -> None:
    """An `OTHER`-classified parameter annotation (a partialled lookup table,
    a string config, …) is not run through the wide-numeric beartype check —
    only structural numeric parameters are.
    """

    def underlying(x: IntColumn, lookup: dict) -> IntColumn:  # noqa: ARG001
        return x

    wrapper = build_beartype_checkable_wrapper(
        underlying,
        annotations={"x": "IntColumn", "lookup": "dict", "return": "IntColumn"},
        node_name="underlying",
    )

    # A non-dict `lookup` would raise if beartype were enforcing the `"dict"`
    # annotation; it does not, because `_is_numeric_annotation("dict")` is
    # `False` and the wrapper only installs checks for numeric parameters.
    result = wrapper(x=np.array([1, 2, 3]), lookup="not a dict")
    np.testing.assert_array_equal(result, np.array([1, 2, 3]))
