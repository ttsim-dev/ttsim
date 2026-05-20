"""Build-time DAG type resolution for auto-generated nodes.

ttsim auto-generates two kinds of DAG nodes: aggregation functions (`x_hh`
from `x`) and auto-vectorized wrappers of scalar policy functions. Both are
built by wrapping a primitive whose runtime signature carries an imprecise
*union* return type (`grouped_sum` is statically `@overload`-ed per input
dtype but its runtime implementation signature widens to
`FloatColumn | IntColumn`).

`dags` renders that union onto the wrapper's `__signature__`, and its
DAG-consistency check (`set_annotations=True`) then rejects a producer typed
`FloatColumn | IntColumn` feeding a consumer parameter typed concretely
(`BoolColumn`), raising `dags.exceptions.AnnotationMismatchError`.

The honest return type of an auto-generated node *is* knowable when the DAG
is built: it follows from the dtype of the source column and the kind of
aggregation. This module performs that resolution sweep so the synthesis
sites in `automatically_added_functions` and `specialized_environment` can
stamp a concrete return annotation onto every wrapper.

The sweep is strict: a node it must resolve but cannot raises
`TypeResolutionError` rather than silently falling back to a union.
"""

from collections.abc import Callable
from enum import Enum, auto
from types import MappingProxyType

from dags import get_annotations

from ttsim.exceptions import TTSIMError
from ttsim.tt.aggregation import AggType


class TypeResolutionError(TTSIMError):
    """Raised when the build-time type-resolution sweep cannot resolve a node.

    The honest output type of an auto-generated DAG node must be derivable
    from the source column's dtype and the aggregation kind. When it is not
    — an unknown annotation string, an aggregation applied to an
    incompatible input kind — the sweep fails loudly here rather than
    emitting an imprecise union annotation that would defeat the DAG's
    type-consistency check.
    """


class ResolvedKind(Enum):
    """The concrete output kind of a DAG node, resolved at build time.

    The column kinds correspond to the canonical column aliases in
    `ttsim.typing` (`FloatColumn`, `IntColumn`, `BoolColumn`); the scalar
    kinds to the scalar Python types. `OTHER` covers nodes whose output is
    neither a numeric column nor a numeric scalar (parameter objects,
    lookup tables, …) — the sweep never needs to stamp an annotation on
    those, so it does not try to narrow them.
    """

    FLOAT_COLUMN = auto()
    INT_COLUMN = auto()
    BOOL_COLUMN = auto()
    FLOAT_SCALAR = auto()
    INT_SCALAR = auto()
    BOOL_SCALAR = auto()
    OTHER = auto()


# Sentinel for "no annotation present" — distinct from a legitimately
# `None`-valued annotation, so a missing annotation can be told apart from
# one a function genuinely declares as `None`.
_EMPTY = object()


_COLUMN_KINDS: frozenset[ResolvedKind] = frozenset(
    {
        ResolvedKind.FLOAT_COLUMN,
        ResolvedKind.INT_COLUMN,
        ResolvedKind.BOOL_COLUMN,
    },
)

# Map a `ResolvedKind` to the canonical column-type alias name. Used to
# stamp a concrete return annotation onto a synthesized wrapper via
# `dags.with_signature`.
_COLUMN_KIND_TO_TYPE_STRING = MappingProxyType(
    {
        ResolvedKind.FLOAT_COLUMN: "FloatColumn",
        ResolvedKind.INT_COLUMN: "IntColumn",
        ResolvedKind.BOOL_COLUMN: "BoolColumn",
    },
)

# Map an annotation string (as it appears on a function's `__signature__`,
# whether scalar-thinking source code or an already-vectorized wrapper) to
# the `ResolvedKind` it denotes.
_ANNOTATION_STRING_TO_KIND = MappingProxyType(
    {
        "FloatColumn": ResolvedKind.FLOAT_COLUMN,
        "IntColumn": ResolvedKind.INT_COLUMN,
        "BoolColumn": ResolvedKind.BOOL_COLUMN,
        "float": ResolvedKind.FLOAT_SCALAR,
        "int": ResolvedKind.INT_SCALAR,
        "bool": ResolvedKind.BOOL_SCALAR,
    },
)

# The beartype claw resolves a stringified column alias to a live
# `jaxtyping` type object. Such an object has no `IntColumn`-style name;
# its repr reads `Int[ndarray, 'n_obs']`. Map the leading dtype tag back to
# the column `ResolvedKind`.
_COLUMN_KIND_OF_JAXTYPING_TEXT = MappingProxyType(
    {
        "Float": ResolvedKind.FLOAT_COLUMN,
        "Int": ResolvedKind.INT_COLUMN,
        "Bool": ResolvedKind.BOOL_COLUMN,
    },
)


# A scalar policy function declares scalar annotations; after
# auto-vectorization the node operates on the corresponding column. Map the
# scalar kind to the column kind it becomes once vectorized.
_SCALAR_KIND_TO_COLUMN_KIND = MappingProxyType(
    {
        ResolvedKind.FLOAT_SCALAR: ResolvedKind.FLOAT_COLUMN,
        ResolvedKind.INT_SCALAR: ResolvedKind.INT_COLUMN,
        ResolvedKind.BOOL_SCALAR: ResolvedKind.BOOL_COLUMN,
        ResolvedKind.FLOAT_COLUMN: ResolvedKind.FLOAT_COLUMN,
        ResolvedKind.INT_COLUMN: ResolvedKind.INT_COLUMN,
        ResolvedKind.BOOL_COLUMN: ResolvedKind.BOOL_COLUMN,
    },
)


def column_kind_to_type_string(kind: ResolvedKind) -> str:
    """Return the canonical column-type alias name for a column `ResolvedKind`.

    Args:
        kind: One of `FLOAT_COLUMN`, `INT_COLUMN`, `BOOL_COLUMN`.

    Returns:
        The alias name (`"FloatColumn"`, `"IntColumn"`, `"BoolColumn"`) to
        stamp onto a wrapper's `__signature__`.

    Raises:
        TypeResolutionError: If `kind` is not a column kind.
    """
    try:
        return _COLUMN_KIND_TO_TYPE_STRING[kind]
    except KeyError:
        msg = (
            f"Cannot stamp a column annotation for resolved kind {kind.name!r}: "
            f"only {sorted(k.name for k in _COLUMN_KINDS)} are column kinds."
        )
        raise TypeResolutionError(msg) from None


def resolve_kind_of_annotation(
    annotation: object,
    *,
    node_name: str,
) -> ResolvedKind:
    """Resolve the `ResolvedKind` denoted by a single type annotation.

    Annotations on ttsim functions are strings under
    `from __future__ import annotations`; live type objects appear too
    (e.g. `int`). Both forms are handled.

    Args:
        annotation: The annotation, a string or a type object.
        node_name: The qualified name of the node, used in error messages.

    Returns:
        The `ResolvedKind` for the annotation; `OTHER` for any annotation
        that denotes neither a numeric column nor a numeric scalar.

    Raises:
        TypeResolutionError: If `annotation` is empty / missing — a node
            that must be resolved has to carry an annotation.
    """
    if annotation is None or annotation == "" or annotation is _EMPTY:
        msg = (
            f"Node {node_name!r} carries no return annotation, so its output "
            f"kind cannot be resolved at DAG-build time. Annotate the function "
            f"with a concrete column or scalar type."
        )
        raise TypeResolutionError(msg)
    key = (
        annotation
        if isinstance(annotation, str)
        else getattr(annotation, "__name__", str(annotation))
    )
    direct = _ANNOTATION_STRING_TO_KIND.get(key)
    if direct is not None:
        return direct
    # The beartype claw resolves stringified column aliases to live
    # `jaxtyping` type objects whose name reads `Int[ndarray, 'n_obs']`
    # rather than `IntColumn`. Probe the textual form for the dtype tag.
    return _COLUMN_KIND_OF_JAXTYPING_TEXT.get(
        _jaxtyping_dtype_tag(key),
        ResolvedKind.OTHER,
    )


def resolve_kind_of_column_function(
    func: Callable[..., object],
    *,
    node_name: str,
) -> ResolvedKind:
    """Resolve the output `ResolvedKind` of a column function.

    The function may be a `ColumnFunction` (which is callable), a bare
    callable, or a `dags` wrapper (`rename_arguments` / `with_signature`)
    whose `__annotations__` is the `*args, **kwargs` forwarder shape;
    `dags.get_annotations` recovers the typed view from `__signature__`.

    Args:
        func: The column function or wrapped callable.
        node_name: The qualified name of the node, for error messages.

    Returns:
        The `ResolvedKind` of the function's return value.

    Raises:
        TypeResolutionError: If the function has no return annotation.
    """
    annotations = get_annotations(func, default="")
    return resolve_kind_of_annotation(
        annotations.get("return", _EMPTY),
        node_name=node_name,
    )


def vectorized_column_kind(kind: ResolvedKind, *, node_name: str) -> ResolvedKind:
    """Return the column kind a node has after auto-vectorization.

    A scalar policy function is written with scalar annotations
    (`int`/`float`/`bool`); after ttsim auto-vectorizes it the node
    operates on the corresponding column. A function that is already
    column-typed keeps its column kind.

    Args:
        kind: The kind resolved from the function's scalar (or column)
            return annotation.
        node_name: The qualified name of the node, for error messages.

    Returns:
        The column `ResolvedKind` the vectorized node produces.

    Raises:
        TypeResolutionError: If `kind` does not correspond to a numeric
            column or scalar (e.g. `OTHER`).
    """
    try:
        return _SCALAR_KIND_TO_COLUMN_KIND[kind]
    except KeyError:
        msg = (
            f"Cannot resolve a vectorized column kind for node {node_name!r}: "
            f"its return annotation resolves to {kind.name!r}, which is neither "
            f"a numeric scalar nor a numeric column."
        )
        raise TypeResolutionError(msg) from None


def resolve_agg_output_kind(
    agg_type: AggType,
    input_kind: ResolvedKind,
    *,
    node_name: str,
) -> ResolvedKind:
    """Resolve the output column kind of an aggregation.

    Applies the hand-written aggregation rule table (`AGG_RULE_TABLE`) for
    `agg_type` to the input column kind. `COUNT` ignores its input and
    always produces an `IntColumn`.

    Args:
        agg_type: The kind of aggregation (`SUM`, `MEAN`, `MAX`, …).
        input_kind: The `ResolvedKind` of the source column.
        node_name: The qualified name of the aggregation node, for error
            messages.

    Returns:
        The output column `ResolvedKind`.

    Raises:
        TypeResolutionError: If `agg_type` applied to `input_kind` has no
            entry in the rule table (e.g. `MAX` of a `BoolColumn`).
    """
    if agg_type == AggType.COUNT:
        return ResolvedKind.INT_COLUMN
    rules = AGG_RULE_TABLE[agg_type]
    try:
        return rules[input_kind]
    except KeyError:
        allowed = sorted(k.name for k in rules)
        msg = (
            f"Aggregation {agg_type.value!r} for node {node_name!r} cannot be "
            f"applied to a source column of kind {input_kind.name!r}. "
            f"Allowed source kinds for {agg_type.value!r}: {allowed}."
        )
        raise TypeResolutionError(msg) from None


# Hand-written aggregation rule table: per `AggType`, the input-column-kind
# to output-column-kind mapping. `COUNT` is handled separately in
# `resolve_agg_output_kind` (it ignores its input). The rules:
#
# - `SUM`: float -> float, int -> int, bool -> int (a bool sum counts trues)
# - `MEAN`: any numeric input -> float
# - `MAX` / `MIN`: float -> float, int -> int (no bool input)
# - `ANY` / `ALL`: int / bool input -> bool
#
# `test_type_resolution.py` cross-checks this table against the `@overload`
# stacks in `ttsim.tt.aggregation` and hard-fails on divergence.
AGG_RULE_TABLE = MappingProxyType(
    {
        AggType.SUM: MappingProxyType(
            {
                ResolvedKind.FLOAT_COLUMN: ResolvedKind.FLOAT_COLUMN,
                ResolvedKind.INT_COLUMN: ResolvedKind.INT_COLUMN,
                ResolvedKind.BOOL_COLUMN: ResolvedKind.INT_COLUMN,
            },
        ),
        AggType.MEAN: MappingProxyType(
            {
                ResolvedKind.FLOAT_COLUMN: ResolvedKind.FLOAT_COLUMN,
                ResolvedKind.INT_COLUMN: ResolvedKind.FLOAT_COLUMN,
                ResolvedKind.BOOL_COLUMN: ResolvedKind.FLOAT_COLUMN,
            },
        ),
        AggType.MAX: MappingProxyType(
            {
                ResolvedKind.FLOAT_COLUMN: ResolvedKind.FLOAT_COLUMN,
                ResolvedKind.INT_COLUMN: ResolvedKind.INT_COLUMN,
            },
        ),
        AggType.MIN: MappingProxyType(
            {
                ResolvedKind.FLOAT_COLUMN: ResolvedKind.FLOAT_COLUMN,
                ResolvedKind.INT_COLUMN: ResolvedKind.INT_COLUMN,
            },
        ),
        AggType.ANY: MappingProxyType(
            {
                ResolvedKind.INT_COLUMN: ResolvedKind.BOOL_COLUMN,
                ResolvedKind.BOOL_COLUMN: ResolvedKind.BOOL_COLUMN,
            },
        ),
        AggType.ALL: MappingProxyType(
            {
                ResolvedKind.INT_COLUMN: ResolvedKind.BOOL_COLUMN,
                ResolvedKind.BOOL_COLUMN: ResolvedKind.BOOL_COLUMN,
            },
        ),
    },
)


def _jaxtyping_dtype_tag(text: str) -> str:
    """Return the leading dtype tag of a `jaxtyping` type object's name.

    A claw-resolved column annotation reads `Int[ndarray, 'n_obs']`; the
    tag is the substring before the first `[`. A non-`jaxtyping` string is
    returned unchanged (and will simply miss the lookup table).
    """
    return text.split("[", 1)[0]
