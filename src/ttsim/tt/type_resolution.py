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

import inspect
import types
import typing
from collections.abc import Callable
from enum import Enum, auto
from types import MappingProxyType
from typing import Any

import numpy
from beartype import beartype
from dags import get_annotations, with_signature
from dags.signature import forwarder_annotations
from jaxtyping import Bool, Float, Int

from ttsim import typing as ttsim_typing
from ttsim._beartype_conf import INTERNAL_CONF
from ttsim.exceptions import TTSIMError
from ttsim.tt.aggregation import AggType

# Backend-agnostic array type: union the (optional) JAX `Array` with
# `numpy.ndarray` so 0-d `Float[_BackendArray, ""]` annotations accept
# scalars from either backend (see `ttsim.typing` column aliases).
try:
    from jax import Array as _JaxArray

    _BackendArray = _JaxArray | numpy.ndarray
except ImportError:
    _BackendArray = numpy.ndarray


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
    # A claw- / `get_annotations`-resolved column alias is a live union of
    # per-backend `jaxtyping` types (`Int[Array, ...] | Int[ndarray, ...]`),
    # whose `__name__` is the unhelpful `"Union"`. Resolve each member; the
    # alias's kind is well-defined only when they all agree.
    if typing.get_origin(annotation) in (types.UnionType, typing.Union):
        member_kinds = {
            resolve_kind_of_annotation(arg, node_name=node_name)
            for arg in typing.get_args(annotation)
        }
        return member_kinds.pop() if len(member_kinds) == 1 else ResolvedKind.OTHER
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


def synthesize_typed_aggregation_wrapper(
    renamed_func: Callable[..., object],
    *,
    agg_type: AggType,
    source_column_kind: ResolvedKind | None,
    column_param_name: str | None,
    node_name: str,
) -> Callable[..., object]:
    """Stamp concrete column-type annotations onto an aggregation wrapper.

    An aggregation primitive (`grouped_sum`, `sum_by_p_id`, …) carries an
    imprecise `FloatColumn | IntColumn`-style union on its runtime
    implementation signature, because its precise per-dtype return types
    live only on `@overload` stacks. Left on the DAG node, that union
    defeats the annotation-consistency check (Bug E). This function
    rewrites the renamed aggregation wrapper's `__signature__` so it
    advertises the concrete column types resolved from the source column's
    kind and the aggregation rule table.

    Args:
        renamed_func: The aggregation primitive already adapted to the
            DAG's argument names via `dags.rename_arguments`.
        agg_type: The kind of aggregation.
        source_column_kind: The `ResolvedKind` of the aggregated source
            column. `None` only for `COUNT`, which has no source column.
        column_param_name: The renamed parameter holding the source
            column. `None` only for `COUNT`.
        node_name: The qualified name of the aggregation node, for error
            messages.

    Returns:
        The wrapper with a concretely typed `__signature__`.

    Raises:
        TypeResolutionError: If `agg_type` cannot be applied to
            `source_column_kind`, or a non-`COUNT` aggregation is missing
            its source column information.
    """
    if agg_type != AggType.COUNT and (
        source_column_kind is None or column_param_name is None
    ):
        msg = (
            f"Aggregation {agg_type.value!r} for node {node_name!r} requires a "
            f"source column to synthesize a typed wrapper."
        )
        raise TypeResolutionError(msg)

    output_kind = resolve_agg_output_kind(
        agg_type,
        source_column_kind if source_column_kind is not None else ResolvedKind.OTHER,
        node_name=node_name,
    )
    return_type_string = column_kind_to_type_string(output_kind)

    args: dict[str, str] = {}
    for name in inspect.signature(renamed_func).parameters:
        if name == column_param_name and source_column_kind is not None:
            args[name] = column_kind_to_type_string(source_column_kind)
        elif name == "num_segments":
            args[name] = "int"
        elif name == "backend":
            args[name] = "Literal['numpy', 'jax']"
        else:
            # Every remaining parameter of an aggregation primitive is a
            # group identifier or a person-pointer column — all integer
            # columns.
            args[name] = "IntColumn"
    return with_signature(
        renamed_func,
        args=args,
        return_annotation=return_type_string,
        enforce=False,
    )


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


# The "any numeric column or scalar" union the vectorized-node forwarder's
# numeric parameters and return are checked against. The runtime check
# guards against structural misuse (a string / mapping / `None` reaching a
# numeric node) without enforcing exact array dtype — ttsim data columns
# are loosely dtyped and a vectorized node broadcasts scalar arguments.
#
# A "scalar" argument is a Python number / NumPy scalar under the NumPy
# backend, but a 0-d array under JAX (policy parameters materialize as 0-d
# `jax.Array`s). The union therefore also admits 0-d jaxtyping arrays.
_WIDE_NUMERIC_ALIAS = "_TTSIMVectorizedNumeric"
_WIDE_NUMERIC_UNION = (
    ttsim_typing.FloatColumn
    | ttsim_typing.IntColumn
    | ttsim_typing.BoolColumn
    | ttsim_typing.ScalarFloat
    | ttsim_typing.ScalarInt
    | ttsim_typing.ScalarBool
    | Float[_BackendArray, ""]
    | Int[_BackendArray, ""]
    | Bool[_BackendArray, ""]
)

# The set of narrow column-type annotation strings the auto-vectorizer
# stamps; together with the un-annotated fallback union they denote a
# numeric node parameter / return that beartype checks against the wide
# numeric union. Any other annotation string is a non-numeric `OTHER`
# pass-through and is left untouched.
_NUMERIC_ANNOTATION_STRINGS: frozenset[str] = frozenset(
    {
        "FloatColumn",
        "IntColumn",
        "BoolColumn",
        "IntColumn | FloatColumn | BoolColumn",
    },
)


def _is_numeric_annotation(annotation: object) -> bool:
    """Return whether an annotation denotes a numeric node parameter / return.

    Two forms reach the typed-wrapper builder:

    - A column-type **string** (`"FloatColumn"`, `"IntColumn"`,
      `"BoolColumn"`) or the un-annotated fallback union string
      `"IntColumn | FloatColumn | BoolColumn"` — both produced by
      `create_vectorized_annotations`.
    - A live `jaxtyping` column type (or `Union` of per-backend
      `jaxtyping` types) read directly off a wrapper's `__signature__`. A
      user-authored `vectorization_strategy="not_required"` function
      *without* `from __future__ import annotations` arrives in this form.

    Anything else is a non-numeric `OTHER` annotation (a partialled
    parameter object, a lookup table, …) and beartype should not enforce
    the wide numeric union for it.
    """
    if isinstance(annotation, str):
        return annotation in _NUMERIC_ANNOTATION_STRINGS
    try:
        kind = resolve_kind_of_annotation(annotation, node_name="<numeric-check>")
    except TypeResolutionError:
        return False
    return kind in _COLUMN_KINDS


def scalar_type_to_array_type(orig_type: str | type) -> str:
    """Convert a scalar (or already-column) type annotation to a column type.

    A scalar policy function declares scalar annotations; after
    vectorization the node operates on the corresponding column.

    Annotations the resolver classifies as `OTHER` — anything that is
    neither a numeric scalar nor a numeric column, including the
    `IntColumn | FloatColumn | BoolColumn` union used as the fallback for
    an un-annotated node — are passed through unchanged.
    """
    if not isinstance(orig_type, str):
        orig_type = getattr(orig_type, "__name__", str(orig_type))
    if not orig_type or orig_type == "_empty":
        return orig_type
    kind = resolve_kind_of_annotation(orig_type, node_name="<vectorized node>")
    if kind == ResolvedKind.OTHER:
        return orig_type
    return column_kind_to_type_string(
        vectorized_column_kind(kind, node_name="<vectorized node>"),
    )


def create_vectorized_annotations(func: Callable[..., Any]) -> dict[str, Any]:
    """Create column-typed annotations for a vectorized wrapper.

    Walks the user function's scalar annotations and maps each to its
    column-type counterpart via `scalar_type_to_array_type`.
    """
    parameters_and_return = ["return", *inspect.signature(func).parameters]
    annotations = get_annotations(func, default="IntColumn | FloatColumn | BoolColumn")
    return {
        name: scalar_type_to_array_type(annotations[name])
        for name in parameters_and_return
    }


def build_beartype_checkable_wrapper(
    wrapped: Callable[..., Any],
    *,
    annotations: dict[str, Any],
    node_name: str,
) -> Callable[..., Any]:
    """Wrap a callable in a directly `@beartype`-decorable forwarder.

    The wrapped callable is typically a `numpy.vectorize` / AST-rewrite
    output, or a rounding wrapper — an *isomorphic* `*args, **kwargs`
    callable whose annotations beartype would resolve against the user
    function's globals (where the column aliases are not importable),
    raising `BeartypeCallHintForwardRefException`.

    This builds a real-parameter forwarder around `wrapped`. Being non-
    isomorphic (its code object declares the actual parameter names)
    beartype stops unwrapping at it; being defined against `ttsim.typing`'s
    namespace its string column annotations resolve.

    `@beartype` resolves the string forward references on `__annotations__`
    into live `jaxtyping` type objects and writes them back. `dags`'
    annotation-consistency check is string-based and would reject those
    live objects against the `"FloatColumn"`-style strings other nodes
    advertise. So after decoration `__annotations__` is reset to the
    `*args, **kwargs` forwarder shape: beartype's check is already compiled
    and survives the reset, while `dags.get_annotations` falls back to the
    `__signature__` — which keeps the concrete column-type *strings*.

    Wide vs narrow split: beartype's runtime check and `dags`'
    annotation-consistency check want different granularity.

    - `dags` compares a producer node's return against a consumer node's
      parameter and must distinguish `FloatColumn` from `IntColumn` from
      `BoolColumn` (a producer feeding an incompatibly typed consumer is a
      real DAG bug). It reads the *narrow* column-type strings off
      `__signature__`.
    - beartype's runtime check guards against *structural* misuse — a
      string / list / mapping / `None` reaching a numeric node. It must
      *not* enforce exact array dtype: ttsim data columns are loosely
      dtyped (an `int`-valued column legitimately feeds a `float`-typed
      policy function), and a vectorized node broadcasts scalar arguments.
      So beartype checks every numeric parameter and the return against the
      *wide* "any numeric column or scalar" union.

    Args:
        wrapped: The callable to forward to.
        annotations: Column-type annotation strings keyed by parameter name
            plus `"return"`, as produced by `create_vectorized_annotations`.
        node_name: The wrapped callable's name, used for the forwarder's
            `__name__` / `__qualname__` (kept dotless so beartype does not
            misclassify the forwarder as a lexically nested callable).

    Returns:
        A typed forwarder, decorated with `@beartype` under `INTERNAL_CONF`.
    """
    sig = inspect.signature(wrapped)
    param_names = list(sig.parameters)

    # `node_name` is interpolated into source compiled by `exec`. Guarantee
    # it is a bare Python identifier so a stray qualified name (`pkg.mod`)
    # or other punctuation cannot turn into an executable expression.
    if not node_name.isidentifier():
        msg = (
            f"node_name must be a Python identifier; got {node_name!r}. "
            "Qualified names (`pkg.mod.func`) and other punctuation are not "
            "allowed for the typed-forwarder symbol."
        )
        raise ValueError(msg)

    params_src = ", ".join(param_names)
    forwarder_name = f"_typed_{node_name}"
    source = (
        f"def {forwarder_name}({params_src}):\n"
        f"    return _ttsim_wrapped_impl({params_src})\n"
    )
    namespace: dict[str, Any] = {
        "_ttsim_wrapped_impl": wrapped,
        _WIDE_NUMERIC_ALIAS: _WIDE_NUMERIC_UNION,
    }
    exec(compile(source, "<ttsim-typed-wrapper>", "exec"), namespace)  # noqa: S102
    forwarder = namespace[forwarder_name]
    # Expose the wrapped callable's name to outside callers — beartype's
    # lexical-nesting heuristic only requires that `__qualname__` stay
    # dotless, which a plain function name already is.
    forwarder.__name__ = node_name
    forwarder.__qualname__ = node_name

    forwarder.__annotations__ = {
        name: _WIDE_NUMERIC_ALIAS
        for name in ("return", *param_names)
        if _is_numeric_annotation(annotations.get(name))
    }
    forwarder.__signature__ = inspect.Signature(
        parameters=[
            inspect.Parameter(
                name=name,
                kind=sig.parameters[name].kind,
                default=sig.parameters[name].default,
                annotation=annotations.get(name, inspect.Parameter.empty),
            )
            for name in param_names
        ],
        return_annotation=annotations.get("return", inspect.Parameter.empty),
    )
    forwarder.__module__ = "ttsim.typing"

    checked = beartype(conf=INTERNAL_CONF)(forwarder)
    forwarder.__annotations__ = forwarder_annotations()
    checked.__annotations__ = forwarder_annotations()
    checked.__signature__ = forwarder.__signature__
    return checked
