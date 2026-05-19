"""Patch jaxtyping for two issues encountered with the beartype claw.

1. Make jaxtyping's anonymous-variadic-dim sentinel survive pickling.

   jaxtyping marks a `"..."` axis with a module-level `object()` sentinel
   (`_anonymous_variadic_dim`). A plain `object()` does not keep its identity
   across a pickle round-trip, so cloudpickling a value whose type annotations
   reference a `Foo[Array, "..."]` type — which the beartype claw makes
   pervasive — yields a type whose variadic-dim marker no longer matches the
   live module global. jaxtyping's shape check then trips
   `assert type(variadic_dim) is _NamedVariadicDim`.

   Replacing the sentinel with a `__reduce__`-backed singleton makes it
   round-trip to the same object, so unpickled annotation types stay valid.

2. Accept a bare `Ellipsis` as a synonym for the string `"..."` in
   `Float[Array, ...]` / `Int[Array, ...]` / `Bool[Array, ...]`.

   When a module uses `from __future__ import annotations`, an annotation
   like `Int[Array, ...]` is stored as the source string `"Int[Array, ...]"`.
   Runtime annotation consumers (beartype, jaxtyping) `eval` that string,
   which produces `Int[Array, Ellipsis]` — but jaxtyping's subscript handler
   calls `dim_str.strip()` and dies with `AttributeError` because
   `Ellipsis` is not a string. Coercing `Ellipsis` to `"..."` inside the
   subscript handler makes both forms behave identically.

This module must be imported before any `jaxtyping`-subscripted type is
created — `ttsim/__init__.py` imports it before every other `ttsim`
submodule.
"""

from typing import Self

from jaxtyping import _array_types


class _AnonymousVariadicDim:
    """Picklable singleton for jaxtyping's `"..."` axis marker."""

    _instance: Self | None = None

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __reduce__(self) -> tuple[type["_AnonymousVariadicDim"], tuple[()]]:
        return (_AnonymousVariadicDim, ())


_array_types._anonymous_variadic_dim = _AnonymousVariadicDim()  # noqa: SLF001


_ORIGINAL_META_DTYPE_GETITEM = _array_types._MetaAbstractDtype.__getitem__  # noqa: SLF001

_ARRAY_TYPE_AND_DIM_PAIR_LEN = 2


def _patched_meta_dtype_getitem(cls: type, item: object) -> object:
    """Translate a bare `Ellipsis` axis spec to the string `"..."`.

    Stringified annotations such as `Int[Array, ...]` (under `from __future__
    import annotations`) eval to `Int[Array, Ellipsis]`. jaxtyping otherwise
    expects the variadic spelling as the literal string `"..."`.
    """
    if isinstance(item, tuple) and len(item) == _ARRAY_TYPE_AND_DIM_PAIR_LEN:
        array_type, dim_spec = item
        if dim_spec is Ellipsis:
            item = (array_type, "...")
    return _ORIGINAL_META_DTYPE_GETITEM(cls, item)


_array_types._MetaAbstractDtype.__getitem__ = _patched_meta_dtype_getitem  # noqa: SLF001
