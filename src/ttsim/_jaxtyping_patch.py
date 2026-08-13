"""Accept a bare `Ellipsis` as a synonym for the string `"..."` in
`Float[Array, ...]` / `Int[Array, ...]` / `Bool[Array, ...]`.

When a module uses `from __future__ import annotations`, an annotation
like `Int[Array, ...]` is stored as the source string `"Int[Array, ...]"`.
Runtime annotation consumers (beartype, jaxtyping) `eval` that string,
which produces `Int[Array, Ellipsis]` — but jaxtyping's subscript handler
calls `dim_str.strip()` and dies with `AttributeError` because `Ellipsis`
is not a string. Coercing `Ellipsis` to `"..."` inside the subscript
handler makes both forms behave identically.

This module must be imported before any `jaxtyping`-subscripted type is
created — `ttsim/__init__.py` imports it before every other `ttsim`
submodule.

The cloudpickle-sentinel issue that previously also lived here is
fixed upstream in `jaxtyping >= 0.3.10` (the floor pin), which replaces
the three module-level `object()` sentinels (`_any_dtype`,
`_anonymous_dim`, `_anonymous_variadic_dim`) with `__reduce__`-backed
singleton classes.

Delete this module — and its import from `src/ttsim/__init__.py` —
when `requires-python` rises to `>=3.14` and `from __future__ import
annotations` is dropped repo-wide. Under PEP 649 deferred evaluation,
annotations are real objects, not strings, so `Int[Array, ...]` never
gets `eval`'d into `Int[Array, Ellipsis]` and the shim becomes a no-op.
See `src/ttsim/typing.py`'s module docstring for the full 3.14-migration
checklist. If jaxtyping accepts a bare `Ellipsis` upstream before then,
the shim becomes removable independently.
"""

from jaxtyping import _array_types

_ORIGINAL_META_DTYPE_GETITEM = _array_types._MetaAbstractDtype.__getitem__  # noqa: SLF001

_ARRAY_TYPE_AND_DIM_PAIR_LEN = 2


def _patched_meta_dtype_getitem(
    cls: _array_types._MetaAbstractDtype,
    item: tuple[object, object],
) -> object:
    """Translate a bare `Ellipsis` axis spec to the string `"..."`.

    Stringified annotations such as `Int[Array, ...]` (under `from __future__
    import annotations`) eval to `Int[Array, Ellipsis]`. jaxtyping otherwise
    expects the variadic spelling as the literal string `"..."`.
    """
    if isinstance(item, tuple) and len(item) == _ARRAY_TYPE_AND_DIM_PAIR_LEN:
        array_type, dim_spec = item
        if dim_spec is Ellipsis:
            item = (array_type, "...")
    return _ORIGINAL_META_DTYPE_GETITEM(cls, item)  # ty: ignore[invalid-argument-type]


_array_types._MetaAbstractDtype.__getitem__ = _patched_meta_dtype_getitem  # noqa: SLF001
