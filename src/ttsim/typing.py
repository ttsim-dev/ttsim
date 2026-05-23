"""Type aliases used across ttsim.

Keep JAX an *optional* runtime dependency of ttsim. That constraint
dictates how the column-type aliases below are defined and how every
module that references them must import them.

The load-bearing mechanism is in `jaxtyping`. `Bool`, `Float`, `Int`, and
`Shaped` can be imported at runtime without pulling JAX in. `jaxtyping.Array`
cannot — its top-level `__getattr__` hook resolves `Array` by importing
`jax`, so any module that touches `Array` at runtime implicitly imports
JAX. Column aliases like `IntColumn = Int[Array | np.ndarray, " n_obs"]`
contain `Array` and would trigger that import if evaluated at runtime.
(The `try / except ImportError` block below softens this: in JAX-free
envs `Array` falls back to `np.ndarray`, so the aliases stay defined.
But the alias *evaluations* still happen at module import time only
because the wider codebase opts into PEP 563 string annotations — see
the rules below.)

The aliases split into two groups:

1. **Runtime-resolvable aliases at module scope.** Column-type, scalar-type,
   simple-name aliases, and the "user-boundary" `User*` aliases. These are
   evaluated when this module is imported, so they must not pull JAX in.
   Safe because `Array | np.ndarray` reaches the `np.ndarray` branch in
   JAX-free envs.
2. **Forward-reference aliases inside `TYPE_CHECKING`.** Anything mentioning
   `ColumnObject`, `ParamFunction`, `ParamObject`, etc. — defined here
   would create import cycles. Referenced from runtime annotations only
   via PEP 563 string evaluation.

Do's and don'ts for files that reference the aliases below:

- DO keep `from __future__ import annotations` at the top of any file
  that uses column-type or `User*` aliases. PEP 563 makes annotations
  string-only at runtime, so beartype resolves them lazily without
  forcing a JAX import.
- DO NOT hoist `jaxtyping.Array` out of `TYPE_CHECKING` in any other
  module. The fallback in this file is the only place that's allowed to
  touch `Array` at module scope.
- DO NOT evaluate column-alias expressions at runtime (e.g., do not
  `isinstance(x, IntColumn)`). The string form via PEP 563 is fine for
  signatures; beartype handles it.
- DO leave the column registry (`ttsim.tt.column_objects_param_function`)
  returning string-form annotations. Eagerly resolving them would
  re-introduce the JAX import.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Annotated, Any, Literal, Protocol, TypeAlias, overload

import numpy as np
import pandas as pd
from beartype.vale import Is
from jaxtyping import Bool, Float, Int, Shaped

# `jax` is an optional runtime dependency; the NumPy-only test envs do not
# install it. Resolve `Array` to the JAX type when available, else fall
# back to `np.ndarray` so the column-type aliases below stay
# runtime-resolvable for beartype either way. `Array` is only ever an
# assignment target (never a direct `import` binding), so ty infers its
# type from both branches instead of flagging a conflicting declaration.
try:
    from jax import Array as _JaxArray

    Array = _JaxArray
except ImportError:  # pragma: no cover - exercised in numpy-only envs
    Array = np.ndarray

# Canonical column types: jaxtyping-tagged 1-d arrays from either backend
# (NumPy or JAX). Union with `np.ndarray` so the NumPy backend's untagged
# arrays satisfy the same alias — without this beartype rejects every
# NumPy-backed column under the package-wide claw.
BoolColumn: TypeAlias = Bool[Array | np.ndarray, " n_obs"]
IntColumn: TypeAlias = Int[Array | np.ndarray, " n_obs"]
FloatColumn: TypeAlias = Float[Array | np.ndarray, " n_obs"]
# jaxtyping has no datetime dtype tag, so a beartype `Is` validator enforces
# the `datetime64` dtype. Hoisted to module scope (like the other column
# aliases) so the claw can resolve it at decoration time.
DatetimeColumn: TypeAlias = Annotated[
    Shaped[np.ndarray, " n_obs"],
    Is[lambda a: np.issubdtype(a.dtype, np.datetime64)],
]

# Canonical scalar types (used inside ttsim once user input has been
# converted to a single concrete numeric kind).
ScalarFloat: TypeAlias = float | np.floating
ScalarInt: TypeAlias = int | np.integer
ScalarBool: TypeAlias = bool | np.bool_

# User-boundary aliases (Decision 8): the wider set ttsim accepts from
# users on the way in. Internal code should narrow to the canonical alias
# above via explicit `_canonicalize_*` helpers.
UserScalarFloat: TypeAlias = float | int | np.floating | np.integer
UserScalarInt: TypeAlias = int | np.integer
UserScalarBool: TypeAlias = bool | np.bool_
UserFloatColumn: TypeAlias = FloatColumn | pd.Series
UserIntColumn: TypeAlias = IntColumn | pd.Series
UserBoolColumn: TypeAlias = BoolColumn | pd.Series

# `DashedISOString`: a string representing a date in the format 'YYYY-MM-DD'.
DashedISOString: TypeAlias = str

# Simple runtime aliases. They sit at module scope (not under
# `TYPE_CHECKING`) so beartype-decorated entry points can resolve them.
#
# - `RawParamValue`: the value field of a `RawParam`.
# - `UnorderedQNames`: a set of qualified names.
# - `OrderedQNames`: a tuple or a list of qualified names.
# - `QNameStrings`: any iterable of qualified names.
RawParamValue: TypeAlias = dict[str | int, Any]
UnorderedQNames: TypeAlias = set[str]
OrderedQNames: TypeAlias = tuple[str, ...] | list[str]
QNameStrings: TypeAlias = Iterable[str]


if TYPE_CHECKING:
    # Real definitions live in `dags.tree.typing` and are recursive
    # `Mapping[str, str | None | <self>]` aliases. ty consumes the narrow
    # form.
    from dags.tree.typing import NestedInputStructureDict, NestedTargetDict
else:
    # beartype cannot resolve the stringified recursive form; widen the
    # runtime alias to a one-level Mapping (Mapping itself is enough to
    # satisfy beartype's isinstance check).
    NestedTargetDict: TypeAlias = Mapping[str, object]
    NestedInputStructureDict: TypeAlias = Mapping[str, object]

# Data-tree aliases. Hoisted out of TYPE_CHECKING so @beartype-decorated
# user-boundary entry points (InputData.tree/.flat/.qname, TTTargets.tree)
# can resolve them at decoration time.
if TYPE_CHECKING:
    # Recursive aliases for ty: precise nested types with the narrow recursive
    # form ttsim's call sites expect.
    NestedData: TypeAlias = Mapping[
        str, "FloatColumn | IntColumn | BoolColumn | NestedData"
    ]
    """Tree mapping TTSIM paths to 1d arrays."""
    NestedStrings: TypeAlias = Mapping[str, "str | None | NestedStrings"]
    """Tree mapping TTSIM paths to df column names, type hints, or `None`.

    A `None` leaf marks a target to compute (vs. a string that renames it);
    see `FlatTTTargets`.
    """
else:
    # Runtime aliases for beartype: the recursive form's stringified inner
    # type is not a valid Python attribute name, so beartype cannot resolve
    # it. Widen to a one-level Mapping; the per-element type still narrows.
    NestedData = Mapping[str, FloatColumn | IntColumn | BoolColumn | Mapping]
    NestedStrings = Mapping[str, str | None | Mapping]

# `FlatData`: flattened tree mapping TTSIM tree paths (tuple) to 1-d arrays.
# `QNameData`: mapping of qualified-name strings to 1-d arrays.
FlatData: TypeAlias = Mapping[tuple[str, ...], FloatColumn | IntColumn | BoolColumn]
QNameData: TypeAlias = Mapping[str, FloatColumn | IntColumn | BoolColumn]

# Results aliases. A results tree's leaves are not only columns: processed
# param values are genuinely heterogeneous (scalars, dicts, lookup arrays,
# dates). Use an honest-wide `object` leaf rather than enumerating a union.
# `QNameResults`: flat mapping of qualified names to heterogeneous values.
QNameResults: TypeAlias = Mapping[str, object]
if TYPE_CHECKING:
    # `NestedResults`: recursive results tree for ty (precise nested form).
    NestedResults: TypeAlias = Mapping[str, "object | NestedResults"]
else:
    # beartype cannot resolve the stringified recursive inner name; widen the
    # runtime alias to a one-level Mapping.
    NestedResults = Mapping[str, object]

# User-boundary data aliases (Decision 8 / GEP-09). User-facing `InputData.*`
# factories accept a wider leaf type than the canonical column aliases above:
# users legitimately pass `pd.Series` and plain Python lists/sequences of
# numbers, which internal code canonicalizes to backend arrays. Keep these
# strictly at the `@beartype`-decorated user boundary; internal call sites use
# the narrow `NestedData` / `FlatData` / `QNameData` forms.
UserColumn: TypeAlias = (
    FloatColumn | IntColumn | BoolColumn | pd.Series | list[float | int | bool]
)
# `UserNestedData` is a recursive tree like `NestedData`; use the
# two-definition pattern (precise for ty, widened for the beartype claw).
# The runtime form uses an honest-wide `object` leaf — like `NestedResults`
# — so the `InputData.tree` beartype boundary admits scalars and other
# malformed-but-plausible leaves, letting ttsim's `fail_if` validators
# raise their curated, path-listing diagnostics instead of a generic
# beartype message. beartype still enforces the `Mapping[str, ...]`
# structure (string keys, dict shape).
if TYPE_CHECKING:
    UserNestedData: TypeAlias = Mapping[str, "UserColumn | UserNestedData"]
else:
    UserNestedData = Mapping[str, object]
# `UserNestedData`: user-boundary tree mapping TTSIM paths to columns, Series,
# or sequences.
# `UserFlatData`: user-boundary flat mapping of tree paths to columns or
# scalars (the latter for users opting into the partial-application path).
# `UserQNameData`: user-boundary mapping of qualified names to the same.
UserFlatData: TypeAlias = Mapping[
    tuple[str, ...],
    UserColumn | UserScalarFloat | UserScalarInt | UserScalarBool,
]
UserQNameData: TypeAlias = Mapping[
    str, UserColumn | UserScalarFloat | UserScalarInt | UserScalarBool
]

# `FlatTTTargets`: a flattened target tree, mapping each qualified name to its
# leaf value (`None` to compute the target, a string to rename it). Produced by
# `dags.tree.flatten_to_qnames` on a `NestedTargetDict` / `NestedStrings`.
FlatTTTargets: TypeAlias = Mapping[str, str | None]

if TYPE_CHECKING:
    # Names below are TYPE_CHECKING-only because they either reference
    # types that would cause an import cycle at runtime (ColumnObject,
    # ParamFunction, ParamObject, PolicyInput, ColumnFunction,
    # InterfaceFunction, InterfaceInput) or use `Iterable` / `Iterator`
    # in ways that beartype need not see (no `@beartype` decorator
    # consumes them in a checked signature).
    import datetime
    from collections.abc import Iterator

    class OrigParamSpec(Protocol):
        """A dictionary with patterns for header and parameters at one point in time."""

        @overload
        def __getitem__(self, key: Literal["type"]) -> str: ...

        @overload
        def __getitem__(
            self, key: str
        ) -> str | None | dict[Literal["de", "en"], str | None]: ...

        @overload
        def __getitem__(
            self, key: datetime.date
        ) -> dict[Literal["note", "reference"] | str | int, Any]: ...

        def __getitem__(
            self, key: str | datetime.date
        ) -> (
            str
            | None
            | dict[Literal["de", "en"], str | None]
            | dict[Literal["note", "reference"] | str | int, Any]
        ): ...

        @overload
        def get(
            self, key: str, default: None = None
        ) -> str | None | dict[Literal["de", "en"], str | None]: ...

        @overload
        def get(
            self, key: str, default: str | bool | float
        ) -> (
            str | None | dict[Literal["de", "en"], str | None] | bool | int | float
        ): ...

        def get(
            self,
            key: str,
            default: str
            | None
            | bool
            | float
            | dict[Literal["de", "en"], str | None] = None,
        ) -> (
            str | None | dict[Literal["de", "en"], str | None] | bool | int | float
        ): ...

        def __contains__(self, key: str | datetime.date) -> bool: ...

        def __iter__(self) -> Iterator[str | datetime.date]: ...

        def keys(self) -> Iterable[str | datetime.date]: ...

    from ttsim.interface_dag_elements.interface_node_objects import (
        InterfaceFunction,
        InterfaceInput,
    )

    FlatInterfaceObjects = Mapping[
        tuple[str, ...], InterfaceFunction | InterfaceInput | "FlatInterfaceObjects"
    ]
    """Flattened tree of interface objects."""

    from ttsim.tt import (
        ColumnFunction,
        ColumnObject,
        ParamFunction,
        ParamObject,
        PolicyInput,
    )

    NestedInputsMapper = Mapping[str, str | bool | int | float | "NestedInputsMapper"]
    """Tree mapping TTSIM paths to df columns or constants."""

    NestedPolicyInputs = Mapping[str, "PolicyInput | NestedPolicyInputs"]
    """Tree of policy inputs."""
    FlatColumnObjects = Mapping[str, ColumnObject]
    """Flat mapping of paths to column objects."""
    FlatColumnObjectsParamFunctions = Mapping[
        tuple[str, ...],
        ColumnObject | ParamFunction,
    ]
    """Flat mapping of paths to column objects or param functions."""
    NestedColumnObjectsParamFunctions = dict[
        str,
        ColumnObject | ParamFunction | "NestedColumnObjectsParamFunctions",
    ]
    """Tree of column objects or param functions."""
    FlatOrigParamSpecs = dict[tuple[str, ...], OrigParamSpec]
    """Flat mapping of paths to yaml contents; the leaf name is also the last element of the key."""  # noqa: E501
    NestedParamObjects = dict[str, "ParamObject | NestedParamObjects"]
    """Tree with param objects."""
    PolicyEnvironment = Mapping[
        str,
        ColumnObject | ParamFunction | ParamObject | "PolicyEnvironment",
    ]
    """Tree of column objects, param functions, and param objects."""
    FlatPolicyEnvironment = Mapping[
        tuple[str, ...], ColumnObject | ParamFunction | ParamObject
    ]
    """Flat mapping of paths to column objects, param functions, and param objects."""
    SpecEnvWithoutTreeLogicAndWithDerivedFunctions = Mapping[
        str,
        ColumnObject | ParamFunction | ParamObject | int | float | bool,
    ]
    """Map qualified names to column objects, param functions, param objects, or scalars from processed data."""  # noqa: E501
    SpecEnvWithProcessedParamsAndScalars = Mapping[str, ColumnObject | Any]
    """Map qualified names to column objects and anything that comes out of processing the params."""  # noqa: E501
    SpecEnvWithPartialledParamsAndScalars = Mapping[str, ColumnFunction]
    """Map qualified names to column functions that depend on columns only."""

if TYPE_CHECKING:
    NestedLookupDict: TypeAlias = dict[int, float | int | bool | "NestedLookupDict"]
else:
    # Recursive aliases stringified as inner attribute names are unresolvable
    # by beartype; widen the runtime form to `dict[int, object]`.
    NestedLookupDict = dict[int, object]


# `DictParamValue`: the value of a `DictParam` read from YAML. Keys are
# strings or integers; leaves are YAML scalars or further nested dicts
# (dict params may be merged recursively across policy dates).
if TYPE_CHECKING:
    DictParamValue: TypeAlias = dict[
        str | int, "int | float | bool | str | DictParamValue"
    ]
else:
    # beartype cannot resolve the stringified recursive inner name; widen the
    # runtime form to a one-level dict with a bare `dict` for nested levels.
    DictParamValue = dict[str | int, int | float | bool | str | dict]


if not TYPE_CHECKING:
    # Loose runtime stubs for the aliases that reference ColumnObject etc.;
    # importing the precise definitions at runtime would create a cycle
    # through ttsim.tt. beartype only needs the alias to exist as a Mapping
    # subtype.
    PolicyEnvironment: TypeAlias = Mapping[str, object]
    FlatPolicyEnvironment: TypeAlias = Mapping[tuple[str, ...], object]
    FlatColumnObjectsParamFunctions: TypeAlias = Mapping[tuple[str, ...], object]
    FlatColumnObjects: TypeAlias = Mapping[str, object]
    NestedPolicyInputs: TypeAlias = Mapping[str, object]
    NestedColumnObjectsParamFunctions: TypeAlias = dict
    NestedParamObjects: TypeAlias = dict
    FlatOrigParamSpecs: TypeAlias = dict
    SpecEnvWithoutTreeLogicAndWithDerivedFunctions: TypeAlias = Mapping[str, object]
    SpecEnvWithProcessedParamsAndScalars: TypeAlias = Mapping[str, object]
    SpecEnvWithPartialledParamsAndScalars: TypeAlias = Mapping[str, object]
    FlatInterfaceObjects: TypeAlias = Mapping[tuple[str, ...], object]
    NestedInputsMapper: TypeAlias = Mapping[str, object]
    OrigParamSpec: TypeAlias = Mapping[object, object]
