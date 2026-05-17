"""Type aliases used across ttsim.

The aliases split into two groups:

1. Runtime-resolvable aliases at module scope. These can be referenced from
   `@beartype`-decorated signatures (column-type, scalar-type, simple-name
   aliases, and the "user-boundary" `User*` aliases that accept the wider
   set of inputs users may pass).
2. Aliases that reference forward types (`ColumnObject`, `ParamFunction`,
   `ParamObject`, …) and would create import cycles at runtime. These stay
   inside the `TYPE_CHECKING` block and must be referenced from runtime
   annotations only via the `__future__.annotations` string form (which
   ttsim's defining modules opt into).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias, overload

import numpy as np
import pandas as pd
from jaxtyping import Bool, Float, Int

# `jax` is an optional runtime dependency; the NumPy-only test envs do not
# install it. Resolve `Array` to the JAX type when available, else fall
# back to `np.ndarray` so the column-type aliases below stay
# runtime-resolvable for beartype either way.
try:
    from jax import Array
except ImportError:  # pragma: no cover - exercised in numpy-only envs
    Array = np.ndarray

# Canonical column types: jaxtyping-tagged 1-d arrays from either backend
# (NumPy or JAX). Union with `np.ndarray` so the NumPy backend's untagged
# arrays satisfy the same alias — without this beartype rejects every
# NumPy-backed column under the package-wide claw.
BoolColumn: TypeAlias = Bool[Array | np.ndarray, " n_obs"]
IntColumn: TypeAlias = Int[Array | np.ndarray, " n_obs"]
FloatColumn: TypeAlias = Float[Array | np.ndarray, " n_obs"]

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

DashedISOString: TypeAlias = str
"""A string representing a date in the format 'YYYY-MM-DD'."""

# Simple runtime aliases. They sit at module scope (not under
# `TYPE_CHECKING`) so beartype-decorated entry points can resolve them.
RawParamValue: TypeAlias = dict[str | int, Any]
"""The value field of a RawParam."""
UnorderedQNames: TypeAlias = set[str]
"""A set of qualified names."""
OrderedQNames: TypeAlias = tuple[str, ...] | list[str]
"""A tuple or a list of qualified names."""
QNameStrings: TypeAlias = Iterable[str]
"""A list, tuple, or set of qualified names."""

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
    NestedStrings: TypeAlias = Mapping[str, "str | NestedStrings"]
    """Tree mapping TTSIM paths to df columns or type hints."""
else:
    # Runtime aliases for beartype: the recursive form's stringified inner
    # type is not a valid Python attribute name, so beartype cannot resolve
    # it. Widen to a one-level Mapping; the per-element type still narrows.
    NestedData = Mapping[str, FloatColumn | IntColumn | BoolColumn | Mapping]
    NestedStrings = Mapping[str, str | Mapping]

FlatData: TypeAlias = Mapping[tuple[str, ...], FloatColumn | IntColumn | BoolColumn]
"""Flattened tree mapping TTSIM paths to 1d arrays."""
QNameData: TypeAlias = Mapping[str, FloatColumn | IntColumn | BoolColumn]
"""Mapping of qualified name paths to 1d arrays."""

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

    from dags.tree.typing import (  # noqa: F401
        NestedInputStructureDict,
        NestedTargetDict,
    )

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

    NestedLookupDict: TypeAlias = dict[int, float | int | bool | "NestedLookupDict"]
