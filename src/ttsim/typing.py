from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypeVar

if TYPE_CHECKING:
    from jaxtyping import Array, Bool, Float, Int

    BoolColumn: TypeAlias = Bool[Array, " n_obs"]
    IntColumn: TypeAlias = Int[Array, " n_obs"]
    FloatColumn: TypeAlias = Float[Array, " n_obs"]

    # Make these available for import from other modules.
    import datetime
    from collections.abc import Iterable

    OrigParamSpec = (
        # Header
        dict[str, str | None | dict[Literal["de", "en"], str | None]]
        |
        # Parameters at one point in time
        dict[
            datetime.date,
            dict[Literal["note", "reference"] | str | int, Any],  # noqa: PYI051
        ]
    )
    DashedISOString = str
    """A string representing a date in the format 'YYYY-MM-DD'."""

    from dags.tree.typing import (  # noqa: F401
        NestedInputStructureDict,
        NestedTargetDict,
    )

    from ttsim.interface_dag_elements.interface_node_objects import (
        InterfaceFunction,
        InterfaceInput,
    )

    FlatInterfaceObjects = dict[
        tuple[str, ...], InterfaceFunction | InterfaceInput | "FlatInterfaceObjects"
    ]
    """Flattened tree of interface objects."""

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Possible leaves of the various trees.
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    from ttsim.tt import (
        ColumnFunction,
        ColumnObject,
        ParamFunction,
        ParamObject,
        PolicyInput,
    )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Tree-like data structures for input, processing, and output; including metadata.
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    NestedData = dict[str, FloatColumn | IntColumn | BoolColumn | "NestedData"]
    """Tree mapping TTSIM paths to 1d arrays."""
    FlatData = dict[tuple[str, ...], FloatColumn | IntColumn | BoolColumn]
    """Flattened tree mapping TTSIM paths to 1d arrays."""
    NestedInputsMapper = dict[str, str | bool | int | float | "NestedInputsMapper"]
    """Tree mapping TTSIM paths to df columns or constants."""
    QNameData = dict[str, FloatColumn | IntColumn | BoolColumn]
    """Mapping of qualified name paths to 1d arrays."""
    QNameStrings = Iterable[str]
    """A list, tuple, or set of qualified names."""

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Collections of names etc.
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    NestedStrings = dict[str, str | "NestedStrings"]
    """Tree mapping TTSIM paths to df columns or type hints."""
    UnorderedQNames = set[str]
    """A set of qualified names."""
    OrderedQNames = TypeVar("OrderedQNames", tuple[str, ...], list[str])
    """A tuple or a list of qualified names."""

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Tree-like data structures for policy objects
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    NestedPolicyInputs = dict[str, PolicyInput | "NestedPolicyInputs"]
    """Tree of policy inputs."""
    FlatColumnObjects = dict[str, ColumnObject]
    """Flat mapping of paths to column objects."""
    FlatColumnObjectsParamFunctions = dict[
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
    NestedParamObjects = dict[str, ParamObject | "NestedParamObjects"]
    """Tree with param objects."""
    PolicyEnvironment = dict[
        str,
        ColumnObject | ParamFunction | ParamObject | "PolicyEnvironment",
    ]
    """Tree of column objects, param functions, and param objects."""
    FlatPolicyEnvironment = dict[
        tuple[str, ...], ColumnObject | ParamFunction | ParamObject
    ]
    """Flat mapping of paths to column objects, param functions, and param objects."""
    SpecEnvWithoutTreeLogicAndWithDerivedFunctions = dict[
        str,
        ColumnObject | ParamFunction | ParamObject | int | float | bool,
    ]
    """Map qualified names to column objects, param functions, param objects, or scalars from processed data."""  # noqa: E501
    SpecEnvWithProcessedParamsAndScalars = dict[str, ColumnObject | Any]
    """Map qualified names to column objects and anything that comes out of processing the params."""  # noqa: E501
    SpecEnvWithPartialledParamsAndScalars = dict[str, ColumnFunction]
    """Map qualified names to column functions that depend on columns only."""

    NestedLookupDict: TypeAlias = dict[int, float | int | bool | "NestedLookupDict"]
