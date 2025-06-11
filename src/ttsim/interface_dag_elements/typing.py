from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, NewType, TypeVar

if TYPE_CHECKING:
    # Make these available for import from other modules.
    import datetime
    from collections.abc import Mapping

    OrigParamSpec = (
        # Header
        dict[str, str | None | dict[Literal["de", "en"], str | None]]
        |
        # Parameters at one point in time
        dict[
            datetime.date, dict[Literal["note", "reference"] | str | int, Any]  # noqa: PYI051
        ]
    )
    DashedISOString = NewType("DashedISOString", str)
    """A string representing a date in the format 'YYYY-MM-DD'."""

    from dags.tree.typing import (  # noqa: F401
        NestedInputStructureDict,
        NestedTargetDict,
    )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Possible leaves of the various trees.
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    from ttsim.tt_dag_elements import (
        ColumnFunction,
        ColumnObject,
        ParamFunction,
        ParamObject,
        TTSIMArray,
    )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Tree-like data structures for input, processing, and output; including metadata.
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    NestedData = Mapping[str, TTSIMArray | "NestedData"]
    """Tree mapping TTSIM paths to 1d arrays."""
    FlatData = Mapping[str, TTSIMArray | "FlatData"]
    """Flattened tree mapping TTSIM paths to 1d arrays."""
    NestedInputsMapper = Mapping[str, str | bool | int | float | "NestedInputsMapper"]
    """Tree mapping TTSIM paths to df columns or constants."""
    QNameData = Mapping[str, TTSIMArray]
    """Mapping of qualified name paths to 1d arrays."""

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Collections of names etc.
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    NestedStrings = Mapping[str, str | "NestedStrings"]
    """Tree mapping TTSIM paths to df columns or type hints."""
    UnorderedQNames = set[str]
    """A set of qualified names."""
    OrderedQNames = TypeVar("OrderedQNames", tuple[str, ...], list[str])
    """A tuple or a list of qualified names."""

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Tree-like data structures for policy objects
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    FlatColumnObjectsParamFunctions = Mapping[
        tuple[str, ...], ColumnObject | ParamFunction
    ]
    """Flat mapping of paths to column objects or param functions."""
    NestedColumnObjectsParamFunctions = Mapping[
        str, ColumnObject | ParamFunction | "NestedColumnObjectsParamFunctions"
    ]
    """Tree of column objects or param functions."""
    FlatOrigParamSpecs = dict[tuple[str, ...], OrigParamSpec]
    """Flat mapping of paths to yaml contents; the leaf name is also the last element of the key."""  # noqa: E501
    NestedParamObjects = Mapping[str, ParamObject | "NestedParamObjects"]
    """Tree with param objects."""
    NestedPolicyEnvironment = Mapping[
        str,
        ColumnObject | ParamFunction | ParamObject | "NestedPolicyEnvironment",
    ]
    """Tree of column objects, param functions, and param objects."""
    QNamePolicyEnvironment = dict[
        str,
        ColumnObject | ParamFunction | ParamObject,
    ]
    """Tree of column objects, param functions, and param objects."""
    QNameCombinedEnvironment0 = Mapping[
        str, ColumnObject | ParamFunction | ParamObject | int | float | bool
    ]
    """Map qualified names to column objects, param functions, param objects, or scalars from processed data."""  # noqa: E501
    QNameCombinedEnvironment1 = Mapping[str, ColumnObject | Any]
    """Map qualified names to column objects and anything that comes out of processing the params."""  # noqa: E501
    QNameCombinedEnvironment2 = Mapping[str, ColumnFunction]
    """Map qualified names to column functions that depend on columns only."""
