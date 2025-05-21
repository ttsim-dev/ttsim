from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, NewType

from ttsim.config import IS_JAX_INSTALLED

if IS_JAX_INSTALLED:
    from jax import Array as TTSIMArray
else:
    from numpy import ndarray as TTSIMArray  # noqa: N812, TC002

if TYPE_CHECKING:
    import datetime
    from collections.abc import Mapping

    # Make these available for import from other modules.
    from dags.tree.typing import (  # noqa: F401
        GenericCallable,
        NestedInputStructureDict,
        NestedTargetDict,
        QualNameTargetList,
    )

    from ttsim.ttsim_objects import (
        ColumnObject,
        ParamFunction,
        ParamObject,
        PolicyInput,
        TTSIMFunction,
    )

    NestedInputs = Mapping[str, str | bool | int | float | "NestedInputs"]
    """Tree mapping TTSIM paths to df columns or constants."""
    NestedStrings = Mapping[str, str | "NestedStrings"]
    """Tree mapping TTSIM paths to df columns or type hints."""
    NestedData = Mapping[str, TTSIMArray | "NestedData"]
    """Tree mapping TTSIM paths to data (1-d arrays)."""
    QualNameData = Mapping[str, TTSIMArray]
    """Mapping of qualified name paths to data (1-d arrays)."""
    NestedColumnObjectsParamFunctions = Mapping[
        str, ColumnObject | ParamFunction | "NestedColumnObjectsParamFunctions"
    ]
    """Tree mapping TTSIM paths to column objects or param functions."""
    FlatColumnObjectsParamFunctions = Mapping[
        tuple[str, ...], ColumnObject | ParamFunction
    ]
    """Mapping of flat paths to column objects or param functions."""
    QualNameColumnObjectsParamFunctions = Mapping[str, ColumnObject | ParamFunction]
    """Mapping of qualified name paths to column objects or param functions."""
    NestedColumnObjects = Mapping[str, ColumnObject | "NestedColumnObjects"]
    """Tree mapping TTSIM paths to column objects."""
    FlatColumnObjects = Mapping[tuple[str, ...], ColumnObject]
    """Mapping of flat paths to column objects."""
    QualNameColumnObjects = Mapping[str, ColumnObject]
    """Mapping of qualified name paths to column objects."""
    NestedTTSIMFunctions = Mapping[str, TTSIMFunction | "NestedTTSIMFunctions"]
    """Tree mapping TTSIM paths to TTSIM functions."""
    QualNameTTSIMFunctions = Mapping[str, TTSIMFunction]
    """Mapping of qualified name paths to TTSIM functions."""
    QualNamePolicyInputDict = Mapping[str, PolicyInput]
    """Mapping of qualified name paths to policy inputs (info about expected data)."""
    OrigParamSpec = (
        dict[str, str | None | dict[Literal["de", "en"], str | None]]  # Header
        | dict[
            datetime.date, dict[Literal["note", "reference"] | str | int, Any]  # noqa: PYI051
        ]  # Parameters at one point in time
    )
    """The contents of a yaml files with parameters, excluding the outermost key."""
    FlatOrigParamSpecs = dict[tuple[str, ...], OrigParamSpec]
    """Flat tree of yaml contents; the last element of the key is the leaf name."""
    OrigParam = dict[str | int, Any]
    """The original contents of YYYY-MM-DD key, after minimal processing."""
    NestedParams = Mapping[str, ParamObject | "NestedParams"]
    """Tree mapping TTSIM paths to TTSIM parameter objects."""
    QualNameParams = Mapping[str, ParamObject]
    """Mapping of qualified name paths to TTSIM parameters."""
    QualNameProcessedParams = Mapping[str, Any]
    """A mapping of qualified names to fully processed TTSIM parameters."""
    DashedISOString = NewType("DashedISOString", str)
    """A string representing a date in the format 'YYYY-MM-DD'."""
