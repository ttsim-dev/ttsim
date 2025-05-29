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

    from ttsim.column_objects_param_function import (
        ColumnFunction,
        ColumnObject,
        ParamFunction,
        ParamObject,
    )

    FlatColumnObjectsParamFunctions = Mapping[
        tuple[str, ...], ColumnObject | ParamFunction
    ]
    """Mapping of flat paths to column objects or param functions."""
    OrigParamSpec = (
        dict[str, str | None | dict[Literal["de", "en"], str | None]]  # Header
        | dict[
            datetime.date, dict[Literal["note", "reference"] | str | int, Any]  # noqa: PYI051
        ]  # Parameters at one point in time
    )
    """The contents of a yaml files with parameters, excluding the outermost key."""
    FlatOrigParamSpecs = dict[tuple[str, ...], OrigParamSpec]
    """Flat tree of yaml contents; the last element of the key is the leaf name."""
    NestedParamObjects = Mapping[str, ParamObject | "NestedParamObjects"]
    """Tree mapping TTSIM paths to parameters."""
    NestedInputs = Mapping[str, str | bool | int | float | "NestedInputs"]
    """Tree mapping TTSIM paths to df columns or constants."""
    NestedStrings = Mapping[str, str | "NestedStrings"]
    """Tree mapping TTSIM paths to df columns or type hints."""
    NestedData = Mapping[str, TTSIMArray | "NestedData"]
    """Tree mapping TTSIM paths to data (1-d arrays)."""
    QualNameData = Mapping[str, TTSIMArray]
    """Mapping of qualified name paths to data (1-d arrays)."""
    QualNameDataColumns = set[str]
    """The set of data columns, represented by qualified names."""
    NestedAnyTTSIMObject = Mapping[
        str,
        ColumnObject
        | ParamFunction
        | ParamObject
        | int
        | float
        | bool
        | TTSIMArray
        | "NestedAnyTTSIMObject",
    ]
    NestedAny = Mapping[str, Any | "NestedAnyTTSIMObject"]
    """Tree mapping TTSIM paths to any type of TTSIM object."""
    NestedColumnObjectsParamFunctions = Mapping[
        str, ColumnObject | ParamFunction | "NestedColumnObjectsParamFunctions"
    ]
    NestedPolicyEnvironment = Mapping[
        str,
        ColumnObject | ParamFunction | ParamObject | "NestedPolicyEnvironment",
    ]
    """Tree mapping TTSIM paths to column objects, param functions, param objects."""
    QualNamePolicyEnvironment = Mapping[str, ColumnObject | ParamFunction | ParamObject]
    """Tree mapping TTSIM paths to column objects, param functions, param objects."""
    QualNameColumnObjectsParamFunctions = Mapping[str, ColumnObject | ParamFunction]
    """Mapping of qualified name paths to column objects or param functions."""
    QualNameColumnFunctionsWithProcessedParamsAndScalars = Mapping[str, Any]
    """A mapping of qualified names to fully processed parameters."""
    QualNameColumnFunctions = Mapping[str, ColumnFunction]
    """Mapping of qualified name paths to functions operating on columns of data."""
    DashedISOString = NewType("DashedISOString", str)
    """A string representing a date in the format 'YYYY-MM-DD'."""
