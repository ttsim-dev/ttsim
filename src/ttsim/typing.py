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

    from ttsim.ttsim_objects import PolicyInput, TTSIMFunction, TTSIMObject, TTSIMParam

    NestedInputsDict = Mapping[str, str | bool | int | float | "NestedInputsDict"]
    """Tree mapping TTSIM paths to df columns or constants."""
    NestedStringDict = Mapping[str, str | "NestedStringDict"]
    """Tree mapping TTSIM paths to df columns or type hints."""
    NestedDataDict = Mapping[str, TTSIMArray | "NestedDataDict"]
    """Tree mapping TTSIM paths to data (1-d arrays)."""
    QualNameDataDict = Mapping[str, TTSIMArray]
    """Mapping of qualified name paths to data (1-d arrays)."""
    NestedTTSIMObjectDict = Mapping[str, TTSIMObject | "NestedTTSIMObjectDict"]
    """Tree mapping TTSIM paths to TTSIM objects."""
    FlatTTSIMObjectDict = Mapping[tuple[str, ...], TTSIMObject]
    """Mapping of flat paths to TTSIM objects."""
    QualNameTTSIMObjectDict = Mapping[str, TTSIMObject]
    """Mapping of qualified name paths to TTSIM objects."""
    NestedTTSIMFunctionDict = Mapping[str, TTSIMFunction | "NestedTTSIMFunctionDict"]
    """Tree mapping TTSIM paths to TTSIM functions."""
    QualNameTTSIMFunctionDict = Mapping[str, TTSIMFunction]
    """Mapping of qualified name paths to TTSIM functions."""
    QualNamePolicyInputDict = Mapping[str, PolicyInput]
    """Mapping of qualified name paths to policy inputs (info about expected data)."""
    OrigParamSpec = (
        dict[str, str | None | dict[Literal["de", "en"], str | None]]
        | dict[
            datetime.date,
            dict[
                str | int,
                Any,
            ],
        ]
    )
    """The contents of a yaml files with parameters, excluding the outermost key."""
    FlatOrigParamSpecDict = dict[tuple[str, ...], OrigParamSpec]
    """Flat tree of yaml contents; the outermost key in a file is the leaf name."""
    NestedTTSIMParamDict = Mapping[str, TTSIMParam | "NestedTTSIMParamDict"]
    """Tree mapping TTSIM paths to TTSIM parameters."""
    QualNameTTSIMParamDict = Mapping[str, TTSIMParam]
    """Mapping of qualified name paths to TTSIM parameters."""

    # continue from here.
    RawParamsRequiringConversion = Mapping[
        str, float | int | bool | str | "RawParamsRequiringConversion"
    ]

    QualNameProcessedParamDict = Mapping[str, Any]
    """A mapping of qualified names to processed TTSIM parameters."""

    DashedISOString = NewType("DashedISOString", str)
    """A string representing a date in the format 'YYYY-MM-DD'."""
