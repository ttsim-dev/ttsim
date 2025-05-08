from __future__ import annotations

from typing import TYPE_CHECKING, Any, NewType

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

    from ttsim.ttsim_objects import PolicyInput, TTSIMFunction, TTSIMObject
    from ttsim.ttsim_params import TTSIMParam

    NestedTTSIMObjectDict = Mapping[str, TTSIMObject | "NestedTTSIMObjectDict"]
    FlatTTSIMObjectDict = Mapping[tuple[str, ...], TTSIMObject]
    QualNameTTSIMObjectDict = Mapping[str, TTSIMObject]

    # Specialise from dags' GenericCallable types to GETTSIM's functions.
    NestedTTSIMFunctionDict = Mapping[str, TTSIMFunction | "NestedTTSIMFunctionDict"]
    QualNameTTSIMFunctionDict = Mapping[str, TTSIMFunction]
    QualNamePolicyInputDict = Mapping[str, PolicyInput]

    # Specialise from dags' NestedInputDict to GETTSIM's types.
    NestedInputsPathsToDfColumns = Mapping[str, Any | "NestedInputsPathsToDfColumns"]
    NestedDataDict = Mapping[str, TTSIMArray | "NestedDataDict"]
    QualNameDataDict = Mapping[str, TTSIMArray]

    DashedISOString = NewType("DashedISOString", str)
    """A string representing a date in the format 'YYYY-MM-DD'."""

    OrigParamSpec = (
        str
        | dict[
            datetime.date,
            dict[
                str | int,
                str | float | int | bool | dict[str | int, float | int | bool],
            ],
        ]
    )
    """The contents of a yaml files with parameters, excluding the outermost key."""
    FlatOrigParamSpecDict = dict[tuple[str, ...], OrigParamSpec]
    """A flat tree of yaml contents; the outermost key in a file is part of the path."""

    NestedTTSIMParamDict = Mapping[str, TTSIMParam | "NestedTTSIMParamDict"]
    """A nested tree of TTSIM parameters."""
