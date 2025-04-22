from typing import TYPE_CHECKING, NewType

from ttsim.config import IS_JAX_INSTALLED

if IS_JAX_INSTALLED:
    from jax import Array as TTSIMArray
else:
    from numpy import ndarray as TTSIMArray


if TYPE_CHECKING:
    from collections.abc import Mapping

    # Make these available for import from other modules.
    from dags.tree.typing import (  # noqa: F401
        GenericCallable,
        NestedInputStructureDict,
        NestedTargetDict,
        QualNameTargetList,
    )

    from ttsim.ttsim_objects import PolicyInput, TTSIMFunction, TTSIMObject

    NestedTTSIMObjectDict = Mapping[str, TTSIMObject | "NestedTTSIMObjectDict"]
    QualNameTTSIMObjectDict = Mapping[str, TTSIMObject]

    # Specialise from dags' GenericCallable types to GETTSIM's functions.
    NestedTTSIMFunctionDict = Mapping[str, TTSIMFunction | "NestedTTSIMFunctionDict"]
    QualNameTTSIMFunctionDict = Mapping[str, TTSIMFunction]
    QualNamePolicyInputDict = Mapping[str, PolicyInput]

    # Specialise from dags' NestedInputDict to GETTSIM's types.
    NestedDataDict = Mapping[str, TTSIMArray | "NestedDataDict"]
    QualNameDataDict = Mapping[str, TTSIMArray]

    DashedISOString = NewType("DashedISOString", str)
    """A string representing a date in the format 'YYYY-MM-DD'."""
