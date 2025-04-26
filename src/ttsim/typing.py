from typing import TYPE_CHECKING, Any, NewType

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd

    # Make these available for import from other modules.
    from dags.tree.typing import (  # noqa: F401
        GenericCallable,
        NestedInputStructureDict,
        NestedTargetDict,
        QualNameTargetList,
    )

    from ttsim.config import numpy_or_jax as np
    from ttsim.ttsim_objects import PolicyInput, TTSIMFunction, TTSIMObject

    NestedTTSIMObjectDict = Mapping[str, TTSIMObject | "NestedTTSIMObjectDict"]
    QualNameTTSIMObjectDict = Mapping[str, TTSIMObject]

    # Specialise from dags' GenericCallable types to GETTSIM's functions.
    NestedTTSIMFunctionDict = Mapping[str, TTSIMFunction | "NestedTTSIMFunctionDict"]
    QualNameTTSIMFunctionDict = Mapping[str, TTSIMFunction]
    QualNamePolicyInputDict = Mapping[str, PolicyInput]

    # Specialise from dags' NestedInputDict to GETTSIM's types.
    NestedInputsPathsToDfColumns = Mapping[str, Any | "NestedInputsPathsToDfColumns"]
    NestedDataDict = Mapping[str, pd.Series | "NestedDataDict"]
    QualNameDataDict = Mapping[str, pd.Series]
    NestedArrayDict = Mapping[str, np.ndarray | "NestedArrayDict"]

    DashedISOString = NewType("DashedISOString", str)
    """A string representing a date in the format 'YYYY-MM-DD'."""
