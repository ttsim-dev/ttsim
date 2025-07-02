from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

    from ttsim.interface_dag_elements.typing import (
        FlatData,
        NestedData,
        QNameData,
    )


def _camel_to_snake(name: str) -> str:
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.lower()


@dataclass(frozen=True)
class ABC:
    def to_dict(self) -> dict[str, Any]:
        name = _camel_to_snake(self.__class__.__name__)
        if len(self.__dict__) == 1:
            return {name: self.data}  # type: ignore[attr-defined]
        return {name: self.__dict__}


@dataclass(frozen=True)
class DfAndMapper(ABC):
    df: pd.DataFrame
    """A dataframe with arbitrary columns."""
    mapper: dict[str, Any]
    """A nested dictionary mapping expected inputs to column names in df."""


@dataclass(frozen=True)
class DfWithNestedColumns(ABC):
    data: pd.DataFrame
    """A df with a MultiIndex in the column dimension, elements correspond to expected tree paths."""  # noqa: E501


@dataclass(frozen=True)
class Tree(ABC):
    data: NestedData
    """A nested dictionary mapping expected input names to vectors of data."""


@dataclass(frozen=True)
class Flat(ABC):
    data: FlatData
    """A dictionary mapping tree paths to vectors of data."""


@dataclass(frozen=True)
class QName(ABC):
    data: QNameData
    """A dictionary mapping qualified names to vectors of data."""
