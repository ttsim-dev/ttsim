from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ttsim.main_args import MainArg

if TYPE_CHECKING:
    import pandas as pd

    from ttsim.interface_dag_elements.typing import (
        FlatData,
        NestedData,
        QNameData,
    )


@dataclass(frozen=True)
class DfAndMapper(MainArg):
    df: pd.DataFrame
    """A dataframe with arbitrary columns."""
    mapper: dict[str, Any]
    """A nested dictionary mapping expected inputs to column names in df."""

    def to_dict(self) -> dict[str, Any]:
        return {"df_and_mapper": self.__dict__}


@dataclass(frozen=True)
class DfWithNestedColumns(MainArg):
    data: pd.DataFrame
    """A df with a MultiIndex in the column dimension, elements correspond to expected tree paths."""  # noqa: E501

    def to_dict(self) -> dict[str, Any]:
        return {"df_with_nested_columns": self.data}


@dataclass(frozen=True)
class Tree(MainArg):
    data: NestedData
    """A nested dictionary mapping expected input names to vectors of data."""

    def to_dict(self) -> dict[str, Any]:
        return {"tree": self.data}


@dataclass(frozen=True)
class Flat(MainArg):
    data: FlatData
    """A dictionary mapping tree paths to vectors of data."""

    def to_dict(self) -> dict[str, Any]:
        return {"flat": self.data}


@dataclass(frozen=True)
class QName(MainArg):
    data: QNameData
    """A dictionary mapping qualified names to vectors of data."""

    def to_dict(self) -> dict[str, Any]:
        return {"qname": self.data}
