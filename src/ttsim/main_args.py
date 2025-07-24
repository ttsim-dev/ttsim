from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import networkx as nx
    import pandas as pd

    from ttsim.typing import (
        FlatColumnObjectsParamFunctions,
        FlatData,
        FlatOrigParamSpecs,
        NestedData,
        NestedStrings,
        OrderedQNames,
        QNameData,
        QNameStrings,
        SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
        SpecEnvWithPartialledParamsAndScalars,
        SpecEnvWithProcessedParamsAndScalars,
        UnorderedQNames,
    )


@dataclass(frozen=True)
class MainArg:
    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


@dataclass(frozen=True)
class DfAndMapper:
    df: pd.DataFrame
    """A dataframe with arbitrary columns."""
    mapper: NestedStrings
    """A nested dictionary mapping expected inputs to column names in df."""

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


@dataclass(frozen=True)
class InputData(MainArg):
    df_and_mapper: DfAndMapper | None
    df_with_nested_columns: pd.DataFrame | None
    tree: NestedData | None
    flat: FlatData | None
    qname: QNameData | None

    def __init__(self, *args: Any, **kwargs: Any):  # noqa: ANN401, ARG002, ANN204
        raise RuntimeError("Use any of the class methods to instantiate this class.")

    @classmethod
    def df_and_mapper(cls, df: pd.DataFrame, mapper: NestedStrings) -> InputData:
        """A df with arbitrary columns and a nested dictionary mapping expected inputs to column names in this df."""  # noqa: E501
        obj = object.__new__(cls)
        object.__setattr__(obj, "df_and_mapper", DfAndMapper(df=df, mapper=mapper))
        object.__setattr__(obj, "df_with_nested_columns", None)
        object.__setattr__(obj, "tree", None)
        object.__setattr__(obj, "flat", None)
        object.__setattr__(obj, "qname", None)
        return obj

    @classmethod
    def df_with_nested_columns(cls, df_with_nested_columns: pd.DataFrame) -> InputData:
        """A df with a MultiIndex in the column dimension, elements correspond to expected tree paths."""  # noqa: E501
        obj = object.__new__(cls)
        object.__setattr__(obj, "df_and_mapper", None)
        object.__setattr__(obj, "df_with_nested_columns", df_with_nested_columns)
        object.__setattr__(obj, "tree", None)
        object.__setattr__(obj, "flat", None)
        object.__setattr__(obj, "qname", None)
        return obj

    @classmethod
    def tree(cls, tree: NestedData) -> InputData:
        """A nested dictionary mapping expected input names to vectors of data."""
        obj = object.__new__(cls)
        object.__setattr__(obj, "df_and_mapper", None)
        object.__setattr__(obj, "df_with_nested_columns", None)
        object.__setattr__(obj, "tree", tree)
        object.__setattr__(obj, "flat", None)
        object.__setattr__(obj, "qname", None)
        return obj

    @classmethod
    def flat(cls, flat: FlatData) -> InputData:
        """A dictionary mapping tree paths to vectors of data."""
        obj = object.__new__(cls)
        object.__setattr__(obj, "df_and_mapper", None)
        object.__setattr__(obj, "df_with_nested_columns", None)
        object.__setattr__(obj, "tree", None)
        object.__setattr__(obj, "flat", flat)
        object.__setattr__(obj, "qname", None)
        return obj

    @classmethod
    def qname(cls, qname: QNameData) -> InputData:
        """A dictionary mapping qualified names to vectors of data."""
        obj = object.__new__(cls)
        object.__setattr__(obj, "df_and_mapper", None)
        object.__setattr__(obj, "df_with_nested_columns", None)
        object.__setattr__(obj, "tree", None)
        object.__setattr__(obj, "flat", None)
        object.__setattr__(obj, "qname", qname)
        return obj

    def to_dict(self) -> dict[str, Any]:
        if self.df_and_mapper is not None:
            return {"df_and_mapper": self.df_and_mapper.to_dict()}
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass(frozen=True)
class OrigPolicyObjects(MainArg):
    root: Path | None = None
    column_objects_and_param_functions: FlatColumnObjectsParamFunctions | None = None
    param_specs: FlatOrigParamSpecs | None = None


@dataclass(frozen=True)
class Labels(MainArg):
    column_targets: OrderedQNames | None = None
    grouping_levels: OrderedQNames | None = None
    input_data_targets: OrderedQNames | None = None
    param_targets: OrderedQNames | None = None
    processed_data_columns: UnorderedQNames | None = None
    input_columns: UnorderedQNames | None = None
    root_nodes: UnorderedQNames | None = None
    top_level_namespace: UnorderedQNames | None = None


@dataclass(frozen=True)
class RawResults(MainArg):
    columns: QNameData | None = None
    params: QNameData | None = None
    from_input_data: QNameData | None = None
    combined: QNameData | None = None


@dataclass(frozen=True)
class Results(MainArg):
    df_with_mapper: pd.DataFrame | None = None
    df_with_nested_columns: pd.DataFrame | None = None
    tree: NestedData | None = None


@dataclass(frozen=True)
class TTTargets(MainArg):
    qname: QNameStrings | None = None
    tree: NestedStrings | None = None


@dataclass(frozen=True)
class SpecializedEnvironment(MainArg):
    without_tree_logic_and_with_derived_functions: (
        SpecEnvWithoutTreeLogicAndWithDerivedFunctions | None
    ) = None
    with_processed_params_and_scalars: SpecEnvWithProcessedParamsAndScalars | None = (
        None
    )
    with_partialled_params_and_scalars: SpecEnvWithPartialledParamsAndScalars | None = (
        None
    )
    tax_transfer_dag: nx.DiGraph | None = None
    tax_transfer_function: Callable[[QNameData], QNameData] | None = None
