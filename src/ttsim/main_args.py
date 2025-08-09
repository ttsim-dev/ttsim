from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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


def _fix_classmethod_namespace_conflicts(obj: MainArg) -> None:
    """Fix namespace conflicts in ttsim main argument classes.

    Main argument classes like TTTargets, OrigPolicyObjects, etc. provide both:

    1. Direct construction: TTTargets(tree=data)
    2. Classmethod construction: TTTargets.tree(data)

    When using direct construction classmethods overwrite dataclass fields with the same
    name. For example, TTTargets(qname=data) might set the qname field to the qname
    classmethod instead of the provided data.

    This function detects such conflicts by checking if any field value is callable
    (indicating it was overwritten by a classmethod) and restores it to None.

    Args:
        obj: A ttsim main argument dataclass instance (TTTargets, OrigPolicyObjects,
        etc.)
    """
    classmethod_names = [
        name
        for name, cls_obj in obj.__class__.__dict__.items()
        if isinstance(cls_obj, classmethod)
    ]
    for field_name in classmethod_names:
        field_value = getattr(obj, field_name, None)
        if callable(field_value):
            object.__setattr__(obj, field_name, None)


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

    def __post_init__(self) -> None:
        _fix_classmethod_namespace_conflicts(self)

    @classmethod
    def root(cls, root: Path) -> OrigPolicyObjects:
        """Root path for policy objects."""
        obj = object.__new__(cls)
        object.__setattr__(obj, "root", root)
        object.__setattr__(obj, "column_objects_and_param_functions", None)
        object.__setattr__(obj, "param_specs", None)
        return obj

    @classmethod
    def column_objects_and_param_functions(
        cls, column_objects_and_param_functions: FlatColumnObjectsParamFunctions
    ) -> OrigPolicyObjects:
        """Column objects and parameter functions."""
        obj = object.__new__(cls)
        object.__setattr__(obj, "root", None)
        object.__setattr__(
            obj,
            "column_objects_and_param_functions",
            column_objects_and_param_functions,
        )
        object.__setattr__(obj, "param_specs", None)
        return obj

    @classmethod
    def param_specs(cls, param_specs: FlatOrigParamSpecs) -> OrigPolicyObjects:
        """Parameter specifications."""
        obj = object.__new__(cls)
        object.__setattr__(obj, "root", None)
        object.__setattr__(obj, "column_objects_and_param_functions", None)
        object.__setattr__(obj, "param_specs", param_specs)
        return obj


@dataclass(frozen=True)
class Labels(MainArg):
    column_targets: OrderedQNames | None = None
    grouping_levels: OrderedQNames | None = None
    input_data_targets: OrderedQNames | None = None
    param_targets: OrderedQNames | None = None
    input_columns: UnorderedQNames | None = None
    root_nodes: UnorderedQNames | None = None
    top_level_namespace: UnorderedQNames | None = None

    def __post_init__(self) -> None:
        _fix_classmethod_namespace_conflicts(self)

    @classmethod
    def column_targets(cls, column_targets: OrderedQNames) -> Labels:
        """Column targets for labeling."""
        obj = object.__new__(cls)
        object.__setattr__(obj, "column_targets", column_targets)
        object.__setattr__(obj, "grouping_levels", None)
        object.__setattr__(obj, "input_data_targets", None)
        object.__setattr__(obj, "param_targets", None)
        object.__setattr__(obj, "input_columns", None)
        object.__setattr__(obj, "root_nodes", None)
        object.__setattr__(obj, "top_level_namespace", None)
        return obj

    @classmethod
    def grouping_levels(cls, grouping_levels: OrderedQNames) -> Labels:
        """Grouping levels for labeling."""
        obj = object.__new__(cls)
        object.__setattr__(obj, "column_targets", None)
        object.__setattr__(obj, "grouping_levels", grouping_levels)
        object.__setattr__(obj, "input_data_targets", None)
        object.__setattr__(obj, "param_targets", None)
        object.__setattr__(obj, "input_columns", None)
        object.__setattr__(obj, "root_nodes", None)
        object.__setattr__(obj, "top_level_namespace", None)
        return obj

    @classmethod
    def input_data_targets(cls, input_data_targets: OrderedQNames) -> Labels:
        """Input data targets for labeling."""
        obj = object.__new__(cls)
        object.__setattr__(obj, "column_targets", None)
        object.__setattr__(obj, "grouping_levels", None)
        object.__setattr__(obj, "input_data_targets", input_data_targets)
        object.__setattr__(obj, "param_targets", None)
        object.__setattr__(obj, "input_columns", None)
        object.__setattr__(obj, "root_nodes", None)
        object.__setattr__(obj, "top_level_namespace", None)
        return obj

    @classmethod
    def param_targets(cls, param_targets: OrderedQNames) -> Labels:
        """Parameter targets for labeling."""
        obj = object.__new__(cls)
        object.__setattr__(obj, "column_targets", None)
        object.__setattr__(obj, "grouping_levels", None)
        object.__setattr__(obj, "input_data_targets", None)
        object.__setattr__(obj, "param_targets", param_targets)
        object.__setattr__(obj, "input_columns", None)
        object.__setattr__(obj, "root_nodes", None)
        object.__setattr__(obj, "top_level_namespace", None)
        return obj

    @classmethod
    def input_columns(cls, input_columns: UnorderedQNames) -> Labels:
        """Input columns for labeling."""
        obj = object.__new__(cls)
        object.__setattr__(obj, "column_targets", None)
        object.__setattr__(obj, "grouping_levels", None)
        object.__setattr__(obj, "input_data_targets", None)
        object.__setattr__(obj, "param_targets", None)
        object.__setattr__(obj, "input_columns", input_columns)
        object.__setattr__(obj, "root_nodes", None)
        object.__setattr__(obj, "top_level_namespace", None)
        return obj

    @classmethod
    def root_nodes(cls, root_nodes: UnorderedQNames) -> Labels:
        """Root nodes for labeling."""
        obj = object.__new__(cls)
        object.__setattr__(obj, "column_targets", None)
        object.__setattr__(obj, "grouping_levels", None)
        object.__setattr__(obj, "input_data_targets", None)
        object.__setattr__(obj, "param_targets", None)
        object.__setattr__(obj, "input_columns", None)
        object.__setattr__(obj, "root_nodes", root_nodes)
        object.__setattr__(obj, "top_level_namespace", None)
        return obj

    @classmethod
    def top_level_namespace(cls, top_level_namespace: UnorderedQNames) -> Labels:
        """Top level namespace for labeling."""
        obj = object.__new__(cls)
        object.__setattr__(obj, "column_targets", None)
        object.__setattr__(obj, "grouping_levels", None)
        object.__setattr__(obj, "input_data_targets", None)
        object.__setattr__(obj, "param_targets", None)
        object.__setattr__(obj, "input_columns", None)
        object.__setattr__(obj, "root_nodes", None)
        object.__setattr__(obj, "top_level_namespace", top_level_namespace)
        return obj


@dataclass(frozen=True)
class RawResults(MainArg):
    columns: QNameData | None = None
    params: QNameData | None = None
    from_input_data: QNameData | None = None
    combined: QNameData | None = None

    def __post_init__(self) -> None:
        _fix_classmethod_namespace_conflicts(self)

    @classmethod
    def columns(cls, columns: QNameData) -> RawResults:
        """Column results data."""
        obj = object.__new__(cls)
        object.__setattr__(obj, "columns", columns)
        object.__setattr__(obj, "params", None)
        object.__setattr__(obj, "from_input_data", None)
        object.__setattr__(obj, "combined", None)
        return obj

    @classmethod
    def params(cls, params: QNameData) -> RawResults:
        """Parameter results data."""
        obj = object.__new__(cls)
        object.__setattr__(obj, "columns", None)
        object.__setattr__(obj, "params", params)
        object.__setattr__(obj, "from_input_data", None)
        object.__setattr__(obj, "combined", None)
        return obj

    @classmethod
    def from_input_data(cls, from_input_data: QNameData) -> RawResults:
        """Results from input data."""
        obj = object.__new__(cls)
        object.__setattr__(obj, "columns", None)
        object.__setattr__(obj, "params", None)
        object.__setattr__(obj, "from_input_data", from_input_data)
        object.__setattr__(obj, "combined", None)
        return obj

    @classmethod
    def combined(cls, combined: QNameData) -> RawResults:
        """Combined results data."""
        obj = object.__new__(cls)
        object.__setattr__(obj, "columns", None)
        object.__setattr__(obj, "params", None)
        object.__setattr__(obj, "from_input_data", None)
        object.__setattr__(obj, "combined", combined)
        return obj


@dataclass(frozen=True)
class Results(MainArg):
    df_with_mapper: pd.DataFrame | None = None
    df_with_nested_columns: pd.DataFrame | None = None
    tree: NestedData | None = None

    def __post_init__(self) -> None:
        _fix_classmethod_namespace_conflicts(self)

    @classmethod
    def df_with_mapper(cls, df_with_mapper: pd.DataFrame) -> Results:
        """Results as a dataframe with mapper."""
        obj = object.__new__(cls)
        object.__setattr__(obj, "df_with_mapper", df_with_mapper)
        object.__setattr__(obj, "df_with_nested_columns", None)
        object.__setattr__(obj, "tree", None)
        return obj

    @classmethod
    def df_with_nested_columns(cls, df_with_nested_columns: pd.DataFrame) -> Results:
        """Results as a dataframe with nested columns."""
        obj = object.__new__(cls)
        object.__setattr__(obj, "df_with_mapper", None)
        object.__setattr__(obj, "df_with_nested_columns", df_with_nested_columns)
        object.__setattr__(obj, "tree", None)
        return obj

    @classmethod
    def tree(cls, tree: NestedData) -> Results:
        """Results as a nested data tree."""
        obj = object.__new__(cls)
        object.__setattr__(obj, "df_with_mapper", None)
        object.__setattr__(obj, "df_with_nested_columns", None)
        object.__setattr__(obj, "tree", tree)
        return obj


@dataclass(frozen=True)
class TTTargets(MainArg):
    qname: QNameStrings | None = None
    tree: NestedStrings | None = None

    def __post_init__(self) -> None:
        _fix_classmethod_namespace_conflicts(self)

    @classmethod
    def qname(cls, qname: QNameStrings) -> TTTargets:
        """TT targets using qualified names."""
        obj = object.__new__(cls)
        object.__setattr__(obj, "qname", qname)
        object.__setattr__(obj, "tree", None)
        return obj

    @classmethod
    def tree(cls, tree: NestedStrings) -> TTTargets:
        """TT targets using nested tree structure."""
        obj = object.__new__(cls)
        object.__setattr__(obj, "qname", None)
        object.__setattr__(obj, "tree", tree)
        return obj


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
    tt_dag: nx.DiGraph | None = None

    def __post_init__(self) -> None:
        _fix_classmethod_namespace_conflicts(self)

    @classmethod
    def without_tree_logic_and_with_derived_functions(
        cls,
        without_tree_logic_and_with_derived_functions: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,  # noqa: E501
    ) -> SpecializedEnvironment:
        """Specialized environment without tree logic and with derived functions."""
        obj = object.__new__(cls)
        object.__setattr__(
            obj,
            "without_tree_logic_and_with_derived_functions",
            without_tree_logic_and_with_derived_functions,
        )
        object.__setattr__(obj, "with_processed_params_and_scalars", None)
        object.__setattr__(obj, "with_partialled_params_and_scalars", None)
        object.__setattr__(obj, "tt_dag", None)
        return obj

    @classmethod
    def with_processed_params_and_scalars(
        cls, with_processed_params_and_scalars: SpecEnvWithProcessedParamsAndScalars
    ) -> SpecializedEnvironment:
        """Specialized environment with processed parameters and scalars."""
        obj = object.__new__(cls)
        object.__setattr__(obj, "without_tree_logic_and_with_derived_functions", None)
        object.__setattr__(
            obj, "with_processed_params_and_scalars", with_processed_params_and_scalars
        )
        object.__setattr__(obj, "with_partialled_params_and_scalars", None)
        object.__setattr__(obj, "tt_dag", None)
        return obj

    @classmethod
    def with_partialled_params_and_scalars(
        cls, with_partialled_params_and_scalars: SpecEnvWithPartialledParamsAndScalars
    ) -> SpecializedEnvironment:
        """Specialized environment with partialled parameters and scalars."""
        obj = object.__new__(cls)
        object.__setattr__(obj, "without_tree_logic_and_with_derived_functions", None)
        object.__setattr__(obj, "with_processed_params_and_scalars", None)
        object.__setattr__(
            obj,
            "with_partialled_params_and_scalars",
            with_partialled_params_and_scalars,
        )
        object.__setattr__(obj, "tt_dag", None)
        return obj

    @classmethod
    def tt_dag(cls, tt_dag: nx.DiGraph) -> SpecializedEnvironment:
        """Specialized environment with TT DAG."""
        obj = object.__new__(cls)
        object.__setattr__(obj, "without_tree_logic_and_with_derived_functions", None)
        object.__setattr__(obj, "with_processed_params_and_scalars", None)
        object.__setattr__(obj, "with_partialled_params_and_scalars", None)
        object.__setattr__(obj, "tt_dag", tt_dag)
        return obj
