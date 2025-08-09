from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

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


T = TypeVar("T", bound="MainArg")


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


def _set_single_field(cls: type[T], field_name: str, field_value: Any) -> T:  # noqa: ANN401
    """Create an instance with one field set and all others set to None."""
    obj = object.__new__(cls)
    all_field_names = cls.__dataclass_fields__.keys()

    for name in all_field_names:
        object.__setattr__(obj, name, None)

    object.__setattr__(obj, field_name, field_value)
    return obj


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
        """A df with arbitrary columns and a nested dictionary mapping expected inputs
        to column names in this df."""
        return _set_single_field(
            cls=cls,
            field_name="df_and_mapper",
            field_value=DfAndMapper(df=df, mapper=mapper),
        )

    @classmethod
    def df_with_nested_columns(cls, df_with_nested_columns: pd.DataFrame) -> InputData:
        """A df with a MultiIndex in the column dimension, elements correspond to
        expected tree paths."""
        return _set_single_field(
            cls=cls,
            field_name="df_with_nested_columns",
            field_value=df_with_nested_columns,
        )

    @classmethod
    def tree(cls, tree: NestedData) -> InputData:
        """A nested dictionary mapping expected input names to vectors of data."""
        return _set_single_field(cls=cls, field_name="tree", field_value=tree)

    @classmethod
    def flat(cls, flat: FlatData) -> InputData:
        """A dictionary mapping tree paths to vectors of data."""
        return _set_single_field(cls=cls, field_name="flat", field_value=flat)

    @classmethod
    def qname(cls, qname: QNameData) -> InputData:
        """A dictionary mapping qualified names to vectors of data."""
        return _set_single_field(cls=cls, field_name="qname", field_value=qname)

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
        return _set_single_field(cls=cls, field_name="root", field_value=root)

    @classmethod
    def column_objects_and_param_functions(
        cls, column_objects_and_param_functions: FlatColumnObjectsParamFunctions
    ) -> OrigPolicyObjects:
        """Column objects and parameter functions."""
        return _set_single_field(
            cls=cls,
            field_name="column_objects_and_param_functions",
            field_value=column_objects_and_param_functions,
        )

    @classmethod
    def param_specs(cls, param_specs: FlatOrigParamSpecs) -> OrigPolicyObjects:
        """Parameter specifications."""
        return _set_single_field(
            cls=cls, field_name="param_specs", field_value=param_specs
        )


@dataclass(frozen=True)
class Labels(MainArg):
    input_columns: UnorderedQNames | None = None
    column_targets: OrderedQNames | None = None
    input_data_targets: OrderedQNames | None = None
    param_targets: OrderedQNames | None = None
    root_nodes: UnorderedQNames | None = None
    top_level_namespace: UnorderedQNames | None = None
    grouping_levels: OrderedQNames | None = None
    all_qnames_in_policy_environment: UnorderedQNames | None = None
    policy_inputs: OrderedQNames | None = None

    def __post_init__(self) -> None:
        _fix_classmethod_namespace_conflicts(self)

    @classmethod
    def column_targets(cls, column_targets: OrderedQNames) -> Labels:
        """Column targets for labeling."""
        return _set_single_field(
            cls=cls, field_name="column_targets", field_value=column_targets
        )

    @classmethod
    def grouping_levels(cls, grouping_levels: OrderedQNames) -> Labels:
        """Grouping levels for labeling."""
        return _set_single_field(
            cls=cls, field_name="grouping_levels", field_value=grouping_levels
        )

    @classmethod
    def input_data_targets(cls, input_data_targets: OrderedQNames) -> Labels:
        """Input data targets for labeling."""
        return _set_single_field(
            cls=cls, field_name="input_data_targets", field_value=input_data_targets
        )

    @classmethod
    def param_targets(cls, param_targets: OrderedQNames) -> Labels:
        """Parameter targets for labeling."""
        return _set_single_field(
            cls=cls, field_name="param_targets", field_value=param_targets
        )

    @classmethod
    def input_columns(cls, input_columns: UnorderedQNames) -> Labels:
        """Input columns for labeling."""
        return _set_single_field(
            cls=cls, field_name="input_columns", field_value=input_columns
        )

    @classmethod
    def root_nodes(cls, root_nodes: UnorderedQNames) -> Labels:
        """Root nodes for labeling."""
        return _set_single_field(
            cls=cls, field_name="root_nodes", field_value=root_nodes
        )

    @classmethod
    def top_level_namespace(cls, top_level_namespace: UnorderedQNames) -> Labels:
        """Top level namespace for labeling."""
        return _set_single_field(
            cls=cls, field_name="top_level_namespace", field_value=top_level_namespace
        )


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
        return _set_single_field(cls=cls, field_name="columns", field_value=columns)

    @classmethod
    def params(cls, params: QNameData) -> RawResults:
        """Parameter results data."""
        return _set_single_field(cls=cls, field_name="params", field_value=params)

    @classmethod
    def from_input_data(cls, from_input_data: QNameData) -> RawResults:
        """Results from input data."""
        return _set_single_field(
            cls=cls, field_name="from_input_data", field_value=from_input_data
        )

    @classmethod
    def combined(cls, combined: QNameData) -> RawResults:
        """Combined results data."""
        return _set_single_field(cls=cls, field_name="combined", field_value=combined)


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
        return _set_single_field(
            cls=cls, field_name="df_with_mapper", field_value=df_with_mapper
        )

    @classmethod
    def df_with_nested_columns(cls, df_with_nested_columns: pd.DataFrame) -> Results:
        """Results as a dataframe with nested columns."""
        return _set_single_field(
            cls=cls,
            field_name="df_with_nested_columns",
            field_value=df_with_nested_columns,
        )

    @classmethod
    def tree(cls, tree: NestedData) -> Results:
        """Results as a nested data tree."""
        return _set_single_field(cls=cls, field_name="tree", field_value=tree)


@dataclass(frozen=True)
class TTTargets(MainArg):
    qname: QNameStrings | None = None
    tree: NestedStrings | None = None

    def __post_init__(self) -> None:
        _fix_classmethod_namespace_conflicts(self)

    @classmethod
    def qname(cls, qname: QNameStrings) -> TTTargets:
        """TT targets using qualified names."""
        return _set_single_field(cls=cls, field_name="qname", field_value=qname)

    @classmethod
    def tree(cls, tree: NestedStrings) -> TTTargets:
        """TT targets using nested tree structure."""
        return _set_single_field(cls=cls, field_name="tree", field_value=tree)


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
        return _set_single_field(
            cls=cls,
            field_name="without_tree_logic_and_with_derived_functions",
            field_value=without_tree_logic_and_with_derived_functions,
        )

    @classmethod
    def with_processed_params_and_scalars(
        cls, with_processed_params_and_scalars: SpecEnvWithProcessedParamsAndScalars
    ) -> SpecializedEnvironment:
        """Specialized environment with processed parameters and scalars."""
        return _set_single_field(
            cls=cls,
            field_name="with_processed_params_and_scalars",
            field_value=with_processed_params_and_scalars,
        )

    @classmethod
    def with_partialled_params_and_scalars(
        cls, with_partialled_params_and_scalars: SpecEnvWithPartialledParamsAndScalars
    ) -> SpecializedEnvironment:
        """Specialized environment with partialled parameters and scalars."""
        return _set_single_field(
            cls=cls,
            field_name="with_partialled_params_and_scalars",
            field_value=with_partialled_params_and_scalars,
        )

    @classmethod
    def tt_dag(cls, tt_dag: nx.DiGraph) -> SpecializedEnvironment:
        """Specialized environment with TT DAG."""
        return _set_single_field(cls=cls, field_name="tt_dag", field_value=tt_dag)


@dataclass(frozen=True)
class SpecializedEnvironmentForPlottingAndTemplates(MainArg):
    qnames_to_derive_functions_from: UnorderedQNames | None
    without_tree_logic_and_with_derived_functions: (
        SpecEnvWithoutTreeLogicAndWithDerivedFunctions | None
    )
    without_input_data_nodes_with_dummy_callables: (
        SpecEnvWithoutTreeLogicAndWithDerivedFunctions | None
    )
    complete_tt_dag: nx.DiGraph | None
    with_processed_params_and_scalars: SpecEnvWithProcessedParamsAndScalars | None
    with_partialled_params_and_scalars: SpecEnvWithPartialledParamsAndScalars | None

    def __post_init__(self) -> None:
        _fix_classmethod_namespace_conflicts(self)

    @classmethod
    def qnames_to_derive_functions_from(
        cls, qnames_to_derive_functions_from: UnorderedQNames
    ) -> SpecializedEnvironmentForPlottingAndTemplates:
        """Qnames to derive functions from."""
        return _set_single_field(
            cls=cls,
            field_name="qnames_to_derive_functions_from",
            field_value=qnames_to_derive_functions_from,
        )

    @classmethod
    def without_tree_logic_and_with_derived_functions(
        cls,
        without_tree_logic_and_with_derived_functions: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,  # noqa: E501
    ) -> SpecializedEnvironmentForPlottingAndTemplates:
        """Specialized environment without tree logic and with derived functions."""
        return _set_single_field(
            cls=cls,
            field_name="without_tree_logic_and_with_derived_functions",
            field_value=without_tree_logic_and_with_derived_functions,
        )

    @classmethod
    def without_input_data_nodes_with_dummy_callables(
        cls,
        without_input_data_nodes_with_dummy_callables: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,  # noqa: E501
    ) -> SpecializedEnvironmentForPlottingAndTemplates:
        """Specialized environment without input data nodes with dummy callables."""
        return _set_single_field(
            cls=cls,
            field_name="without_input_data_nodes_with_dummy_callables",
            field_value=without_input_data_nodes_with_dummy_callables,
        )

    @classmethod
    def complete_tt_dag(
        cls, complete_tt_dag: nx.DiGraph
    ) -> SpecializedEnvironmentForPlottingAndTemplates:
        """Specialized environment with complete TT DAG."""
        return _set_single_field(
            cls=cls, field_name="complete_tt_dag", field_value=complete_tt_dag
        )

    @classmethod
    def with_processed_params_and_scalars(
        cls, with_processed_params_and_scalars: SpecEnvWithProcessedParamsAndScalars
    ) -> SpecializedEnvironmentForPlottingAndTemplates:
        """Specialized environment with processed parameters and scalars."""
        return _set_single_field(
            cls=cls,
            field_name="with_processed_params_and_scalars",
            field_value=with_processed_params_and_scalars,
        )

    @classmethod
    def with_partialled_params_and_scalars(
        cls, with_partialled_params_and_scalars: SpecEnvWithPartialledParamsAndScalars
    ) -> SpecializedEnvironmentForPlottingAndTemplates:
        """Specialized environment with partialled parameters and scalars."""
        return _set_single_field(
            cls=cls,
            field_name="with_partialled_params_and_scalars",
            field_value=with_partialled_params_and_scalars,
        )
