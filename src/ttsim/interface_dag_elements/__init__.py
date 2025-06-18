from __future__ import annotations

import datetime
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, get_type_hints

import pandas as pd

__all__ = []  # type: ignore[var-annotated]

from ttsim.interface_dag_elements.interface_node_objects import (
    FailOrWarnFunction,
    InterfaceFunction,
    InterfaceInput,
)


class NestedInit:
    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        for name, type_ in get_type_hints(cls).items():
            if hasattr(type_, "__origin__") and type_.__origin__ is type:
                setattr(cls, name, type_())

    def __setattr__(self, name: str, value: Any) -> None:
        if name in get_type_hints(self.__class__):
            object.__setattr__(self, name, value)
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' has no attribute '{name}'"
            )


@dataclass
class WarnIf(NestedInit):
    functions_and_data_columns_overlap: FailOrWarnFunction | None = None


@dataclass
class FailIf(NestedInit):
    active_periods_overlap: FailOrWarnFunction | None = None
    any_paths_are_invalid: FailOrWarnFunction | None = None
    paths_are_missing_in_targets_tree_mapper: FailOrWarnFunction | None = None
    environment_is_invalid: FailOrWarnFunction | None = None
    foreign_keys_are_invalid_in_data: FailOrWarnFunction | None = None
    group_ids_are_outside_top_level_namespace: FailOrWarnFunction | None = None
    group_variables_are_not_constant_within_groups: FailOrWarnFunction | None = None
    input_data_tree_is_invalid: FailOrWarnFunction | None = None
    input_df_has_bool_or_numeric_column_names: FailOrWarnFunction | None = None
    input_df_mapper_columns_missing_in_df: FailOrWarnFunction | None = None
    input_df_mapper_has_incorrect_format: FailOrWarnFunction | None = None
    non_convertible_objects_in_results_tree: FailOrWarnFunction | None = None
    root_nodes_are_missing: FailOrWarnFunction | None = None
    targets_are_not_in_specialized_environment_or_data: FailOrWarnFunction | None = None
    targets_tree_is_invalid: FailOrWarnFunction | None = None


@dataclass
class Results(NestedInit):
    df_with_mapper: InterfaceFunction | None = None
    df_with_nested_columns: InterfaceFunction | None = None
    tree: InterfaceFunction | None = None


@dataclass
class RawResults(NestedInit):
    columns: InterfaceFunction | None = None
    combined: InterfaceFunction | None = None
    from_input_data: InterfaceFunction | None = None
    params: InterfaceFunction | None = None


@dataclass
class SpecializedEnvironment(NestedInit):
    without_tree_logic_and_with_derived_functions: InterfaceFunction | None = None
    with_processed_params_and_scalars: InterfaceFunction | None = None
    with_partialled_params_and_scalars: InterfaceFunction | None = None
    tax_transfer_dag: InterfaceFunction | None = None
    tax_transfer_function: InterfaceFunction | None = None


@dataclass
class Targets(NestedInit):
    qname: InterfaceFunction | None = None
    tree: dict[str, Any] | None = None


@dataclass
class Labels(NestedInit):
    column_targets: InterfaceFunction | None = None
    grouping_levels: InterfaceFunction | None = None
    input_data_targets: InterfaceFunction | None = None
    param_targets: InterfaceFunction | None = None
    processed_data_columns: InterfaceFunction | None = None
    input_columns: InterfaceFunction | None = None
    root_nodes: InterfaceFunction | None = None
    top_level_namespace: InterfaceFunction | None = None


@dataclass
class DfAndMapper(NestedInit):
    df: pd.DataFrame | None = None
    mapper: dict[str, Any] | None = None


@dataclass
class InputData(NestedInit):
    df_and_mapper: DfAndMapper = field(default_factory=DfAndMapper)
    df_with_nested_columns: InterfaceFunction | None = None
    flat: InterfaceFunction | None = None
    tree: InterfaceFunction | None = None


@dataclass
class OrigPolicyObjects(NestedInit):
    column_objects_and_param_functions: InterfaceFunction | None = None
    param_specs: InterfaceFunction | None = None
    root: Path | None = None


@dataclass
class Templates(NestedInit):
    input_data_dtypes: InterfaceFunction | None = None


@dataclass
class InterfaceDAGElements:
    backend: Literal["numpy", "jax"] = "numpy"
    """The backend to use for computations."""
    date_str: str | None = None
    input_data: InputData = field(default_factory=InputData)
    targets: Targets = field(default_factory=Targets)
    orig_policy_objects: OrigPolicyObjects = field(default_factory=OrigPolicyObjects)
    raw_results: RawResults = field(default_factory=RawResults)
    results: Results = field(default_factory=Results)
    specialized_environment: SpecializedEnvironment = field(
        default_factory=SpecializedEnvironment
    )
    policy_environment: InterfaceFunction | None = None
    processed_data: InterfaceFunction | None = None
    dnp: InterfaceFunction | None = None
    xnp: InterfaceFunction | None = None
    date: datetime.date | None = None
    labels: Labels = field(default_factory=Labels)
    rounding: bool = True
    templates: Templates = field(default_factory=Templates)
    warn_if: WarnIf = field(default_factory=WarnIf)
    fail_if: FailIf = field(default_factory=FailIf)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__dataclass_fields__:
            object.__setattr__(self, name, value)
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' has no attribute '{name}'"
            )
