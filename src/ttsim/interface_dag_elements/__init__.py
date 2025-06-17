from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

__all__ = []  # type: ignore[var-annotated]

from ttsim.interface_dag_elements.interface_node_objects import (
    FailOrWarnFunction,
    InterfaceFunction,
    InterfaceInput,
)


@dataclass(frozen=True)
class WarnIf:
    functions_and_data_columns_overlap: FailOrWarnFunction | None


@dataclass(frozen=True)
class FailIf:
    active_periods_overlap: FailOrWarnFunction | None
    any_paths_are_invalid: FailOrWarnFunction | None
    data_paths_are_missing_in_paths_to_column_names: FailOrWarnFunction | None
    environment_is_invalid: FailOrWarnFunction | None
    foreign_keys_are_invalid_in_data: FailOrWarnFunction | None
    group_ids_are_outside_top_level_namespace: FailOrWarnFunction | None
    group_variables_are_not_constant_within_groups: FailOrWarnFunction | None
    input_data_tree_is_invalid: FailOrWarnFunction | None
    input_df_has_bool_or_numeric_column_names: FailOrWarnFunction | None
    input_df_mapper_columns_missing_in_df: FailOrWarnFunction | None
    input_df_mapper_has_incorrect_format: FailOrWarnFunction | None
    non_convertible_objects_in_results_tree: FailOrWarnFunction | None
    root_nodes_are_missing: FailOrWarnFunction | None
    targets_are_not_in_policy_environment_or_data: FailOrWarnFunction | None
    targets_tree_is_invalid: FailOrWarnFunction | None


@dataclass(frozen=True)
class Results:
    df_with_mapper: InterfaceFunction | None
    df_with_nested_columns: InterfaceFunction | None
    tree: InterfaceFunction | None


@dataclass(frozen=True)
class RawResults:
    columns: InterfaceFunction | None
    combined: InterfaceFunction | None
    from_input_data: InterfaceFunction | None
    params: InterfaceFunction | None


@dataclass(frozen=True)
class SpecializedEnvironment:
    without_tree_logic_and_with_derived_functions: InterfaceFunction | None
    with_processed_params_and_scalars: InterfaceFunction | None
    with_partialled_params_and_scalars: InterfaceFunction | None
    tax_transfer_dag: InterfaceFunction | None
    tax_transfer_function: InterfaceFunction | None


@dataclass(frozen=True)
class Targets:
    qname: InterfaceFunction | None
    tree: InterfaceFunction | None


@dataclass(frozen=True)
class Labels:
    column_targets: InterfaceFunction | None
    grouping_levels: InterfaceFunction | None
    input_data_targets: InterfaceFunction | None
    param_targets: InterfaceFunction | None
    processed_data_columns: InterfaceFunction | None
    root_nodes: InterfaceFunction | None
    top_level_namespace: InterfaceFunction | None


@dataclass(frozen=True)
class DfAndMapper:
    df: InterfaceFunction | None
    mapper: InterfaceFunction | None


@dataclass(frozen=True)
class InputData:
    df_and_mapper: DfAndMapper | None
    df_with_nested_columns: InterfaceFunction | None
    flat: InterfaceFunction | None
    tree: InterfaceFunction | None


@dataclass(frozen=True)
class OrigPolicyObjects:
    column_objects_and_param_functions: InterfaceFunction | None
    param_specs: InterfaceFunction | None
    root: Path | None


@dataclass(frozen=True)
class InterfaceDAGElements:
    processed_data: InterfaceFunction | None
    results: Results
    raw_results: RawResults
    backend: Literal["numpy", "jax"]
    dnp: InterfaceFunction | None
    xnp: InterfaceFunction | None
    specialized_environment: dict[str, Any]
    targets: dict[str, Any]
    policy_environment: InterfaceFunction | None
    date: datetime.date | None
    date_str: str | None
    labels: dict[str, Any]
    input_data: dict[str, Any]
    orig_policy_objects: dict[str, Any]
    rounding: bool
    warn_if: WarnIf
    fail_if: FailIf
