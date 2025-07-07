from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class _ABC:
    @classmethod
    def to_dict(cls) -> dict[str, Any]:
        return {
            k: v.to_dict() if isinstance(v, type(_ABC)) else v
            for k, v in cls.__dict__.items()
            if not k.startswith("_")
        }

    def __post_init__(self) -> None:
        raise NotImplementedError("Do not instantiate this class directly.")


@dataclass(frozen=True)
class WarnIf(_ABC):
    functions_and_data_columns_overlap: str = (
        "warn_if__functions_and_data_columns_overlap"
    )


@dataclass(frozen=True)
class FailIf(_ABC):
    active_periods_overlap: str = "fail_if__active_periods_overlap"
    any_paths_are_invalid: str = "fail_if__any_paths_are_invalid"
    environment_is_invalid: str = "fail_if__environment_is_invalid"
    foreign_keys_are_invalid_in_data: str = "fail_if__foreign_keys_are_invalid_in_data"
    group_ids_are_outside_top_level_namespace: str = (
        "fail_if__group_ids_are_outside_top_level_namespace"
    )
    group_variables_are_not_constant_within_groups: str = (
        "fail_if__group_variables_are_not_constant_within_groups"
    )
    input_arrays_have_different_lengths: str = (
        "fail_if__input_arrays_have_different_lengths"
    )
    input_data_tree_is_invalid: str = "fail_if__input_data_tree_is_invalid"
    input_df_has_bool_or_numeric_column_names: str = (
        "fail_if__input_df_has_bool_or_numeric_column_names"
    )
    input_df_mapper_columns_missing_in_df: str = (
        "fail_if__input_df_mapper_columns_missing_in_df"
    )
    input_df_mapper_has_incorrect_format: str = (
        "fail_if__input_df_mapper_has_incorrect_format"
    )
    invalid_p_id_values: str = "fail_if__invalid_p_id_values"
    non_convertible_objects_in_results_tree: str = (
        "fail_if__non_convertible_objects_in_results_tree"
    )
    paths_are_missing_in_targets_tree_mapper: str = (
        "fail_if__paths_are_missing_in_targets_tree_mapper"
    )
    root_nodes_are_missing: str = "fail_if__root_nodes_are_missing"
    targets_are_not_in_specialized_environment_or_data: str = (
        "fail_if__targets_are_not_in_specialized_environment_or_data"
    )
    targets_tree_is_invalid: str = "fail_if__targets_tree_is_invalid"


@dataclass(frozen=True)
class Results(_ABC):
    df_with_mapper: str = "results__df_with_mapper"
    df_with_nested_columns: str = "results__df_with_nested_columns"
    tree: str = "results__tree"


@dataclass(frozen=True)
class RawResults(_ABC):
    columns: str = "raw_results__columns"
    combined: str = "raw_results__combined"
    from_input_data: str = "raw_results__from_input_data"
    params: str = "raw_results__params"


@dataclass(frozen=True)
class SpecializedEnvironment(_ABC):
    without_tree_logic_and_with_derived_functions: str = (
        "specialized_environment__without_tree_logic_and_with_derived_functions"
    )
    with_processed_params_and_scalars: str = (
        "specialized_environment__with_processed_params_and_scalars"
    )
    with_partialled_params_and_scalars: str = (
        "specialized_environment__with_partialled_params_and_scalars"
    )
    tax_transfer_dag: str = "specialized_environment__tax_transfer_dag"
    tax_transfer_function: str = "specialized_environment__tax_transfer_function"


@dataclass(frozen=True)
class Targets(_ABC):
    qname: str = "targets__qname"
    tree: str = "targets__tree"


@dataclass(frozen=True)
class Labels(_ABC):
    column_targets: str = "labels__column_targets"
    grouping_levels: str = "labels__grouping_levels"
    input_data_targets: str = "labels__input_data_targets"
    param_targets: str = "labels__param_targets"
    processed_data_columns: str = "labels__processed_data_columns"
    input_columns: str = "labels__input_columns"
    root_nodes: str = "labels__root_nodes"
    top_level_namespace: str = "labels__top_level_namespace"


@dataclass(frozen=True)
class DfAndMapper(_ABC):
    df: str = "input_data__df_and_mapper__df"
    mapper: str = "input_data__df_and_mapper__mapper"


@dataclass(frozen=True)
class InputData(_ABC):
    df_and_mapper: type[DfAndMapper] = field(default=DfAndMapper)
    df_with_nested_columns: str = "input_data__df_with_nested_columns"
    flat: str = "input_data__flat"
    tree: str = "input_data__tree"


@dataclass(frozen=True)
class OrigPolicyObjects(_ABC):
    column_objects_and_param_functions: str = (
        "orig_policy_objects__column_objects_and_param_functions"
    )
    param_specs: str = "orig_policy_objects__param_specs"
    # Do not include root here, will be pre-defined in user-facing implementations.


@dataclass(frozen=True)
class Templates(_ABC):
    input_data_dtypes: str = "templates__input_data_dtypes"


@dataclass(frozen=True)
class AllOutputNames(_ABC):
    results: type[Results] = field(default=Results)
    policy_environment: str = "policy_environment"
    templates: type[Templates] = field(default=Templates)
    orig_policy_objects: type[OrigPolicyObjects] = field(default=OrigPolicyObjects)
    specialized_environment: type[SpecializedEnvironment] = field(
        default=SpecializedEnvironment
    )
    processed_data: str = "processed_data"
    raw_results: type[RawResults] = field(default=RawResults)
    labels: type[Labels] = field(default=Labels)
    input_data: type[InputData] = field(default=InputData)
    targets: type[Targets] = field(default=Targets)
    backend: str = "backend"
    date_str: str = "date_str"
    date: str = "date"
    evaluation_date_str: str = "evaluation_date_str"
    evaluation_date: str = "evaluation_date"
    policy_date_str: str = "policy_date_str"
    policy_date: str = "policy_date"
    xnp: str = "xnp"
    dnp: str = "dnp"
    num_segments: str = "num_segments"
    rounding: str = "rounding"
    warn_if: type[WarnIf] = field(default=WarnIf)
    fail_if: type[FailIf] = field(default=FailIf)
