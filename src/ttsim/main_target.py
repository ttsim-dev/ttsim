from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MainTargetABC:
    @classmethod
    def to_dict(cls) -> dict[str, Any]:
        return {
            k: v.to_dict() if isinstance(v, type(MainTargetABC)) else v
            for k, v in cls.__dict__.items()
            if not k.startswith("_")
        }

    def __post_init__(self) -> None:
        raise NotImplementedError("Do not instantiate this class directly.")


@dataclass(frozen=True)
class WarnIf(MainTargetABC):
    functions_and_data_columns_overlap: str = (
        "warn_if__functions_and_data_columns_overlap"
    )
    evaluation_date_set_in_multiple_places: str = (
        "warn_if__evaluation_date_set_in_multiple_places"
    )
    tt_dag_includes_function_with_warn_msg_if_included_set: str = (
        "warn_if__tt_dag_includes_function_with_warn_msg_if_included_set"
    )


@dataclass(frozen=True)
class FailIf(MainTargetABC):
    active_periods_overlap: str = "fail_if__active_periods_overlap"
    any_paths_are_invalid: str = "fail_if__any_paths_are_invalid"
    backend_has_changed: str = "fail_if__backend_has_changed"
    environment_is_invalid: str = "fail_if__environment_is_invalid"
    foreign_keys_are_invalid_in_data: str = "fail_if__foreign_keys_are_invalid_in_data"
    group_ids_are_outside_top_level_namespace: str = (
        "fail_if__group_ids_are_outside_top_level_namespace"
    )
    group_variables_are_not_constant_within_groups: str = (
        "fail_if__group_variables_are_not_constant_within_groups"
    )
    input_data_is_invalid: str = "fail_if__input_data_is_invalid"
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
    input_df_mapper_p_id_is_missing: str = "fail_if__input_df_mapper_p_id_is_missing"
    non_convertible_objects_in_results_tree: str = (
        "fail_if__non_convertible_objects_in_results_tree"
    )
    param_function_depends_on_column_objects: str = (
        "fail_if__param_function_depends_on_column_objects"
    )
    paths_are_missing_in_targets_tree_mapper: str = (
        "fail_if__paths_are_missing_in_targets_tree_mapper"
    )
    endogenous_p_id_among_targets: str = "fail_if__endogenous_p_id_among_targets"
    tt_root_nodes_are_missing: str = "fail_if__tt_root_nodes_are_missing"
    targets_are_not_in_specialized_environment_or_data: str = (
        "fail_if__targets_are_not_in_specialized_environment_or_data"
    )
    targets_tree_is_invalid: str = "fail_if__targets_tree_is_invalid"
    tt_dag_includes_function_with_fail_msg_if_included_set: str = (
        "fail_if__tt_dag_includes_function_with_fail_msg_if_included_set"
    )


@dataclass(frozen=True)
class Results(MainTargetABC):
    df_with_mapper: str = "results__df_with_mapper"
    df_with_nested_columns: str = "results__df_with_nested_columns"
    tree: str = "results__tree"


@dataclass(frozen=True)
class RawResults(MainTargetABC):
    columns: str = "raw_results__columns"
    from_input_data: str = "raw_results__from_input_data"
    params: str = "raw_results__params"


@dataclass(frozen=True)
class SpecializedEnvironment(MainTargetABC):
    without_tree_logic_and_with_derived_functions: str = (
        "specialized_environment__without_tree_logic_and_with_derived_functions"
    )
    with_processed_params_and_scalars: str = (
        "specialized_environment__with_processed_params_and_scalars"
    )
    with_partialled_params_and_scalars: str = (
        "specialized_environment__with_partialled_params_and_scalars"
    )
    tt_dag: str = "specialized_environment__tt_dag"


@dataclass(frozen=True)
class SpecializedEnvrionmentForPlottingAndTemplates(MainTargetABC):
    qnames_to_derive_functions_from: str = "specialized_environment_for_plotting_and_templates__qnames_to_derive_functions_from"  # noqa: E501
    without_tree_logic_and_with_derived_functions: str = "specialized_environment_for_plotting_and_templates__without_tree_logic_and_with_derived_functions"  # noqa: E501
    without_input_data_nodes_with_dummy_callables: str = "specialized_environment_for_plotting_and_templates__without_input_data_nodes_with_dummy_callables"  # noqa: E501
    complete_tt_dag: str = (
        "specialized_environment_for_plotting_and_templates__complete_tt_dag"
    )
    with_processed_params_and_scalars: str = "specialized_environment_for_plotting_and_templates__with_processed_params_and_scalars"  # noqa: E501
    with_partialled_params_and_scalars: str = "specialized_environment_for_plotting_and_templates__with_partialled_params_and_scalars"  # noqa: E501


@dataclass(frozen=True)
class Targets(MainTargetABC):
    qname: str = "tt_targets__qname"
    tree: str = "tt_targets__tree"


@dataclass(frozen=True)
class Labels(MainTargetABC):
    input_columns: str = "labels__input_columns"
    column_targets: str = "labels__column_targets"
    input_data_targets: str = "labels__input_data_targets"
    param_targets: str = "labels__param_targets"
    root_nodes: str = "labels__root_nodes"
    top_level_namespace: str = "labels__top_level_namespace"
    grouping_levels: str = "labels__grouping_levels"
    all_qnames_in_policy_environment: str = "labels__all_qnames_in_policy_environment"
    policy_inputs: str = "labels__policy_inputs"


@dataclass(frozen=True)
class DfAndMapper(MainTargetABC):
    df: str = "input_data__df_and_mapper__df"
    mapper: str = "input_data__df_and_mapper__mapper"


@dataclass(frozen=True)
class InputData(MainTargetABC):
    df_and_mapper: type[DfAndMapper] = field(default=DfAndMapper)
    df_with_nested_columns: str = "input_data__df_with_nested_columns"
    flat: str = "input_data__flat"
    sort_indices: str = "input_data__sort_indices"
    tree: str = "input_data__tree"


@dataclass(frozen=True)
class OrigPolicyObjects(MainTargetABC):
    column_objects_and_param_functions: str = (
        "orig_policy_objects__column_objects_and_param_functions"
    )
    param_specs: str = "orig_policy_objects__param_specs"
    # Do not include root here, will be pre-defined in user-facing implementations.


@dataclass(frozen=True)
class InputDataDtypes(MainTargetABC):
    tree: str = "templates__input_data_dtypes__tree"
    df_with_nested_columns: str = "templates__input_data_dtypes__df_with_nested_columns"


@dataclass(frozen=True)
class Templates(MainTargetABC):
    input_data_dtypes: type[InputDataDtypes] = field(default=InputDataDtypes)


@dataclass(frozen=True)
class MainTarget(MainTargetABC):
    results: type[Results] = field(default=Results)
    templates: type[Templates] = field(default=Templates)
    policy_environment: str = "policy_environment"
    specialized_environment: type[SpecializedEnvironment] = field(
        default=SpecializedEnvironment
    )
    specialized_environment_for_plotting_and_templates: type[
        SpecializedEnvrionmentForPlottingAndTemplates
    ] = field(
        default=SpecializedEnvrionmentForPlottingAndTemplates,
    )
    orig_policy_objects: type[OrigPolicyObjects] = field(default=OrigPolicyObjects)
    processed_data: str = "processed_data"
    raw_results: type[RawResults] = field(default=RawResults)
    labels: type[Labels] = field(default=Labels)
    input_data: type[InputData] = field(default=InputData)
    tt_targets: type[Targets] = field(default=Targets)
    num_segments: str = "num_segments"
    backend: str = "backend"
    evaluation_date_str: str = "evaluation_date_str"
    evaluation_date: str = "evaluation_date"
    policy_date_str: str = "policy_date_str"
    policy_date: str = "policy_date"
    xnp: str = "xnp"
    dnp: str = "dnp"
    rounding: str = "rounding"
    tt_function: str = "tt_function"
    tt_function_set_annotations: str = "tt_function_set_annotations"
    warn_if: type[WarnIf] = field(default=WarnIf)
    fail_if: type[FailIf] = field(default=FailIf)
