from __future__ import annotations

from ttsim.aggregation import AggType
from ttsim.automatically_added_functions import create_time_conversion_functions
from ttsim.column_objects_param_function import (
    AggByGroupFunction,
    AggByPIDFunction,
    FKType,
    GroupCreationFunction,
    ParamFunction,
    PolicyFunction,
    PolicyInput,
    TimeConversionFunction,
    agg_by_group_function,
    agg_by_p_id_function,
    group_creation_function,
    param_function,
    policy_function,
    policy_input,
)
from ttsim.compute_taxes_and_transfers import (
    FunctionsAndDataOverlapWarning,
    _add_derived_functions,
    compute_taxes_and_transfers,
)
from ttsim.convert_nested_data import dataframe_to_nested_data, nested_data_to_dataframe
from ttsim.param_objects import (
    ConsecutiveInt1dLookupTableParam,
    ConsecutiveInt1dLookupTableParamValue,
    ConsecutiveInt2dLookupTableParamValue,
    DictParam,
    ParamObject,
    PiecewisePolynomialParam,
    PiecewisePolynomialParamValue,
    RawParam,
    ScalarParam,
)
from ttsim.piecewise_polynomial import (
    get_piecewise_parameters,
    piecewise_polynomial,
)
from ttsim.plot_dag import plot_dag
from ttsim.policy_environment import (
    OrigTreesWithFileNames,
    active_tree,
    get_consecutive_int_1d_lookup_table_param_value,
    get_consecutive_int_2d_lookup_table_param_value,
    get_month_based_phase_inout_of_age_thresholds_param_value,
    get_year_based_phase_inout_of_age_thresholds_param_value,
    set_up_policy_environment,
)
from ttsim.rounding import RoundingSpec
from ttsim.shared import (
    insert_path_and_value,
    join,
    merge_trees,
    to_datetime,
    upsert_path_and_value,
    upsert_tree,
)

__all__ = [
    "AggByGroupFunction",
    "AggByPIDFunction",
    "AggType",
    "ConsecutiveInt1dLookupTableParam",
    "ConsecutiveInt1dLookupTableParamValue",
    "ConsecutiveInt2dLookupTableParamValue",
    "DictParam",
    "FKType",
    "FunctionsAndDataOverlapWarning",
    "GroupCreationFunction",
    "OrigTreesWithFileNames",
    "ParamFunction",
    "ParamObject",
    "PiecewisePolynomialParam",
    "PiecewisePolynomialParamValue",
    "PolicyFunction",
    "PolicyInput",
    "RawParam",
    "RoundingSpec",
    "ScalarParam",
    "TimeConversionFunction",
    "_add_derived_functions",
    "active_tree",
    "agg_by_group_function",
    "agg_by_p_id_function",
    "compute_taxes_and_transfers",
    "create_time_conversion_functions",
    "dataframe_to_nested_data",
    "get_consecutive_int_1d_lookup_table_param_value",
    "get_consecutive_int_2d_lookup_table_param_value",
    "get_month_based_phase_inout_of_age_thresholds_param_value",
    "get_piecewise_parameters",
    "get_year_based_phase_inout_of_age_thresholds_param_value",
    "group_creation_function",
    "insert_path_and_value",
    "join",
    "merge_trees",
    "nested_data_to_dataframe",
    "param_function",
    "piecewise_polynomial",
    "plot_dag",
    "policy_function",
    "policy_input",
    "set_up_policy_environment",
    "to_datetime",
    "upsert_path_and_value",
    "upsert_tree",
]
