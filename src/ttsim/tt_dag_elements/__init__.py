from ttsim.config import IS_JAX_INSTALLED

if IS_JAX_INSTALLED:
    from jax import Array as TTSIMArray
else:
    from numpy import ndarray as TTSIMArray  # noqa: N812

from ttsim.tt_dag_elements.aggregation import AggType
from ttsim.tt_dag_elements.column_objects_param_function import (
    AggByGroupFunction,
    AggByPIDFunction,
    ColumnFunction,
    ColumnObject,
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
from ttsim.tt_dag_elements.param_objects import (
    ConsecutiveInt1dLookupTableParam,
    ConsecutiveInt1dLookupTableParamValue,
    ConsecutiveInt2dLookupTableParam,
    ConsecutiveInt2dLookupTableParamValue,
    DictParam,
    ParamObject,
    PiecewisePolynomialParam,
    PiecewisePolynomialParamValue,
    RawParam,
    ScalarParam,
    get_consecutive_int_1d_lookup_table_param_value,
    get_consecutive_int_2d_lookup_table_param_value,
    get_month_based_phase_inout_of_age_thresholds_param_value,
    get_year_based_phase_inout_of_age_thresholds_param_value,
)
from ttsim.tt_dag_elements.piecewise_polynomial import (
    get_piecewise_parameters,
    piecewise_polynomial,
)
from ttsim.tt_dag_elements.rounding import RoundingSpec
from ttsim.tt_dag_elements.shared import join

__all__ = [
    "AggByGroupFunction",
    "AggByPIDFunction",
    "AggType",
    "ColumnFunction",
    "ColumnObject",
    "ConsecutiveInt1dLookupTableParam",
    "ConsecutiveInt1dLookupTableParamValue",
    "ConsecutiveInt2dLookupTableParam",
    "ConsecutiveInt2dLookupTableParamValue",
    "DictParam",
    "FKType",
    "GroupCreationFunction",
    "ParamFunction",
    "ParamObject",
    "PiecewisePolynomialParam",
    "PiecewisePolynomialParamValue",
    "PolicyFunction",
    "PolicyInput",
    "RawParam",
    "RoundingSpec",
    "ScalarParam",
    "TTSIMArray",
    "TimeConversionFunction",
    "agg_by_group_function",
    "agg_by_p_id_function",
    "get_consecutive_int_1d_lookup_table_param_value",
    "get_consecutive_int_2d_lookup_table_param_value",
    "get_month_based_phase_inout_of_age_thresholds_param_value",
    "get_piecewise_parameters",
    "get_year_based_phase_inout_of_age_thresholds_param_value",
    "group_creation_function",
    "join",
    "param_function",
    "piecewise_polynomial",
    "policy_function",
    "policy_input",
]
