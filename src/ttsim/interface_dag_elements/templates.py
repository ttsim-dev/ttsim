from __future__ import annotations

from typing import TYPE_CHECKING

import dags.tree as dt

from ttsim.interface_dag_elements.interface_node_objects import interface_function
from ttsim.tt_dag_elements.column_objects_param_function import PolicyInput
from ttsim.tt_dag_elements.vectorization import scalar_type_to_array_type

if TYPE_CHECKING:
    from ttsim.interface_dag_elements.typing import (
        NestedInputStructureDict,
        OrderedQNames,
        PolicyEnvironment,
        SpecEnvWithPartialledParamsAndScalars,
        UnorderedQNames,
    )


@interface_function()
def input_data_dtypes(
    specialized_environment__with_partialled_params_and_scalars: SpecEnvWithPartialledParamsAndScalars,  # noqa: E501
    policy_environment: PolicyEnvironment,
    tt_targets__qname: OrderedQNames,
    labels__top_level_namespace: UnorderedQNames,
) -> NestedInputStructureDict:
    base_dtype_tree = dt.create_tree_with_input_types(
        functions=dt.unflatten_from_qnames(
            specialized_environment__with_partialled_params_and_scalars,
        ),
        targets=tt_targets__qname,
        top_level_inputs=labels__top_level_namespace,
    )

    # Replace dtypes of PolicyInputs that have the generic type 'FloatColumn | IntColumn
    # | BoolColumn' with the actual dtype found in the policy environment.
    flat_policy_env = dt.flatten_to_tree_paths(policy_environment)
    flat_dtype_tree = dt.flatten_to_tree_paths(base_dtype_tree)
    out = {}
    for p, derived_dtype_in_base in flat_dtype_tree.items():
        policy_env_element = flat_policy_env[p]
        if p[0] in {"evaluation_year", "evaluation_month", "evaluation_day"}:
            continue
        if isinstance(policy_env_element, PolicyInput) and "|" in derived_dtype_in_base:
            out[p] = scalar_type_to_array_type(policy_env_element.data_type)
        else:
            out[p] = derived_dtype_in_base

    return dt.unflatten_from_tree_paths(out)
