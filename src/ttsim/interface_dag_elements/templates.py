from __future__ import annotations

from typing import TYPE_CHECKING

import dags.tree as dt

from ttsim.interface_dag_elements.interface_node_objects import interface_function
from ttsim.tt.column_objects_param_function import PolicyInput
from ttsim.tt.vectorization import scalar_type_to_array_type

if TYPE_CHECKING:
    from ttsim.typing import (
        NestedInputStructureDict,
        OrderedQNames,
        PolicyEnvironment,
        SpecEnvWithPartialledParamsAndScalars,
        UnorderedQNames,
    )


@interface_function()
def input_data_dtypes(
    specialized_environment_from_policy_inputs__with_partialled_params_and_scalars: SpecEnvWithPartialledParamsAndScalars,  # noqa: E501
    policy_environment: PolicyEnvironment,
    tt_targets__qname: OrderedQNames,
    labels__top_level_namespace: UnorderedQNames,
) -> NestedInputStructureDict:
    """
    A template of the required input data and their expected types.

    Parameters
    ----------
    specialized_environment__with_partialled_params_and_scalars
        The specialized environment with partialled parameters and scalars.
    policy_environment
        The policy environment containing functions and parameters.
    tt_targets__qname
        Ordered qualified names of the targets.
    labels__top_level_namespace
        Unordered qualified names of the top-level namespace.

    Returns
    -------
    NestedInputStructureDict
        A nested dictionary mapping input paths to their data types.
    """
    base_dtype_tree = dt.create_tree_with_input_types(
        functions=dt.unflatten_from_qnames(
            specialized_environment_from_policy_inputs__with_partialled_params_and_scalars,
        ),
        targets=tt_targets__qname,
        top_level_inputs=labels__top_level_namespace,
    )

    qname_policy_env = dt.flatten_to_qnames(policy_environment)
    qname_dtype_tree = dt.flatten_to_qnames(base_dtype_tree)
    policy_inputs = {
        k: v for k, v in qname_policy_env.items() if isinstance(v, PolicyInput)
    }

    cleaned_qname_dtype_tree: dict[str, str] = {}

    for qn, derived_dtype_in_base in qname_dtype_tree.items():
        if qn in {"evaluation_year", "evaluation_month", "evaluation_day"}:
            continue

        if qn in policy_inputs:
            # Replace dtypes of PolicyInputs that have the generic type 'FloatColumn |
            # IntColumn | BoolColumn' with the actual dtype found in the policy
            # environment.
            cleaned_qname_dtype_tree[qn] = scalar_type_to_array_type(
                policy_inputs[qn].data_type
            )
        else:
            cleaned_qname_dtype_tree[qn] = derived_dtype_in_base

    return dt.unflatten_from_qnames(cleaned_qname_dtype_tree)
