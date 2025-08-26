from __future__ import annotations

from typing import TYPE_CHECKING

import dags.tree as dt
import pandas as pd

from ttsim.interface_dag_elements.data_converters import (
    nested_data_to_df_with_nested_columns,
)
from ttsim.interface_dag_elements.interface_node_objects import interface_function
from ttsim.tt.column_objects_param_function import PolicyInput
from ttsim.tt.vectorization import scalar_type_to_array_type

if TYPE_CHECKING:
    from types import ModuleType

    from ttsim.typing import (
        NestedInputStructureDict,
        OrderedQNames,
        PolicyEnvironment,
        SpecEnvWithPartialledParamsAndScalars,
        UnorderedQNames,
    )


@interface_function(leaf_name="tree")
def input_data_dtypes__tree(
    specialized_environment_for_plotting_and_templates__with_partialled_params_and_scalars: SpecEnvWithPartialledParamsAndScalars,  # noqa: E501
    policy_environment: PolicyEnvironment,
    tt_targets__qname: OrderedQNames,
    labels__top_level_namespace: UnorderedQNames,
) -> NestedInputStructureDict:
    """A template of the required input data and their expected types."""
    base_dtype_tree = dt.create_tree_with_input_types(
        functions=dt.unflatten_from_qnames(
            specialized_environment_for_plotting_and_templates__with_partialled_params_and_scalars,
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


@interface_function(leaf_name="df_with_nested_columns")
def input_data_dtypes__df_with_nested_columns(
    tree: NestedInputStructureDict,
    xnp: ModuleType,
) -> pd.DataFrame:
    """A template of the required input data and their expected types."""
    return nested_data_to_df_with_nested_columns(
        nested_data_to_convert=tree,
        index=pd.Index(xnp.array([1])),
    )
