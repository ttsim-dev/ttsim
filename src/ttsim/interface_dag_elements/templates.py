from __future__ import annotations

from typing import TYPE_CHECKING

import dags.tree as dt

from ttsim.interface_dag_elements.automatically_added_functions import TIME_UNIT_LABELS
from ttsim.interface_dag_elements.interface_node_objects import interface_function
from ttsim.interface_dag_elements.shared import (
    get_re_pattern_for_all_time_units_and_groupings,
)
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
    specialized_environment__with_partialled_params_and_scalars: SpecEnvWithPartialledParamsAndScalars,  # noqa: E501
    policy_environment: PolicyEnvironment,
    tt_targets__qname: OrderedQNames,
    labels__grouping_levels: OrderedQNames,
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
    labels__grouping_levels
        Ordered qualified names of grouping levels.
    labels__top_level_namespace
        Unordered qualified names of the top-level namespace.

    Returns
    -------
    NestedInputStructureDict
        A nested dictionary mapping input paths to their data types.
    """
    print(f"\n=== DEBUG: input_data_dtypes function called ===")
    print(f"tt_targets__qname: {tt_targets__qname}")
    print(f"labels__grouping_levels: {labels__grouping_levels}")
    print(f"labels__top_level_namespace length: {len(labels__top_level_namespace)}")
    print(f"labels__top_level_namespace: {sorted(labels__top_level_namespace)}")
    
    base_dtype_tree = dt.create_tree_with_input_types(
        functions=dt.unflatten_from_qnames(
            specialized_environment__with_partialled_params_and_scalars,
        ),
        targets=tt_targets__qname,
        top_level_inputs=labels__top_level_namespace,
    )
    
    print(f"\nbase_dtype_tree: {base_dtype_tree}")

    qname_policy_env = dt.flatten_to_qnames(policy_environment)
    qname_dtype_tree = dt.flatten_to_qnames(base_dtype_tree)
    policy_inputs = {
        k: v for k, v in qname_policy_env.items() if isinstance(v, PolicyInput)
    }

    print(f"\nqname_dtype_tree length: {len(qname_dtype_tree)}")
    print(f"qname_dtype_tree keys: {sorted(qname_dtype_tree.keys())}")
    print(f"policy_inputs length: {len(policy_inputs)}")
    print(f"policy_inputs keys: {sorted(policy_inputs.keys())}")

    cleaned_qname_dtype_tree: dict[str, str] = {}

    pattern_all = get_re_pattern_for_all_time_units_and_groupings(
        time_units=list(TIME_UNIT_LABELS),
        grouping_levels=labels__grouping_levels,
    )

    print(f"\n=== Processing variables ===")
    for qn, derived_dtype_in_base in qname_dtype_tree.items():
        print(f"\nProcessing: {qn} -> {derived_dtype_in_base}")
        
        if qn in {"evaluation_year", "evaluation_month", "evaluation_day"}:
            print(f"  Skipping evaluation time variable: {qn}")
            continue

        match = pattern_all.fullmatch(qn)
        base_name = match.group("base_name") if match else qn
        print(f"  base_name: {base_name}")
        print(f"  match: {match}")
        if match:
            print(f"    match groups: {match.groups()}")

        # Skip aggregated variables when both a base name and an aggregated
        # form of the base name is required by the DAG. Fixes #24.
        if (
            match
            and match.group("grouping")
            and base_name in policy_inputs
            and qn not in policy_inputs
        ):
            print(f"  Case 1: Skipping aggregated variable {qn}, will add base {base_name}")
            # Add the base variable to the template instead of the aggregated one
            if base_name not in cleaned_qname_dtype_tree:
                cleaned_qname_dtype_tree[base_name] = scalar_type_to_array_type(
                    policy_inputs[base_name].data_type
                )
                print(f"    Added base variable: {base_name} -> {cleaned_qname_dtype_tree[base_name]}")
            # Skip this aggregated variable - the base variable will be added instead
            continue

        if (
            base_name not in qname_dtype_tree
            and base_name not in cleaned_qname_dtype_tree
            and base_name in policy_inputs
        ):
            print(f"  Case 2: Missing policy input {base_name}, adding it")
            # If some input data is provided, we create aggregation functions
            # automatically only if the source node is part of the input data. Hence, if
            # the user provides incomplete input data (i.e. some policy inputs are
            # missing) and those policy inputs are sources of automatic aggregation
            # functions, dt.create_tree_with_input_types will return the name of the
            # aggregation function as root node. The policy input is not in the output.
            # We take care of this here.
            cleaned_qname_dtype_tree[base_name] = scalar_type_to_array_type(
                policy_inputs[base_name].data_type
            )
            print(f"    Added missing policy input: {base_name} -> {cleaned_qname_dtype_tree[base_name]}")

        elif qn in policy_inputs:
            print(f"  Case 3: Policy input replacement for {qn}")
            # Replace dtypes of PolicyInputs that have the generic type 'FloatColumn |
            # IntColumn | BoolColumn' with the actual dtype found in the policy
            # environment.
            cleaned_qname_dtype_tree[qn] = scalar_type_to_array_type(
                policy_inputs[qn].data_type
            )
            print(f"    Replaced dtype: {qn} -> {cleaned_qname_dtype_tree[qn]}")
        else:
            print(f"  Case 4: Using derived dtype for {qn}")
            cleaned_qname_dtype_tree[qn] = derived_dtype_in_base
            print(f"    Using derived: {qn} -> {cleaned_qname_dtype_tree[qn]}")

    print(f"\n=== Final Results ===")
    print(f"cleaned_qname_dtype_tree: {cleaned_qname_dtype_tree}")
    
    result = dt.unflatten_from_qnames(cleaned_qname_dtype_tree)
    print(f"Final result: {result}")
    print(f"=== END DEBUG ===\n")

    return result
