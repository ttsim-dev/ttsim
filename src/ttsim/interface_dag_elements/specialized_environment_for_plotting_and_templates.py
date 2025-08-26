"""Builds the specialized environment specifically for creating input templates or plots
of the DAG.

The main difference to `specialized_environment` is that input data is optional. Derived
functions are created based on policy inputs or the qnames of the input columns. This is
useful for applications where users are interested in the DAG itself (not its
execution).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

import dags.tree as dt
import networkx as nx
from dags import create_dag

from ttsim.interface_dag_elements import specialized_environment
from ttsim.interface_dag_elements.automatically_added_functions import TIME_UNIT_LABELS
from ttsim.interface_dag_elements.interface_node_objects import (
    InterfaceFunction,
    InterfaceInput,
    interface_function,
)
from ttsim.interface_dag_elements.shared import (
    get_re_pattern_for_all_time_units_and_groupings,
)
from ttsim.interface_dag_elements.specialized_environment import (
    _add_derived_functions,
    _remove_tree_logic_from_policy_environment,
)
from ttsim.tt.column_objects_param_function import (
    ParamFunction,
    PolicyFunction,
    PolicyInput,
    param_function,
    policy_function,
)
from ttsim.tt.param_objects import ParamObject

if TYPE_CHECKING:
    import datetime
    from collections.abc import Callable
    from types import ModuleType
    from typing import Any

    from ttsim.typing import (
        OrderedQNames,
        PolicyEnvironment,
        SpecEnvWithoutTreeLogicAndWithDerivedFunctions,
        SpecEnvWithPartialledParamsAndScalars,
        SpecEnvWithProcessedParamsAndScalars,
        UnorderedQNames,
    )


@interface_function()
def qnames_to_derive_functions_from(
    labels__input_columns: UnorderedQNames,
    labels__policy_inputs: UnorderedQNames,
    labels__grouping_levels: OrderedQNames,
) -> UnorderedQNames:
    """The qnames to derive functions from.

    Derived functions should be created based on the actual input columns, and the
    policy inputs (if their base name is not already in the input columns).
    """
    pattern_all = get_re_pattern_for_all_time_units_and_groupings(
        time_units=list(TIME_UNIT_LABELS),
        grouping_levels=labels__grouping_levels,
    )
    base_names_input_columns = {
        pattern_all.fullmatch(qn).group("base_name") for qn in labels__input_columns
    }

    out = set(labels__input_columns)
    for pi in labels__policy_inputs:
        match = pattern_all.fullmatch(pi)
        base_name = match.group("base_name")
        if base_name in base_names_input_columns:
            continue
        out.add(pi)
    return out


@interface_function()
def without_tree_logic_and_with_derived_functions(
    policy_environment: PolicyEnvironment,
    qnames_to_derive_functions_from: UnorderedQNames,
    tt_targets__qname: OrderedQNames,
    labels__input_columns: UnorderedQNames,
    labels__all_qnames_in_policy_environment: UnorderedQNames,
    labels__top_level_namespace: UnorderedQNames,
    labels__grouping_levels: OrderedQNames,
) -> SpecEnvWithoutTreeLogicAndWithDerivedFunctions:
    """A flat policy environment with derived functions.

    The difference to the corresponding function in `specialized_environment` is that
    policy inputs may be considered like actual inputs.
    """
    qname_env_without_tree_logic = _remove_tree_logic_from_policy_environment(
        qname_env=dt.flatten_to_qnames(policy_environment),
        labels__top_level_namespace=labels__top_level_namespace,
    )
    return _add_derived_functions(
        qname_env_without_tree_logic=qname_env_without_tree_logic,
        tt_targets=(
            (labels__all_qnames_in_policy_environment | set(tt_targets__qname))
            - set(labels__input_columns)
        ),
        input_columns=qnames_to_derive_functions_from,
        grouping_levels=labels__grouping_levels,
    )


@interface_function()
def without_input_data_nodes_with_dummy_callables(
    without_tree_logic_and_with_derived_functions: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,  # noqa: E501
    labels__input_columns: UnorderedQNames,
) -> SpecEnvWithoutTreeLogicAndWithDerivedFunctions:
    """An environment where non-callable are transformed into callables so we can use
    dags to set up the DAG for plotting and templates.
    """
    return {
        qn: dummy_callable(obj=n, leaf_name=dt.tree_path_from_qname(qn)[-1])
        if not callable(n)
        else n
        for qn, n in without_tree_logic_and_with_derived_functions.items()
        if qn not in labels__input_columns
    }


@interface_function()
def complete_tt_dag(
    without_input_data_nodes_with_dummy_callables: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,  # noqa: E501
    tt_targets__qname: OrderedQNames,
    labels__all_qnames_in_policy_environment: UnorderedQNames,
    labels__input_columns: UnorderedQNames,
) -> nx.DiGraph:
    """The complete DAG, which includes parameters and param_functions.

    If tt_targets are specified, only nodes that are ancestors or descendants of the
    targets are kept. If no tt_targets are specified, the entire DAG is returned.
    """
    dag = create_dag(
        functions=without_input_data_nodes_with_dummy_callables,
        # We need to specify all possible nodes as targets here because of the selection
        # types 'descendants' and 'neighbors'. In these cases 'tt_targets__qname' is not
        # the end point of the DAG, but the start point (descendants) or the middle
        # point (neighbors).
        targets=(
            (labels__all_qnames_in_policy_environment | set(tt_targets__qname))
            - set(labels__input_columns)
        ),
    )

    # If no input columns are specified, return the entire DAG
    if not labels__input_columns:
        return dag

    # Keep only nodes that are ancestors or descendants of the tt_targets. This is
    # necessary because we specified all nodes as targets in the initial DAG.
    target_nodes = set(tt_targets__qname)
    nodes_to_keep = set()

    for target in target_nodes:
        if target in dag.nodes():
            # Add the target itself
            nodes_to_keep.add(target)
            # Add all ancestors (nodes that the target depends on)
            nodes_to_keep.update(nx.ancestors(dag, target))
            # Add all descendants (nodes that depend on the target)
            nodes_to_keep.update(nx.descendants(dag, target))

    # Remove nodes that are not ancestors or descendants of any target
    dag.remove_nodes_from(set(dag.nodes()) - nodes_to_keep)

    return dag


@interface_function()
def with_processed_params_and_scalars(
    without_tree_logic_and_with_derived_functions: SpecEnvWithoutTreeLogicAndWithDerivedFunctions,  # noqa: E501
    labels__input_columns: UnorderedQNames,
    backend: Literal["numpy", "jax"],
    xnp: ModuleType,
    dnp: ModuleType,
    evaluation_date: datetime.date | None,
) -> SpecEnvWithProcessedParamsAndScalars:
    """The environment where all parameters and param functions have been processed.

    All RawParams have been removed (note that a RawParam object is pointless without a
    param function making use of it).

    The difference to the corresponding function in `specialized_environment` is that
    policy inputs may be considered like actual inputs.
    """
    return specialized_environment.with_processed_params_and_scalars(
        without_tree_logic_and_with_derived_functions=without_tree_logic_and_with_derived_functions,
        processed_data=dict.fromkeys(labels__input_columns),
        backend=backend,
        xnp=xnp,
        dnp=dnp,
        evaluation_date=evaluation_date,
    )


@interface_function()
def with_partialled_params_and_scalars(
    with_processed_params_and_scalars: SpecEnvWithProcessedParamsAndScalars,
    rounding: bool,
    backend: Literal["numpy", "jax"],
    xnp: ModuleType,
    dnp: ModuleType,
) -> SpecEnvWithPartialledParamsAndScalars:
    """The policy environment where all parameters and scalars have been partialed into
    the column functions.

    The difference to the corresponding function in `specialized_environment` is that
    policy inputs may be considered like actual inputs.
    """
    return specialized_environment.with_partialled_params_and_scalars(
        with_processed_params_and_scalars=with_processed_params_and_scalars,
        rounding=rounding,
        num_segments=1,
        backend=backend,
        xnp=xnp,
        dnp=dnp,
    )


@overload
def dummy_callable(obj: PolicyInput, leaf_name: str) -> PolicyFunction: ...


@overload
def dummy_callable(obj: ParamObject, leaf_name: str) -> ParamFunction: ...


@overload
def dummy_callable(obj: InterfaceInput, leaf_name: str) -> InterfaceFunction: ...


def dummy_callable(
    obj: PolicyInput | ParamObject | InterfaceInput, leaf_name: str
) -> Callable[[], Any]:
    """Dummy callable, for plotting and checking DAG completeness."""

    def dummy():  # type: ignore[no-untyped-def]  # noqa: ANN202
        pass

    # Extract docstring from the appropriate attribute based on object type
    if isinstance(obj, PolicyInput):
        original_docstring = obj.docstring
        if original_docstring:
            dummy.__doc__ = original_docstring
        return policy_function(
            leaf_name=leaf_name,
            start_date=obj.start_date,
            end_date=obj.end_date,
            foreign_key_type=obj.foreign_key_type,
        )(dummy)
    if isinstance(obj, ParamObject):
        # Use description["en"] for ParamObjects
        original_docstring = obj.description.get("en") if obj.description else None
        if original_docstring:
            dummy.__doc__ = original_docstring
        return param_function(
            leaf_name=leaf_name,
            start_date=obj.start_date,
            end_date=obj.end_date,
        )(dummy)
    if isinstance(obj, InterfaceInput):
        original_docstring = obj.docstring
        if original_docstring:
            dummy.__doc__ = original_docstring
        return interface_function(
            leaf_name=leaf_name,
            in_top_level_namespace=obj.in_top_level_namespace,
        )(dummy)
    return dummy
