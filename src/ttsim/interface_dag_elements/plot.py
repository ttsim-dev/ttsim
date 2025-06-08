from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import dags
import dags.tree as dt

from ttsim.interface_dag_elements.interface_node_objects import (
    interface_function,
    interface_input,
)
from ttsim.plot_dag import plot_dag
from ttsim.tt_dag_elements import ParamObject, PolicyInput

if TYPE_CHECKING:
    from pathlib import Path

    from ttsim.interface_dag_elements.typing import (
        NestedPolicyEnvironment,
        UnorderedQNames,
    )

from ttsim.interface_dag_elements.fail_if import format_list_linewise


@interface_input()
def full_policy_environment_path() -> Path:
    """The path to save the full policy environment DAG in."""


@interface_function()
def full_policy_environment(
    policy_environment: NestedPolicyEnvironment,
    full_policy_environment_path: Path,
    names__top_level_namespace: UnorderedQNames,
) -> None:
    """Plot the full policy environment DAG."""

    tree_path_nodes = {
        p: n.dummy_callable() if isinstance(n, PolicyInput | ParamObject) else n
        for p, n in dt.flatten_to_tree_paths(policy_environment).items()
    }
    # TODO: Must call `with_derived_functions_and_processed_input_nodes`
    # Better approach though: Make it dependend on that instead of policy environment
    # and parameterize the overall call accordingly

    nodes = {}
    for tree_path, node in tree_path_nodes.items():
        nodes[dt.qual_name_from_tree_path(tree_path)] = node.remove_tree_logic(
            tree_path=tree_path,
            top_level_namespace=names__top_level_namespace,
        )

    dag = dags.create_dag(functions=nodes, targets=None)
    f = dags.concatenate_functions(
        dag=dag,
        functions=nodes,
        targets=None,
        return_type="dict",
        enforce_signature=False,
        set_annotations=False,
    )
    args = inspect.signature(f).parameters
    if args:
        raise ValueError(
            "The policy environment DAG should include all root nodes but requires "
            f"inputs:\n\n{format_list_linewise(args.keys())}"
        )
    fig = plot_dag(dag)
    if full_policy_environment_path.suffix == ".html":
        fig.write_html(full_policy_environment_path)
    else:
        raise ValueError(
            f"Unsupported file extension: {full_policy_environment_path.suffix}"
        )
