from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from ttsim.interface_dag_elements.labels import (
    all_qnames_in_policy_environment,
    column_targets,
    grouping_levels,
    input_columns_from_input_data,
    input_columns_is_empty_set,
    input_data_targets,
    param_targets,
    policy_inputs,
    root_nodes,
    top_level_namespace,
)
from ttsim.tt import param_function, policy_function, policy_input


def identity(x: int) -> int:
    return x


@policy_input()
def fam_id() -> int:
    pass


@pytest.mark.parametrize(
    (
        "policy_environment",
        "expected",
    ),
    [
        (
            {
                "foo_m": policy_function(leaf_name="foo_m")(identity),
                "fam_id": fam_id,
            },
            {"foo_m", "foo_y", "foo_m_fam", "foo_y_fam"},
        ),
        (
            {
                "foo": policy_function(leaf_name="foo")(identity),
                "fam_id": fam_id,
            },
            {"foo", "foo_fam"},
        ),
    ],
)
def test_get_top_level_namespace(policy_environment, expected):
    result = top_level_namespace(
        policy_environment=policy_environment,
        grouping_levels=grouping_levels(policy_environment),
    )
    assert all(name in result for name in expected)


# =============================================================================
# grouping_levels tests
# =============================================================================
def test_grouping_levels_extracts_id_columns():
    @policy_input()
    def p_id() -> int:
        pass

    @policy_input()
    def hh_id() -> int:
        pass

    @policy_input()
    def fam_id() -> int:
        pass

    policy_env = {
        "p_id": p_id,
        "hh_id": hh_id,
        "fam_id": fam_id,
    }

    result = grouping_levels(policy_env)

    # p_id should not be included (it's the primary identifier)
    assert "p" not in result
    # hh and fam should be included
    assert "hh" in result
    assert "fam" in result


def test_grouping_levels_excludes_p_id():
    @policy_input()
    def p_id() -> int:
        pass

    policy_env = {"p_id": p_id}
    result = grouping_levels(policy_env)

    assert "p" not in result
    assert len(result) == 0


def test_grouping_levels_empty_environment():
    result = grouping_levels({})
    assert result == ()


# =============================================================================
# input_columns_from_input_data tests
# =============================================================================
def test_input_columns_from_input_data_returns_processed_data_keys():
    processed_data = {
        "p_id": [0, 1, 2],
        "income": [100, 200, 300],
    }

    result = input_columns_from_input_data(processed_data=processed_data)

    assert result == {"p_id", "income"}


def test_input_columns_from_input_data_empty():
    result = input_columns_from_input_data(processed_data={})
    assert result == set()


# =============================================================================
# input_columns_is_empty_set tests
# =============================================================================
def test_input_columns_is_empty_set_returns_empty():
    result = input_columns_is_empty_set(xnp=np)
    assert result == set()


# =============================================================================
# all_qnames_in_policy_environment tests
# =============================================================================
def test_all_qnames_in_policy_environment_flat():
    @policy_function()
    def col_a(x: int) -> int:
        return x

    @policy_input()
    def col_b() -> int:
        pass

    policy_env = {"col_a": col_a, "col_b": col_b}
    result = all_qnames_in_policy_environment(policy_env)

    assert "col_a" in result
    assert "col_b" in result


def test_all_qnames_in_policy_environment_nested():
    @policy_function()
    def nested_col(x: int) -> int:
        return x

    policy_env = {"namespace": {"nested_col": nested_col}}
    result = all_qnames_in_policy_environment(policy_env)

    assert "namespace__nested_col" in result


# =============================================================================
# policy_inputs tests
# =============================================================================
def test_policy_inputs_returns_only_policy_input_qnames():
    @policy_function()
    def col_func(x: int) -> int:
        return x

    @policy_input()
    def input_col() -> int:
        pass

    @param_function()
    def param_func() -> int:
        return 42

    policy_env = {
        "col_func": col_func,
        "input_col": input_col,
        "param_func": param_func,
    }

    result = policy_inputs(policy_env)

    assert "input_col" in result
    assert "col_func" not in result
    assert "param_func" not in result


def test_policy_inputs_handles_nested():
    @policy_input()
    def nested_input() -> int:
        pass

    policy_env = {"namespace": {"nested_input": nested_input}}
    result = policy_inputs(policy_env)

    assert "namespace__nested_input" in result


# =============================================================================
# root_nodes tests
# =============================================================================
def test_root_nodes_filters_to_dag_roots():
    # Create a simple DAG where only "input_a" has no incoming edges
    dag = nx.DiGraph()
    dag.add_edge("input_a", "intermediate")
    dag.add_edge("intermediate", "output")

    input_columns = {"input_a", "intermediate", "output", "extra_input"}

    result = root_nodes(
        specialized_environment__tt_dag=dag,
        input_columns=input_columns,
    )

    # Only input_a is a root node (in_degree == 0) AND in input_columns
    assert result == {"input_a"}


def test_root_nodes_excludes_non_input_roots():
    # Root node that's not in input_columns should be excluded
    dag = nx.DiGraph()
    dag.add_node("root_not_in_input")
    dag.add_edge("root_in_input", "output")

    input_columns = {"root_in_input"}

    result = root_nodes(
        specialized_environment__tt_dag=dag,
        input_columns=input_columns,
    )

    assert "root_in_input" in result
    assert "root_not_in_input" not in result


def test_root_nodes_empty_dag():
    dag = nx.DiGraph()
    input_columns = {"some_input"}

    result = root_nodes(
        specialized_environment__tt_dag=dag,
        input_columns=input_columns,
    )

    assert result == set()


def test_root_nodes_multiple_roots():
    dag = nx.DiGraph()
    dag.add_edge("root_a", "output")
    dag.add_edge("root_b", "output")

    input_columns = {"root_a", "root_b", "output"}

    result = root_nodes(
        specialized_environment__tt_dag=dag,
        input_columns=input_columns,
    )

    assert result == {"root_a", "root_b"}


# =============================================================================
# input_data_targets tests
# =============================================================================
def test_input_data_targets_filters_to_input_columns():
    tt_targets = ["col_a", "col_b", "col_c"]
    input_columns = {"col_a", "col_c"}

    result = input_data_targets(
        tt_targets__qname=tt_targets,
        input_columns=input_columns,
    )

    assert "col_a" in result
    assert "col_c" in result
    assert "col_b" not in result


def test_input_data_targets_preserves_order():
    tt_targets = ["z", "a", "m"]
    input_columns = {"z", "a", "m"}

    result = input_data_targets(
        tt_targets__qname=tt_targets,
        input_columns=input_columns,
    )

    assert result == ["z", "a", "m"]


def test_input_data_targets_empty_intersection():
    tt_targets = ["col_a", "col_b"]
    input_columns = {"col_c", "col_d"}

    result = input_data_targets(
        tt_targets__qname=tt_targets,
        input_columns=input_columns,
    )

    assert result == []


# =============================================================================
# column_targets tests
# =============================================================================
def test_column_targets_excludes_input_data_targets():
    specialized_env = {
        "col_a": identity,
        "col_b": identity,
        "input_col": identity,
    }
    tt_targets = ["col_a", "col_b", "input_col"]
    input_data_targets_list = ["input_col"]

    result = column_targets(
        specialized_environment__with_partialled_params_and_scalars=specialized_env,
        tt_targets__qname=tt_targets,
        input_data_targets=input_data_targets_list,
    )

    assert "col_a" in result
    assert "col_b" in result
    assert "input_col" not in result


def test_column_targets_filters_to_specialized_environment():
    specialized_env = {
        "col_a": identity,
        # col_b not in specialized_env
    }
    tt_targets = ["col_a", "col_b"]
    input_data_targets_list = []

    result = column_targets(
        specialized_environment__with_partialled_params_and_scalars=specialized_env,
        tt_targets__qname=tt_targets,
        input_data_targets=input_data_targets_list,
    )

    assert "col_a" in result
    assert "col_b" not in result


def test_column_targets_preserves_order():
    specialized_env = {"z": identity, "a": identity, "m": identity}
    tt_targets = ["z", "a", "m"]
    input_data_targets_list = []

    result = column_targets(
        specialized_environment__with_partialled_params_and_scalars=specialized_env,
        tt_targets__qname=tt_targets,
        input_data_targets=input_data_targets_list,
    )

    assert result == ["z", "a", "m"]


# =============================================================================
# param_targets tests
# =============================================================================
def test_param_targets_excludes_column_and_input_targets():
    specialized_env = {
        "param_a": 100,
        "param_b": 200,
        "col_func": identity,
        "input_col": identity,
    }
    tt_targets = ["param_a", "param_b", "col_func", "input_col"]
    column_targets_list = ["col_func"]
    input_data_targets_list = ["input_col"]

    result = param_targets(
        specialized_environment__without_tree_logic_and_with_derived_functions=specialized_env,
        tt_targets__qname=tt_targets,
        column_targets=column_targets_list,
        input_data_targets=input_data_targets_list,
    )

    assert "param_a" in result
    assert "param_b" in result
    assert "col_func" not in result
    assert "input_col" not in result


def test_param_targets_filters_to_specialized_environment():
    specialized_env = {
        "param_a": 100,
        # param_b not in specialized_env
    }
    tt_targets = ["param_a", "param_b"]
    column_targets_list = []
    input_data_targets_list = []

    result = param_targets(
        specialized_environment__without_tree_logic_and_with_derived_functions=specialized_env,
        tt_targets__qname=tt_targets,
        column_targets=column_targets_list,
        input_data_targets=input_data_targets_list,
    )

    assert "param_a" in result
    assert "param_b" not in result


def test_param_targets_preserves_order():
    specialized_env = {"z": 1, "a": 2, "m": 3}
    tt_targets = ["z", "a", "m"]
    column_targets_list = []
    input_data_targets_list = []

    result = param_targets(
        specialized_environment__without_tree_logic_and_with_derived_functions=specialized_env,
        tt_targets__qname=tt_targets,
        column_targets=column_targets_list,
        input_data_targets=input_data_targets_list,
    )

    assert result == ["z", "a", "m"]


def test_param_targets_empty_when_all_excluded():
    specialized_env = {
        "col_func": identity,
        "input_col": identity,
    }
    tt_targets = ["col_func", "input_col"]
    column_targets_list = ["col_func"]
    input_data_targets_list = ["input_col"]

    result = param_targets(
        specialized_environment__without_tree_logic_and_with_derived_functions=specialized_env,
        tt_targets__qname=tt_targets,
        column_targets=column_targets_list,
        input_data_targets=input_data_targets_list,
    )

    assert result == []
