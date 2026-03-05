from __future__ import annotations

from ttsim.interface_dag_elements.interface_node_objects import InterfaceFunction
from ttsim.interface_dag_elements.tt_targets import qname, tree
from ttsim.tt import param_function, policy_function, policy_input


def identity(x: int) -> int:
    return x


# =============================================================================
# tree() function tests
# =============================================================================
def test_tree_is_interface_function():
    assert isinstance(tree, InterfaceFunction)


def test_tree_extracts_policy_functions_from_environment():
    """tree() extracts ColumnFunction instances (e.g., PolicyFunction)."""

    @policy_function()
    def my_col(x: int) -> int:
        return x

    policy_env = {
        "my_col": my_col,
    }

    result = tree(policy_env)

    assert "my_col" in result


def test_tree_excludes_policy_inputs():
    """tree() only includes ColumnFunction instances, not PolicyInput."""

    @policy_function()
    def my_col(x: int) -> int:
        return x

    @policy_input()
    def my_input() -> int:
        pass

    policy_env = {
        "my_col": my_col,
        "my_input": my_input,
    }

    result = tree(policy_env)

    # PolicyFunction is included (it's a ColumnFunction)
    assert "my_col" in result
    # PolicyInput is NOT included (it's a ColumnObject but not a ColumnFunction)
    assert "my_input" not in result


def test_tree_excludes_param_functions():
    @policy_function()
    def my_col(x: int) -> int:
        return x

    @param_function()
    def my_param() -> int:
        return 42

    policy_env = {
        "my_col": my_col,
        "my_param": my_param,
    }

    result = tree(policy_env)

    assert "my_col" in result
    assert "my_param" not in result


def test_tree_handles_nested_policy_environment():
    @policy_function()
    def nested_col(x: int) -> int:
        return x

    policy_env = {
        "namespace1": {
            "nested_col": nested_col,
        }
    }

    result = tree(policy_env)

    assert "namespace1" in result
    assert "nested_col" in result["namespace1"]


def test_tree_returns_none_as_leaf_values():
    @policy_function()
    def my_col(x: int) -> int:
        return x

    policy_env = {"my_col": my_col}
    result = tree(policy_env)

    assert result["my_col"] is None


def test_tree_empty_environment():
    result = tree({})
    assert result == {}


def test_tree_environment_with_only_param_functions():
    @param_function()
    def my_param() -> int:
        return 42

    policy_env = {"my_param": my_param}
    result = tree(policy_env)

    assert result == {}


def test_tree_environment_with_only_policy_inputs():
    @policy_input()
    def my_input() -> int:
        pass

    policy_env = {"my_input": my_input}
    result = tree(policy_env)

    # PolicyInput is not a ColumnFunction, so result should be empty
    assert result == {}


# =============================================================================
# qname() function tests
# =============================================================================
def test_qname_is_interface_function():
    assert isinstance(qname, InterfaceFunction)


def test_qname_flattens_tree_to_qnames():
    target_tree = {
        "col_a": None,
        "col_b": None,
    }

    result = qname(target_tree)

    assert "col_a" in result
    assert "col_b" in result


def test_qname_handles_nested_tree():
    target_tree = {
        "namespace1": {
            "col_a": None,
        },
        "col_b": None,
    }

    result = qname(target_tree)

    assert "namespace1__col_a" in result
    assert "col_b" in result


def test_qname_handles_deeply_nested_tree():
    target_tree = {
        "level1": {
            "level2": {
                "level3": {
                    "col": None,
                },
            },
        },
    }

    result = qname(target_tree)

    assert "level1__level2__level3__col" in result


def test_qname_handles_empty_tree():
    """qname returns a dict-like object (empty dict) for empty tree input."""
    result = qname({})
    # The result is an empty dict-like object, not a list
    assert len(result) == 0


def test_qname_preserves_order():
    """The qname function should return an ordered collection."""
    target_tree = {
        "col_a": None,
        "col_b": None,
        "col_c": None,
    }

    result = qname(target_tree)

    # Result should be iterable and contain all keys
    result_list = list(result)
    assert len(result_list) == 3
    assert set(result_list) == {"col_a", "col_b", "col_c"}


def test_qname_handles_mixed_flat_and_nested():
    target_tree = {
        "flat_col": None,
        "namespace": {
            "nested_col": None,
        },
    }

    result = qname(target_tree)

    assert "flat_col" in result
    assert "namespace__nested_col" in result


# =============================================================================
# tree + qname
# =============================================================================
def test_tree_then_qname_extracts_column_function_qnames():
    @policy_function()
    def col_a(x: int) -> int:
        return x

    @policy_function()
    def col_b(x: int) -> int:
        return x

    @policy_input()
    def input_col() -> int:
        pass

    @param_function()
    def param_func() -> int:
        return 42

    policy_env = {
        "col_a": col_a,
        "col_b": col_b,
        "input_col": input_col,
        "param_func": param_func,
    }

    target_tree = tree(policy_env)
    target_qnames = qname(target_tree)

    # PolicyFunction included (ColumnFunction)
    assert "col_a" in target_qnames
    assert "col_b" in target_qnames

    # PolicyInput not included (ColumnObject but not ColumnFunction)
    assert "input_col" not in target_qnames

    # Param function excluded
    assert "param_func" not in target_qnames


def test_tree_then_qname_with_nested_policy_environment():
    @policy_function()
    def nested_col(x: int) -> int:
        return x

    policy_env = {
        "namespace": {
            "nested_col": nested_col,
        }
    }

    target_tree = tree(policy_env)
    target_qnames = qname(target_tree)

    assert "namespace__nested_col" in target_qnames
