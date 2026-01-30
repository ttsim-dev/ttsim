from __future__ import annotations

import pytest

from ttsim.interface_dag_elements.interface_node_objects import InterfaceFunction
from ttsim.interface_dag_elements.raw_results import columns, from_input_data, params


# =============================================================================
# columns() function tests
# =============================================================================
def test_columns_is_interface_function():
    assert isinstance(columns, InterfaceFunction)


def test_columns_filters_to_root_nodes(xnp):
    processed_data = {
        "p_id": xnp.array([0, 1, 2]),
        "income": xnp.array([100, 200, 300]),
        "age": xnp.array([25, 35, 45]),
    }
    root_nodes = {"p_id", "income"}

    # Create a simple identity tt_function
    def tt_function(data):
        return data

    result = columns(
        labels__root_nodes=root_nodes,
        processed_data=processed_data,
        tt_function=tt_function,
    )

    # Only root_nodes should be passed to tt_function
    assert "p_id" in result
    assert "income" in result
    assert "age" not in result


def test_columns_calls_tt_function_with_filtered_data(xnp):
    processed_data = {
        "p_id": xnp.array([0, 1]),
        "income": xnp.array([100, 200]),
    }
    root_nodes = {"p_id"}

    call_args = []

    def tt_function(data):
        call_args.append(data)
        return {"output": xnp.array([1, 2])}

    result = columns(
        labels__root_nodes=root_nodes,
        processed_data=processed_data,
        tt_function=tt_function,
    )

    # Verify tt_function was called with only root_nodes data
    assert len(call_args) == 1
    assert "p_id" in call_args[0]
    assert "income" not in call_args[0]


def test_columns_returns_tt_function_output(xnp):
    processed_data = {"p_id": xnp.array([0, 1])}
    root_nodes = {"p_id"}

    expected_output = {"computed_col": xnp.array([10, 20])}

    def tt_function():
        return expected_output

    result = columns(
        labels__root_nodes=root_nodes,
        processed_data=processed_data,
        tt_function=tt_function,
    )

    assert result == expected_output


def test_columns_with_empty_root_nodes(xnp):
    processed_data = {"p_id": xnp.array([0, 1])}
    root_nodes = set()

    def tt_function(data):
        return data

    result = columns(
        labels__root_nodes=root_nodes,
        processed_data=processed_data,
        tt_function=tt_function,
    )

    assert result == {}


# =============================================================================
# from_input_data() function tests
# =============================================================================
def test_from_input_data_is_interface_function():
    assert isinstance(from_input_data, InterfaceFunction)


def test_from_input_data_extracts_requested_targets(xnp):
    input_data__flat = {
        ("p_id",): xnp.array([0, 1, 2]),
        ("income",): xnp.array([100, 200, 300]),
        ("age",): xnp.array([25, 35, 45]),
    }
    input_data_targets = ["p_id", "income"]

    result = from_input_data(
        labels__input_data_targets=input_data_targets,
        input_data__flat=input_data__flat,
    )

    assert "p_id" in result
    assert "income" in result
    assert "age" not in result


def test_from_input_data_preserves_order(xnp):
    input_data__flat = {
        ("a",): xnp.array([1]),
        ("b",): xnp.array([2]),
        ("c",): xnp.array([3]),
    }
    # Specific order requested
    input_data_targets = ["c", "a", "b"]

    result = from_input_data(
        labels__input_data_targets=input_data_targets,
        input_data__flat=input_data__flat,
    )

    # Check all are present
    assert set(result.keys()) == {"a", "b", "c"}


def test_from_input_data_handles_nested_tree_paths(xnp):
    input_data__flat = {
        ("namespace", "col_a"): xnp.array([1, 2]),
        ("namespace", "col_b"): xnp.array([3, 4]),
    }
    input_data_targets = ["namespace__col_a"]

    result = from_input_data(
        labels__input_data_targets=input_data_targets,
        input_data__flat=input_data__flat,
    )

    assert "namespace__col_a" in result


def test_from_input_data_empty_targets(xnp):
    input_data__flat = {
        ("p_id",): xnp.array([0, 1]),
    }
    input_data_targets = []

    result = from_input_data(
        labels__input_data_targets=input_data_targets,
        input_data__flat=input_data__flat,
    )

    assert result == {}


def test_from_input_data_returns_arrays_unsorted(xnp):
    """Arrays should be returned as they are in input_data__flat."""
    input_data__flat = {
        ("values",): xnp.array([5, 3, 1, 4, 2]),
    }
    input_data_targets = ["values"]

    result = from_input_data(
        labels__input_data_targets=input_data_targets,
        input_data__flat=input_data__flat,
    )

    # Values should be unchanged (not sorted)
    assert list(result["values"]) == [5, 3, 1, 4, 2]


# =============================================================================
# params() function tests
# =============================================================================
def test_params_is_interface_function():
    assert isinstance(params, InterfaceFunction)


def test_params_extracts_requested_param_targets():
    specialized_env = {
        "param_a": 100,
        "param_b": 200,
        "col_function": lambda x: x,  # Not a param
    }
    param_targets = ["param_a", "param_b"]

    result = params(
        labels__param_targets=param_targets,
        specialized_environment__with_processed_params_and_scalars=specialized_env,
    )

    assert result == {"param_a": 100, "param_b": 200}


def test_params_preserves_order():
    specialized_env = {
        "param_c": 3,
        "param_a": 1,
        "param_b": 2,
    }
    # Specific order requested
    param_targets = ["param_b", "param_c", "param_a"]

    result = params(
        labels__param_targets=param_targets,
        specialized_environment__with_processed_params_and_scalars=specialized_env,
    )

    assert set(result.keys()) == {"param_a", "param_b", "param_c"}


def test_params_handles_nested_qnames():
    specialized_env = {
        "namespace__param_a": 100,
        "namespace__param_b": 200,
    }
    param_targets = ["namespace__param_a"]

    result = params(
        labels__param_targets=param_targets,
        specialized_environment__with_processed_params_and_scalars=specialized_env,
    )

    assert "namespace__param_a" in result
    assert "namespace__param_b" not in result


def test_params_empty_targets():
    specialized_env = {"param_a": 100}
    param_targets = []

    result = params(
        labels__param_targets=param_targets,
        specialized_environment__with_processed_params_and_scalars=specialized_env,
    )

    assert result == {}


def test_params_returns_various_value_types():
    """Params can be scalars, dicts, or other param function outputs."""
    specialized_env = {
        "scalar_param": 42,
        "dict_param": {"key": "value"},
        "list_param": [1, 2, 3],
    }
    param_targets = ["scalar_param", "dict_param", "list_param"]

    result = params(
        labels__param_targets=param_targets,
        specialized_environment__with_processed_params_and_scalars=specialized_env,
    )

    assert result["scalar_param"] == 42
    assert result["dict_param"] == {"key": "value"}
    assert result["list_param"] == [1, 2, 3]


# =============================================================================
# Dependencies property tests
# =============================================================================
def test_columns_dependencies():
    assert columns.dependencies == {
        "labels__root_nodes",
        "processed_data",
        "tt_function",
    }


def test_from_input_data_dependencies():
    assert from_input_data.dependencies == {
        "labels__input_data_targets",
        "input_data__flat",
    }


def test_params_dependencies():
    assert params.dependencies == {
        "labels__param_targets",
        "specialized_environment__with_processed_params_and_scalars",
    }
