from __future__ import annotations

from pathlib import Path

import ttsim.interface_dag_elements
from ttsim.interface_dag_elements.interface_node_objects import InterfaceFunction
from ttsim.interface_dag_elements.orig_policy_objects import load_module

# The package claw rewrites normally-imported `interface_dag_elements` modules,
# binding each callable `InterfaceFunction` instance into a bound method. Re-load
# the module via `load_module` — claw-free, exactly how the interface DAG obtains
# these objects — to assert on the pristine instance.
_IFACE_DIR = Path(ttsim.interface_dag_elements.__file__).parent
len_p_id = load_module(_IFACE_DIR / "len_p_id.py", _IFACE_DIR).len_p_id


def test_len_p_id_is_interface_function():
    assert isinstance(len_p_id, InterfaceFunction)


def test_len_p_id_in_top_level_namespace():
    assert len_p_id.in_top_level_namespace is True


def test_len_p_id_returns_data_length(xnp):
    processed_data = {
        "p_id": xnp.array([0, 1, 2, 3, 4]),
        "income": xnp.array([100, 200, 300, 400, 500]),
    }

    result = len_p_id(processed_data)

    assert result == 5


def test_len_p_id_single_row(xnp):
    processed_data = {
        "p_id": xnp.array([0]),
    }

    result = len_p_id(processed_data)

    assert result == 1


def test_len_p_id_empty_data_returns_sentinel():
    """When processed_data is empty, return 11111 as a recognizable sentinel value."""
    processed_data = {}

    result = len_p_id(processed_data)

    # The sentinel value is used for jittability tests
    assert result == 11111


def test_len_p_id_uses_first_array_length(xnp):
    """len_p_id should use the length of the first value in processed_data."""
    processed_data = {
        "first_col": xnp.array([0, 1, 2]),
        "second_col": xnp.array([0, 1, 2]),
    }

    result = len_p_id(processed_data)

    assert result == 3


def test_len_p_id_dependencies():
    assert len_p_id.dependencies == {"processed_data"}


def test_len_p_id_leaf_name():
    assert len_p_id.leaf_name == "len_p_id"
