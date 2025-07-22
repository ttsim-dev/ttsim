from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import pytest

from gettsim import main
from ttsim import copy_environment
from ttsim.interface_dag_elements import MainTarget
from ttsim.tt_dag_elements.param_objects import ScalarParam

if TYPE_CHECKING:
    from ttsim.interface_dag_elements.typing import PolicyEnvironment


def test_copy_single_scalar_param():
    """Copy a ScalarParam and verify content equality but object independence."""
    original = ScalarParam(value=0.186)
    copied = copy_environment(original)

    # Content should be identical
    assert copied.value == original.value
    assert copied.leaf_name == original.leaf_name

    # But objects should be independent
    assert copied is not original


def test_copy_nested_dict_with_params():
    """Copy nested dictionary with parameters and verify structural preservation."""
    original = {
        "level1": {
            "level2": {
                "param1": ScalarParam(value=0.5),
                "param2": ScalarParam(value=1.0),
            }
        }
    }

    copied = copy_environment(original)

    # Structure should be preserved
    assert "level1" in copied
    assert "level2" in copied["level1"]
    assert "param1" in copied["level1"]["level2"]
    assert "param2" in copied["level1"]["level2"]

    # Values should be preserved
    assert copied["level1"]["level2"]["param1"].value == 0.5
    assert copied["level1"]["level2"]["param2"].value == 1.0

    # Modifications to copy should not affect original
    copied["level1"]["level2"]["param1"] = ScalarParam(value=2.0)
    assert original["level1"]["level2"]["param1"].value == 0.5
    assert copied["level1"]["level2"]["param1"].value == 2.0


def test_copy_full_policy_environment():
    """Copy complete policy environment and verify independence of nested parameters."""
    # Load policy environment
    policy_env = main(
        date_str="2025-01-01",
        main_target=MainTarget.policy_environment,
    )

    copied_env = copy_environment(policy_env)

    # Verify structure is intact
    assert isinstance(copied_env, dict)
    assert "sozialversicherung" in copied_env
    assert "rente" in copied_env["sozialversicherung"]
    assert "beitrag" in copied_env["sozialversicherung"]["rente"]
    assert "beitragssatz" in copied_env["sozialversicherung"]["rente"]["beitrag"]

    # Get reference to nested parameter in both versions
    original_param = policy_env["sozialversicherung"]["rente"]["beitrag"][
        "beitragssatz"
    ]
    copied_param = copied_env["sozialversicherung"]["rente"]["beitrag"]["beitragssatz"]

    # Values should be equal initially
    assert copied_param.value == original_param.value

    # Modify copy - should not affect original
    copied_env["sozialversicherung"]["rente"]["beitrag"]["beitragssatz"] = ScalarParam(
        value=0.3
    )

    assert (
        policy_env["sozialversicherung"]["rente"]["beitrag"]["beitragssatz"].value
        == original_param.value
    )
    assert (
        copied_env["sozialversicherung"]["rente"]["beitrag"]["beitragssatz"].value
        == 0.3
    )


def test_deepcopy_fails_on_policy_environment():
    """Verify copy.deepcopy fails on policy environments due to unpickleable objects."""
    policy_env = main(
        date_str="2025-01-01",
        main_target=MainTarget.policy_environment,
    )

    with pytest.raises((TypeError, AttributeError)) as excinfo:
        deepcopy(policy_env)

    # Should be a pickling-related error
    error_message = str(excinfo.value).lower()
    assert any(word in error_message for word in ["pickle", "module"])


def test_copy_environment_works_where_deepcopy_fails():
    """Verify copy_environment succeeds on objects that break copy.deepcopy."""
    policy_env = main(
        date_str="2025-01-01",
        main_target=MainTarget.policy_environment,
    )

    # Confirm deepcopy fails
    with pytest.raises((TypeError, AttributeError)):
        deepcopy(policy_env)

    # But copy_environment should work
    copied_env = copy_environment(policy_env)
    assert isinstance(copied_env, dict)
    assert len(copied_env) > 0


def test_copy_empty_dict():
    """Copy empty dictionary and verify independence."""
    original: dict[str, Any] = {}
    copied = copy_environment(original)

    assert copied == {}
    assert copied is not original


def test_copy_mixed_data_types():
    """Copy dictionary with various data types and verify all are handled correctly."""
    original: dict[str, Any] = {
        "scalar": ScalarParam(value=42),
        "string": "hello",
        "number": 123,
        "list": [1, 2, 3],
        "nested": {
            "param": ScalarParam(value=3.14),
            "bool": True,
        },
    }

    copied = copy_environment(original)

    # All types should be preserved
    assert copied["scalar"].value == 42
    assert copied["string"] == "hello"
    assert copied["number"] == 123
    assert copied["list"] == [1, 2, 3]
    assert copied["nested"]["param"].value == 3.14
    assert copied["nested"]["bool"] is True

    # Modifications should be independent
    copied["nested"]["param"] = ScalarParam(value=2.71)
    assert original["nested"]["param"].value == 3.14
    assert copied["nested"]["param"].value == 2.71


def test_policy_environment_type_inference():
    """Verify type hints work correctly for PolicyEnvironment input/output."""
    policy_env = main(
        date_str="2025-01-01",
        main_target=MainTarget.policy_environment,
    )

    # Type checker should infer PolicyEnvironment -> PolicyEnvironment
    copied_env: PolicyEnvironment = copy_environment(policy_env)

    # Function should work correctly
    assert isinstance(copied_env, dict)
    assert "sozialversicherung" in copied_env
