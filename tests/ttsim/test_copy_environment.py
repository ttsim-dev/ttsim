"""Tests for the copy_environment function."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from gettsim import main
from ttsim import copy_environment
from ttsim.interface_dag_elements import MainTarget
from ttsim.tt_dag_elements.param_objects import ScalarParam


class TestCopyEnvironment:
    """Tests for the copy_environment function."""

    def test_copy_simple_scalar_param(self) -> None:
        """Test copying a single ScalarParam object."""
        original = ScalarParam(value=0.186)
        copied = copy_environment(original)

        # Should be equal in content
        assert copied.value == original.value
        assert copied.leaf_name == original.leaf_name

        # But should be different objects
        assert copied is not original

    def test_copy_nested_dict_with_params(self) -> None:
        """Test copying a nested dictionary containing parameter objects."""
        original = {
            "level1": {
                "level2": {
                    "param1": ScalarParam(value=0.5),
                    "param2": ScalarParam(value=1.0),
                }
            }
        }

        copied = copy_environment(original)

        # Check structure is preserved
        assert "level1" in copied
        assert "level2" in copied["level1"]
        assert "param1" in copied["level1"]["level2"]
        assert "param2" in copied["level1"]["level2"]

        # Check parameter values are preserved
        assert copied["level1"]["level2"]["param1"].value == 0.5
        assert copied["level1"]["level2"]["param2"].value == 1.0

        # Check independence - modifying copy shouldn't affect original
        copied["level1"]["level2"]["param1"] = ScalarParam(value=2.0)
        assert original["level1"]["level2"]["param1"].value == 0.5
        assert copied["level1"]["level2"]["param1"].value == 2.0

    def test_copy_policy_environment(self) -> None:
        """Test copying a full policy environment (integration test)."""
        # Load a policy environment
        policy_env = main(
            date_str="2025-01-01",
            main_target=MainTarget.policy_environment,
        )

        # Copy it
        copied_env = copy_environment(policy_env)

        # Check that basic structure is preserved
        assert isinstance(copied_env, dict)
        assert "sozialversicherung" in copied_env
        assert "rente" in copied_env["sozialversicherung"]
        assert "beitrag" in copied_env["sozialversicherung"]["rente"]
        assert "beitragssatz" in copied_env["sozialversicherung"]["rente"]["beitrag"]

        # Get the nested parameter
        original_param = policy_env["sozialversicherung"]["rente"]["beitrag"][
            "beitragssatz"
        ]
        copied_param = copied_env["sozialversicherung"]["rente"]["beitrag"][
            "beitragssatz"
        ]

        # Check values are equal
        assert copied_param.value == original_param.value

        # Test independence by modification
        new_param = ScalarParam(value=0.3)
        copied_env["sozialversicherung"]["rente"]["beitrag"]["beitragssatz"] = new_param

        # Original should be unchanged
        assert (
            policy_env["sozialversicherung"]["rente"]["beitrag"]["beitragssatz"].value
            == original_param.value
        )
        # Copy should be changed
        assert (
            copied_env["sozialversicherung"]["rente"]["beitrag"]["beitragssatz"].value
            == 0.3
        )

    def test_deepcopy_fails_on_policy_environment(self) -> None:
        """Verify that copy.deepcopy fails on policy environments (regression test)."""
        # Load a policy environment
        policy_env = main(
            date_str="2025-01-01",
            main_target=MainTarget.policy_environment,
        )

        # copy.deepcopy should fail
        with pytest.raises((TypeError, AttributeError)) as excinfo:
            deepcopy(policy_env)

        # Should be a pickling-related error
        assert any(word in str(excinfo.value).lower() for word in ["pickle", "module"])

    def test_copy_environment_succeeds_where_deepcopy_fails(self) -> None:
        """Test that copy_environment works on objects where deepcopy fails."""
        # Load a policy environment that deepcopy can't handle
        policy_env = main(
            date_str="2025-01-01",
            main_target=MainTarget.policy_environment,
        )

        # Verify deepcopy fails
        with pytest.raises((TypeError, AttributeError)):
            deepcopy(policy_env)

        # But copy_environment should work
        copied_env = copy_environment(policy_env)

        # And should be functional
        assert isinstance(copied_env, dict)
        assert len(copied_env) > 0

    def test_copy_empty_dict(self) -> None:
        """Test copying an empty dictionary."""
        original: dict[str, Any] = {}
        copied = copy_environment(original)

        assert copied == {}
        assert copied is not original

    def test_copy_mixed_types(self) -> None:
        """Test copying a tree with mixed object types."""
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

        # Check all types are preserved
        assert copied["scalar"].value == 42
        assert copied["string"] == "hello"
        assert copied["number"] == 123
        assert copied["list"] == [1, 2, 3]
        assert copied["nested"]["param"].value == 3.14
        assert copied["nested"]["bool"] is True

        # Check independence
        copied["nested"]["param"] = ScalarParam(value=2.71)
        assert original["nested"]["param"].value == 3.14
        assert copied["nested"]["param"].value == 2.71
