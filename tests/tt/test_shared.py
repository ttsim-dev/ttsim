from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy
import optree
import pytest
from mettsim import middle_earth

from ttsim import MainTarget, OrigPolicyObjects, copy_environment, main
from ttsim.tt import RawParam, ScalarParam, join

if TYPE_CHECKING:
    from types import ModuleType

    from ttsim.typing import IntColumn, PolicyEnvironment


@pytest.mark.parametrize(
    (
        "foreign_key",
        "primary_key",
        "target",
        "value_if_foreign_key_is_missing",
        "expected",
    ),
    [
        (
            numpy.array([1, 2, 3]),
            numpy.array([1, 2, 3]),
            numpy.array([1, 2, 3]),
            4,
            numpy.array([1, 2, 3]),
        ),
        (
            numpy.array([3, 2, 1]),
            numpy.array([1, 2, 3]),
            numpy.array([1, 2, 3]),
            4,
            numpy.array([3, 2, 1]),
        ),
        (
            numpy.array([1, 1, 1]),
            numpy.array([1, 2, 3]),
            numpy.array([1, 2, 3]),
            4,
            numpy.array([1, 1, 1]),
        ),
        (
            numpy.array([-1]),
            numpy.array([1]),
            numpy.array([1]),
            4,
            numpy.array([4]),
        ),
    ],
)
def test_join(
    foreign_key: IntColumn,
    primary_key: IntColumn,
    target: IntColumn,
    value_if_foreign_key_is_missing: int,
    expected: IntColumn,
    xnp: ModuleType,
):
    assert numpy.array_equal(
        join(
            foreign_key=xnp.asarray(foreign_key),
            primary_key=xnp.asarray(primary_key),
            target=xnp.asarray(target),
            value_if_foreign_key_is_missing=value_if_foreign_key_is_missing,
            xnp=xnp,
        ),
        expected,
    )


def test_copy_single_scalar_param():
    """Copy a ScalarParam and verify content equality but object independence."""
    original = {"param": ScalarParam(value=0.186)}
    copied = copy_environment(original)

    # Content should be identical
    assert copied["param"].value == original["param"].value

    # But objects should be independent
    assert copied["param"] is not original["param"]


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
    # Load policy environment (mettsim)
    policy_env = main(
        main_target=MainTarget.policy_environment,
        policy_date_str="2025-01-01",
        orig_policy_objects=OrigPolicyObjects.root(middle_earth.ROOT_PATH),
    )

    copied_env = copy_environment(policy_env)

    # Verify skeletons (tree structure) are identical
    assert set(optree.tree_paths(policy_env)) == set(optree.tree_paths(copied_env))

    # Get reference to nested parameter in both versions
    original_param = policy_env["orc_hunting_bounty"]["raw_bounties_per_orc"]
    copied_param = copied_env["orc_hunting_bounty"]["raw_bounties_per_orc"]

    # Values should be equal initially
    assert copied_param.value == original_param.value

    # Modify copy - should not affect original
    copied_env["orc_hunting_bounty"]["raw_bounties_per_orc"] = RawParam(
        value={
            "small_orc": 500,
            "large_orc": {"peasant_hunter": 500, "noble_hunter": 500},
        }
    )

    assert (
        policy_env["orc_hunting_bounty"]["raw_bounties_per_orc"].value
        == original_param.value
    )
    assert copied_env["orc_hunting_bounty"]["raw_bounties_per_orc"].value == {
        "small_orc": 500,
        "large_orc": {"peasant_hunter": 500, "noble_hunter": 500},
    }


def test_deepcopy_fails_on_policy_environment():
    """Verify copy.deepcopy fails on policy environments due to unpickleable objects."""
    policy_env = main(
        main_target=MainTarget.policy_environment,
        policy_date_str="2025-01-01",
        orig_policy_objects=OrigPolicyObjects.root(middle_earth.ROOT_PATH),
    )

    with pytest.raises((TypeError, AttributeError)) as excinfo:
        deepcopy(policy_env)

    # Should be a pickling-related error
    error_message = str(excinfo.value).lower()
    assert any(word in error_message for word in ["pickle", "module"])


def test_copy_environment_works_where_deepcopy_fails():
    """Verify copy_environment succeeds on objects that break copy.deepcopy."""
    policy_env = main(
        main_target=MainTarget.policy_environment,
        policy_date_str="2025-01-01",
        orig_policy_objects=OrigPolicyObjects.root(middle_earth.ROOT_PATH),
    )

    # Confirm deepcopy fails
    with pytest.raises((TypeError, AttributeError)):
        deepcopy(policy_env)

    # But copy_environment should work
    copied_env = copy_environment(policy_env)
    assert set(optree.tree_paths(policy_env)) == set(optree.tree_paths(copied_env))
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
    """Verify type hints work correctly for PolicyEnvironment input/output (mettsim)."""
    policy_env = main(
        main_target=MainTarget.policy_environment,
        policy_date_str="2025-01-01",
        orig_policy_objects=OrigPolicyObjects.root(middle_earth.ROOT_PATH),
    )

    # Type checker should infer PolicyEnvironment -> PolicyEnvironment
    copied_env: PolicyEnvironment = copy_environment(policy_env)

    # Function should work correctly
    assert isinstance(copied_env, dict)
    assert "payroll_tax" in copied_env


# =============================================================================
# Additional join() edge case tests
# =============================================================================


def test_join_empty_arrays(xnp: ModuleType):
    """Test join with empty arrays."""
    foreign_key = xnp.asarray(numpy.array([], dtype=int))
    primary_key = xnp.asarray(numpy.array([], dtype=int))
    target = xnp.asarray(numpy.array([], dtype=int))

    result = join(
        foreign_key=foreign_key,
        primary_key=primary_key,
        target=target,
        value_if_foreign_key_is_missing=0,
        xnp=xnp,
    )

    assert len(result) == 0


def test_join_single_element_arrays(xnp: ModuleType):
    """Test join with single-element arrays."""
    foreign_key = xnp.asarray(numpy.array([5]))
    primary_key = xnp.asarray(numpy.array([5]))
    target = xnp.asarray(numpy.array([100]))

    result = join(
        foreign_key=foreign_key,
        primary_key=primary_key,
        target=target,
        value_if_foreign_key_is_missing=-1,
        xnp=xnp,
    )

    numpy.testing.assert_array_equal(result, numpy.array([100]))


def test_join_all_foreign_keys_missing(xnp: ModuleType):
    """Test join when all foreign keys are missing (don't match any primary key)."""
    foreign_key = xnp.asarray(numpy.array([10, 20, 30]))
    primary_key = xnp.asarray(numpy.array([1, 2, 3]))
    target = xnp.asarray(numpy.array([100, 200, 300]))

    result = join(
        foreign_key=foreign_key,
        primary_key=primary_key,
        target=target,
        value_if_foreign_key_is_missing=-999,
        xnp=xnp,
    )

    # All foreign keys don't match, so all get the missing value
    numpy.testing.assert_array_equal(result, numpy.array([-999, -999, -999]))


def test_join_float_target(xnp: ModuleType):
    """Test join with float target values."""
    foreign_key = xnp.asarray(numpy.array([1, 2, 3]))
    primary_key = xnp.asarray(numpy.array([1, 2, 3]))
    target = xnp.asarray(numpy.array([1.5, 2.5, 3.5]))

    result = join(
        foreign_key=foreign_key,
        primary_key=primary_key,
        target=target,
        value_if_foreign_key_is_missing=0.0,
        xnp=xnp,
    )

    numpy.testing.assert_array_almost_equal(result, numpy.array([1.5, 2.5, 3.5]))


def test_join_bool_target(xnp: ModuleType):
    """Test join with boolean target values."""
    foreign_key = xnp.asarray(numpy.array([1, 2, -1]))
    primary_key = xnp.asarray(numpy.array([1, 2, 3]))
    target = xnp.asarray(numpy.array([True, False, True]))

    result = join(
        foreign_key=foreign_key,
        primary_key=primary_key,
        target=target,
        value_if_foreign_key_is_missing=False,
        xnp=xnp,
    )

    numpy.testing.assert_array_equal(result, numpy.array([True, False, False]))


def test_join_unsorted_primary_keys(xnp: ModuleType):
    """Test join with unsorted primary keys."""
    foreign_key = xnp.asarray(numpy.array([3, 1, 2]))
    primary_key = xnp.asarray(numpy.array([3, 1, 2]))  # Unsorted
    target = xnp.asarray(numpy.array([300, 100, 200]))

    result = join(
        foreign_key=foreign_key,
        primary_key=primary_key,
        target=target,
        value_if_foreign_key_is_missing=-1,
        xnp=xnp,
    )

    # Should correctly find matching targets regardless of order
    numpy.testing.assert_array_equal(result, numpy.array([300, 100, 200]))


def test_join_partial_matches(xnp: ModuleType):
    """Test join with some matching and some non-matching foreign keys."""
    foreign_key = xnp.asarray(numpy.array([1, 5, 2, 10, 3]))
    primary_key = xnp.asarray(numpy.array([1, 2, 3]))
    target = xnp.asarray(numpy.array([100, 200, 300]))

    result = join(
        foreign_key=foreign_key,
        primary_key=primary_key,
        target=target,
        value_if_foreign_key_is_missing=-1,
        xnp=xnp,
    )

    # 1->100, 5->-1 (missing), 2->200, 10->-1 (missing), 3->300
    numpy.testing.assert_array_equal(result, numpy.array([100, -1, 200, -1, 300]))


def test_join_large_primary_key_values(xnp: ModuleType):
    """Test join with large primary key values."""
    foreign_key = xnp.asarray(numpy.array([1000000, 2000000]))
    primary_key = xnp.asarray(numpy.array([1000000, 2000000]))
    target = xnp.asarray(numpy.array([1, 2]))

    result = join(
        foreign_key=foreign_key,
        primary_key=primary_key,
        target=target,
        value_if_foreign_key_is_missing=-1,
        xnp=xnp,
    )

    numpy.testing.assert_array_equal(result, numpy.array([1, 2]))
