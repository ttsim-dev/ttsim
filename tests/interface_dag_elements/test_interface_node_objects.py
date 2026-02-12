from __future__ import annotations

import inspect

import pytest

from ttsim.interface_dag_elements.interface_node_objects import (
    FailFunction,
    InputDependentInterfaceFunction,
    InterfaceFunction,
    InterfaceInput,
    WarnFunction,
    fail_function,
    input_dependent_interface_function,
    interface_function,
    interface_input,
    warn_function,
)


# =============================================================================
# InterfaceInput tests
# =============================================================================
def test_interface_input_decorator_creates_interface_input():
    @interface_input()
    def my_input() -> int:
        """Docstring for my_input."""

    assert isinstance(my_input, InterfaceInput)


def test_interface_input_uses_function_name_as_leaf_name():
    @interface_input()
    def my_input() -> int:
        """Docstring."""

    assert my_input.leaf_name == "my_input"


def test_interface_input_with_custom_leaf_name():
    @interface_input(leaf_name="custom_name")
    def my_input() -> int:
        """Docstring."""

    assert my_input.leaf_name == "custom_name"


def test_interface_input_extracts_return_type_from_annotation():
    @interface_input()
    def my_input() -> float:
        """Docstring."""

    # In Python 3.14 with __future__ annotations, the return type is stored as a string
    assert my_input.return_type == "float" or my_input.return_type is float


def test_interface_input_extracts_docstring():
    @interface_input()
    def my_input() -> int:
        """This is the docstring."""

    assert my_input.docstring == "This is the docstring."


def test_interface_input_in_top_level_namespace_defaults_false():
    @interface_input()
    def my_input() -> int:
        """Docstring."""

    assert my_input.in_top_level_namespace is False


def test_interface_input_in_top_level_namespace_can_be_set_true():
    @interface_input(in_top_level_namespace=True)
    def my_input() -> int:
        """Docstring."""

    assert my_input.in_top_level_namespace is True


def test_interface_input_remove_tree_logic_returns_self():
    @interface_input()
    def my_input() -> int:
        """Docstring."""

    result = my_input.remove_tree_logic(
        tree_path=("a", "b"),
        top_level_namespace={"some_name"},
    )
    assert result is my_input


# =============================================================================
# InterfaceFunction tests
# =============================================================================
def test_interface_function_decorator_creates_interface_function():
    @interface_function()
    def my_func(x: int) -> int:
        return x + 1

    assert isinstance(my_func, InterfaceFunction)


def test_interface_function_uses_function_name_as_leaf_name():
    @interface_function()
    def my_func(x: int) -> int:
        return x + 1

    assert my_func.leaf_name == "my_func"


def test_interface_function_with_custom_leaf_name():
    @interface_function(leaf_name="custom_func")
    def my_func(x: int) -> int:
        return x + 1

    assert my_func.leaf_name == "custom_func"


def test_interface_function_is_callable():
    @interface_function()
    def my_func(x: int) -> int:
        return x + 1

    assert my_func(5) == 6


def test_interface_function_dependencies_property():
    @interface_function()
    def my_func(a: int, b: str, c: float) -> int:  # noqa: ARG001
        return a

    assert my_func.dependencies == {"a", "b", "c"}


def test_interface_function_original_function_name_property():
    @interface_function(leaf_name="different_name")
    def original_name(x: int) -> int:
        return x

    assert original_name.original_function_name == "original_name"


def test_interface_function_preserves_signature():
    @interface_function()
    def my_func(a: int, b: str = "default") -> int:  # noqa: ARG001
        return a

    sig = inspect.signature(my_func)
    params = list(sig.parameters.keys())
    assert params == ["a", "b"]
    assert sig.parameters["b"].default == "default"


def test_interface_function_preserves_docstring():
    @interface_function()
    def my_func(x: int) -> int:
        """This is my docstring."""
        return x

    assert my_func.__doc__ == "This is my docstring."


def test_interface_function_in_top_level_namespace_defaults_false():
    @interface_function()
    def my_func(x: int) -> int:
        return x

    assert my_func.in_top_level_namespace is False


def test_interface_function_in_top_level_namespace_can_be_set_true():
    @interface_function(in_top_level_namespace=True)
    def my_func(x: int) -> int:
        return x

    assert my_func.in_top_level_namespace is True


def test_interface_function_remove_tree_logic_returns_new_interface_function():
    @interface_function()
    def my_func(x: int) -> int:
        return x

    # With empty tree_path and namespace, the function is returned unchanged
    result = my_func.remove_tree_logic(
        tree_path=(),
        top_level_namespace=set(),
    )

    assert isinstance(result, InterfaceFunction)
    assert result.leaf_name == my_func.leaf_name


# =============================================================================
# InputDependentInterfaceFunction tests
# =============================================================================
def test_input_dependent_requires_at_least_one_condition():
    with pytest.raises(ValueError, match="At least one of"):

        @input_dependent_interface_function()
        def my_func(x: int) -> int:
            return x


def test_input_dependent_with_include_if_any_input_present():
    @input_dependent_interface_function(include_if_any_input_present=["input_a"])
    def my_func(x: int) -> int:
        return x

    assert isinstance(my_func, InputDependentInterfaceFunction)
    assert my_func.include_if_any_input_present == ["input_a"]


def test_input_dependent_with_include_if_all_inputs_present():
    @input_dependent_interface_function(
        include_if_all_inputs_present=["input_a", "input_b"]
    )
    def my_func(x: int) -> int:
        return x

    assert isinstance(my_func, InputDependentInterfaceFunction)
    assert my_func.include_if_all_inputs_present == ["input_a", "input_b"]


def test_input_dependent_with_include_if_no_input_present():
    @input_dependent_interface_function(
        include_if_no_input_present=["input_a", "input_b"]
    )
    def my_func(x: int) -> int:
        return x

    assert isinstance(my_func, InputDependentInterfaceFunction)
    assert my_func.include_if_no_input_present == ["input_a", "input_b"]


def test_include_condition_satisfied_all_inputs_present_true():
    @input_dependent_interface_function(
        include_if_all_inputs_present=["input_a", "input_b"]
    )
    def my_func(x: int) -> int:
        return x

    # Both inputs present
    assert my_func.include_condition_satisfied(["input_a", "input_b", "other"]) is True


def test_include_condition_satisfied_all_inputs_present_false():
    @input_dependent_interface_function(
        include_if_all_inputs_present=["input_a", "input_b"]
    )
    def my_func(x: int) -> int:
        return x

    # Missing input_b
    assert my_func.include_condition_satisfied(["input_a", "other"]) is False


def test_include_condition_satisfied_any_input_present_true():
    @input_dependent_interface_function(
        include_if_any_input_present=["input_a", "input_b"]
    )
    def my_func(x: int) -> int:
        return x

    # At least one present
    assert my_func.include_condition_satisfied(["input_a", "other"]) is True


def test_include_condition_satisfied_any_input_present_false():
    @input_dependent_interface_function(
        include_if_any_input_present=["input_a", "input_b"]
    )
    def my_func(x: int) -> int:
        return x

    # Neither present
    assert my_func.include_condition_satisfied(["other", "something"]) is False


def test_include_condition_satisfied_no_input_present_true():
    @input_dependent_interface_function(
        include_if_no_input_present=["input_a", "input_b"]
    )
    def my_func(x: int) -> int:
        return x

    # None of the specified inputs present
    assert my_func.include_condition_satisfied(["other", "something"]) is True


def test_include_condition_satisfied_no_input_present_false():
    @input_dependent_interface_function(
        include_if_no_input_present=["input_a", "input_b"]
    )
    def my_func(x: int) -> int:
        return x

    # One of the specified inputs is present
    assert my_func.include_condition_satisfied(["input_a", "other"]) is False


def test_include_condition_satisfied_combined_conditions_or_logic():
    """Test that conditions are combined with OR logic."""

    @input_dependent_interface_function(
        include_if_all_inputs_present=["all_a", "all_b"],
        include_if_any_input_present=["any_a", "any_b"],
    )
    def my_func(x: int) -> int:
        return x

    # Only any_a present - any condition satisfied
    assert my_func.include_condition_satisfied(["any_a"]) is True

    # all_a and all_b present - all condition satisfied
    assert my_func.include_condition_satisfied(["all_a", "all_b"]) is True

    # Neither condition satisfied
    assert my_func.include_condition_satisfied(["other"]) is False


def test_input_dependent_is_callable():
    @input_dependent_interface_function(include_if_any_input_present=["input_a"])
    def my_func(x: int) -> int:
        return x * 2

    assert my_func(5) == 10


def test_input_dependent_remove_tree_logic_returns_input_dependent():
    @input_dependent_interface_function(include_if_any_input_present=["input_a"])
    def my_func(x: int) -> int:
        return x

    result = my_func.remove_tree_logic(
        tree_path=(),
        top_level_namespace=set(),
    )

    assert isinstance(result, InputDependentInterfaceFunction)
    assert result.include_if_any_input_present == my_func.include_if_any_input_present


# =============================================================================
# FailFunction tests
# =============================================================================
def test_fail_function_decorator_creates_fail_function():
    @fail_function()
    def my_fail_func(x: int) -> int:
        return x

    assert isinstance(my_fail_func, FailFunction)


def test_fail_function_stores_include_conditions():
    @fail_function(
        include_if_any_element_present=["elem_a"],
        include_if_all_elements_present=["elem_b", "elem_c"],
    )
    def my_fail_func(x: int) -> int:
        return x

    assert my_fail_func.include_if_any_element_present == ["elem_a"]
    assert my_fail_func.include_if_all_elements_present == ["elem_b", "elem_c"]


def test_fail_function_uses_function_name_as_leaf_name():
    @fail_function()
    def my_fail_func(x: int) -> int:
        return x

    assert my_fail_func.leaf_name == "my_fail_func"


def test_fail_function_with_custom_leaf_name():
    @fail_function(leaf_name="custom_fail")
    def my_fail_func(x: int) -> int:
        return x

    assert my_fail_func.leaf_name == "custom_fail"


def test_fail_function_is_callable():
    @fail_function()
    def my_fail_func(x: int) -> int:
        return x + 10

    assert my_fail_func(5) == 15


def test_fail_function_remove_tree_logic_returns_fail_function():
    @fail_function(include_if_any_element_present=["elem_a"])
    def my_fail_func(x: int) -> int:
        return x

    result = my_fail_func.remove_tree_logic(
        tree_path=(),
        top_level_namespace=set(),
    )

    assert isinstance(result, FailFunction)
    assert (
        result.include_if_any_element_present
        == my_fail_func.include_if_any_element_present
    )


# =============================================================================
# WarnFunction tests
# =============================================================================
def test_warn_function_decorator_creates_warn_function():
    @warn_function()
    def my_warn_func(x: int) -> int:
        return x

    assert isinstance(my_warn_func, WarnFunction)


def test_warn_function_stores_include_conditions():
    @warn_function(
        include_if_any_element_present=["elem_a"],
        include_if_all_elements_present=["elem_b", "elem_c"],
    )
    def my_warn_func(x: int) -> int:
        return x

    assert my_warn_func.include_if_any_element_present == ["elem_a"]
    assert my_warn_func.include_if_all_elements_present == ["elem_b", "elem_c"]


def test_warn_function_uses_function_name_as_leaf_name():
    @warn_function()
    def my_warn_func(x: int) -> int:
        return x

    assert my_warn_func.leaf_name == "my_warn_func"


def test_warn_function_with_custom_leaf_name():
    @warn_function(leaf_name="custom_warn")
    def my_warn_func(x: int) -> int:
        return x

    assert my_warn_func.leaf_name == "custom_warn"


def test_warn_function_is_callable():
    @warn_function()
    def my_warn_func(x: int) -> int:
        return x + 20

    assert my_warn_func(5) == 25


def test_warn_function_remove_tree_logic_returns_warn_function():
    @warn_function(include_if_any_element_present=["elem_a"])
    def my_warn_func(x: int) -> int:
        return x

    result = my_warn_func.remove_tree_logic(
        tree_path=(),
        top_level_namespace=set(),
    )

    assert isinstance(result, WarnFunction)
    assert (
        result.include_if_any_element_present
        == my_warn_func.include_if_any_element_present
    )
