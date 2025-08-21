from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, ParamSpec, TypeVar

import dags.tree as dt

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from ttsim.typing import UnorderedQNames


FunArgTypes = ParamSpec("FunArgTypes")
ReturnType = TypeVar("ReturnType")


@dataclass(frozen=True)
class InterfaceNodeObject:
    """Base class for all objects operating on columns of data.

    Examples
    --------
    - PolicyInputs
    - PolicyFunctions
    - GroupCreationFunctions
    - AggByGroupFunctions
    - AggByPIDFunctions
    - TimeConversionFunctions

    Parameters are not ColumnObjectParamFunctions.

    """

    leaf_name: str
    in_top_level_namespace: bool

    def remove_tree_logic(
        self,
        tree_path: tuple[str, ...],
        top_level_namespace: UnorderedQNames,
    ) -> InterfaceNodeObject:
        """Remove tree logic from the function and update the function signature."""
        raise NotImplementedError("Subclasses must implement this method.")


@dataclass(frozen=True)
class InterfaceInput(InterfaceNodeObject):
    """A dummy function representing an input node."""

    return_type: type
    docstring: str

    def remove_tree_logic(
        self,
        tree_path: tuple[str, ...],  # noqa: ARG002
        top_level_namespace: UnorderedQNames,  # noqa: ARG002
    ) -> InterfaceInput:
        return self


def interface_input(
    leaf_name: str | None = None,
    in_top_level_namespace: bool = False,
) -> Callable[[Callable[..., Any]], InterfaceInput]:
    """
    Decorator that makes a (dummy) function an `InterfaceInput`.

    Returns
    -------
    A decorator that returns an InterfaceInput object.
    """

    def inner(func: Callable[..., Any]) -> InterfaceInput:
        return InterfaceInput(
            leaf_name=leaf_name if leaf_name else func.__name__,
            in_top_level_namespace=in_top_level_namespace,
            return_type=func.__annotations__["return"],
            docstring=inspect.getdoc(func),  # type: ignore[arg-type]
        )

    return inner


def _frozen_safe_update_wrapper(wrapper: object, wrapped: Callable[..., Any]) -> None:
    """Update a frozen wrapper dataclass to look like the wrapped function.

    This is necessary because the wrapper is a frozen dataclass, so we cannot
    use the `functools.update_wrapper` function or `self.__signature__ = ...`
    assignments in the `__post_init__` method.

    Args:
        wrapper: The wrapper dataclass to update.
        wrapped: The function to update the wrapper to.

    """
    object.__setattr__(wrapper, "__signature__", inspect.signature(wrapped))

    WRAPPER_ASSIGNMENTS = (  # noqa: N806
        "__globals__",
        "__closure__",
        "__doc__",
        "__name__",
        "__QName__",
        "__module__",
        "__annotations__",
        "__type_params__",
    )
    for attr in WRAPPER_ASSIGNMENTS:
        if hasattr(wrapped, attr):
            object.__setattr__(wrapper, attr, getattr(wrapped, attr))

    getattr(wrapper, "__dict__", {}).update(getattr(wrapped, "__dict__", {}))


@dataclass(frozen=True)
class InterfaceFunction(InterfaceNodeObject, Generic[FunArgTypes, ReturnType]):
    """
    Base class for all functions operating on columns of data.
    """

    function: Callable[FunArgTypes, ReturnType]

    def __post_init__(self) -> None:
        # Expose the signature of the wrapped function for dependency resolution
        _frozen_safe_update_wrapper(self, self.function)

    def __call__(
        self,
        *args: FunArgTypes.args,
        **kwargs: FunArgTypes.kwargs,
    ) -> ReturnType:
        return self.function(*args, **kwargs)

    @property
    def dependencies(self) -> UnorderedQNames:
        """The names of input variables that the function depends on."""
        return set(inspect.signature(self).parameters)

    @property
    def original_function_name(self) -> str:
        """The name of the wrapped function."""
        return self.function.__name__

    def remove_tree_logic(
        self,
        tree_path: tuple[str, ...],
        top_level_namespace: UnorderedQNames,
    ) -> InterfaceFunction:  # type: ignore[type-arg]
        """Remove tree logic from the function and update the function signature."""
        return InterfaceFunction(
            leaf_name=self.leaf_name,
            function=dt.one_function_without_tree_logic(
                function=self.function,
                tree_path=tree_path,
                top_level_namespace=top_level_namespace,
            ),
            in_top_level_namespace=self.in_top_level_namespace,
        )


def interface_function(
    *,
    leaf_name: str | None = None,
    in_top_level_namespace: bool = False,
) -> Callable[[Callable[..., Any]], InterfaceFunction[..., Any]]:
    """
    Decorator that makes an `InterfaceFunction` from a function.

    Parameters
    ----------
    leaf_name
        The name that should be used as the PolicyFunction's leaf name in the DAG. If
        omitted, we use the name of the function as defined.
    in_top_level_namespace:
        Whether the function is in the top-level namespace of the interface-DAG.

    Returns
    -------
    A decorator that returns an InterfaceFunction object.
    """

    def inner(func: Callable[..., Any]) -> InterfaceFunction:  # type: ignore[type-arg]
        return InterfaceFunction(
            leaf_name=leaf_name if leaf_name else func.__name__,
            function=func,
            in_top_level_namespace=in_top_level_namespace,
        )

    return inner


@dataclass(frozen=True)
class InputDependentInterfaceFunction(InterfaceFunction[FunArgTypes, ReturnType]):
    """A function that dynamically changes its behavior based on which InterfaceInput
    nodes are given by the user."""

    include_if_any_input_present: Iterable[str]
    include_if_all_inputs_present: Iterable[str]
    include_if_no_input_present: Iterable[str]

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            not self.include_if_all_inputs_present
            and not self.include_if_any_input_present
            and not self.include_if_no_input_present
        ):
            raise ValueError(
                "At least one of `include_if_all_inputs_present`, "
                "`include_if_any_input_present`, or `include_if_no_input_present` "
                "must be specified."
            )

    def include_condition_satisfied(self, input_names: Iterable[str]) -> bool:
        """Check if the input names match the include condition."""
        # all(()) evaluates to True, so include first bit
        all_cond = self.include_if_all_inputs_present and all(
            i in input_names for i in self.include_if_all_inputs_present
        )
        any_cond = any(i in input_names for i in self.include_if_any_input_present)
        no_cond = bool(self.include_if_no_input_present) and not any(
            i in input_names for i in self.include_if_no_input_present
        )
        return all_cond or any_cond or no_cond

    def remove_tree_logic(
        self,
        tree_path: tuple[str, ...],
        top_level_namespace: UnorderedQNames,
    ) -> InputDependentInterfaceFunction:  # type: ignore[type-arg]
        """Remove tree logic from the function and update the function signature."""
        return InputDependentInterfaceFunction(
            leaf_name=self.leaf_name,
            function=dt.one_function_without_tree_logic(
                function=self.function,
                tree_path=tree_path,
                top_level_namespace=top_level_namespace,
            ),
            in_top_level_namespace=self.in_top_level_namespace,
            include_if_any_input_present=self.include_if_any_input_present,
            include_if_all_inputs_present=self.include_if_all_inputs_present,
            include_if_no_input_present=self.include_if_no_input_present,
        )


def input_dependent_interface_function(
    *,
    include_if_any_input_present: Iterable[str] = (),
    include_if_all_inputs_present: Iterable[str] = (),
    include_if_no_input_present: Iterable[str] = (),
    leaf_name: str | None = None,
    in_top_level_namespace: bool = False,
) -> Callable[
    [Callable[..., Any]], InputDependentInterfaceFunction[FunArgTypes, ReturnType]
]:
    """
    Decorator that makes an `InputDependentInterfaceFunction` from a function.

    Parameters
    ----------
    include_if_any_input_present
        List of input names that must be present for the function to be used if any of
        the inputs are present.
    include_if_all_inputs_present
        List of input names that must be present for the function to be used if all of
        the inputs are present.
    include_if_no_input_present
        List of input names that must not be present for the function to be used if no
        inputs are present.
    leaf_name
        The name that should be used as the function's leaf name in the DAG. If omitted,
        we use the name of the function as defined.
    in_top_level_namespace
        Whether the function is in the top-level namespace of the interface-DAG.

    Returns
    -------
    A decorator that returns an InputDependentInterfaceFunction object.
    """

    def inner(
        func: Callable[..., Any],
    ) -> InputDependentInterfaceFunction[FunArgTypes, ReturnType]:
        return InputDependentInterfaceFunction(
            leaf_name=leaf_name if leaf_name else func.__name__,
            function=func,
            in_top_level_namespace=in_top_level_namespace,
            include_if_any_input_present=include_if_any_input_present,
            include_if_all_inputs_present=include_if_all_inputs_present,
            include_if_no_input_present=include_if_no_input_present,
        )

    return inner


@dataclass(frozen=True)
class FailFunction(InterfaceFunction):  # type: ignore[type-arg]
    """An interface function that fails under certain conditions."""

    include_if_any_element_present: Iterable[str]
    include_if_all_elements_present: Iterable[str]

    def remove_tree_logic(
        self,
        tree_path: tuple[str, ...],
        top_level_namespace: UnorderedQNames,
    ) -> FailFunction:
        """Remove tree logic from the function and update the function signature."""
        return FailFunction(
            leaf_name=self.leaf_name,
            function=dt.one_function_without_tree_logic(
                function=self.function,
                tree_path=tree_path,
                top_level_namespace=top_level_namespace,
            ),
            in_top_level_namespace=self.in_top_level_namespace,
            include_if_any_element_present=self.include_if_any_element_present,
            include_if_all_elements_present=self.include_if_all_elements_present,
        )


def fail_function(
    *,
    include_if_any_element_present: Iterable[str] = (),
    include_if_all_elements_present: Iterable[str] = (),
    leaf_name: str | None = None,
    in_top_level_namespace: bool = False,
) -> Callable[[Callable[..., Any]], FailFunction]:
    """
    Decorator that makes a `FailFunction` from a function.
    """

    def inner(func: Callable[..., Any]) -> FailFunction:
        return FailFunction(
            include_if_any_element_present=include_if_any_element_present,
            include_if_all_elements_present=include_if_all_elements_present,
            leaf_name=leaf_name if leaf_name else func.__name__,
            function=func,
            in_top_level_namespace=in_top_level_namespace,
        )

    return inner


@dataclass(frozen=True)
class WarnFunction(InterfaceFunction):  # type: ignore[type-arg]
    """An interface function that warns under certain conditions."""

    include_if_any_element_present: Iterable[str]
    include_if_all_elements_present: Iterable[str]

    def remove_tree_logic(
        self,
        tree_path: tuple[str, ...],
        top_level_namespace: UnorderedQNames,
    ) -> WarnFunction:
        """Remove tree logic from the function and update the function signature."""
        return WarnFunction(
            leaf_name=self.leaf_name,
            function=dt.one_function_without_tree_logic(
                function=self.function,
                tree_path=tree_path,
                top_level_namespace=top_level_namespace,
            ),
            in_top_level_namespace=self.in_top_level_namespace,
            include_if_any_element_present=self.include_if_any_element_present,
            include_if_all_elements_present=self.include_if_all_elements_present,
        )


def warn_function(
    *,
    include_if_any_element_present: Iterable[str] = (),
    include_if_all_elements_present: Iterable[str] = (),
    leaf_name: str | None = None,
    in_top_level_namespace: bool = False,
) -> Callable[[Callable[..., Any]], WarnFunction]:
    """
    Decorator that makes a `WarnFunction` from a function.
    """

    def inner(func: Callable[..., Any]) -> WarnFunction:
        return WarnFunction(
            include_if_any_element_present=include_if_any_element_present,
            include_if_all_elements_present=include_if_all_elements_present,
            leaf_name=leaf_name if leaf_name else func.__name__,
            function=func,
            in_top_level_namespace=in_top_level_namespace,
        )

    return inner
