from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, ParamSpec, TypeVar

import dags.tree as dt

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ttsim.interface_dag_elements.typing import GenericCallable, UnorderedQNames

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

    def remove_tree_logic(
        self,
        tree_path: tuple[str, ...],  # noqa: ARG002
        top_level_namespace: UnorderedQNames,  # noqa: ARG002
    ) -> InterfaceInput:
        return self


def interface_input(
    in_top_level_namespace: bool = False,
) -> GenericCallable[[GenericCallable], InterfaceInput]:
    """
    Decorator that makes a (dummy) function an `InterfaceInput`.

    Returns
    -------
    A decorator that returns an InterfaceInput object.
    """

    def inner(func: GenericCallable) -> InterfaceInput:
        return InterfaceInput(
            leaf_name=func.__name__,
            in_top_level_namespace=in_top_level_namespace,
            return_type=func.__annotations__["return"],
        )

    return inner


def _frozen_safe_update_wrapper(wrapper: object, wrapped: GenericCallable) -> None:
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

    function: GenericCallable[FunArgTypes, ReturnType]

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
) -> GenericCallable[[GenericCallable], InterfaceFunction]:
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

    def inner(func: GenericCallable) -> InterfaceFunction:  # type: ignore[type-arg]
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

    specs: list[InterfaceFunctionVariant]

    def resolve_to_static_interface_function(
        self, user_input_qnames: list[str]
    ) -> InterfaceFunction[FunArgTypes, ReturnType]:
        """Generate a static function based on the user inputs."""
        _fail_if_not_exactly_one_function_variant_matches_inputs(
            specs=self.specs,
            user_input_qnames=user_input_qnames,
        )
        for spec in self.specs:
            if set(spec.required_input_qnames) <= set(user_input_qnames):
                func: InterfaceFunction[FunArgTypes, ReturnType] = InterfaceFunction(
                    leaf_name=self.leaf_name,
                    function=spec.function,
                    in_top_level_namespace=self.in_top_level_namespace,
                )
                break
        return func


def input_dependent_interface_function(
    *,
    variants: list[InterfaceFunctionVariant],
    leaf_name: str | None = None,
    in_top_level_namespace: bool = False,
) -> GenericCallable[[GenericCallable], InputDependentInterfaceFunction]:
    """
    Decorator that makes an `InputDependentInterfaceFunction` from a function.

    Parameters
    ----------
    variants
        List of function variants that define different behaviors based on input
        availability.
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
        func: GenericCallable,
    ) -> InputDependentInterfaceFunction[FunArgTypes, ReturnType]:
        return InputDependentInterfaceFunction(
            leaf_name=leaf_name if leaf_name else func.__name__,
            function=func,
            in_top_level_namespace=in_top_level_namespace,
            specs=variants,
        )

    return inner


@dataclass(frozen=True)
class InterfaceFunctionVariant:
    required_input_qnames: list[str]
    function: GenericCallable


def _fail_if_not_exactly_one_function_variant_matches_inputs(
    specs: list[InterfaceFunctionVariant],
    user_input_qnames: list[str],
) -> None:
    """Validate that exactly one function variant matches the provided user inputs.

    This function ensures that the user has provided the correct combination of inputs
    to uniquely determine which function variant should be used.
    """
    potential_qnames = [spec.required_input_qnames for spec in specs]
    qnames_from_user_satisfying_specs = [
        spec.required_input_qnames
        for spec in specs
        if set(spec.required_input_qnames) <= set(user_input_qnames)
    ]

    base_msg = (
        "Exactly one of the following sets of inputs is required:\n\n"
        f"{'\n'.join([f'[{", ".join(s)}]' for s in potential_qnames])}"
    )

    if len(qnames_from_user_satisfying_specs) > 1:
        msg = (
            base_msg
            + "\n\n"
            + (
                "Multiple sets of inputs were found that satisfy the requirements:\n\n"
                f"{
                    '\n'.join(
                        [f'[{", ".join(s)}]' for s in qnames_from_user_satisfying_specs]
                    )
                }\n\n"
                "Please provide only one of these."
            )
        )
        raise ValueError(msg)
    if len(qnames_from_user_satisfying_specs) == 0:
        msg = (
            base_msg
            + "\n\n"
            + (
                "None of the required input sets were found in the provided inputs.\n"
                "Please provide one of the sets of inputs listed above."
            )
        )
        raise ValueError(msg)


@dataclass(frozen=True)
class FailOrWarnFunction(InterfaceFunction):  # type: ignore[type-arg]
    """
    Base class for all functions operating on columns of data.
    """

    include_if_any_element_present: Iterable[str]
    include_if_all_elements_present: Iterable[str]

    def remove_tree_logic(
        self,
        tree_path: tuple[str, ...],
        top_level_namespace: UnorderedQNames,
    ) -> FailOrWarnFunction:
        """Remove tree logic from the function and update the function signature."""
        return FailOrWarnFunction(
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


def fail_or_warn_function(
    *,
    include_if_any_element_present: Iterable[str] = (),
    include_if_all_elements_present: Iterable[str] = (),
    leaf_name: str | None = None,
    in_top_level_namespace: bool = False,
) -> GenericCallable[[GenericCallable], FailOrWarnFunction]:
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

    def inner(func: GenericCallable) -> FailOrWarnFunction:
        return FailOrWarnFunction(
            include_if_any_element_present=include_if_any_element_present,
            include_if_all_elements_present=include_if_all_elements_present,
            leaf_name=leaf_name if leaf_name else func.__name__,
            function=func,
            in_top_level_namespace=in_top_level_namespace,
        )

    return inner
