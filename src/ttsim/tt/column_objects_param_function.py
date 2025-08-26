from __future__ import annotations

import datetime
import functools
import inspect
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Generic, Literal, ParamSpec, TypeVar

import dags.tree as dt
from dags import rename_arguments

from ttsim.interface_dag_elements.shared import to_datetime
from ttsim.tt.aggregation import (
    AggType,
    all_by_p_id,
    any_by_p_id,
    count_by_p_id,
    grouped_all,
    grouped_any,
    grouped_count,
    grouped_max,
    grouped_mean,
    grouped_min,
    grouped_sum,
    max_by_p_id,
    mean_by_p_id,
    min_by_p_id,
    sum_by_p_id,
)
from ttsim.tt.rounding import RoundingSpec
from ttsim.tt.vectorization import vectorize_function

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from ttsim.typing import (
        DashedISOString,
        IntColumn,
        UnorderedQNames,
    )

FunArgTypes = ParamSpec("FunArgTypes")
ReturnType = TypeVar("ReturnType")

DEFAULT_START_DATE = datetime.date(1900, 1, 1)
DEFAULT_END_DATE = datetime.date(2099, 12, 31)


class FKType(StrEnum):
    """
    Enum for foreign key types.
    """

    IRRELEVANT = "irrelevant"
    MAY_POINT_TO_SELF = "may point to self"
    MUST_NOT_POINT_TO_SELF = "must not point to self"


@dataclass(frozen=True)
class ColumnObject:
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
    start_date: datetime.date
    end_date: datetime.date
    description: str

    def is_active(self, policy_date: datetime.date) -> bool:
        """Check if the function is active at a given date."""
        return self.start_date <= policy_date <= self.end_date

    def remove_tree_logic(
        self,
        tree_path: tuple[str, ...],
        top_level_namespace: UnorderedQNames,
    ) -> ColumnObject:
        """Remove tree logic from the function and update the function signature."""
        raise NotImplementedError("Subclasses must implement this method.")


@dataclass(frozen=True)
class PolicyInput(ColumnObject):
    """
    A dummy function representing an input variable.

    Parameters
    ----------
    data_type:
        The data type of the input variable.
    start_date:
        The date from which the input is relevant / active (inclusive).
    end_date:
        The date until which the input is relevant / active (inclusive).
    foreign_key_type:
        Whether this is a foreign key and, if so, whether it may point to itself.
    """

    data_type: type[float | int | bool]
    foreign_key_type: FKType = FKType.IRRELEVANT
    warn_msg_if_included: str | None = None
    fail_msg_if_included: str | None = None
    docstring: str = ""

    def remove_tree_logic(
        self,
        tree_path: tuple[str, ...],  # noqa: ARG002
        top_level_namespace: UnorderedQNames,  # noqa: ARG002
    ) -> PolicyInput:
        return self


def policy_input(
    *,
    start_date: str | datetime.date = DEFAULT_START_DATE,
    end_date: str | datetime.date = DEFAULT_END_DATE,
    foreign_key_type: FKType = FKType.IRRELEVANT,
    warn_msg_if_included: str | None = None,
    fail_msg_if_included: str | None = None,
) -> Callable[[Callable[..., Any]], PolicyInput]:
    """
    Decorator that makes a (dummy) function a `PolicyInput`.

    **Dates active (start_date, end_date):**

    Specifies that a PolicyInput is only active between two dates, `start` and `end`.

    **Rounding spec (params_key_for_rounding):**

    Adds the location of the rounding specification to a PolicyInput.

    Parameters
    ----------
    start_date
        The start date (inclusive) in the format YYYY-MM-DD (part of ISO 8601).
    end_date
        The end date (inclusive) in the format YYYY-MM-DD (part of ISO 8601).

    Returns
    -------
    A decorator that returns a PolicyInput object.
    """
    start_date, end_date = _convert_and_validate_dates(start_date, end_date)

    def inner(func: Callable[..., Any]) -> PolicyInput:
        return PolicyInput(
            leaf_name=func.__name__,
            data_type=func.__annotations__["return"],
            start_date=start_date,
            end_date=end_date,
            foreign_key_type=foreign_key_type,
            description=str(inspect.getdoc(func)),
            docstring=inspect.getdoc(func),  # type: ignore[arg-type]
            warn_msg_if_included=warn_msg_if_included,
            fail_msg_if_included=fail_msg_if_included,
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
        "__code__",
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
class ColumnFunction(ColumnObject, Generic[FunArgTypes, ReturnType]):
    """
    Base class for all functions operating on columns of data.
    """

    function: Callable[FunArgTypes, ReturnType]
    rounding_spec: RoundingSpec | None = None
    foreign_key_type: FKType = FKType.IRRELEVANT
    warn_msg_if_included: str | None = None
    fail_msg_if_included: str | None = None

    def __post_init__(self) -> None:
        _fail_if_rounding_has_wrong_type(self.rounding_spec)
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

    def is_active(self, policy_date: datetime.date) -> bool:
        """Check if the function is active at a given date."""
        return self.start_date <= policy_date <= self.end_date


def _fail_if_rounding_has_wrong_type(rounding_spec: RoundingSpec | None) -> None:
    """Check if rounding_spec has the correct type.

    Parameters
    ----------
    rounding_spec
        The rounding specification to check.

    Raises
    ------
    TypeError
        If rounding_spec is not a RoundingSpec or None.
    """
    if not isinstance(rounding_spec, RoundingSpec | None):
        raise TypeError(
            f"`rounding_spec` must be a `RoundingSpec` or `None`, got {rounding_spec}"
        )


@dataclass(frozen=True)
class PolicyFunction(ColumnFunction):  # type: ignore[type-arg]
    """
    Computes a column based on at least one input column and/or parameters.

    Parameters
    ----------
    leaf_name:
        The leaf name of the function in the functions tree.
    function:
        The function that is called when the PolicyFunction is evaluated.
    start_date:
        The date from which the function is active (inclusive).
    end_date:
        The date until which the function is active (inclusive).
    rounding_spec:
        The rounding specification.
    """

    vectorization_strategy: Literal["loop", "vectorize", "not_required"] = "vectorize"

    def remove_tree_logic(
        self,
        tree_path: tuple[str, ...],
        top_level_namespace: UnorderedQNames,
    ) -> PolicyFunction:
        """Remove tree logic from the function and update the function signature."""

        function_without_tree_logic = dt.one_function_without_tree_logic(
            function=self.function,
            tree_path=tree_path,
            top_level_namespace=top_level_namespace,
        )
        # All functions that will be vectorized require the globals attribute to be
        # the same as for the initially defined function, since otherwise global
        # variables or imported functions cannot be found after vectorization.
        # This is not done by dt.one_function_without_tree_logic, so we do it here.
        function_without_tree_logic.__globals__.update(self.function.__globals__)

        return PolicyFunction(
            leaf_name=self.leaf_name,
            function=function_without_tree_logic,
            start_date=self.start_date,
            end_date=self.end_date,
            description=self.description,
            rounding_spec=self.rounding_spec,
            foreign_key_type=self.foreign_key_type,
            vectorization_strategy=self.vectorization_strategy,
            warn_msg_if_included=self.warn_msg_if_included,
            fail_msg_if_included=self.fail_msg_if_included,
        )

    def vectorize(self, backend: str, xnp: ModuleType) -> PolicyFunction:
        func = (
            self.function
            if self.vectorization_strategy == "not_required"
            else vectorize_function(
                self.function,
                vectorization_strategy=self.vectorization_strategy,
                backend=backend,
                xnp=xnp,
            )
        )
        return PolicyFunction(
            leaf_name=self.leaf_name,
            function=func,
            start_date=self.start_date,
            end_date=self.end_date,
            description=self.description,
            rounding_spec=self.rounding_spec,
            foreign_key_type=self.foreign_key_type,
            vectorization_strategy="not_required",
            warn_msg_if_included=self.warn_msg_if_included,
            fail_msg_if_included=self.fail_msg_if_included,
        )


def policy_function(
    *,
    leaf_name: str | None = None,
    start_date: str | datetime.date = DEFAULT_START_DATE,
    end_date: str | datetime.date = DEFAULT_END_DATE,
    rounding_spec: RoundingSpec | None = None,
    vectorization_strategy: Literal["loop", "vectorize", "not_required"] = "vectorize",
    foreign_key_type: FKType = FKType.IRRELEVANT,
    warn_msg_if_included: str | None = None,
    fail_msg_if_included: str | None = None,
) -> Callable[[Callable[..., Any]], PolicyFunction]:
    """
    Decorator that makes a `PolicyFunction` from a function.

    PolicyFunctions are typically defined on scalars, but work on data columns (i.e.,
    arrays of the same length as `p_id`). TTSIM will handle this (see
    `vectorization_strategy` below). Use `param_function` / `ParamFunction` for
    functions that convert the parameters of the taxes and transfers system, which do
    not require any columns from the data.

    Parameters
    ----------
    leaf_name
        The name that should be used as the PolicyFunction's leaf name in the DAG. If
        omitted, we use the name of the function as defined.
    start_date
        The start date (inclusive) in the format YYYY-MM-DD (part of ISO 8601).
    end_date
        The end date (inclusive) in the format YYYY-MM-DD (part of ISO 8601).
    rounding_spec
        The specification to be used for rounding.
    vectorization_strategy:
        Whether and how the function should be vectorized. Typically, functions will be
        defined on scalars and will be vectorized by TTSIM. Stick to the default of
        'vectorize'. Exceptions: 'loop' for constructs that cannot be vectorized by
        numpy or jax; 'not_required' if the function works natively with arrays (e.g.,
        joining two columns).
    foreign_key_type:
        Whether this is a foreign key and, if so, whether it may point to itself.

    Returns
    -------
    A decorator that returns a PolicyFunction object.
    """
    start_date, end_date = _convert_and_validate_dates(start_date, end_date)

    def inner(func: Callable[..., Any]) -> PolicyFunction:
        return PolicyFunction(
            leaf_name=leaf_name if leaf_name else func.__name__,
            function=func,
            start_date=start_date,
            end_date=end_date,
            description=str(inspect.getdoc(func)),
            rounding_spec=rounding_spec,
            foreign_key_type=foreign_key_type,
            vectorization_strategy=vectorization_strategy,
            warn_msg_if_included=warn_msg_if_included,
            fail_msg_if_included=fail_msg_if_included,
        )

    return inner


def reorder_ids(ids: IntColumn, xnp: ModuleType) -> IntColumn:
    """Make ID's consecutively numbered.

    Takes the given IDs and replaces them by consecutive numbers
    starting at 0.

    [43,44,70,50] -> [0,1,3,2]

    """
    sorting = xnp.argsort(ids)
    ids_sorted = ids[sorting]
    index_after_sort = xnp.arange(ids.shape[0])[sorting]

    # Look for difference from previous entry in sorted array
    diff_to_prev = xnp.where(xnp.diff(ids_sorted) >= 1, 1, 0)

    # Sum up all differences to get new id
    consecutive_ids = xnp.concatenate((xnp.asarray([0]), xnp.cumsum(diff_to_prev)))

    return consecutive_ids[xnp.argsort(index_after_sort)]


@dataclass(frozen=True)
class GroupCreationFunction(ColumnFunction):  # type: ignore[type-arg]
    """
    A function that computes endogenous group_by IDs.

    Parameters
    ----------
    leaf_name:
        The leaf name of the function in the functions tree.
    function:
        The function calculating the group_by IDs.
    start_date:
        The date from which the function is active (inclusive).
    end_date:
        The date until which the function is active (inclusive).
    """

    def remove_tree_logic(
        self,
        tree_path: tuple[str, ...],
        top_level_namespace: UnorderedQNames,
    ) -> GroupCreationFunction:
        """Remove tree logic from the function and update the function signature."""
        return GroupCreationFunction(
            leaf_name=self.leaf_name,
            function=dt.one_function_without_tree_logic(
                function=self.function,
                tree_path=tree_path,
                top_level_namespace=top_level_namespace,
            ),
            start_date=self.start_date,
            end_date=self.end_date,
            description=self.description,
            rounding_spec=self.rounding_spec,
            foreign_key_type=self.foreign_key_type,
            warn_msg_if_included=self.warn_msg_if_included,
            fail_msg_if_included=self.fail_msg_if_included,
        )


def group_creation_function(
    *,
    leaf_name: str | None = None,
    start_date: str | datetime.date = DEFAULT_START_DATE,
    end_date: str | datetime.date = DEFAULT_END_DATE,
    reorder: bool = True,
    warn_msg_if_included: str | None = None,
    fail_msg_if_included: str | None = None,
) -> Callable[[Callable[..., Any]], GroupCreationFunction]:
    """
    Decorator that creates a group_by function from a function.

    Parameters
    ----------
    leaf_name:
        The leaf name of the function in the functions tree.
    start_date:
        The date from which the function is active (inclusive).
    end_date:
        The date until which the function is active (inclusive).
    reorder:
        Whether the created Group ID's should be reordered to be
        consecutively numbered starting from 0.
    """
    start_date, end_date = _convert_and_validate_dates(start_date, end_date)

    def decorator(func: Callable[..., Any]) -> GroupCreationFunction:
        _leaf_name = func.__name__ if leaf_name is None else leaf_name
        func_with_reorder = lambda **kwargs: reorder_ids(  # noqa: E731
            ids=func(**kwargs),
            xnp=kwargs["xnp"],
        )
        functools.update_wrapper(func_with_reorder, func)

        return GroupCreationFunction(
            leaf_name=_leaf_name,
            function=func_with_reorder if reorder else func,
            start_date=start_date,
            end_date=end_date,
            description=str(inspect.getdoc(func)),
            warn_msg_if_included=warn_msg_if_included,
            fail_msg_if_included=fail_msg_if_included,
        )

    return decorator


@dataclass(frozen=True)
class AggByGroupFunction(ColumnFunction):  # type: ignore[type-arg]
    """
    A function that is an aggregation of another column by some group id.

    Parameters
    ----------
    leaf_name:
        The leaf name of the function in the functions tree.
    function:
        The function performing the aggregation.
    start_date:
        The date from which the function is active (inclusive).
    end_date:
        The date until which the function is active (inclusive).
    params_key_for_rounding:
        The key in the params dictionary that should be used for rounding.
    skip_vectorization:
        Whether the function should be vectorized.
    orig_location:
        The original location of the function, or "automatically generated".
    """

    # Default value is necessary because we have defaults in the superclass.
    orig_location: str = "automatically generated"

    def remove_tree_logic(
        self,
        tree_path: tuple[str, ...],
        top_level_namespace: UnorderedQNames,
    ) -> AggByGroupFunction:
        """Remove tree logic from the function and update the function signature."""
        return AggByGroupFunction(
            leaf_name=self.leaf_name,
            function=dt.one_function_without_tree_logic(
                function=self.function,
                tree_path=tree_path,
                top_level_namespace=top_level_namespace,
            ),
            start_date=self.start_date,
            end_date=self.end_date,
            description=self.description,
            rounding_spec=self.rounding_spec,
            foreign_key_type=self.foreign_key_type,
            orig_location=self.orig_location,
            warn_msg_if_included=self.warn_msg_if_included,
            fail_msg_if_included=self.fail_msg_if_included,
        )


def agg_by_group_function(
    *,
    leaf_name: str | None = None,
    start_date: str | datetime.date = DEFAULT_START_DATE,
    end_date: str | datetime.date = DEFAULT_END_DATE,
    agg_type: AggType,
    warn_msg_if_included: str | None = None,
    fail_msg_if_included: str | None = None,
) -> Callable[[Callable[..., Any]], AggByGroupFunction]:
    start_date, end_date = _convert_and_validate_dates(start_date, end_date)

    agg_registry = {
        AggType.SUM: grouped_sum,
        AggType.MEAN: grouped_mean,
        AggType.MAX: grouped_max,
        AggType.MIN: grouped_min,
        AggType.ANY: grouped_any,
        AggType.ALL: grouped_all,
        AggType.COUNT: grouped_count,
    }

    def inner(func: Callable[..., Any]) -> AggByGroupFunction:
        orig_location = f"{func.__module__}.{func.__name__}"
        args = set(inspect.signature(func).parameters)
        group_ids = {p for p in args if p.endswith("_id")}
        _fail_if_group_id_is_invalid(group_ids, orig_location)
        group_id = group_ids.pop()
        other_args = args - {group_id, "num_segments", "backend"}
        if agg_type == AggType.COUNT:
            _fail_if_other_arg_is_present(other_args, orig_location)
            mapper = {"group_id": group_id}
        else:
            _fail_if_other_arg_is_invalid(other_args, orig_location)
            mapper = {"group_id": group_id, "column": other_args.pop()}
        agg_func = rename_arguments(
            func=agg_registry[agg_type],
            mapper=mapper,
        )
        return AggByGroupFunction(
            leaf_name=leaf_name if leaf_name else func.__name__,
            function=agg_func,
            start_date=start_date,
            end_date=end_date,
            description=str(inspect.getdoc(func)),
            foreign_key_type=FKType.IRRELEVANT,
            orig_location=f"{func.__module__}.{func.__name__}",
            warn_msg_if_included=warn_msg_if_included,
            fail_msg_if_included=fail_msg_if_included,
        )

    return inner


def _fail_if_group_id_is_invalid(
    group_ids: UnorderedQNames,
    orig_location: str,
) -> None:
    if len(group_ids) != 1:
        raise ValueError(
            "Require exactly one group identifier ending with '_id' for "
            "aggregation by group. Got "
            f"{', '.join(group_ids) if group_ids else 'nothing'} in {orig_location}.",
        )


def _fail_if_other_arg_is_present(
    other_args: UnorderedQNames,
    orig_location: str,
) -> None:
    if other_args:
        raise ValueError(
            "There must be no argument besides identifiers for counting. Got: "
            f"{', '.join(other_args) if other_args else 'nothing'} in {orig_location}.",
        )


def _fail_if_other_arg_is_invalid(
    other_args: UnorderedQNames,
    orig_location: str,
) -> None:
    if len(other_args) != 1:
        raise ValueError(
            "There must be exactly one argument besides identifiers, num_segments, and "
            "backend for aggregations. Got: "
            f"{', '.join(other_args) if other_args else 'nothing'} in {orig_location}.",
        )


@dataclass(frozen=True)
class AggByPIDFunction(ColumnFunction):  # type: ignore[type-arg]
    """
    A function that is an aggregation of another column by some group id.

    Parameters
    ----------
    leaf_name:
        The leaf name of the function in the functions tree.
    function:
        The function performing the aggregation.
    start_date:
        The date from which the function is active (inclusive).
    end_date:
        The date until which the function is active (inclusive).
    params_key_for_rounding:
        The key in the params dictionary that should be used for rounding.
    skip_vectorization:
        Whether the function should be vectorized.
    orig_location:
        The original location of the function, or "automatically generated".
    """

    # Default value is necessary because we have defaults in the superclass.
    orig_location: str = "automatically generated"

    def remove_tree_logic(
        self,
        tree_path: tuple[str, ...],
        top_level_namespace: UnorderedQNames,
    ) -> AggByGroupFunction:
        """Remove tree logic from the function and update the function signature."""
        return AggByGroupFunction(
            leaf_name=self.leaf_name,
            function=dt.one_function_without_tree_logic(
                function=self.function,
                tree_path=tree_path,
                top_level_namespace=top_level_namespace,
            ),
            start_date=self.start_date,
            end_date=self.end_date,
            description=self.description,
            rounding_spec=self.rounding_spec,
            foreign_key_type=self.foreign_key_type,
            orig_location=self.orig_location,
            warn_msg_if_included=self.warn_msg_if_included,
            fail_msg_if_included=self.fail_msg_if_included,
        )


def agg_by_p_id_function(
    *,
    leaf_name: str | None = None,
    start_date: str | datetime.date = DEFAULT_START_DATE,
    end_date: str | datetime.date = DEFAULT_END_DATE,
    agg_type: AggType,
    warn_msg_if_included: str | None = None,
    fail_msg_if_included: str | None = None,
) -> Callable[[Callable[..., Any]], AggByPIDFunction]:
    start_date, end_date = _convert_and_validate_dates(start_date, end_date)

    agg_registry = {
        AggType.SUM: sum_by_p_id,
        AggType.MEAN: mean_by_p_id,
        AggType.MAX: max_by_p_id,
        AggType.MIN: min_by_p_id,
        AggType.ANY: any_by_p_id,
        AggType.ALL: all_by_p_id,
        AggType.COUNT: count_by_p_id,
    }

    def inner(func: Callable[..., Any]) -> AggByPIDFunction:
        orig_location = f"{func.__module__}.{func.__name__}"
        args = set(inspect.signature(func).parameters)
        other_p_ids = {
            p
            for p in args
            if any(e.startswith("p_id_") for e in dt.tree_path_from_qname(p))
        }
        other_args = args - {*other_p_ids, "p_id", "num_segments", "backend"}
        _fail_if_p_id_is_not_present(args, orig_location)
        _fail_if_other_p_id_is_invalid(other_p_ids, orig_location)
        if agg_type == AggType.COUNT:
            _fail_if_other_arg_is_present(other_args, orig_location)
            mapper = {
                "p_id_to_aggregate_by": other_p_ids.pop(),
                "p_id_to_store_by": "p_id",
                "num_segments": "num_segments",
                "backend": "backend",
            }
        else:
            _fail_if_other_arg_is_invalid(other_args, orig_location)
            mapper = {
                "column": other_args.pop(),
                "p_id_to_aggregate_by": other_p_ids.pop(),
                "p_id_to_store_by": "p_id",
                "num_segments": "num_segments",
                "backend": "backend",
            }
        agg_func = rename_arguments(
            func=agg_registry[agg_type],
            mapper=mapper,
        )
        return AggByPIDFunction(
            leaf_name=leaf_name if leaf_name else func.__name__,
            function=agg_func,
            start_date=start_date,
            end_date=end_date,
            description=str(inspect.getdoc(func)),
            foreign_key_type=FKType.IRRELEVANT,
            orig_location=f"{func.__module__}.{func.__name__}",
            warn_msg_if_included=warn_msg_if_included,
            fail_msg_if_included=fail_msg_if_included,
        )

    return inner


def _fail_if_p_id_is_not_present(args: UnorderedQNames, orig_location: str) -> None:
    if "p_id" not in args:
        raise ValueError(
            "The function must have the argument named 'p_id' for aggregation by p_id. "
            f"Got {', '.join(args) if args else 'nothing'} in {orig_location}.",
        )


def _fail_if_other_p_id_is_invalid(
    other_p_ids: UnorderedQNames,
    orig_location: str,
) -> None:
    if len(other_p_ids) != 1:
        raise ValueError(
            "Require exactly one identifier starting with 'p_id_' for "
            "aggregation by p_id. Got: "
            f"{', '.join(other_p_ids) if other_p_ids else 'nothing'} in {orig_location}.",  # noqa: E501
        )


@dataclass(frozen=True)
class TimeConversionFunction(ColumnFunction):  # type: ignore[type-arg]
    """
    A function that is a time conversion of another function.

    Parameters
    ----------
    leaf_name:
        The leaf name of the function in the functions tree.
    function:
        The function performing the time conversion.
    source:
        The name of the source function or data column.
    start_date:
        The date from which the function is active (inclusive).
    end_date:
        The date until which the function is active (inclusive).
    """

    source: str | None = None

    def __post_init__(self) -> None:
        if self.source is None:
            raise ValueError("The source must be specified.")
        super().__post_init__()

    def remove_tree_logic(
        self,
        tree_path: tuple[str, ...],
        top_level_namespace: UnorderedQNames,
    ) -> TimeConversionFunction:
        """Remove tree logic from the function and update the function signature."""
        return TimeConversionFunction(
            source=self.source,
            leaf_name=self.leaf_name,
            function=dt.one_function_without_tree_logic(
                function=self.function,
                tree_path=tree_path,
                top_level_namespace=top_level_namespace,
            ),
            start_date=self.start_date,
            end_date=self.end_date,
            description=self.description,
            rounding_spec=self.rounding_spec,
            foreign_key_type=self.foreign_key_type,
            warn_msg_if_included=self.warn_msg_if_included,
            fail_msg_if_included=self.fail_msg_if_included,
        )


def _convert_and_validate_dates(
    start_date: datetime.date | DashedISOString,
    end_date: datetime.date | DashedISOString,
) -> tuple[datetime.date, datetime.date]:
    """Convert and validate date strings to datetime.date objects.

    Parameters
    ----------
    start_date
        The start date (inclusive) in the format YYYY-MM-DD (part of ISO 8601).
    end_date
        The end date (inclusive) in the format YYYY-MM-DD (part of ISO 8601).

    Returns
    -------
    tuple[datetime.date, datetime.date]
        The converted and validated start and end dates.
    """
    start_date = to_datetime(start_date)
    end_date = to_datetime(end_date)

    if start_date > end_date:
        raise ValueError(
            f"The start date {start_date} must be before the end date {end_date}.",
        )

    return start_date, end_date


@dataclass(frozen=True)
class ParamFunction(Generic[FunArgTypes, ReturnType]):
    """
    Compute a scalar or custom object from parameters of the taxes and transfers system.

    Parameters
    ----------
    leaf_name:
        The leaf name of the function in the objects tree.
    start_date:
        The date from which the function is active (inclusive).
    end_date:
        The date until which the function is active (inclusive).
    function:
        The function that is called when the ParamFunction is evaluated.
    """

    leaf_name: str
    start_date: datetime.date
    end_date: datetime.date
    function: Callable[FunArgTypes, ReturnType]
    description: str
    warn_msg_if_included: str | None = None
    fail_msg_if_included: str | None = None

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

    def is_active(self, policy_date: datetime.date) -> bool:
        """Check if the function is active at a given date."""
        return self.start_date <= policy_date <= self.end_date

    def remove_tree_logic(
        self,
        tree_path: tuple[str, ...],
        top_level_namespace: UnorderedQNames,
    ) -> ParamFunction:  # type: ignore[type-arg]
        """Remove tree logic from the function and update the function signature."""
        return ParamFunction(
            leaf_name=self.leaf_name,
            function=dt.one_function_without_tree_logic(
                function=self.function,
                tree_path=tree_path,
                top_level_namespace=top_level_namespace,
            ),
            start_date=self.start_date,
            end_date=self.end_date,
            description=self.description,
            warn_msg_if_included=self.warn_msg_if_included,
            fail_msg_if_included=self.fail_msg_if_included,
        )


# Never returns a column, require precise annotation
def param_function(
    *,
    leaf_name: str | None = None,
    start_date: str | datetime.date = DEFAULT_START_DATE,
    end_date: str | datetime.date = DEFAULT_END_DATE,
    warn_msg_if_included: str | None = None,
    fail_msg_if_included: str | None = None,
) -> Callable[[Callable[..., Any]], ParamFunction[..., Any]]:
    """
    Decorator that makes a `ParamFunction` from a function.

    ParamFunctions convert complex parameters (i.e., anything that is not a scalar, a
    flat homogenous dictionary, or a set of parameters of a piecewise polynomial
    function) to custom representations. They must not use any data columns (i.e.,
    arrays of the same length as `p_id`). Use `policy_function` / `PolicyFunction` for
    functions that operate on data columns.

    As a consequence, the arguments of the decorated function must be found in the
    params tree. They are typically defined as outermost keys in the yaml files with
    parameters of the taxes and transfers system.

    Parameters
    ----------
    leaf_name
        The name that should be used as the ParamFunction's leaf name in the DAG. If
        omitted, we use the name of the function as defined.
    start_date
        The start date (inclusive) in the format YYYY-MM-DD (part of ISO 8601).
    end_date
        The end date (inclusive) in the format YYYY-MM-DD (part of ISO 8601).

    Returns
    -------
    A decorator that returns a ParamFunction object.
    """
    start_date, end_date = _convert_and_validate_dates(start_date, end_date)

    def inner(func: Callable[..., Any]) -> ParamFunction:  # type: ignore[type-arg]
        return ParamFunction(
            leaf_name=leaf_name if leaf_name else func.__name__,
            function=func,
            start_date=start_date,
            end_date=end_date,
            description=str(inspect.getdoc(func)),
            warn_msg_if_included=warn_msg_if_included,
            fail_msg_if_included=fail_msg_if_included,
        )

    return inner
