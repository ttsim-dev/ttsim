from __future__ import annotations

import datetime
import functools
import inspect
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Generic, Literal, ParamSpec, TypeVar

import dags
import dags.tree as dt
import numpy
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_float_dtype,
    is_integer_dtype,
)

from ttsim.aggregation import (
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
from ttsim.config import IS_JAX_INSTALLED
from ttsim.rounding import RoundingSpec
from ttsim.shared import to_datetime, validate_date_range
from ttsim.vectorization import make_vectorizable

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

    from ttsim.config import numpy_or_jax as np
    from ttsim.typing import DashedISOString

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
class TTSIMObject:
    """Abstract base class for all TTSIM Functions and Data Inputs."""

    leaf_name: str
    start_date: datetime.date
    end_date: datetime.date

    def is_active(self, date: datetime.date) -> bool:
        """Check if the function is active at a given date."""
        return self.start_date <= date <= self.end_date

    def remove_tree_logic(
        self,
        tree_path: tuple[str, ...],
        top_level_namespace: set[str],
    ) -> TTSIMObject:
        """Remove tree logic from the function and update the function signature."""
        raise NotImplementedError("Subclasses must implement this method.")


@dataclass(frozen=True)
class PolicyInput(TTSIMObject):
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

    def remove_tree_logic(
        self,
        tree_path: tuple[str, ...],  # noqa: ARG002
        top_level_namespace: set[str],  # noqa: ARG002
    ) -> PolicyInput:
        return self


def policy_input(
    *,
    start_date: str | datetime.date = DEFAULT_START_DATE,
    end_date: str | datetime.date = DEFAULT_END_DATE,
    foreign_key_type: FKType = FKType.IRRELEVANT,
) -> Callable[[Callable], PolicyInput]:
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

    def inner(func: Callable) -> PolicyInput:
        data_type = func.__annotations__["return"]
        return PolicyInput(
            leaf_name=func.__name__,
            data_type=data_type,
            start_date=start_date,
            end_date=end_date,
            foreign_key_type=foreign_key_type,
        )

    return inner


def _frozen_safe_update_wrapper(wrapper: object, wrapped: Callable) -> None:
    """Update a frozen wrapper dataclass to look like the wrapped function.

    This is necessary because the wrapper is a frozen dataclass, so we cannot
    use the `functools.update_wrapper` function or `self.__signature__ = ...`
    assigments in the `__post_init__` method.

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
        "__qualname__",
        "__module__",
        "__annotations__",
        "__type_params__",
    )
    for attr in WRAPPER_ASSIGNMENTS:
        if hasattr(wrapped, attr):
            object.__setattr__(wrapper, attr, getattr(wrapped, attr))

    getattr(wrapper, "__dict__", {}).update(getattr(wrapped, "__dict__", {}))


@dataclass(frozen=True)
class TTSIMFunction(TTSIMObject, Generic[FunArgTypes, ReturnType]):
    """
    Base class for all TTSIM functions.
    """

    function: Callable[FunArgTypes, ReturnType]
    rounding_spec: RoundingSpec | None = None
    foreign_key_type: FKType = FKType.IRRELEVANT

    def __post_init__(self) -> None:
        self._fail_if_rounding_has_wrong_type(self.rounding_spec)
        # Expose the signature of the wrapped function for dependency resolution
        _frozen_safe_update_wrapper(self, self.function)

    def _fail_if_rounding_has_wrong_type(
        self, rounding_spec: RoundingSpec | None
    ) -> None:
        """Check if rounding_spec has the correct type.

        Parameters
        ----------
        rounding_spec
            The rounding specification to check.

        Raises
        ------
        AssertionError
            If rounding_spec is not a RoundingSpec or None.
        """
        assert isinstance(rounding_spec, RoundingSpec | None), (
            f"rounding_spec must be a RoundingSpec or None, got {rounding_spec}"
        )

    def __call__(
        self, *args: FunArgTypes.args, **kwargs: FunArgTypes.kwargs
    ) -> ReturnType:
        return self.function(*args, **kwargs)

    @property
    def dependencies(self) -> set[str]:
        """The names of input variables that the function depends on."""
        return set(inspect.signature(self).parameters)

    @property
    def original_function_name(self) -> str:
        """The name of the wrapped function."""
        return self.function.__name__

    def is_active(self, date: datetime.date) -> bool:
        """Check if the function is active at a given date."""
        return self.start_date <= date <= self.end_date


@dataclass(frozen=True)
class PolicyFunction(TTSIMFunction):
    """
    A function that computes an output vector based on some input vectors and/or
    parameters.

    Parameters
    ----------
    leaf_name:
        The leaf name of the function in the functions tree.
    function:
        The function to wrap. Argument values of the `@policy_function` are reused
        unless explicitly overwritten.
    start_date:
        The date from which the function is active (inclusive).
    end_date:
        The date until which the function is active (inclusive).
    rounding_spec:
        The rounding specification.
    """

    def remove_tree_logic(
        self,
        tree_path: tuple[str, ...],
        top_level_namespace: set[str],
    ) -> PolicyFunction:
        """Remove tree logic from the function and update the function signature."""
        return PolicyFunction(
            leaf_name=self.leaf_name,
            function=dt.one_function_without_tree_logic(
                function=self.function,
                tree_path=tree_path,
                top_level_namespace=top_level_namespace,
            ),
            start_date=self.start_date,
            end_date=self.end_date,
            rounding_spec=self.rounding_spec,
            foreign_key_type=self.foreign_key_type,
        )


# Never returns a column, require precise annotation
def params_function(
    *,
    leaf_name: str | None = None,
    start_date: str | datetime.date = DEFAULT_START_DATE,
    end_date: str | datetime.date = DEFAULT_END_DATE,
) -> Callable[[Callable], ParamsFunction]:
    """
    Decorator that makes a `ParamsFunction` from a function.
    """
    start_date, end_date = _convert_and_validate_dates(start_date, end_date)

    def inner(func: Callable) -> ParamsFunction:
        return ParamsFunction(
            leaf_name=leaf_name if leaf_name else func.__name__,
            function=func,
            start_date=start_date,
            end_date=end_date,
        )

    return inner


# Always returns a column
def policy_function(
    *,
    leaf_name: str | None = None,
    start_date: str | datetime.date = DEFAULT_START_DATE,
    end_date: str | datetime.date = DEFAULT_END_DATE,
    rounding_spec: RoundingSpec | None = None,
    vectorization_strategy: Literal["loop", "vectorize", "not_required"] = "vectorize",
    foreign_key_type: FKType = FKType.IRRELEVANT,
) -> Callable[[Callable], PolicyFunction]:
    """
    Decorator that makes a `PolicyFunction` from a function.

    **Dates active (start_date, end_date, leaf_name):**

    Specifies that a PolicyFunction is only active between two dates, `start` and `end`.
    By using the `leaf_name` argument, you can specify a different name for the
    PolicyFunction in the functions tree.

    Note that even if you use this decorator with the `leaf_name` argument, you must
    ensure that the function name is unique in the file where it is defined. Otherwise,
    the function would be overwritten by the last function with the same name.

    **Rounding specification (rounding_spec):**

    Adds the way rounding is to be done to a PolicyFunction.

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
        Whether and how the function should be vectorized.
    foreign_key_type:
        Whether this is a foreign key and, if so, whether it may point to itself.

    Returns
    -------
    A decorator that returns a PolicyFunction object.
    """

    start_date, end_date = _convert_and_validate_dates(start_date, end_date)

    def inner(func: Callable) -> PolicyFunction:
        func = (
            func
            if vectorization_strategy == "not_required"
            else _vectorize_func(func, vectorization_strategy=vectorization_strategy)
        )
        return PolicyFunction(
            leaf_name=leaf_name if leaf_name else func.__name__,
            function=func,
            start_date=start_date,
            end_date=end_date,
            rounding_spec=rounding_spec,
            foreign_key_type=foreign_key_type,
        )

    return inner


def _vectorize_func(
    func: Callable, vectorization_strategy: Literal["loop", "vectorize"]
) -> Callable:
    if vectorization_strategy == "loop":
        vectorized = functools.wraps(func)(numpy.vectorize(func))
        vectorized.__signature__ = inspect.signature(func)  # type: ignore[attr-defined]
        vectorized.__globals__ = func.__globals__  # type: ignore[attr-defined]
        vectorized.__closure__ = func.__closure__  # type: ignore[attr-defined]
    elif vectorization_strategy == "vectorize":
        backend = "jax" if IS_JAX_INSTALLED else "numpy"
        vectorized = make_vectorizable(func, backend=backend)
    else:
        raise ValueError(
            f"Vectorization strategy {vectorization_strategy} is not supported. "
            "Use 'loop' or 'vectorize'."
        )
    return vectorized


@dataclass(frozen=True)
class GroupCreationFunction(TTSIMFunction):
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
        top_level_namespace: set[str],
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
            rounding_spec=self.rounding_spec,
            foreign_key_type=self.foreign_key_type,
        )


def group_creation_function(
    *,
    leaf_name: str | None = None,
    start_date: str | datetime.date = DEFAULT_START_DATE,
    end_date: str | datetime.date = DEFAULT_END_DATE,
) -> Callable[[Callable], GroupCreationFunction]:
    """
    Decorator that creates a group_by function from a function.
    """
    start_date, end_date = _convert_and_validate_dates(start_date, end_date)

    def decorator(func: Callable) -> GroupCreationFunction:
        _leaf_name = func.__name__ if leaf_name is None else leaf_name
        return GroupCreationFunction(
            leaf_name=_leaf_name,
            function=func,
            start_date=start_date,
            end_date=end_date,
        )

    return decorator


@dataclass(frozen=True)
class AggByGroupFunction(TTSIMFunction):
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
        top_level_namespace: set[str],
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
            rounding_spec=self.rounding_spec,
            foreign_key_type=self.foreign_key_type,
            orig_location=self.orig_location,
        )


def agg_by_group_function(
    *,
    leaf_name: str | None = None,
    start_date: str | datetime.date = DEFAULT_START_DATE,
    end_date: str | datetime.date = DEFAULT_END_DATE,
    agg_type: AggType,
) -> Callable[[Callable], AggByGroupFunction]:
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

    def inner(func: Callable) -> AggByGroupFunction:
        orig_location = f"{func.__module__}.{func.__name__}"
        args = set(inspect.signature(func).parameters)
        group_ids = {p for p in args if p.endswith("_id")}
        _fail_if_group_id_is_invalid(group_ids, orig_location)
        group_id = group_ids.pop()
        other_args = args - {group_id}
        if agg_type == AggType.COUNT:
            _fail_if_other_arg_is_present(other_args, orig_location)
            mapper = {"group_id": group_id}
        else:
            _fail_if_other_arg_is_invalid(other_args, orig_location)
            mapper = {"group_id": group_id, "column": other_args.pop()}
        if IS_JAX_INSTALLED:
            mapper["num_segments"] = f"{group_id}_num_segments"
        agg_func = dags.rename_arguments(
            func=agg_registry[agg_type],
            mapper=mapper,
        )

        return AggByGroupFunction(
            leaf_name=leaf_name if leaf_name else func.__name__,
            function=agg_func,
            start_date=start_date,
            end_date=end_date,
            foreign_key_type=FKType.IRRELEVANT,
            orig_location=f"{func.__module__}.{func.__name__}",
        )

    return inner


def _fail_if_group_id_is_invalid(group_ids: set[str], orig_location: str) -> None:
    if len(group_ids) != 1:
        raise ValueError(
            "Require exactly one group identifier ending with '_id' for "
            "aggregation by group. Got "
            f"{', '.join(group_ids) if group_ids else 'nothing'} in {orig_location}."
        )


def _fail_if_other_arg_is_present(other_args: set[str], orig_location: str) -> None:
    if other_args:
        raise ValueError(
            "There must be no argument besides identifiers for counting. Got: "
            f"{', '.join(other_args) if other_args else 'nothing'} in {orig_location}."
        )


def _fail_if_other_arg_is_invalid(other_args: set[str], orig_location: str) -> None:
    if len(other_args) != 1:
        raise ValueError(
            "There must be exactly one argument besides identifiers for aggregations. "
            "Got: "
            f"{', '.join(other_args) if other_args else 'nothing'} in {orig_location}."
        )


@dataclass(frozen=True)
class AggByPIDFunction(TTSIMFunction):
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
        top_level_namespace: set[str],
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
            rounding_spec=self.rounding_spec,
            foreign_key_type=self.foreign_key_type,
            orig_location=self.orig_location,
        )


def agg_by_p_id_function(
    *,
    leaf_name: str | None = None,
    start_date: str | datetime.date = DEFAULT_START_DATE,
    end_date: str | datetime.date = DEFAULT_END_DATE,
    agg_type: AggType,
) -> Callable[[Callable], AggByPIDFunction]:
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

    def inner(func: Callable) -> AggByPIDFunction:
        orig_location = f"{func.__module__}.{func.__name__}"
        args = set(inspect.signature(func).parameters)
        other_p_ids = {
            p
            for p in args
            if any(e.startswith("p_id_") for e in dt.tree_path_from_qual_name(p))
        }
        other_args = args - {*other_p_ids, "p_id"}
        _fail_if_p_id_is_not_present(args, orig_location)
        _fail_if_other_p_id_is_invalid(other_p_ids, orig_location)
        if agg_type == AggType.COUNT:
            _fail_if_other_arg_is_present(other_args, orig_location)
            mapper = {
                "p_id_to_aggregate_by": other_p_ids.pop(),
                "p_id_to_store_by": "p_id",
            }
        else:
            _fail_if_other_arg_is_invalid(other_args, orig_location)
            mapper = {
                "column": other_args.pop(),
                "p_id_to_aggregate_by": other_p_ids.pop(),
                "p_id_to_store_by": "p_id",
            }
        agg_func = dags.rename_arguments(
            func=agg_registry[agg_type],
            mapper=mapper,
        )

        functools.update_wrapper(agg_func, func)
        agg_func.__signature__ = inspect.signature(func)  # type: ignore[attr-defined]

        return AggByPIDFunction(
            leaf_name=leaf_name if leaf_name else func.__name__,
            function=agg_func,
            start_date=start_date,
            end_date=end_date,
            foreign_key_type=FKType.IRRELEVANT,
            orig_location=f"{func.__module__}.{func.__name__}",
        )

    return inner


def _fail_if_p_id_is_not_present(args: set[str], orig_location: str) -> None:
    if "p_id" not in args:
        raise ValueError(
            "The function must have the argument named 'p_id' for aggregation by p_id. "
            f"Got {', '.join(args) if args else 'nothing'} in {orig_location}."
        )


def _fail_if_other_p_id_is_invalid(other_p_ids: set[str], orig_location: str) -> None:
    if len(other_p_ids) != 1:
        raise ValueError(
            "Require exactly one identifier starting with 'p_id_' for "
            "aggregation by p_id. Got: "
            f"{', '.join(other_p_ids) if other_p_ids else 'nothing'} in {orig_location}."  # noqa: E501
        )


@dataclass(frozen=True)
class TimeConversionFunction(TTSIMFunction):
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
        top_level_namespace: set[str],
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
            rounding_spec=self.rounding_spec,
            foreign_key_type=self.foreign_key_type,
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

    validate_date_range(start_date, end_date)

    return start_date, end_date


def check_series_has_expected_type(series: pd.Series, internal_type: np.dtype) -> bool:
    """Checks whether used series has already expected internal type.

    Parameters
    ----------
    series : pandas.Series or pandas.DataFrame or dict of pandas.Series
        Data provided by the user.
    internal_type : TypeVar
        One of the internal gettsim types.

    Returns
    -------
    Bool

    """
    if (
        (internal_type == float) & (is_float_dtype(series))
        or (internal_type == int) & (is_integer_dtype(series))
        or (internal_type == bool) & (is_bool_dtype(series))
        or (internal_type == numpy.datetime64) & (is_datetime64_any_dtype(series))
    ):
        out = True
    else:
        out = False

    return out
