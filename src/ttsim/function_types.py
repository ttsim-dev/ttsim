from __future__ import annotations

import datetime
import functools
import inspect
import re
from collections.abc import Callable
from typing import Literal, TypeVar

import dags.tree as dt
import numpy

from ttsim.config import IS_JAX_INSTALLED
from ttsim.rounding import RoundingSpec
from ttsim.vectorization import make_vectorizable

T = TypeVar("T")


class PolicyFunction(Callable):
    """
    A function that computes an output vector based on some input vectors and/or
    parameters.

    Parameters
    ----------
    function:
        The function to wrap. Argument values of the `@policy_function` are reused
        unless explicitly overwritten.
    leaf_name:
        The leaf name of the function in the functions tree.
    start_date:
        The date from which the function is active (inclusive).
    end_date:
        The date until which the function is active (inclusive).
    rounding_spec:
        The rounding specification.
    skip_vectorization:
        Whether the function should be vectorized.
    """

    def __init__(
        self,
        *,
        function: Callable,
        leaf_name: str,
        start_date: datetime.date,
        end_date: datetime.date,
        vectorization_strategy: Literal["loop", "vectorize", "not_required"],
        rounding_spec: RoundingSpec | None,
    ):
        self.vectorization_strategy: Literal["loop", "vectorize", "not_required"] = (
            vectorization_strategy
        )
        self.function = (
            function
            if self.vectorization_strategy == "not_required"
            else _vectorize_func(
                function, vectorization_strategy=self.vectorization_strategy
            )
        )
        self.leaf_name: str = leaf_name if leaf_name else function.__name__
        self.start_date: datetime.date = start_date
        self.end_date: datetime.date = end_date
        self._fail_if_rounding_has_wrong_type(rounding_spec)
        self.rounding_spec: RoundingSpec | None = rounding_spec

        # Expose the signature of the wrapped function for dependency resolution
        functools.update_wrapper(self, self.function)
        self.__signature__ = inspect.signature(self.function)
        self.__globals__ = self.function.__globals__
        self.__closure__ = self.function.__closure__

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

    def __call__(self, *args, **kwargs):
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


def policy_function(
    *,
    start_date: str | datetime.date = "1900-01-01",
    end_date: str | datetime.date = "2100-12-31",
    leaf_name: str | None = None,
    vectorization_strategy: Literal["loop", "vectorize", "not_required"] = "loop",
    rounding_spec: RoundingSpec | None = None,
) -> PolicyFunction:
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
    start_date
        The start date (inclusive) in the format YYYY-MM-DD (part of ISO 8601).
    end_date
        The end date (inclusive) in the format YYYY-MM-DD (part of ISO 8601).
    leaf_name
        The name that should be used as the PolicyFunction's leaf name in the DAG. If
        omitted, we use the name of the function as defined.
    rounding_spec
        The specification to be used for rounding.
    skip_vectorization
        Whether the function is already vectorized and, thus, should not be vectorized
        again.

    Returns
    -------
    A PolicyFunction object.
    """

    _validate_dashed_iso_date(start_date)
    _validate_dashed_iso_date(end_date)

    start_date = datetime.date.fromisoformat(start_date)
    end_date = datetime.date.fromisoformat(end_date)

    _validate_date_range(start_date, end_date)

    def inner(func: Callable) -> PolicyFunction:
        return PolicyFunction(
            function=func,
            leaf_name=leaf_name if leaf_name else func.__name__,
            start_date=start_date,
            end_date=end_date,
            vectorization_strategy=vectorization_strategy,
            rounding_spec=rounding_spec,
        )

    return inner


_DASHED_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _validate_dashed_iso_date(date: str | datetime.date):
    if not _DASHED_ISO_DATE.match(date):
        raise ValueError(f"Date {date} does not match the format YYYY-MM-DD.")


def _validate_date_range(start: datetime.date, end: datetime.date):
    if start > end:
        raise ValueError(f"The start date {start} must be before the end date {end}.")


def _vectorize_func(
    func: Callable, vectorization_strategy: Literal["loop", "vectorize"]
) -> Callable:
    if vectorization_strategy == "loop":
        vectorized = functools.wraps(func)(numpy.vectorize(func))
        vectorized.__signature__ = inspect.signature(func)
        vectorized.__globals__ = func.__globals__
        vectorized.__closure__ = func.__closure__
    elif vectorization_strategy == "vectorize":
        backend = "jax" if IS_JAX_INSTALLED else "numpy"
        vectorized = make_vectorizable(func, backend=backend)
    else:
        raise ValueError(
            f"Vectorization strategy {vectorization_strategy} is not supported. "
            "Use 'loop' or 'vectorize'."
        )
    return vectorized


class GroupByFunction(Callable):
    """
    A function that computes endogenous group_by IDs.

    Parameters
    ----------
    function:
        The group_by function.
    """

    def __init__(
        self,
        *,
        function: Callable,
        leaf_name: str | None = None,
    ):
        self.function = function
        self.leaf_name = leaf_name if leaf_name else function.__name__

        # Expose the signature of the wrapped function for dependency resolution
        self.__annotations__ = function.__annotations__
        self.__module__ = function.__module__
        self.__name__ = function.__name__
        self.__signature__ = inspect.signature(self.function)

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

    @property
    def dependencies(self) -> set[str]:
        """The names of input variables that the function depends on."""
        return set(inspect.signature(self).parameters)


def group_by_function() -> GroupByFunction:
    """
    Decorator that creates a group_by function from a function.
    """

    def decorator(func: Callable) -> GroupByFunction:
        return GroupByFunction(function=func)

    return decorator


class DerivedAggregationFunction(PolicyFunction):
    """
    A function that is an aggregation of another function.

    Parameters
    ----------
    function:
        The function to wrap. Argument values of the `@policy_function` are reused
        unless explicitly overwritten.
    aggregation_target:
        The qualified name of the aggregation target.
    source_function:
        The function from which the new function is derived.
    source:
        The name of the source function or data column.
    aggregation_method:
        The method of aggregation used.
    """

    def __init__(
        self,
        *,
        function: Callable,
        source_function: PolicyFunction
        | DerivedTimeConversionFunction
        | DerivedAggregationFunction
        | None = None,
        source: str,
        aggregation_target: str,
        aggregation_method: Literal["count", "sum", "mean", "min", "max", "any", "all"],
    ):
        super().__init__(
            function=function,
            leaf_name=dt.tree_path_from_qual_name(aggregation_target)[-1],
            start_date=source_function.start_date if source_function else None,
            end_date=source_function.end_date if source_function else None,
            vectorization_strategy="not_required",
            rounding_spec=None,
        )

        self.source = source
        self.aggregation_method = aggregation_method


class DerivedTimeConversionFunction(PolicyFunction):
    """
    A function that is a time conversion of another function.

    Parameters
    ----------
    function:
        The function to wrap. Argument values of the `@policy_function` are reused
        unless explicitly overwritten.
    source_function:
        The function from which the new function is derived.
    source:
        The name of the source function or data column.
    conversion_target:
        The qualified name of the conversion target.
    """

    def __init__(
        self,
        *,
        function: Callable,
        source_function: PolicyFunction
        | DerivedTimeConversionFunction
        | DerivedAggregationFunction
        | None = None,
        source: str,
        conversion_target: str,
    ):
        super().__init__(
            function=function,
            leaf_name=dt.tree_path_from_qual_name(conversion_target)[-1],
            start_date=source_function.start_date if source_function else None,
            end_date=source_function.end_date if source_function else None,
            vectorization_strategy="not_required",
            rounding_spec=None,
        )

        self.source = source
