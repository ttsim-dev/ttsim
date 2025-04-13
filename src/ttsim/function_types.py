from __future__ import annotations

import datetime
import functools
import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeVar

import numpy

from ttsim.rounding import RoundingSpec
from ttsim.shared import to_datetime, validate_date_range

if TYPE_CHECKING:
    from collections.abc import Callable

    from ttsim.typing import DashedISOString

T = TypeVar("T")

DEFAULT_START_DATE = datetime.date(1900, 1, 1)
DEFAULT_END_DATE = datetime.date(2100, 12, 31)


@dataclass
class TTSIMObject:
    """
    Abstract base class for all TTSIM Functions and Inputs.
    """

    leaf_name: str
    start_date: datetime.date
    end_date: datetime.date

    def is_active(self, date: datetime.date) -> bool:
        """Check if the function is active at a given date."""
        return self.start_date <= date <= self.end_date


@dataclass
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
    """

    data_type: type[float | int | bool]


def policy_input(
    *,
    start_date: str | datetime.date = DEFAULT_START_DATE,
    end_date: str | datetime.date = DEFAULT_END_DATE,
) -> PolicyInput:
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
    A PolicyInput object.
    """
    start_date, end_date = _convert_and_validate_dates(start_date, end_date)

    def inner(func: Callable) -> PolicyInput:
        data_type = func.__annotations__["return"]
        return PolicyInput(
            leaf_name=func.__name__,
            data_type=data_type,
            start_date=start_date,
            end_date=end_date,
        )

    return inner


@dataclass
class TTSIMFunction(TTSIMObject):
    """
    Base class for all TTSIM functions.
    """

    function: Callable
    skip_vectorization: bool = False

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


@dataclass
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
    skip_vectorization:
        Whether the function should be vectorized.
    """

    rounding_spec: RoundingSpec | None = None

    def __post_init__(self):
        self._fail_if_rounding_has_wrong_type(self.rounding_spec)
        self.function = (
            self.function if self.skip_vectorization else _vectorize_func(self.function)
        )

        # Expose the signature of the wrapped function for dependency resolution
        self.__annotations__ = self.function.__annotations__
        self.__module__ = self.function.__module__
        self.__name__ = self.function.__name__
        self.__signature__ = inspect.signature(self.function)

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
    leaf_name: str | None = None,
    start_date: str | datetime.date = DEFAULT_START_DATE,
    end_date: str | datetime.date = DEFAULT_END_DATE,
    rounding_spec: RoundingSpec | None = None,
    skip_vectorization: bool = False,
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
    leaf_name
        The name that should be used as the PolicyFunction's leaf name in the DAG. If
        omitted, we use the name of the function as defined.
    start_date
        The start date (inclusive) in the format YYYY-MM-DD (part of ISO 8601).
    end_date
        The end date (inclusive) in the format YYYY-MM-DD (part of ISO 8601).
    rounding_spec
        The specification to be used for rounding.
    skip_vectorization
        Whether the function is already vectorized and, thus, should not be vectorized
        again.

    Returns
    -------
    A PolicyFunction object.
    """

    start_date, end_date = _convert_and_validate_dates(start_date, end_date)

    def inner(func: Callable) -> PolicyFunction:
        return PolicyFunction(
            leaf_name=leaf_name if leaf_name else func.__name__,
            function=func,
            start_date=start_date,
            end_date=end_date,
            rounding_spec=rounding_spec,
            skip_vectorization=skip_vectorization,
        )

    return inner


def _vectorize_func(func: Callable) -> Callable:
    # What should work once that Jax backend is fully supported
    signature = inspect.signature(func)
    func_vec = numpy.vectorize(func)

    @functools.wraps(func)
    def wrapper_vectorize_func(*args, **kwargs):
        return func_vec(*args, **kwargs)

    wrapper_vectorize_func.__signature__ = signature

    return wrapper_vectorize_func


@dataclass
class GroupByFunction(TTSIMFunction):
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

    def __post_init__(self):
        # Expose the signature of the wrapped function for dependency resolution
        self.__annotations__ = self.function.__annotations__
        self.__module__ = self.function.__module__
        self.__name__ = self.function.__name__
        self.__signature__ = inspect.signature(self.function)

    @property
    def dependencies(self) -> set[str]:
        """The names of input variables that the function depends on."""
        return set(inspect.signature(self).parameters)


def group_by_function(
    *,
    leaf_name: str | None = None,
    start_date: str | datetime.date = DEFAULT_START_DATE,
    end_date: str | datetime.date = DEFAULT_END_DATE,
) -> GroupByFunction:
    """
    Decorator that creates a group_by function from a function.
    """
    start_date, end_date = _convert_and_validate_dates(start_date, end_date)

    def decorator(func: Callable) -> GroupByFunction:
        _leaf_name = func.__name__ if leaf_name is None else leaf_name
        return GroupByFunction(
            leaf_name=_leaf_name,
            function=func,
            start_date=start_date,
            end_date=end_date,
        )

    return decorator


@dataclass
class DerivedAggregationFunction(TTSIMFunction):
    """
    A function that is an aggregation of another function.

    Parameters
    ----------
    leaf_name:
        The leaf name of the function in the functions tree.
    function:
        The function performing the aggregation.
    source:
        The name of the source function or data column.
    aggregation_method:
        The method of aggregation used.
    start_date:
        The date from which the function is active (inclusive).
    end_date:
        The date until which the function is active (inclusive).
    params_key_for_rounding:
        The key in the params dictionary that should be used for rounding.
    skip_vectorization:
        Whether the function should be vectorized.
    """

    source: str | None = None
    aggregation_method: (
        Literal["count", "sum", "mean", "min", "max", "any", "all"] | None
    ) = None

    def __post_init__(self):
        if self.aggregation_method is None:
            raise ValueError("The aggregation method must be specified.")
        if self.source is None and self.aggregation_method != "count":
            raise ValueError("The source must be specified.")

        # Expose the signature of the wrapped function for dependency resolution
        self.__annotations__ = self.function.__annotations__
        self.__module__ = self.function.__module__
        self.__name__ = self.function.__name__
        self.__signature__ = inspect.signature(self.function)


@dataclass
class DerivedTimeConversionFunction(TTSIMFunction):
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
    params_key_for_rounding:
        The key in the params dictionary that should be used for rounding.
    skip_vectorization:
        Whether the function should be vectorized.
    """

    source: str | None = None

    def __post_init__(self):
        if self.source is None:
            raise ValueError("The source must be specified.")

        # Expose the signature of the wrapped function for dependency resolution
        self.__annotations__ = self.function.__annotations__
        self.__module__ = self.function.__module__
        self.__name__ = self.function.__name__
        self.__signature__ = inspect.signature(self.function)


def _convert_and_validate_dates(
    start_date: DashedISOString | datetime.date,
    end_date: DashedISOString | datetime.date,
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
    if isinstance(start_date, str):
        to_datetime(start_date)
        start_date = datetime.date.fromisoformat(start_date)
    if isinstance(end_date, str):
        to_datetime(end_date)
        end_date = datetime.date.fromisoformat(end_date)

    validate_date_range(start_date, end_date)

    return start_date, end_date
