from __future__ import annotations

from dataclasses import dataclass, field
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from jaxtyping import Bool, Float, Int

from ttsim.exceptions import UnitDefinitionError
from ttsim.tt.units import (
    UNSET_UNIT,
    CompositeUnit,
    coerce_unit_token,
)
from ttsim.typing import DictParamValue, NestedLookupDict

# Backend-agnostic array type: union the (optional) JAX `Array` with
# `np.ndarray` so `Float[Array, ...]`/`Int[Array, ...]` annotations accept
# NumPy arrays under the JAX env. Bare `jax.Array` would make the package
# claw reject every NumPy-backed value (see `ttsim.typing` column aliases).
try:
    from jax import Array as _JaxArray

    Array = _JaxArray | np.ndarray
except ImportError:
    Array = np.ndarray

PLACEHOLDER_VALUE: Any = object()
# `Any` so dataclass fields of any narrow type accept the sentinel default.
# The runtime check (`is PLACEHOLDER_VALUE`) enforces the real constraint.
PLACEHOLDER_FIELD: Any = field(default_factory=lambda: PLACEHOLDER_VALUE)

if TYPE_CHECKING:
    import datetime


@dataclass(frozen=True)
class ParamObject:
    """
    Abstract base class for all types of parameters.
    """

    start_date: datetime.date | None = None
    end_date: datetime.date | None = None
    unit: CompositeUnit | str | dict[str | int, Any] = UNSET_UNIT
    """The parameter's compositional unit, e.g. ``CURRENCY_PER_MONTH``,
    ``SILVER_PENNY_PER_FAM``, or a bare base ``DIMENSIONLESS``. A parameter
    spells period *and* level fully. A dict parameter with heterogeneous leaves
    declares a mapping from leaf names to units instead. A concrete currency base
    (``SILVER_PENNY``, ``DM``, …) also names the currency the numbers are written
    in, which the build-time currency conversion reads off. YAML strings are
    coerced to :class:`CompositeUnit` at construction; :data:`UNSET_UNIT` until
    annotated."""
    name: dict[Literal["de", "en"], str] | None = None
    description: dict[Literal["de", "en"], str] | None = None

    def __post_init__(self) -> None:
        if getattr(self, "value", PLACEHOLDER_VALUE) is PLACEHOLDER_VALUE:
            raise ValueError(
                "'value' field must be specified for any type of 'ParamObject'"
            )
        # object.__setattr__ because the dataclass is frozen.
        object.__setattr__(
            self, "unit", _coerce_declared_unit(declared=self.unit, obj=self)
        )


def _coerce_declared_unit(
    declared: Any,  # noqa: ANN401 (raw YAML value)
    obj: ParamObject,
) -> CompositeUnit | dict[str | int, Any]:
    """Coerce a raw ``unit:`` declaration to a CompositeUnit, recursing into
    mappings."""
    name_en = (obj.name or {}).get("en")
    where = f"Parameter {name_en}" if name_en else "Parameter"
    if declared is UNSET_UNIT:
        return UNSET_UNIT
    if isinstance(declared, dict):
        return {
            key: _coerce_declared_unit(declared=sub, obj=obj)
            if isinstance(sub, dict)
            # Present leaves are tokens (``DIMENSIONLESS`` for a dimensionless leaf).
            else coerce_unit_token(value=sub, where=f"{where} (unit of leaf {key!r})")
            for key, sub in declared.items()
        }
    return coerce_unit_token(value=declared, where=where)


@dataclass(frozen=True)
class ScalarParam(ParamObject):
    """
    A scalar parameter directly read from a YAML file.
    """

    value: bool | int | float = PLACEHOLDER_FIELD
    note: str | None = None
    reference: str | None = None


@dataclass(frozen=True)
class DictParam(ParamObject):
    """
    A parameter directly read from a YAML file that is a (possibly nested)
    dictionary.
    """

    value: DictParamValue = PLACEHOLDER_FIELD
    note: str | None = None
    reference: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if any(x in self.value for x in ["note", "reference"]):
            raise ValueError(
                "'note' and 'reference' cannot be keys in the value dictionary"
            )


@dataclass(frozen=True)
class ParamMappingObject(ParamObject):
    """Base class for parameters that are functions between quantities.

    A schedule or lookup table has a domain and a codomain, so it declares
    ``input_unit:`` and ``output_unit:`` (one token per axis) instead of
    ``unit:``. The build-time currency conversion rescales interval bounds on
    the input axis and intercepts on the output axis.
    """

    input_unit: CompositeUnit | str = UNSET_UNIT
    """The unit of the input axis (what the parameter is evaluated at), e.g.
    ``CURRENCY_PER_YEAR`` or ``HECTARE``. :data:`UNSET_UNIT` until annotated."""
    output_unit: CompositeUnit | str = UNSET_UNIT
    """The unit of the output axis (what the parameter yields).
    :data:`UNSET_UNIT` until annotated."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.unit is not UNSET_UNIT:
            raise UnitDefinitionError(
                f"{type(self).__name__} is a function between quantities and "
                f"declares `input_unit:` / `output_unit:` instead of `unit:` "
                f"(GEP 10); got unit={self.unit!r}."
            )
        for axis in ("input_unit", "output_unit"):
            raw = getattr(self, axis)
            if isinstance(raw, dict):
                raise UnitDefinitionError(
                    f"{type(self).__name__}: per-axis declarations are single "
                    f"tokens, not mappings (GEP 10); got {axis}={raw!r}."
                )
            object.__setattr__(
                self, axis, _coerce_declared_unit(declared=raw, obj=self)
            )


@dataclass(frozen=True)
class PiecewisePolynomialParam(ParamMappingObject):
    """A parameter with its contents read and converted from a YAML file.

    Its value is a PiecewisePolynomialParamValue object, i.e., it contains the
    parameters for calling `piecewise_polynomial`.
    """

    value: PiecewisePolynomialParamValue = PLACEHOLDER_FIELD
    note: str | None = None
    reference: str | None = None


@dataclass(frozen=True)
class ConsecutiveIntLookupTableParam(ParamMappingObject):
    """A parameter with its contents read and converted from a YAML file.

    Its value is a ConsecutiveIntLookupTableParamValue object, i.e., it contains the
    parameters for calling `lookup_table`.
    """

    value: ConsecutiveIntLookupTableParamValue = PLACEHOLDER_FIELD
    note: str | None = None
    reference: str | None = None


class ConsecutiveIntLookupTableParamValue:
    """The `value` for ConsecutiveIntLookupTable."""

    __slots__ = (
        "bases_to_subtract",
        "lookup_multipliers",
        "values_to_look_up",
        "xnp",
    )

    bases_to_subtract: Int[Array, "n_rows n_cols"]
    lookup_multipliers: Int[Array, "n_rows n_cols"]
    values_to_look_up: (
        Float[Array | np.ndarray, ...]
        | Int[Array | np.ndarray, ...]
        | Bool[Array | np.ndarray, ...]
    )
    xnp: ModuleType

    def __init__(
        self,
        xnp: ModuleType,
        values_to_look_up: Float[Array | np.ndarray, ...]
        | Int[Array | np.ndarray, ...]
        | Bool[Array | np.ndarray, ...],
        bases_to_subtract: Int[Array | np.ndarray, ...],
    ) -> None:
        self.xnp = xnp
        self.values_to_look_up = values_to_look_up.flatten()
        self.bases_to_subtract = xnp.expand_dims(bases_to_subtract, axis=1)
        self.lookup_multipliers = xnp.concatenate(
            [
                (xnp.cumprod(xnp.asarray(values_to_look_up.shape)[::-1])[::-1])[1:],
                xnp.asarray([1]),
            ]
        )

    def look_up(
        self: ConsecutiveIntLookupTableParamValue,
        *args: int | np.integer | Int[Array | np.ndarray, ...],
    ) -> (
        float
        | int
        | bool
        | np.floating
        | np.integer
        | np.bool_
        | Float[Array | np.ndarray, ...]
        | Int[Array | np.ndarray, ...]
        | Bool[Array | np.ndarray, ...]
    ):
        """Look up value(s) for the given index argument(s).

        Each argument is a scalar integer index or an integer array of
        indices (one per table dimension). Scalar arguments yield a scalar
        result; array arguments yield an array of the looked-up values.
        """
        scalar_input = all(getattr(a, "ndim", 0) == 0 for a in args)
        index = self.xnp.asarray(args)
        if scalar_input:
            index = index.reshape(-1, 1)
        corrected_index = self.xnp.dot(
            (index - self.bases_to_subtract).T, self.lookup_multipliers
        )
        result = self.values_to_look_up[corrected_index]
        if scalar_input:
            return result.flatten()[0]
        return result


@dataclass(frozen=True)
class RawParam(ParamObject):
    """
    A parameter directly read from a YAML file that is an arbitrarily nested
    dictionary.

    A ``require_converter`` is handed to a ``param_function`` that knows its
    structure. For currency conversion it declares one of three shapes: a single
    ``unit:`` token if the whole structure is homogeneously one unit (scaled
    uniformly), a per-leaf ``unit:`` mapping if the structure mixes units (each
    numeric leaf scaled by its own token), or ``input_unit:`` / ``output_unit:``
    axes if its converter produces a function-like value (a piecewise schedule
    or lookup table) whose output is converted per-axis.
    """

    value: dict[str | int, Any] = PLACEHOLDER_FIELD
    note: str | None = None
    reference: str | None = None
    input_unit: CompositeUnit | str = UNSET_UNIT
    """The input-axis unit of a function-like converter's output; mutually
    exclusive with :attr:`unit`. :data:`UNSET_UNIT` until annotated."""
    output_unit: CompositeUnit | str = UNSET_UNIT
    """The output-axis unit of a function-like converter's output; mutually
    exclusive with :attr:`unit`. :data:`UNSET_UNIT` until annotated."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if any(x in self.value for x in ["note", "reference"]):
            raise ValueError(
                "'note' and 'reference' cannot be keys in the value dictionary"
            )
        declares_axes = (
            self.input_unit is not UNSET_UNIT or self.output_unit is not UNSET_UNIT
        )
        if declares_axes and self.unit is not UNSET_UNIT:
            raise UnitDefinitionError(
                "A require_converter declares either `unit:` (a single token or "
                "a per-leaf mapping, scaled leaf by leaf) or `input_unit:` / "
                "`output_unit:` axes (a function-like output, converted "
                f"per-axis), not both (GEP 10); got unit={self.unit!r}, "
                f"input_unit={self.input_unit!r}, output_unit={self.output_unit!r}."
            )
        for axis in ("input_unit", "output_unit"):
            object.__setattr__(
                self,
                axis,
                _coerce_declared_unit(declared=getattr(self, axis), obj=self),
            )


@dataclass(frozen=True)
class PiecewisePolynomialInterval:
    """A single interval of a piecewise polynomial."""

    intercept: float | Float[Array, ""] | Int[Array, ""]
    coefficients: Float[Array, " n_coefficients"] | Int[Array, " n_coefficients"]

    _MIN_COEFFICIENTS_LINEAR = 1
    _MIN_COEFFICIENTS_QUADRATIC = 2
    _MIN_COEFFICIENTS_CUBIC = 3

    @property
    def slope(self) -> float | Float[Array, ""] | Int[Array, ""]:
        """The first coefficient (linear term)."""
        if self.coefficients.shape[0] < self._MIN_COEFFICIENTS_LINEAR:
            raise AttributeError("No slope coefficient for piecewise_constant.")
        return self.coefficients[0]

    @property
    def quadratic(self) -> float | Float[Array, ""] | Int[Array, ""]:
        """The second coefficient (quadratic term)."""
        if self.coefficients.shape[0] < self._MIN_COEFFICIENTS_QUADRATIC:
            raise AttributeError(
                "No quadratic coefficient; requires piecewise_quadratic or higher."
            )
        return self.coefficients[1]

    @property
    def cubic(self) -> float | Float[Array, ""] | Int[Array, ""]:
        """The third coefficient (cubic term)."""
        if self.coefficients.shape[0] < self._MIN_COEFFICIENTS_CUBIC:
            raise AttributeError("No cubic coefficient; requires piecewise_cubic.")
        return self.coefficients[2]


@dataclass(frozen=True)
class PiecewisePolynomialParamValue:
    """The parameters expected by `piecewise_polynomial`.

    thresholds:
        Boundary points defining the pieces / different segments.
    intercepts:
        Intercepts of the polynomial on each segment (one per interval).
    coefficients:
        Coefficients of the polynomial on each segment, shape
        (n_intervals, n_coefficients). For piecewise_constant, this is
        (n_intervals, 1) with all zeros.
    """

    # Thresholds, intercepts, and coefficients are parsed from YAML and may be
    # integer- or float-dtyped, so each accepts both jaxtyping array kinds.
    thresholds: Float[Array, " n_thresholds"] | Int[Array, " n_thresholds"]
    intercepts: Float[Array, " n_intervals"] | Int[Array, " n_intervals"]
    coefficients: (
        Float[Array, "n_intervals n_coefficients"]
        | Int[Array, "n_intervals n_coefficients"]
    )

    def __getitem__(self, index: int) -> PiecewisePolynomialInterval:
        return PiecewisePolynomialInterval(
            intercept=self.intercepts[index],
            coefficients=self.coefficients[index],
        )


def get_consecutive_int_lookup_table_param_value(
    raw: NestedLookupDict,
    xnp: ModuleType,
) -> ConsecutiveIntLookupTableParamValue:
    """Get the parameters for a N-dimensional lookup table."""
    bases_to_substract = {}

    # Function is recursive to step through all levels of dict. The leaves of
    # `NestedLookupDict` may be int, float, or bool, so the produced array can
    # be of any of those dtypes -- match `values_to_look_up`'s union.
    def process_level(
        i: int, level_i_dict: NestedLookupDict
    ) -> (
        Float[Array | np.ndarray, ...]
        | Int[Array | np.ndarray, ...]
        | Bool[Array | np.ndarray, ...]
    ):
        sorted_keys = sorted(level_i_dict.keys())
        bases_to_substract[i] = min(xnp.asarray(sorted_keys))
        if isinstance(level_i_dict[sorted_keys[0]], dict):
            return xnp.concatenate(
                [
                    xnp.expand_dims(process_level(i + 1, level_i_dict[key]), axis=0)
                    for key in level_i_dict
                ]
            )
        return xnp.asarray([level_i_dict[k] for k in sorted_keys])

    values = process_level(0, raw)
    return ConsecutiveIntLookupTableParamValue(
        xnp=xnp,
        values_to_look_up=values,
        bases_to_subtract=xnp.asarray(
            [bases_to_substract[key] for key in sorted(bases_to_substract.keys())]
        ),
    )


def _year_fraction(spec: dict[Literal["years", "months"], int]) -> float:
    return spec["years"] + spec["months"] / 12


def get_month_based_phase_inout_of_age_thresholds_param_value(
    raw: dict[str | int, Any],
    xnp: ModuleType,
) -> ConsecutiveIntLookupTableParamValue:
    """Get the parameters for month-based phase-in/phase-out of age thresholds.

    Fills up months for which no parameters are given with the last given value.
    """

    def _m_since_ad(y: int, m: int) -> int:
        return y * 12 + (m - 1)

    def _fill_phase_inout(
        raw: dict[int, dict[int, dict[Literal["years", "months"], int]]],
        first_m_since_ad_phase_inout: int,
        last_m_since_ad_phase_inout: int,
    ) -> dict[int, float]:
        lookup_table = {}
        for y, m_dict in raw.items():
            for m, v in m_dict.items():
                lookup_table[_m_since_ad(y=y, m=m)] = _year_fraction(v)
        for m in range(first_m_since_ad_phase_inout, last_m_since_ad_phase_inout):
            if m not in lookup_table:
                lookup_table[m] = lookup_table[m - 1]
        return lookup_table

    first_m_since_ad_to_consider = _m_since_ad(y=raw.pop("first_year_to_consider"), m=1)
    last_m_since_ad_to_consider = _m_since_ad(y=raw.pop("last_year_to_consider"), m=12)
    if not all(isinstance(k, int) for k in raw):
        raise ValueError("All keys must be integers")
    int_raw = cast("dict[int, Any]", raw)
    first_year_phase_inout: int = min(int_raw.keys())
    first_month_phase_inout: int = min(int_raw[first_year_phase_inout].keys())
    first_m_since_ad_phase_inout = _m_since_ad(
        y=first_year_phase_inout,
        m=first_month_phase_inout,
    )
    last_year_phase_inout: int = max(int_raw.keys())
    last_month_phase_inout: int = max(int_raw[last_year_phase_inout].keys())
    last_m_since_ad_phase_inout = _m_since_ad(
        y=last_year_phase_inout,
        m=last_month_phase_inout,
    )
    if first_m_since_ad_to_consider > first_m_since_ad_phase_inout:
        raise ValueError(
            "`first_m_since_ad_to_consider` must be less than or equal to "
            "`first_m_since_ad_phase_inout`."
        )
    if last_m_since_ad_to_consider < last_m_since_ad_phase_inout:
        raise ValueError(
            "`last_m_since_ad_to_consider` must be greater than or equal to "
            "`last_m_since_ad_phase_inout`."
        )
    before_phase_inout: dict[int, float] = {
        b_m: _year_fraction(int_raw[first_year_phase_inout][first_month_phase_inout])
        for b_m in range(first_m_since_ad_to_consider, first_m_since_ad_phase_inout)
    }
    during_phase_inout: dict[int, float] = _fill_phase_inout(
        raw=int_raw,
        first_m_since_ad_phase_inout=first_m_since_ad_phase_inout,
        last_m_since_ad_phase_inout=last_m_since_ad_phase_inout,
    )
    after_phase_inout: dict[int, float] = {
        b_m: _year_fraction(int_raw[last_year_phase_inout][last_month_phase_inout])
        for b_m in range(
            last_m_since_ad_phase_inout + 1,
            last_m_since_ad_to_consider + 1,
        )
    }
    return get_consecutive_int_lookup_table_param_value(
        raw={**before_phase_inout, **during_phase_inout, **after_phase_inout},
        xnp=xnp,
    )


def get_year_based_phase_inout_of_age_thresholds_param_value(
    raw: dict[str | int, Any],
    xnp: ModuleType,
) -> ConsecutiveIntLookupTableParamValue:
    """Get the parameters for year-based phase-in/phase-out of age thresholds.

    Requires all years to be given.
    """
    first_year_to_consider = raw.pop("first_year_to_consider")
    last_year_to_consider = raw.pop("last_year_to_consider")
    if not all(isinstance(k, int) for k in raw):
        raise ValueError("All keys must be integers")
    int_raw = cast("dict[int, Any]", raw)
    first_year_phase_inout: int = sorted(int_raw)[0]
    last_year_phase_inout: int = sorted(int_raw)[-1]
    if first_year_to_consider > first_year_phase_inout:
        raise ValueError(
            "`first_year_to_consider` must be less than or equal to "
            "`first_year_phase_inout`."
        )
    if last_year_to_consider < last_year_phase_inout:
        raise ValueError(
            "`last_year_to_consider` must be greater than or equal to "
            "`last_year_phase_inout`."
        )
    before_phase_inout: dict[int, float] = {
        b_y: _year_fraction(int_raw[first_year_phase_inout])
        for b_y in range(first_year_to_consider, first_year_phase_inout)
    }
    during_phase_inout: dict[int, float] = {
        b_y: _year_fraction(spec) for b_y, spec in int_raw.items()
    }
    after_phase_inout: dict[int, float] = {
        b_y: _year_fraction(int_raw[last_year_phase_inout])
        for b_y in range(last_year_phase_inout + 1, last_year_to_consider + 1)
    }
    return get_consecutive_int_lookup_table_param_value(
        raw={**before_phase_inout, **during_phase_inout, **after_phase_inout},
        xnp=xnp,
    )


def convert_sparse_to_consecutive_int_lookup_table(
    raw: dict[int | Literal["min_int_in_table", "max_int_in_table"], Any],
    xnp: ModuleType,
) -> ConsecutiveIntLookupTableParamValue:
    """Convert sparse dict to consecutive int lookup table.

    Converts a dict with sparse integer keys to a consecutive int lookup table by
    filling in the gaps with the last value that is explicitly defined.

    Args:
        raw: Dictionary with sparse integer keys and float values. Special keys
             are 'min_int_in_table' and 'max_int_in_table'.
        xnp: NumPy-like module (numpy or jax.numpy)

    Returns:
        ConsecutiveIntLookupTableParamValue: A lookup table with consecutive integer
        keys from min_int_in_table to max_int_in_table, with gaps filled using the
        last defined value.

    Example:
        >>> result = convert_sparse_to_consecutive_int_lookup_table(
        ...     raw={
        ...         1: 1,
        ...         3: 3,
        ...         "min_int_in_table": 0,
        ...         "max_int_in_table": 5,
        ...     },
        ...     xnp=xnp,
        ...     )
        >>> result.value
        {
            0: 1,
            1: 1,
            2: 1,
            3: 3,
            4: 3,
            5: 3
        }
    """
    tmp: dict[int | Literal["min_int_in_table", "max_int_in_table"], Any] = raw.copy()

    _fail_if_raw_not_dict(tmp)
    _fail_if_raw_missing_min_max_int_in_table_keys(tmp)

    min_int_in_table: int = tmp.pop("min_int_in_table")
    max_int_in_table: int = tmp.pop("max_int_in_table")

    base_spec = cast("dict[int, Any]", tmp)
    _fail_if_raw_incompatible_with_min_max_int_in_table(
        raw=base_spec,
        min_int_in_table=min_int_in_table,
        max_int_in_table=max_int_in_table,
    )
    keys_in_base_spec: list[int] = sorted(base_spec.keys())
    full_table: dict[int, Any] = {}
    for a in range(min_int_in_table, max_int_in_table + 1):
        if a < min(keys_in_base_spec):
            full_table[a] = base_spec[min(keys_in_base_spec)]
        elif a not in keys_in_base_spec:
            full_table[a] = full_table[a - 1]
        else:
            full_table[a] = base_spec[a]
    return get_consecutive_int_lookup_table_param_value(
        raw=full_table,
        xnp=xnp,
    )


def _fail_if_raw_not_dict(
    raw: dict[int | Literal["min_int_in_table", "max_int_in_table"], Any],
) -> None:
    if not isinstance(raw, dict):
        msg = f"The raw dictionary must be a dictionary. You provided: {type(raw)}"
        raise TypeError(msg)


def _fail_if_raw_missing_min_max_int_in_table_keys(
    raw: dict[int | Literal["min_int_in_table", "max_int_in_table"], Any],
) -> None:
    if "min_int_in_table" not in raw or "max_int_in_table" not in raw:
        msg = (
            "The raw dictionary must contain 'min_int_in_table' and 'max_int_in_table' "
            "keys."
        )
        raise TypeError(msg)
    if not isinstance(raw["min_int_in_table"], int) or not isinstance(
        raw["max_int_in_table"], int
    ):
        msg = "The 'min_int_in_table' and 'max_int_in_table' values must be integers."
        raise TypeError(msg)


def _fail_if_raw_incompatible_with_min_max_int_in_table(
    raw: dict[int, Any],
    min_int_in_table: int,
    max_int_in_table: int,
) -> None:
    key_types = {type(k) for k in raw}
    if key_types != {int}:
        msg = (
            "The raw object must be a dictionary with int keys. You provided keys "
            f"of type: {key_types}"
        )
        raise TypeError(msg)
    if min(raw.keys()) < min_int_in_table:
        msg = (
            "The smallest integer in the lookup table must not be larger than the "
            "smallest key in the raw dictionary. You provided the following values: "
            f"min_int_in_table={min_int_in_table}, min(raw.keys())={min(raw.keys())}"
        )
        raise ValueError(msg)
    if max(raw.keys()) >= max_int_in_table:
        msg = (
            "The largest integer in the lookup table must not be smaller than the "
            "largest key in the raw dictionary. You provided the following values: "
            f"max_int_in_table={max_int_in_table}, max(raw.keys())={max(raw.keys())}"
        )
        raise ValueError(msg)
