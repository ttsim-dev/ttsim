from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy

if TYPE_CHECKING:
    import datetime
    from types import ModuleType

    from jaxtyping import Array, Float, Int


@dataclass(frozen=True)
class ParamObject:
    """
    Abstract base class for all types of parameters.
    """

    leaf_name: str
    start_date: datetime.date
    end_date: datetime.date
    unit: (
        None
        | Literal[
            "Euros",
            "DM",
            "Share",
            "Percent",
            "Years",
            "Months",
            "Hours",
            "Square Meters",
            "Euros / Square Meter",
        ]
    )
    reference_period: None | Literal["Year", "Quarter", "Month", "Week", "Day"]
    name: dict[Literal["de", "en"], str]
    description: dict[Literal["de", "en"], str]


@dataclass(frozen=True)
class ScalarParam(ParamObject):
    """
    A scalar parameter directly read from a YAML file.
    """

    value: bool | int | float
    note: str | None = None
    reference: str | None = None


@dataclass(frozen=True)
class DictParam(ParamObject):
    """
    A parameter directly read from a YAML file that is a flat dictionary.
    """

    value: (
        dict[str, int]
        | dict[str, float]
        | dict[str, bool]
        | dict[int, int]
        | dict[int, float]
        | dict[int, bool]
    )
    note: str | None = None
    reference: str | None = None

    def __post_init__(self) -> None:
        assert all(x not in self.value for x in ["note", "reference"])


@dataclass(frozen=True)
class PiecewisePolynomialParam(ParamObject):
    """A parameter with its contents read and converted from a YAML file.

    Its value is a PiecewisePolynomialParamValue object, i.e., it contains the
    parameters for calling `piecewise_polynomial`.
    """

    value: PiecewisePolynomialParamValue
    note: str | None = None
    reference: str | None = None


@dataclass(frozen=True)
class ConsecutiveInt1dLookupTableParam(ParamObject):
    """A parameter with its contents read and converted from a YAML file.

    Its value is a ConsecutiveInt1dLookupTableParamValue object, i.e., it contains the
    parameters for calling `lookup_table`.
    """

    value: ConsecutiveInt1dLookupTableParamValue
    note: str | None = None
    reference: str | None = None


@dataclass(frozen=True)
class ConsecutiveInt2dLookupTableParam(ParamObject):
    """A parameter with its contents read and converted from a YAML file.

    Its value is a ConsecutiveInt2dLookupTableParamValue object, i.e., it contains the
    parameters for calling `lookup_table`.
    """

    value: ConsecutiveInt2dLookupTableParamValue
    note: str | None = None
    reference: str | None = None


@dataclass(frozen=True)
class ConsecutiveIntNdLookupTableParam(ParamObject):
    """A parameter with its contents read and converted from a YAML file.

    Its value is a ConsecutiveInt2dLookupTableParamValue object, i.e., it contains the
    parameters for calling `lookup_table`.
    """

    value: ConsecutiveIntNdLookupTableParamValue | None = None
    note: str | None = None
    reference: str | None = None


class ConsecutiveIntNdLookupTableParamValue:
    """The parameters expected by lookup_table"""

    bases_to_subtract: Int[Array, "n_rows n_cols"]
    lookup_multipliers: Int[Array, "n_rows n_cols"]
    values_to_look_up: Float[Array, "n_rows n_cols"]
    xnp: ModuleType

    def __init__(
        self,
        xnp: ModuleType,
        values_to_look_up: Float[Array, "n_rows n_cols"],
        bases_to_subtract: Int[Array, "n_rows n_cols"],
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

    def lookup(
        self: ConsecutiveIntNdLookupTableParamValue, *args: int
    ) -> Float[Array, "n_rows n_cols"]:
        index = self.xnp.asarray(args)
        corrected_index = self.xnp.dot(
            (index - self.bases_to_subtract).T, self.lookup_multipliers
        )
        return self.values_to_look_up[corrected_index]


@dataclass(frozen=True)
class RawParam(ParamObject):
    """
    A parameter directly read from a YAML file that is an arbitrarily nested
    dictionary.
    """

    value: dict[str | int, Any]
    note: str | None = None
    reference: str | None = None

    def __post_init__(self) -> None:
        assert all(x not in self.value for x in ["note", "reference"])


@dataclass(frozen=True)
class PiecewisePolynomialParamValue:
    """The parameters expected by `piecewise_polynomial`.

    thresholds:
        Thresholds defining the pieces / different segments on the real line.
    intercepts:
        Intercepts of the polynomial on each segment.
    rates:
        Slope and higher-order coefficients of the polynomial on each segment.
    """

    thresholds: Float[Array, " n_segments"]
    intercepts: Float[Array, " n_segments"]
    rates: Float[Array, " n_segments"]


@dataclass(frozen=True)
class ConsecutiveInt1dLookupTableParamValue:
    """The parameters expected by lookup_table"""

    base_to_subtract: int
    values_to_look_up: Float[Array, " n_values_to_look_up"]


@dataclass(frozen=True)
class ConsecutiveInt2dLookupTableParamValue:
    """The parameters expected by lookup_table"""

    base_to_subtract_rows: int
    base_to_subtract_cols: int
    values_to_look_up: Float[Array, "n_rows n_cols"]


def get_consecutive_int_1d_lookup_table_param_value(
    raw: dict[int, float | int | bool],
    xnp: ModuleType,
) -> ConsecutiveInt1dLookupTableParamValue:
    """Get the parameters for a 1-dimensional lookup table."""
    lookup_keys = numpy.asarray(sorted(raw))
    assert (lookup_keys - min(lookup_keys) == numpy.arange(len(lookup_keys))).all(), (
        "Dictionary keys must be consecutive integers."
    )

    return ConsecutiveInt1dLookupTableParamValue(
        base_to_subtract=min(lookup_keys).item(),
        values_to_look_up=xnp.asarray([raw[k] for k in lookup_keys]),
    )


def get_consecutive_int_2d_lookup_table_param_value(
    raw: dict[int, dict[int, float | int | bool]],
    xnp: ModuleType,
) -> ConsecutiveIntNdLookupTableParamValue:
    """Get the parameters for a 2-dimensional lookup table."""
    lookup_keys_rows = xnp.asarray(sorted(raw.keys()))
    lookup_keys_cols = xnp.asarray(sorted(raw[lookup_keys_rows[0].item()].keys()))
    for col_value in raw.values():
        lookup_keys_this_col = xnp.asarray(sorted(col_value.keys()))
        assert (lookup_keys_cols == lookup_keys_this_col).all(), (
            "Column keys must be the same in each column, got:"
            f"{lookup_keys_cols} and {lookup_keys_this_col}"
        )
    for lookup_keys in lookup_keys_rows, lookup_keys_cols:
        assert (lookup_keys - min(lookup_keys) == xnp.arange(len(lookup_keys))).all(), (
            f"Dictionary keys must be consecutive integers, got: {lookup_keys}"
        )
    return ConsecutiveIntNdLookupTableParamValue(
        xnp=xnp,
        bases_to_subtract=xnp.asarray(
            [min(lookup_keys_rows).item(), min(lookup_keys_cols).item()]
        ),
        values_to_look_up=xnp.array(
            [
                raw[row.item()][col.item()]
                for row, col in itertools.product(lookup_keys_rows, lookup_keys_cols)
            ],
        ).reshape(len(lookup_keys_rows), len(lookup_keys_cols)),
    )


def get_consecutive_int_nd_lookup_table_param_value(
    raw: dict[int, dict[int, float | int | bool]],
    n_dims: int,
    xnp: ModuleType,
) -> ConsecutiveIntNdLookupTableParamValue:
    """Get the parameters for a 2-dimensional lookup table."""
    bases_to_substract = numpy.zeros(n_dims, dtype=numpy.int32)

    def process_level(i: int, level_i_dict) -> Float[Array, "n_rows n_cols"]:
        bases_to_substract[i - 1] = min(level_i_dict.keys())
        if i < n_dims:
            return xnp.concatenate(
                [
                    xnp.expand_dims(process_level(i + 1, level_i_dict[key]), axis=0)
                    for key in level_i_dict
                ]
            )
        return xnp.asarray(list(level_i_dict.values()))

    values = process_level(1, raw)
    return ConsecutiveIntNdLookupTableParamValue(
        xnp=xnp, values_to_look_up=values, bases_to_subtract=bases_to_substract
    )


def _year_fraction(r: dict[Literal["years", "months"], int]) -> float:
    return r["years"] + r["months"] / 12


def get_month_based_phase_inout_of_age_thresholds_param_value(
    raw: dict[str | int, Any],
    xnp: ModuleType,
) -> ConsecutiveInt1dLookupTableParamValue:
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
    assert all(isinstance(k, int) for k in raw)
    first_year_phase_inout: int = min(raw.keys())  # type: ignore[assignment]
    first_month_phase_inout: int = min(raw[first_year_phase_inout].keys())
    first_m_since_ad_phase_inout = _m_since_ad(
        y=first_year_phase_inout,
        m=first_month_phase_inout,
    )
    last_year_phase_inout: int = max(raw.keys())  # type: ignore[assignment]
    last_month_phase_inout: int = max(raw[last_year_phase_inout].keys())
    last_m_since_ad_phase_inout = _m_since_ad(
        y=last_year_phase_inout,
        m=last_month_phase_inout,
    )
    assert first_m_since_ad_to_consider <= first_m_since_ad_phase_inout
    assert last_m_since_ad_to_consider >= last_m_since_ad_phase_inout
    before_phase_inout: dict[int, float] = {
        b_m: _year_fraction(raw[first_year_phase_inout][first_month_phase_inout])
        for b_m in range(first_m_since_ad_to_consider, first_m_since_ad_phase_inout)
    }
    during_phase_inout: dict[int, float] = _fill_phase_inout(
        raw=raw,  # type: ignore[arg-type]
        first_m_since_ad_phase_inout=first_m_since_ad_phase_inout,
        last_m_since_ad_phase_inout=last_m_since_ad_phase_inout,
    )
    after_phase_inout: dict[int, float] = {
        b_m: _year_fraction(raw[last_year_phase_inout][last_month_phase_inout])
        for b_m in range(
            last_m_since_ad_phase_inout + 1,
            last_m_since_ad_to_consider + 1,
        )
    }
    return get_consecutive_int_1d_lookup_table_param_value(
        raw={**before_phase_inout, **during_phase_inout, **after_phase_inout},
        xnp=xnp,
    )


def get_year_based_phase_inout_of_age_thresholds_param_value(
    raw: dict[str | int, Any],
    xnp: ModuleType,
) -> ConsecutiveInt1dLookupTableParamValue:
    """Get the parameters for year-based phase-in/phase-out of age thresholds.

    Requires all years to be given.
    """
    first_year_to_consider = raw.pop("first_year_to_consider")
    last_year_to_consider = raw.pop("last_year_to_consider")
    assert all(isinstance(k, int) for k in raw)
    first_year_phase_inout: int = sorted(raw)[0]  # type: ignore[assignment]
    last_year_phase_inout: int = sorted(raw)[-1]  # type: ignore[assignment]
    assert first_year_to_consider <= first_year_phase_inout
    assert last_year_to_consider >= last_year_phase_inout
    before_phase_inout: dict[int, float] = {
        b_y: _year_fraction(raw[first_year_phase_inout])
        for b_y in range(first_year_to_consider, first_year_phase_inout)
    }
    during_phase_inout: dict[int, float] = {
        b_y: _year_fraction(raw[b_y])  # type: ignore[misc]
        for b_y in raw
    }
    after_phase_inout: dict[int, float] = {
        b_y: _year_fraction(raw[last_year_phase_inout])
        for b_y in range(last_year_phase_inout + 1, last_year_to_consider + 1)
    }
    return get_consecutive_int_1d_lookup_table_param_value(
        raw={**before_phase_inout, **during_phase_inout, **after_phase_inout},
        xnp=xnp,
    )
