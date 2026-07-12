"""Build-time currency conversion of parameters (GEP 10).

The single home for restating parameters in the run currency at build time. A
parameter is written in its legal currency (DM, silver penny, …); running the
system in another currency of the same family converts every parameter here,
once, so the numeric runtime stays single-currency.

Conversion happens at two points in a parameter's life, both routed through this
module:

- **at read-in** (``policy_environment._get_one_param``): scalars, dicts, and
  ``require_converter`` parameters with a plain ``unit:`` are scaled as soon as
  they are parsed; piecewise schedules and lookup tables are converted per axis.
- **on the built schedule** (``_restate_converter_outputs_in_run_currency``): a
  ``require_converter`` that declares ``input_unit:`` / ``output_unit:`` cannot
  be scaled from its raw value — the polynomial convention is unknown until its
  param function has built the typed schedule — so it is converted afterwards, on
  that output.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from types import ModuleType
from typing import Any

import dags.tree as dt

from ttsim.exceptions import UnitDefinitionError
from ttsim.tt.column_objects_param_function import ParamFunction
from ttsim.tt.currencies import currency_conversion_factor
from ttsim.tt.param_objects import (
    ConsecutiveIntLookupTableParamValue,
    ParamObject,
    PiecewisePolynomialParamValue,
    RawParam,
)
from ttsim.tt.units import (
    UNSET_UNIT,
    CompositeUnit,
    coerce_to_composite_unit,
    token_source_currency,
)


def currency_conversion_factor_for_token(
    raw_token: Any,  # noqa: ANN401
    run_currency: str,
) -> float:
    """Factor converting a declaration's currency into the run currency (GEP 10).

    A non-currency declaration (dimensionless, an area, a time, or none) and one
    already in the run currency carry nothing to convert, so the factor is
    ``1.0``. An agnostic currency token pins down no source currency and also
    yields ``1.0`` here; the rule that a parameter must name a concrete currency
    is enforced by the unit checks (``_fail_if_param_token_is_agnostic_currency``).
    """
    if raw_token is None or raw_token is UNSET_UNIT:
        return 1.0
    token = coerce_to_composite_unit(value=raw_token, where="currency conversion")
    source = token_source_currency(token)
    if source is None or source == run_currency:
        return 1.0
    return currency_conversion_factor(source_currency=source, run_currency=run_currency)


def axis_factors(
    cleaned_spec: dict[str, Any],
    run_currency: str,
) -> tuple[float, float]:
    """The (input, output) currency conversion factors for a per-axis spec."""
    return (
        currency_conversion_factor_for_token(
            raw_token=cleaned_spec.get("input_unit"), run_currency=run_currency
        ),
        currency_conversion_factor_for_token(
            raw_token=cleaned_spec.get("output_unit"), run_currency=run_currency
        ),
    )


def scale_numeric_leaves(
    value: Any,  # noqa: ANN401
    factor: float,
) -> Any:  # noqa: ANN401
    """Scale every numeric leaf of a (possibly nested) value by ``factor``.

    Booleans and non-numeric leaves pass through untouched. A factor of ``1.0``
    (no currency conversion) returns the value verbatim, so int leaves — GEP-3
    integer thresholds — keep their type instead of being coerced to float.
    """
    if factor == 1.0:
        return value
    if isinstance(value, Mapping):
        return {
            key: scale_numeric_leaves(value=sub_value, factor=factor)
            for key, sub_value in value.items()
        }
    if isinstance(value, bool) or not isinstance(value, int | float):
        return value
    return value * factor


def _token_for_leaf(unit: Any, path: tuple[str | int, ...]) -> Any:  # noqa: ANN401
    """The unit token governing the leaf at ``path``.

    Walk the (possibly coarser or sparser) ``unit`` tree alongside the path: a
    scalar token covers everything below it, a missing key yields ``None`` (a
    factor of ``1.0``).
    """
    node = unit
    for key in path:
        if not isinstance(node, Mapping):
            break
        node = node.get(key)
    return node


def dict_param_value_in_run_currency(
    value: Any,  # noqa: ANN401
    unit: Any,  # noqa: ANN401
    run_currency: str,
) -> Any:  # noqa: ANN401
    """Restate a dict param's value in the run currency, leaf by leaf (GEP 10).

    A per-leaf ``unit:`` mapping rescales each currency-denominated leaf by its
    own token; a scalar token rescales the whole value uniformly. Non-currency
    leaves keep their numbers (their factor is ``1.0``).
    """
    if not isinstance(value, Mapping):
        factor = currency_conversion_factor_for_token(
            raw_token=unit, run_currency=run_currency
        )
        return scale_numeric_leaves(value=value, factor=factor)
    return dt.unflatten_from_tree_paths(
        {
            path: scale_numeric_leaves(
                value=leaf,
                factor=currency_conversion_factor_for_token(
                    raw_token=_token_for_leaf(unit=unit, path=path),
                    run_currency=run_currency,
                ),
            )
            for path, leaf in dt.flatten_to_tree_paths(value).items()
        }
    )


def piecewise_param_value_in_run_currency(
    value: PiecewisePolynomialParamValue,
    input_factor: float,
    output_factor: float,
    xnp: ModuleType,
) -> PiecewisePolynomialParamValue:
    """Restate a schedule in the run currency, axis by axis (GEP 10).

    The piecewise form is ``y = intercept_i + Σ_j c_{i,j} (x - t_i)^j``.
    Scaling the input axis by ``f_in`` and the output axis by ``f_out``
    rescales the thresholds by ``f_in``, the intercepts by ``f_out``, and the
    order-``j`` coefficient by ``f_out / f_in**j`` — for an income schedule
    (both axes the same currency) the slopes are invariant, for an area
    schedule the slopes carry the full output factor. A non-currency axis has
    a factor of ``1.0`` and is left as is.
    """
    orders = xnp.arange(1, value.coefficients.shape[1] + 1)
    return PiecewisePolynomialParamValue(
        thresholds=value.thresholds * input_factor,
        intercepts=value.intercepts * output_factor,
        coefficients=value.coefficients * (output_factor / input_factor**orders),
    )


def lookup_table_value_in_run_currency(
    value: ConsecutiveIntLookupTableParamValue,
    output_factor: float,
) -> ConsecutiveIntLookupTableParamValue:
    """Restate a lookup table in the run currency (GEP 10).

    A lookup table is keyed by consecutive integers, so its input axis is never a
    currency (rejected when the parameter is read in). Only the looked-up values
    are scaled, by ``output_factor``. A factor of ``1.0`` returns the table
    verbatim, keeping int values int; a real conversion returns a copy, never
    mutating the input.
    """
    if output_factor == 1.0:
        return value
    converted = copy.copy(value)
    converted.values_to_look_up = value.values_to_look_up * output_factor
    return converted


def function_like_converter_output_in_run_currency(
    value: Any,  # noqa: ANN401
    *,
    input_unit: Any,  # noqa: ANN401
    output_unit: Any,  # noqa: ANN401
    run_currency: str,
    xnp: ModuleType,
    leaf_name: str,
) -> Any:  # noqa: ANN401
    """Restate a function-like ``require_converter``'s output in the run currency.

    A ``require_converter`` that declares ``input_unit:`` / ``output_unit:`` is
    handed to a converter producing a function-like value (a schedule or a
    lookup table). ttsim cannot read the polynomial convention out of the raw
    value, so it is left unscaled and the conversion happens here, on
    the *typed* output, per axis — the input axis by ``input_unit``, the output
    axis by ``output_unit`` (GEP 10). This is what makes the order-``j``
    polynomial coefficient scale by ``f_out / f_in**j`` rather than by a single
    uniform factor.

    Raises:
        UnitDefinitionError: If the converter did not produce a convertible
            function-like value.
    """
    input_factor = currency_conversion_factor_for_token(
        raw_token=input_unit, run_currency=run_currency
    )
    output_factor = currency_conversion_factor_for_token(
        raw_token=output_unit, run_currency=run_currency
    )
    if input_factor == 1.0 and output_factor == 1.0:
        return value
    if isinstance(value, PiecewisePolynomialParamValue):
        return piecewise_param_value_in_run_currency(
            value=value,
            input_factor=input_factor,
            output_factor=output_factor,
            xnp=xnp,
        )
    if isinstance(value, ConsecutiveIntLookupTableParamValue):
        return lookup_table_value_in_run_currency(
            value=value, output_factor=output_factor
        )
    raise UnitDefinitionError(
        f"Parameter {leaf_name!r} declares `input_unit:` / `output_unit:`, so "
        f"its converter must produce a PiecewisePolynomialParamValue or a "
        f"ConsecutiveIntLookupTableParamValue that ttsim can convert per axis; "
        f"got {type(value).__name__} (GEP 10). A converter returning some "
        f"other structure must instead declare a single homogeneous `unit:`."
    )


def _unit_declares_a_currency(unit: Any) -> bool:  # noqa: ANN401
    """Whether a plain ``unit:`` declaration (a single token or a per-leaf
    mapping) pins down a concrete currency anywhere."""
    if isinstance(unit, Mapping):
        return any(_unit_declares_a_currency(sub) for sub in unit.values())
    return isinstance(unit, CompositeUnit) and token_source_currency(unit) is not None


def _fail_if_a_param_function_reads_multiple_axes_declaring_converters(
    params: dict[str, ParamObject],
    param_functions: dict[str, ParamFunction],
) -> None:
    """Reject a param function that reads more than one input/output-unit parameter.

    A ``require_converter`` that declares ``input_unit:`` / ``output_unit:`` maps
    between two units, and the framework restates its output in the run currency
    per axis. A param function reading two of them would have its output rescaled
    once for each, so at most one is allowed (GEP 10).
    """
    converters_by_param_function: dict[str, list[str]] = {}
    for pf_name, pf in param_functions.items():
        for raw_qname in pf.dependencies:
            raw = params.get(raw_qname)
            if isinstance(raw, RawParam) and (
                raw.input_unit is not UNSET_UNIT or raw.output_unit is not UNSET_UNIT
            ):
                converters_by_param_function.setdefault(pf_name, []).append(raw_qname)
    for pf_name, converter_qnames in converters_by_param_function.items():
        if len(converter_qnames) > 1:
            names = ", ".join(f"{qname!r}" for qname in sorted(converter_qnames))
            raise UnitDefinitionError(
                f"Param function {pf_name!r} reads {len(converter_qnames)} "
                f"parameters that map between an input and an output unit "
                f"({names}); its output would be restated in the run currency "
                f"once for each, so at most one is allowed (GEP 10)."
            )


def restate_converter_outputs_in_run_currency(
    *,
    outputs: dict[str, Any],
    params: dict[str, ParamObject],
    param_functions: dict[str, ParamFunction],
    run_currency: str,
    xnp: ModuleType,
) -> None:
    """Restate the outputs of ``require_converter`` schedules in the run currency.

    A ``require_converter`` that declares ``input_unit:`` / ``output_unit:`` is
    left in its raw currency at read-in; only once its param function has built
    the typed schedule (a piecewise polynomial or a lookup table) can it be
    restated in the run currency — per axis, here, the one place that knows both
    the built schedule and the run currency.

    A ``require_converter`` with a plain currency ``unit:`` (a single token or a
    per-leaf mapping) is converted leaf by leaf at read-in instead. If it none
    the less builds a schedule, that leaf-by-leaf conversion is wrong — a
    polynomial's order-``j`` coefficient scales by ``f_out / f_in**j``, not by one
    factor — so it is rejected, pointing the author at ``input_unit:`` /
    ``output_unit:`` (GEP 10).
    """
    _fail_if_a_param_function_reads_multiple_axes_declaring_converters(
        params=params, param_functions=param_functions
    )
    for raw_qname, raw in params.items():
        if not isinstance(raw, RawParam):
            continue
        declares_axes = (
            raw.input_unit is not UNSET_UNIT or raw.output_unit is not UNSET_UNIT
        )
        consumers = [
            pf_name
            for pf_name, pf in param_functions.items()
            if raw_qname in pf.dependencies
        ]
        for pf_name in consumers:
            if declares_axes:
                outputs[pf_name] = function_like_converter_output_in_run_currency(
                    value=outputs[pf_name],
                    input_unit=raw.input_unit,
                    output_unit=raw.output_unit,
                    run_currency=run_currency,
                    xnp=xnp,
                    leaf_name=raw_qname,
                )
            elif _unit_declares_a_currency(raw.unit) and isinstance(
                outputs[pf_name],
                PiecewisePolynomialParamValue | ConsecutiveIntLookupTableParamValue,
            ):
                raise UnitDefinitionError(
                    f"require_converter {raw_qname!r} declares a plain currency "
                    f"`unit:` ({raw.unit}), which is converted leaf by leaf, but "
                    f"its param function {pf_name!r} builds a "
                    f"{type(outputs[pf_name]).__name__} — a schedule whose "
                    f"coefficients do not all scale by the same factor. Declare "
                    f"`input_unit:` / `output_unit:` so each axis is converted "
                    f"correctly (GEP 10)."
                )
