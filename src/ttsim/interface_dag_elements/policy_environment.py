from __future__ import annotations

import copy
import datetime
from collections.abc import Callable, Mapping
from types import ModuleType
from typing import TYPE_CHECKING, Any

import dags.tree as dt
import numpy

from ttsim.exceptions import UnitDefinitionError
from ttsim.interface_dag_elements.interface_node_objects import interface_function
from ttsim.interface_dag_elements.shared import (
    merge_trees,
    param_has_substantive_content,
    upsert_tree,
)
from ttsim.tt import (
    UNSET_UNIT,
    ConsecutiveIntLookupTableParam,
    ConsecutiveIntLookupTableParamValue,
    DictParam,
    ParamObject,
    PiecewisePolynomialParam,
    PolicyInput,
    RawParam,
    ScalarParam,
    convert_sparse_to_consecutive_int_lookup_table,
    get_consecutive_int_lookup_table_param_value,
    get_month_based_phase_inout_of_age_thresholds_param_value,
    get_year_based_phase_inout_of_age_thresholds_param_value,
)
from ttsim.tt.column_objects_param_function import (
    DEFAULT_END_DATE,
)
from ttsim.tt.interval_utils import merge_piecewise_intervals
from ttsim.tt.param_objects import PiecewisePolynomialParamValue
from ttsim.tt.piecewise_polynomial import PIECEWISE_TYPES, get_piecewise_parameters
from ttsim.tt.units import (
    coerce_unit_token,
    currency_conversion_factor,
    token_is_agnostic_currency,
    token_source_currency,
)
from ttsim.typing import (
    FlatColumnObjectsParamFunctions,
    FlatOrigParamSpecs,
    NestedColumnObjectsParamFunctions,
    NestedParamObjects,
    OrigParamSpec,
)

#: YAML ``type:``\ s whose parameters are functions between quantities
#: (GEP 10): they declare ``input_unit:``/``output_unit:`` instead of
#: ``unit:``.
PARAM_MAPPING_OBJECT_TYPES: frozenset[str] = PIECEWISE_TYPES | {
    "consecutive_int_lookup_table",
    "sparse_to_consecutive_int_lookup_table",
    "month_based_phase_inout_of_age_thresholds",
    "year_based_phase_inout_of_age_thresholds",
}

if TYPE_CHECKING:
    from types import ModuleType

    from ttsim.typing import (
        FlatColumnObjectsParamFunctions,
        FlatOrigParamSpecs,
        NestedColumnObjectsParamFunctions,
        NestedParamObjects,
        OrigParamSpec,
        PolicyEnvironment,
    )


@interface_function(in_top_level_namespace=True)
def policy_environment(
    orig_policy_objects__column_objects_and_param_functions: FlatColumnObjectsParamFunctions,  # noqa: E501
    orig_policy_objects__param_specs: FlatOrigParamSpecs,
    policy_date: datetime.date,
    xnp: ModuleType,
    currency: str | None,
) -> PolicyEnvironment:
    """The policy environment at a particular date."""
    return {
        "policy_year": ScalarParam(
            value=policy_date.year,
            start_date=policy_date,
            end_date=policy_date,
        ),
        "policy_month": ScalarParam(
            value=policy_date.month,
            start_date=policy_date,
            end_date=policy_date,
        ),
        "policy_day": ScalarParam(
            value=policy_date.day,
            start_date=policy_date,
            end_date=policy_date,
        ),
        "evaluation_year": PolicyInput(
            leaf_name="evaluation_year",
            data_type=int,
            start_date=policy_date,
            end_date=policy_date,
            description="The evaluation year, will typically be set via `main`.",
        ),
        "evaluation_month": PolicyInput(
            leaf_name="evaluation_month",
            data_type=int,
            start_date=policy_date,
            end_date=policy_date,
            description="The evaluation month, will typically be set via `main`.",
        ),
        "evaluation_day": PolicyInput(
            leaf_name="evaluation_day",
            data_type=int,
            start_date=policy_date,
            end_date=policy_date,
            description="The evaluation day, will typically be set via `main`.",
        ),
        **merge_trees(
            left=_active_column_objects_and_param_functions(
                orig=orig_policy_objects__column_objects_and_param_functions,
                policy_date=policy_date,
            ),
            right=_active_param_objects(
                orig=orig_policy_objects__param_specs,
                policy_date=policy_date,
                xnp=xnp,
                currency=currency,
            ),
        ),
    }


def _active_column_objects_and_param_functions(
    orig: FlatColumnObjectsParamFunctions,
    policy_date: datetime.date,
) -> NestedColumnObjectsParamFunctions:
    """Traverse `root` and return all ColumnObjectParamFunctions for a given date.

    Args:
        root: The directory to traverse.
        policy_date: The date for which policy objects should be loaded.

    Returns:
        A tree of active ColumnObjectParamFunctions.

    """
    flat_objects_tree: dict[tuple[str, ...], Any] = {
        (*orig_path[:-2], obj.leaf_name): obj
        for orig_path, obj in orig.items()
        if obj.is_active(policy_date)
    }

    return dt.unflatten_from_tree_paths(flat_objects_tree)


def _active_param_objects(
    orig: FlatOrigParamSpecs,
    policy_date: datetime.date,
    xnp: ModuleType,
    currency: str | None = None,
) -> NestedParamObjects:
    """Parse the original yaml tree."""
    flat_tree_with_params = {}
    for orig_path, orig_params_spec in orig.items():
        path_to_keep = orig_path[:-2]
        leaf_name = orig_path[-1]
        param = _get_one_param(
            leaf_name=leaf_name,
            spec=orig_params_spec,
            policy_date=policy_date,
            xnp=xnp,
            currency=currency,
        )
        if param is not None:
            flat_tree_with_params[(*path_to_keep, leaf_name)] = param
        if orig_params_spec.get("add_jahresanfang", False):
            date_jan1 = policy_date.replace(month=1, day=1)
            leaf_name_jan1 = f"{leaf_name}_jahresanfang"
            param = _get_one_param(
                leaf_name=leaf_name_jan1,
                spec=orig_params_spec,
                policy_date=date_jan1,
                xnp=xnp,
                currency=currency,
            )
            if param is not None:
                flat_tree_with_params[(*path_to_keep, leaf_name_jan1)] = param
    return dt.unflatten_from_tree_paths(flat_tree_with_params)


def _currency_conversion_factor_for_token(
    raw_token: Any,  # noqa: ANN401
    run_currency: str | None,
) -> float:
    """Factor converting a declaration's currency into the run currency (GEP 10).

    A non-currency declaration (dimensionless, an area, a time, or none) carries
    no currency to convert, so the factor is ``1.0``. A currency declaration
    must pin down a concrete, registered currency: an agnostic token
    (``CURRENCY``, ``CURRENCY_FLOW``) or an unknown one is rejected.
    """
    if run_currency is None or raw_token is None or raw_token is UNSET_UNIT:
        return 1.0
    token = coerce_unit_token(raw_token, where="currency conversion")
    if token_is_agnostic_currency(token):
        raise UnitDefinitionError(
            f"Currency conversion: a parameter must pin down the concrete "
            f"currency its numbers are written in; the agnostic token {token} "
            f"cannot be converted to {run_currency!r} (GEP 10)."
        )
    source = token_source_currency(token)
    if source is None or source == run_currency:
        return 1.0
    return currency_conversion_factor(source_currency=source, run_currency=run_currency)


def _scale_numeric_leaves(
    value: Any,  # noqa: ANN401
    factor: float,
) -> Any:  # noqa: ANN401
    """Scale every numeric leaf of a (possibly nested) value by ``factor``.

    Booleans and non-numeric leaves pass through untouched.
    """
    if isinstance(value, Mapping):
        return {
            key: _scale_numeric_leaves(value=sub_value, factor=factor)
            for key, sub_value in value.items()
        }
    if isinstance(value, bool) or not isinstance(value, int | float):
        return value
    return value * factor


def _dict_param_value_in_run_currency(
    value: Any,  # noqa: ANN401
    unit: Any,  # noqa: ANN401
    run_currency: str | None,
) -> Any:  # noqa: ANN401
    """Restate a dict param's value in the run currency, leaf by leaf (GEP 10).

    A per-leaf ``unit:`` mapping rescales each currency-denominated leaf by its
    own token; a scalar token rescales the whole value uniformly. Non-currency
    leaves keep their numbers (their factor is ``1.0``).
    """
    if isinstance(unit, Mapping) and isinstance(value, Mapping):
        return {
            key: _dict_param_value_in_run_currency(
                value=sub_value,
                unit=unit.get(key),
                run_currency=run_currency,
            )
            for key, sub_value in value.items()
        }
    factor = _currency_conversion_factor_for_token(
        raw_token=unit, run_currency=run_currency
    )
    return _scale_numeric_leaves(value=value, factor=factor)


def _piecewise_param_value_in_run_currency(
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


def function_like_converter_output_in_run_currency(
    value: Any,  # noqa: ANN401
    *,
    input_unit: Any,  # noqa: ANN401
    output_unit: Any,  # noqa: ANN401
    run_currency: str | None,
    xnp: ModuleType,
    leaf_name: str,
) -> Any:  # noqa: ANN401
    """Restate a function-like ``require_converter``'s output in the run currency.

    A ``require_converter`` that declares ``input_unit:`` / ``output_unit:`` is
    handed to a converter producing a function-like value (a schedule or a
    lookup table). ttsim cannot read the polynomial convention out of the raw
    blob, so its raw value is left unscaled and the conversion happens here, on
    the *typed* output, per axis — the input axis by ``input_unit``, the output
    axis by ``output_unit`` (GEP 10). This is what makes the order-``j``
    polynomial coefficient scale by ``f_out / f_in**j`` rather than by a single
    uniform factor.

    Raises:
        UnitDefinitionError: If the converter did not produce a convertible
            function-like value, or if a currency conversion is requested for
            an integer-keyed lookup input axis.
    """
    input_factor = _currency_conversion_factor_for_token(
        raw_token=input_unit, run_currency=run_currency
    )
    output_factor = _currency_conversion_factor_for_token(
        raw_token=output_unit, run_currency=run_currency
    )
    if input_factor == 1.0 and output_factor == 1.0:
        return value
    if isinstance(value, PiecewisePolynomialParamValue):
        return _piecewise_param_value_in_run_currency(
            value=value,
            input_factor=input_factor,
            output_factor=output_factor,
            xnp=xnp,
        )
    if isinstance(value, ConsecutiveIntLookupTableParamValue):
        if input_factor != 1.0:
            raise UnitDefinitionError(
                f"Parameter {leaf_name!r}: lookup-table input axes are "
                f"integer-keyed and cannot be converted between currencies; a "
                f"concrete currency `input_unit:` is not supported (GEP 10)."
            )
        value.values_to_look_up = value.values_to_look_up * output_factor
        return value
    raise UnitDefinitionError(
        f"Parameter {leaf_name!r} declares `input_unit:` / `output_unit:`, so "
        f"its converter must produce a PiecewisePolynomialParamValue or a "
        f"ConsecutiveIntLookupTableParamValue that ttsim can convert per axis; "
        f"got {type(value).__name__} (GEP 10). A converter returning some "
        f"other structure must instead declare a single homogeneous `unit:`."
    )


def _get_one_param(
    leaf_name: str,
    spec: OrigParamSpec,
    policy_date: datetime.date,
    xnp: ModuleType,
    currency: str | None = None,
) -> ParamObject | None:
    """Parse the original specification found in the yaml tree to a ParamObject."""
    cleaned_spec = _clean_one_param_spec(spec=spec, policy_date=policy_date)

    if cleaned_spec is None:
        return None

    param_type = spec["type"]

    if param_type == "scalar":
        cleaned_spec["value"] = _scale_numeric_leaves(
            value=cleaned_spec["value"],
            factor=_currency_conversion_factor_for_token(
                raw_token=cleaned_spec.get("unit"), run_currency=currency
            ),
        )
        return ScalarParam(**cleaned_spec)
    if param_type == "dict":
        cleaned_spec["value"] = _dict_param_value_in_run_currency(
            value=cleaned_spec["value"],
            unit=cleaned_spec.get("unit"),
            run_currency=currency,
        )
        return DictParam(**cleaned_spec)
    if param_type == "require_converter":
        # A homogeneous require_converter (single currency `unit:`) is scaled
        # uniformly here, leaf by leaf. A function-like one (`input_unit:` /
        # `output_unit:`) is left raw: ttsim cannot read the polynomial
        # convention out of an arbitrary blob, so its converter's *typed*
        # output is converted per-axis in `with_processed_params_and_scalars`
        # instead (GEP 10).
        declares_axes = (
            cleaned_spec.get("input_unit", UNSET_UNIT) is not UNSET_UNIT
            or cleaned_spec.get("output_unit", UNSET_UNIT) is not UNSET_UNIT
        )
        if not declares_axes:
            cleaned_spec["value"] = _dict_param_value_in_run_currency(
                value=cleaned_spec["value"],
                unit=cleaned_spec.get("unit"),
                run_currency=currency,
            )
        return RawParam(**cleaned_spec)
    if param_type in PIECEWISE_TYPES:
        cleaned_spec["value"] = _piecewise_param_value_in_run_currency(
            value=get_piecewise_parameters(
                leaf_name=leaf_name,
                func_type=param_type,  # ty: ignore[invalid-argument-type]
                parameter_list=cleaned_spec["value"],
                xnp=xnp,
            ),
            input_factor=_currency_conversion_factor_for_token(
                raw_token=cleaned_spec.get("input_unit"), run_currency=currency
            ),
            output_factor=_currency_conversion_factor_for_token(
                raw_token=cleaned_spec.get("output_unit"), run_currency=currency
            ),
            xnp=xnp,
        )
        return PiecewisePolynomialParam(**cleaned_spec)
    lookup_table_converters: dict[
        str, Callable[..., ConsecutiveIntLookupTableParamValue]
    ] = {
        "consecutive_int_lookup_table": get_consecutive_int_lookup_table_param_value,
        "month_based_phase_inout_of_age_thresholds": (
            get_month_based_phase_inout_of_age_thresholds_param_value
        ),
        "year_based_phase_inout_of_age_thresholds": (
            get_year_based_phase_inout_of_age_thresholds_param_value
        ),
        "sparse_to_consecutive_int_lookup_table": (
            convert_sparse_to_consecutive_int_lookup_table
        ),
    }
    if param_type in lookup_table_converters:
        input_factor = _currency_conversion_factor_for_token(
            raw_token=cleaned_spec.get("input_unit"), run_currency=currency
        )
        if input_factor != 1.0:
            raise UnitDefinitionError(
                f"Parameter {leaf_name!r}: lookup-table input axes are "
                f"integer-keyed and cannot be converted between currencies; "
                f"a concrete currency `input_unit:` is not supported (GEP 10)."
            )
        converter = lookup_table_converters[param_type]
        table = converter(raw=cleaned_spec["value"], xnp=xnp)
        output_factor = _currency_conversion_factor_for_token(
            raw_token=cleaned_spec.get("output_unit"), run_currency=currency
        )
        table.values_to_look_up = table.values_to_look_up * output_factor
        cleaned_spec["value"] = table
        return ConsecutiveIntLookupTableParam(**cleaned_spec)

    raise ValueError(f"Unknown parameter type: {param_type} for {leaf_name}")


def _unit_fields_from_spec(spec: OrigParamSpec) -> dict[str, Any]:
    """Map a spec's ``unit:`` / ``input_unit:`` / ``output_unit:`` to ParamObject
    kwargs (GEP 10).

    Mapping parameters declare one token per axis; a require_converter declares
    either a single ``unit:`` (homogeneous, scaled uniformly) or per-axis tokens
    (a function-like output, converted per axis) — RawParam enforces the
    exclusivity; everything else declares a single ``unit:``. A stray ``unit:``
    on a mapping parameter is passed through so that ParamMappingObject rejects
    it with a precise message.
    """
    if spec["type"] in PARAM_MAPPING_OBJECT_TYPES:
        fields: dict[str, Any] = {
            "input_unit": spec.get("input_unit", UNSET_UNIT),
            "output_unit": spec.get("output_unit", UNSET_UNIT),
        }
        if "unit" in spec:
            fields["unit"] = spec["unit"]
        return fields
    if spec["type"] == "require_converter":
        return {
            "unit": spec.get("unit", UNSET_UNIT),
            "input_unit": spec.get("input_unit", UNSET_UNIT),
            "output_unit": spec.get("output_unit", UNSET_UNIT),
        }
    return {"unit": spec.get("unit", UNSET_UNIT)}


def _clean_one_param_spec(
    spec: OrigParamSpec,
    policy_date: datetime.date,
) -> dict[str, Any] | None:
    """Prepare the specification of one parameter for creating a ParamObject."""
    date_keys = [key for key in spec if isinstance(key, datetime.date)]
    policy_dates_dt64 = numpy.sort([numpy.datetime64(d) for d in date_keys])
    idx = int(
        numpy.searchsorted(
            policy_dates_dt64, numpy.datetime64(policy_date), side="right"
        )
    )
    policy_dates = [
        datetime.date.fromisoformat(str(d.astype("datetime64[D]")))
        for d in policy_dates_dt64
    ]
    if idx == 0:
        return None

    out: dict[str, Any] = {}
    out["start_date"] = policy_dates[idx - 1]
    out["end_date"] = (
        policy_dates[idx] - datetime.timedelta(days=1)
        if len(policy_dates) > idx
        else DEFAULT_END_DATE
    )
    out.update(_unit_fields_from_spec(spec))
    out["reference_period"] = spec.get("reference_period", None)
    out["reference_level"] = spec.get("reference_level", None)
    out["name"] = spec["name"]
    out["description"] = spec["description"]

    current_spec: dict[str | int, Any] = copy.deepcopy(spec[policy_dates[idx - 1]])
    out["note"] = current_spec.pop("note", None)
    out["reference"] = current_spec.pop("reference", None)
    # A dated entry may restate the unit field(s), overriding the top-level
    # declaration for that entry's numbers — this is how a currency
    # changeover within one parameter's history is written (GEP 10): old
    # entries denominated in the legacy currency, entries from the reform
    # date in the new one.
    for unit_key in ("unit", "input_unit", "output_unit"):
        if unit_key in current_spec:
            out[unit_key] = current_spec.pop(unit_key)

    if not param_has_substantive_content(current_spec):
        return None

    param_type = spec["type"]
    if param_type == "scalar":
        if current_spec.pop("updates_previous", False):
            raise ValueError(
                "'updates_previous' cannot be specified for scalar parameters."
            )
        out["value"] = current_spec["value"]
    elif param_type in PIECEWISE_TYPES:
        relevant_specs: list[dict[str | int, Any]] = [
            copy.deepcopy(spec[policy_dates[i]]) for i in range(idx)
        ]
        out["value"] = _get_param_value_piecewise(relevant_specs)
    else:
        relevant_specs: list[dict[str | int, Any]] = [
            copy.deepcopy(spec[policy_dates[i]]) for i in range(idx)
        ]
        out["value"] = _get_param_value(relevant_specs)
    return out


def _pop_unit_overrides(
    current: dict[str | int, Any],
    updates_previous: bool,
) -> None:
    """Strip a dated entry's unit override keys from its value dict (GEP 10).

    The override itself is consumed in :func:`_clean_one_param_spec`; here it
    must not leak into the assembled value. Combining an override with
    ``updates_previous`` is rejected: the merged value would mix numbers
    denominated in different currencies.
    """
    restates_unit = False
    for unit_key in ("unit", "input_unit", "output_unit"):
        if unit_key in current:
            # A present unit key is a restatement whatever its value — hence
            # the membership test rather than a `pop` default.
            del current[unit_key]
            restates_unit = True
    if restates_unit and updates_previous:
        raise UnitDefinitionError(
            "`updates_previous` cannot cross a unit (currency) changeover: "
            "an entry that restates the unit declaration must restate the "
            "full value (GEP 10)."
        )


def _get_param_value(
    relevant_specs: list[dict[str | int, Any]],
) -> dict[str | int, Any]:
    """Resolve parameter value, handling `updates_previous` chains."""
    current = relevant_specs[-1]
    current.pop("note", None)
    current.pop("reference", None)
    updates_previous = current.pop("updates_previous", False)
    _pop_unit_overrides(current=current, updates_previous=updates_previous)

    if updates_previous and len(relevant_specs) <= 1:
        raise ValueError(
            "'updates_previous' cannot be specified on the initial date entry."
        )

    if not updates_previous:
        return current

    base_value = _get_param_value(relevant_specs[:-1])
    return upsert_tree(base=base_value, to_upsert=current)


def _get_param_value_piecewise(
    relevant_specs: list[dict[str | int, Any]],
) -> list[dict[str, Any]]:
    """Resolve piecewise parameter value, handling `updates_previous` chains."""
    current = relevant_specs[-1]
    current.pop("note", None)
    current.pop("reference", None)
    updates_previous = current.pop("updates_previous", False)
    _pop_unit_overrides(current=current, updates_previous=updates_previous)

    if updates_previous and len(relevant_specs) <= 1:
        raise ValueError(
            "'updates_previous' cannot be specified on the initial date entry."
        )

    if not updates_previous:
        return current.get("intervals", [])

    base_intervals = _get_param_value_piecewise(relevant_specs[:-1])
    return merge_piecewise_intervals(
        base=base_intervals,
        update=current.get("intervals", []),
    )
