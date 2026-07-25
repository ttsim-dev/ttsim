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
    UNIT_DECLARATION_KEYS,
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
from ttsim.tt.piecewise_polynomial import PIECEWISE_TYPES, get_piecewise_parameters
from ttsim.tt.units import ttsim_unit_currency, ttsim_unit_from_yaml_value
from ttsim.typing import (
    FlatColumnObjectsParamFunctions,
    FlatOrigParamSpecs,
    NestedColumnObjectsParamFunctions,
    NestedParamObjects,
    OrigParamSpec,
)

#: Lookup-table converters keyed by the YAML ``type:`` that selects them.
LOOKUP_TABLE_CONVERTERS: dict[
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

PARAM_MAPPING_OBJECT_TYPES: frozenset[str] = PIECEWISE_TYPES | frozenset(
    LOOKUP_TABLE_CONVERTERS
)


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
    computation_currency: str,
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
                computation_currency=computation_currency,
            ),
            right=_active_param_objects(
                orig=orig_policy_objects__param_specs,
                policy_date=policy_date,
                xnp=xnp,
                computation_currency=computation_currency,
            ),
        ),
    }


def _active_column_objects_and_param_functions(
    orig: FlatColumnObjectsParamFunctions,
    policy_date: datetime.date,
    computation_currency: str,
) -> NestedColumnObjectsParamFunctions:
    """Traverse `root` and return all ColumnObjectParamFunctions for a given date.

    Args:
        root: The directory to traverse.
        policy_date: The date for which policy objects should be loaded.
        computation_currency: The statutory currency at the policy date.

    Returns:
        A tree of active ColumnObjectParamFunctions.

    """
    flat_objects_tree: dict[tuple[str, ...], Any] = {}
    for orig_path, obj in orig.items():
        if not obj.is_active(policy_date):
            continue
        _fail_if_rounding_spec_currency_is_not_statutory(
            obj=obj,
            policy_date=policy_date,
            computation_currency=computation_currency,
        )
        flat_objects_tree[(*orig_path[:-2], obj.leaf_name)] = obj

    return dt.unflatten_from_tree_paths(flat_objects_tree)


def _fail_if_rounding_spec_currency_is_not_statutory(
    obj: Any,  # noqa: ANN401
    policy_date: datetime.date,
    computation_currency: str,
) -> None:
    """Reject a rounding spec declared in a non-statutory currency."""
    spec = getattr(obj, "rounding_spec", None)
    if spec is None or spec.unit is None:
        return
    source = ttsim_unit_currency(spec.unit)
    if source is None or source == computation_currency:
        return
    raise UnitDefinitionError(
        f"The rounding spec on {obj.leaf_name!r} declares its magnitudes in "
        f"{source!r}, but the statutory currency at {policy_date.isoformat()} "
        f"is {computation_currency!r}. Rounding magnitudes are never converted "
        f"(GEP 10): split the function at the currency changeover and declare "
        f"the restated spec."
    )


def _active_param_objects(
    orig: FlatOrigParamSpecs,
    policy_date: datetime.date,
    xnp: ModuleType,
    computation_currency: str,
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
            )
            if param is not None:
                flat_tree_with_params[(*path_to_keep, leaf_name_jan1)] = param
    _fail_if_param_currencies_are_not_statutory(
        params=flat_tree_with_params,
        policy_date=policy_date,
        computation_currency=computation_currency,
    )
    return dt.unflatten_from_tree_paths(flat_tree_with_params)


def _collect_currencies_in_param_units(raw_token: Any) -> set[str]:  # noqa: ANN401
    """The concrete currencies a parameter's ``unit:`` value pins down.

    A scalar spelling contributes at most one currency; a per-leaf mapping is
    walked recursively: ``{"4": {"betrag": "DM_PER_MONTH"}}`` yields ``{"DM"}``.
    """
    if raw_token is None or raw_token is UNSET_UNIT:
        return set()
    if isinstance(raw_token, Mapping):
        return {
            currency
            for sub_token in raw_token.values()
            for currency in _collect_currencies_in_param_units(sub_token)
        }
    token = ttsim_unit_from_yaml_value(
        value=raw_token, where="the statutory-currency check"
    )
    source = ttsim_unit_currency(token)
    return {source} if source is not None else set()


def _fail_if_param_currencies_are_not_statutory(
    params: Mapping[tuple[str, ...], ParamObject],
    policy_date: datetime.date,
    computation_currency: str,
) -> None:
    """Reject every parameter declared in a non-statutory currency, in one error.

    One traversal over all parameters active at the policy date, so an author who
    misses a currency changeover sees each offending parameter at once instead of
    one per run.
    """
    offenders = {
        dt.qname_from_tree_path(path): non_statutory
        for path, param in params.items()
        if (
            non_statutory := sorted(
                {
                    currency
                    for key in UNIT_DECLARATION_KEYS
                    for currency in _collect_currencies_in_param_units(
                        getattr(param, key, UNSET_UNIT)
                    )
                }
                - {computation_currency}
            )
        )
    }
    if not offenders:
        return
    listed = "\n".join(
        f"    {qname}: {', '.join(repr(c) for c in currencies)}"
        for qname, currencies in sorted(offenders.items())
    )
    raise UnitDefinitionError(
        f"The following parameters declare their numbers in a currency other "
        f"than the statutory currency at {policy_date.isoformat()}, which is "
        f"{computation_currency!r}:\n\n{listed}\n\n"
        f"Parameters are never converted (GEP 10): add a dated entry restating "
        f"the value in the statutory currency."
    )


def _get_one_param(
    leaf_name: str,
    spec: OrigParamSpec,
    policy_date: datetime.date,
    xnp: ModuleType,
) -> ParamObject | None:
    """Parse the original specification found in the yaml tree to a ParamObject."""
    cleaned_spec = _clean_one_param_spec(
        leaf_name=leaf_name, spec=spec, policy_date=policy_date
    )

    if cleaned_spec is None:
        return None

    param_type = spec["type"]

    if param_type == "scalar":
        return ScalarParam(**cleaned_spec)
    if param_type == "dict":
        return DictParam(**cleaned_spec)
    if param_type == "require_converter":
        return RawParam(**cleaned_spec)
    if param_type in PIECEWISE_TYPES:
        cleaned_spec["value"] = get_piecewise_parameters(
            leaf_name=leaf_name,
            func_type=param_type,  # ty: ignore[invalid-argument-type]
            parameter_list=cleaned_spec["value"],
            xnp=xnp,
        )
        return PiecewisePolynomialParam(**cleaned_spec)
    if param_type in LOOKUP_TABLE_CONVERTERS:
        converter = LOOKUP_TABLE_CONVERTERS[param_type]
        cleaned_spec["value"] = converter(raw=cleaned_spec["value"], xnp=xnp)
        return ConsecutiveIntLookupTableParam(**cleaned_spec)

    raise ValueError(f"Unknown parameter type: {param_type} for {leaf_name}")


def _unit_fields_from_spec(spec: OrigParamSpec) -> dict[str, Any]:
    """Map a spec's ``unit:`` / ``input_unit:`` / ``output_unit:`` to ParamObject
    kwargs.

    Mapping parameters declare one token per axis; a require_converter declares
    either ``unit:`` (a single token or a per-leaf mapping, one token per leaf)
    or per-axis tokens (a function-like output, one token per axis) — RawParam
    enforces the exclusivity; everything else declares a single ``unit:``. A
    stray ``unit:`` on a mapping parameter is passed through so that
    ParamMappingObject rejects it with a precise message.
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


def _forward_fill_unit_fields(
    leaf_name: str,
    spec: OrigParamSpec,
    active_dates: list[datetime.date],
) -> dict[str, Any]:
    """Resolve a parameter's unit(s) at the active date by forward-fill (GEP 10).

    Each dated entry inherits the most recent *earlier* unit declaration; the
    top-level ``unit:`` is the seed, and a dated entry that restates the unit
    becomes the seed from its date onward. So a unit that holds across many dates
    is spelled once, not on every entry. Resolution only ever looks backward — a
    gap with no earlier declaration and no top-level seed stays unset, and the
    mandatory-unit gate fires downstream.

    A dated restatement replaces the previous declaration as a whole; a per-leaf
    ``unit:`` mapping must therefore restate every leaf, else the omitted leaves
    would silently keep the previous unit
    (:func:`_fail_if_partial_unit_mapping_restatement`).
    """
    resolved = _unit_fields_from_spec(spec)
    for date in active_dates:
        entry = spec[date]
        _fail_if_updates_previous_restates_unit(
            leaf_name=leaf_name, entry=entry, date=date
        )
        for unit_key in UNIT_DECLARATION_KEYS:
            if unit_key not in entry:
                continue
            _fail_if_partial_unit_mapping_restatement(
                leaf_name=leaf_name,
                previous=resolved.get(unit_key, UNSET_UNIT),
                restated=entry[unit_key],
                unit_key=unit_key,
                date=date,
            )
            resolved[unit_key] = entry[unit_key]
    return resolved


def _fail_if_partial_unit_mapping_restatement(
    leaf_name: str,
    previous: Any,  # noqa: ANN401
    restated: Any,  # noqa: ANN401
    unit_key: str,
    date: datetime.date,
) -> None:
    """Reject a dated restatement of a per-leaf unit mapping that drops leaves.

    A unit declaration is replaced as a whole, so an incomplete mapping would
    silently leave the omitted leaves on the previous unit (GEP 10).
    """
    if (
        isinstance(previous, Mapping)
        and isinstance(restated, Mapping)
        and set(restated) != set(previous)
    ):
        raise UnitDefinitionError(
            f"Parameter {leaf_name!r}: the dated `{unit_key}:` mapping at {date} "
            f"must restate every leaf of the unit it replaces "
            f"(got {sorted(restated)}, expected {sorted(previous)}); a unit "
            f"declaration is replaced as a whole (GEP 10)."
        )


def _fail_if_updates_previous_restates_unit(
    leaf_name: str,
    entry: Mapping[str | int, Any],
    date: datetime.date,
) -> None:
    """Reject a dated entry that both merges values and restates the unit.

    ``updates_previous`` merges the entry's leaves onto the previous value, so
    a leaf it does not restate carries forward from the previous currency yet
    now wears the restated unit — a silent mis-scaling invisible to the
    statutory-currency guard. A unit change must restate the value in full
    (GEP 10).
    """
    if entry.get("updates_previous", False) and any(
        unit_key in entry for unit_key in UNIT_DECLARATION_KEYS
    ):
        raise UnitDefinitionError(
            f"Parameter {leaf_name!r}: the dated entry at {date} both merges "
            f"values (`updates_previous: true`) and restates the unit; a merge "
            f"would carry un-restated leaves forward under the new unit. Restate "
            f"the value in full at a unit change (GEP 10)."
        )


def _clean_one_param_spec(
    leaf_name: str,
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
    out.update(
        _forward_fill_unit_fields(
            leaf_name=leaf_name, spec=spec, active_dates=policy_dates[:idx]
        )
    )
    out["name"] = spec["name"]
    out["description"] = spec["description"]

    current_spec: dict[str | int, Any] = copy.deepcopy(spec[policy_dates[idx - 1]])
    out["note"] = current_spec.pop("note", None)
    out["reference"] = current_spec.pop("reference", None)
    _strip_unit_overrides(current=current_spec)

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


def _get_param_value(
    relevant_specs: list[dict[str | int, Any]],
) -> dict[str | int, Any]:
    """Resolve parameter value, handling `updates_previous` chains."""
    current = relevant_specs[-1]
    current.pop("note", None)
    current.pop("reference", None)
    updates_previous = current.pop("updates_previous", False)
    _strip_unit_overrides(current=current)

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
    _strip_unit_overrides(current=current)

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


def _strip_unit_overrides(current: dict[str | int, Any]) -> None:
    """Strip a dated entry's unit override keys from its value dict."""
    for unit_key in UNIT_DECLARATION_KEYS:
        current.pop(unit_key, None)
