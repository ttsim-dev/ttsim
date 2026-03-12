from __future__ import annotations

import copy
import datetime
from typing import TYPE_CHECKING, Any

import dags.tree as dt
import numpy

from ttsim.interface_dag_elements.interface_node_objects import interface_function
from ttsim.interface_dag_elements.shared import (
    merge_trees,
    upsert_tree,
)
from ttsim.tt import (
    ConsecutiveIntLookupTableParam,
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

if TYPE_CHECKING:
    from types import FunctionType, ModuleType

    from ttsim.tt import ConsecutiveIntLookupTableParamValue
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
            ),
        ),
    }


def _active_column_objects_and_param_functions(
    orig: FlatColumnObjectsParamFunctions,
    policy_date: datetime.date,
) -> NestedColumnObjectsParamFunctions:
    """
    Traverse `root` and return all ColumnObjectParamFunctions for a given date.

    Parameters
    ----------
    root:
        The directory to traverse.
    policy_date:
        The date for which policy objects should be loaded.

    Returns
    -------
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
    return dt.unflatten_from_tree_paths(flat_tree_with_params)


def _get_one_param(
    leaf_name: str,
    spec: OrigParamSpec,
    policy_date: datetime.date,
    xnp: ModuleType,
) -> ParamObject | None:
    """Parse the original specification found in the yaml tree to a ParamObject."""
    cleaned_spec = _clean_one_param_spec(spec=spec, policy_date=policy_date)

    if cleaned_spec is None:
        return None

    param_type = spec["type"]

    if param_type == "scalar":
        return ScalarParam(**cleaned_spec)
    if param_type == "dict":
        return DictParam(**cleaned_spec)
    if param_type in PIECEWISE_TYPES:
        cleaned_spec["value"] = get_piecewise_parameters(
            leaf_name=leaf_name,
            func_type=param_type,  # ty: ignore[invalid-argument-type]
            parameter_list=cleaned_spec["value"],
            xnp=xnp,
        )
        return PiecewisePolynomialParam(**cleaned_spec)
    lookup_table_converters: dict[
        str, FunctionType[..., ConsecutiveIntLookupTableParamValue]
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
        converter = lookup_table_converters[param_type]
        cleaned_spec["value"] = converter(raw=cleaned_spec["value"], xnp=xnp)
        return ConsecutiveIntLookupTableParam(**cleaned_spec)
    if param_type == "require_converter":
        return RawParam(**cleaned_spec)

    raise ValueError(f"Unknown parameter type: {param_type} for {leaf_name}")


def _get_param_value_piecewise(
    relevant_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Resolve piecewise parameter value, handling `updates_previous` chains."""
    current = relevant_specs[-1]
    current.pop("note", None)
    current.pop("reference", None)
    updates_previous = current.pop("updates_previous", False)

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


def _get_param_value(
    relevant_specs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Resolve parameter value, handling `updates_previous` chains."""
    current = relevant_specs[-1]
    current.pop("note", None)
    current.pop("reference", None)
    updates_previous = current.pop("updates_previous", False)

    if updates_previous and len(relevant_specs) <= 1:
        raise ValueError(
            "'updates_previous' cannot be specified on the initial date entry."
        )

    if not updates_previous:
        return current

    base_value = _get_param_value(relevant_specs[:-1])
    return upsert_tree(base=base_value, to_upsert=current)


def _clean_one_param_spec(
    spec: OrigParamSpec,
    policy_date: datetime.date,
) -> dict[str, Any] | None:
    """Prepare the specification of one parameter for creating a ParamObject."""
    date_keys = [key for key in spec if isinstance(key, datetime.date)]
    policy_dates_dt64 = numpy.sort([numpy.datetime64(d) for d in date_keys])
    idx = numpy.searchsorted(
        policy_dates_dt64, numpy.datetime64(policy_date), side="right"
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
    out["unit"] = spec.get("unit", None)
    out["reference_period"] = spec.get("reference_period", None)
    out["name"] = spec["name"]
    out["description"] = spec["description"]

    current_spec: dict[str, Any] = copy.deepcopy(spec[policy_dates[idx - 1]])  # ty: ignore[invalid-assignment]
    out["note"] = current_spec.pop("note", None)
    out["reference"] = current_spec.pop("reference", None)

    # A date entry with only note/reference metadata and no value signals that the
    # parameter is no longer active at this date (e.g. "Ceased to exist" or
    # "Replaced by new rule").
    remaining = {k: v for k, v in current_spec.items() if k != "updates_previous"}
    if not remaining:
        return None

    param_type = spec["type"]
    if param_type == "scalar":
        if current_spec.pop("updates_previous", False):
            raise ValueError(
                "'updates_previous' cannot be specified for scalar parameters."
            )
        out["value"] = current_spec["value"]
    elif param_type in PIECEWISE_TYPES:
        relevant_specs: list[dict[str, Any]] = [
            copy.deepcopy(spec[policy_dates[i]]) for i in range(idx)
        ]  # ty: ignore[invalid-assignment]
        out["value"] = _get_param_value_piecewise(relevant_specs)
    else:
        relevant_specs: list[dict[str, Any]] = [
            copy.deepcopy(spec[policy_dates[i]]) for i in range(idx)
        ]  # ty: ignore[invalid-assignment]
        out["value"] = _get_param_value(relevant_specs)
    return out
