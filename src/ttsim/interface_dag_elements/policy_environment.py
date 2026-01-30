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
from ttsim.tt.interval_utils import extend_intervals_to_real_line
from ttsim.tt.piecewise_polynomial import get_piecewise_parameters

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
    flat_objects_tree = {
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
) -> ParamObject:
    """Parse the original specification found in the yaml tree to a ParamObject."""
    cleaned_spec = _clean_one_param_spec(spec=spec, policy_date=policy_date)

    if cleaned_spec is None:
        return None

    param_type = spec["type"]

    if param_type == "scalar":
        return ScalarParam(**cleaned_spec)
    if param_type == "dict":
        return DictParam(**cleaned_spec)
    if param_type in {
        "piecewise_constant",
        "piecewise_linear",
        "piecewise_quadratic",
        "piecewise_cubic",
    }:
        cleaned_spec["value"] = get_piecewise_parameters(
            leaf_name=leaf_name,
            func_type=param_type,  # ty: ignore[invalid-argument-type]
            parameter_list=cleaned_spec["value"],
            xnp=xnp,
        )
        return PiecewisePolynomialParam(**cleaned_spec)
    lookup_table_converters = {
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
    raw_current = copy.deepcopy(spec[policy_dates[idx - 1]])
    if isinstance(raw_current, list):
        # List-based spec (no note/reference at this level)
        out["note"] = None
        out["reference"] = None
        out["value"] = _get_param_value([spec[d] for d in policy_dates[:idx]])
        return out

    current_spec: dict[str | int, Any] = raw_current
    out["note"] = current_spec.pop("note", None)
    out["reference"] = current_spec.pop("reference", None)
    if not current_spec:
        return None
    if len(current_spec) == 1 and "updates_previous" in current_spec:
        raise ValueError(
            "'updates_previous' cannot be specified as the only element, found:\n\n"
            f"{spec}\n\n",
        )
    if spec["type"] == "scalar":
        if "updates_previous" in current_spec:
            raise ValueError(
                "'updates_previous' cannot be specified for scalar parameters"
            )
        out["value"] = current_spec["value"]
    else:
        out["value"] = _get_param_value([spec[d] for d in policy_dates[:idx]])
    return out


def _get_param_value(
    relevant_specs: list[dict[str | int, Any] | list[dict[str, Any]]],
) -> dict[str | int, Any] | list[dict[str, Any]]:
    """Get the value of a parameter.

    Implementation is a recursion in order to handle the 'updates_previous' machinery.

    Supports both dict-based and list-based (piecewise) specs. When the raw spec
    is a list (no reference/note fields), it's used directly. When it's a dict
    with integer keys (has reference/note alongside), the integer-keyed entries
    are converted to a list.

    """
    raw_spec = relevant_specs[-1]
    if isinstance(raw_spec, list):
        # Already a list (YAML date entry was a plain list)
        return raw_spec

    current_spec = raw_spec.copy()
    updates_previous = current_spec.pop("updates_previous", False)
    current_spec.pop("note", None)
    current_spec.pop("reference", None)
    if updates_previous:
        if len(relevant_specs) <= 1:
            raise ValueError(
                "'updates_previous' cannot be missing in the initial spec, found "
                f"{relevant_specs}"
            )
        base = _get_param_value(relevant_specs=relevant_specs[:-1])
        if isinstance(base, list):
            # List-based spec: convert list to dict with integer keys for merging
            base_dict = dict(enumerate(base))
            merged = upsert_tree(
                base=base_dict,
                to_upsert=current_spec,  # ty: ignore[invalid-argument-type]
            )
            result = [
                merged[i] for i in sorted(k for k in merged if isinstance(k, int))
            ]
            return extend_intervals_to_real_line(result)
        return upsert_tree(
            base=base,  # ty: ignore[invalid-argument-type]
            to_upsert=current_spec,  # ty: ignore[invalid-argument-type]
        )

    # Convert integer-keyed dict to list when keys are consecutive ints 0..n-1
    # and values are dicts (piecewise polynomial interval specs).
    # Do not convert when values are scalars (consecutive_int_lookup_table)
    # or keys are non-consecutive (partial overlay specs like {3: {...}}).
    if (
        current_spec
        and all(isinstance(k, int) for k in current_spec)
        and all(isinstance(v, dict) for v in current_spec.values())
        and sorted(current_spec) == list(range(len(current_spec)))
    ):
        return [current_spec[i] for i in range(len(current_spec))]

    return current_spec
