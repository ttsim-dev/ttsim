from __future__ import annotations

import copy
import datetime
from typing import TYPE_CHECKING, Any

import dags.tree as dt
import numpy

from ttsim.interface_dag_elements.shared import (
    merge_trees,
    to_datetime,
    upsert_tree,
)
from ttsim.tt_dag_elements import (
    ConsecutiveInt1dLookupTableParam,
    DictParam,
    ParamObject,
    PiecewisePolynomialParam,
    RawParam,
    ScalarParam,
    get_consecutive_int_1d_lookup_table_param_value,
    get_consecutive_int_2d_lookup_table_param_value,
    get_month_based_phase_inout_of_age_thresholds_param_value,
    get_year_based_phase_inout_of_age_thresholds_param_value,
)
from ttsim.tt_dag_elements.column_objects_param_function import (
    DEFAULT_END_DATE,
)
from ttsim.tt_dag_elements.piecewise_polynomial import get_piecewise_parameters

if TYPE_CHECKING:
    from ttsim.interface_dag_elements.typing import (
        DashedISOString,
        FlatColumnObjectsParamFunctions,
        FlatOrigParamSpecs,
        NestedColumnObjectsParamFunctions,
        NestedParamObjects,
        NestedPolicyEnvironment,
        OrigParamSpec,
    )


def policy_environment(
    orig_policy_objects__column_objects_and_param_functions: NestedColumnObjectsParamFunctions,  # noqa: E501
    orig_policy_objects__param_specs: FlatOrigParamSpecs,
    date: datetime.date | DashedISOString,
) -> NestedPolicyEnvironment:
    """
    Set up the policy environment for a particular date.

    Parameters
    ----------
    root
        The directory to load the policy environment from.
    date
        The date for which the policy system is set up. An integer is
        interpreted as the year.

    Returns
    -------
    The policy environment for the specified date.
    """
    # Check policy date for correct format and convert to datetime.date
    date = to_datetime(date)

    a_tree = merge_trees(
        left=_active_column_objects_and_param_functions(
            orig=orig_policy_objects__column_objects_and_param_functions,
            date=date,
        ),
        right=_active_param_objects(
            orig=orig_policy_objects__param_specs,
            date=date,
        ),
    )

    assert "evaluationsjahr" not in a_tree, "evaluationsjahr must not be specified"
    a_tree["evaluationsjahr"] = ScalarParam(
        leaf_name="evaluationsjahr",
        start_date=date,
        end_date=date,
        value=date.year,
        name={"de": "Evaluationsjahr. Implementation wird noch verbessert."},
        description={"de": "Der Zeitpunkt, für den die Berechnung durchgeführt wird."},
        unit="Year",
        reference_period=None,
        note=None,
        reference=None,
    )
    return a_tree


def _active_column_objects_and_param_functions(
    orig: FlatColumnObjectsParamFunctions,
    date: datetime.date,
) -> NestedColumnObjectsParamFunctions:
    """
    Traverse `root` and return all ColumnObjectParamFunctions for a given date.

    Parameters
    ----------
    root:
        The directory to traverse.
    date:
        The date for which policy objects should be loaded.

    Returns
    -------
    A tree of active ColumnObjectParamFunctions.
    """

    flat_objects_tree = {
        (*orig_path[:-2], obj.leaf_name): obj
        for orig_path, obj in orig.items()
        if obj.is_active(date)
    }

    return dt.unflatten_from_tree_paths(flat_objects_tree)


def _active_param_objects(
    orig: FlatOrigParamSpecs,
    date: datetime.date,
) -> NestedParamObjects:
    """Parse the original yaml tree."""
    flat_tree_with_params = {}
    for orig_path, orig_params_spec in orig.items():
        path_to_keep = orig_path[:-2]
        leaf_name = orig_path[-1]
        param = _get_one_param(
            leaf_name=leaf_name,
            spec=orig_params_spec,
            date=date,
        )
        if param is not None:
            flat_tree_with_params[(*path_to_keep, leaf_name)] = param
        if orig_params_spec.get("add_jahresanfang", False):
            date_jan1 = date.replace(month=1, day=1)
            leaf_name_jan1 = f"{leaf_name}_jahresanfang"
            param = _get_one_param(
                leaf_name=leaf_name_jan1,
                spec=orig_params_spec,
                date=date_jan1,
            )
            if param is not None:
                flat_tree_with_params[(*path_to_keep, leaf_name_jan1)] = param
    return dt.unflatten_from_tree_paths(flat_tree_with_params)


def _get_one_param(  # noqa: PLR0911
    leaf_name: str,
    spec: OrigParamSpec,
    date: datetime.date,
) -> ParamObject:
    """Parse the original specification found in the yaml tree to a ParamObject."""
    cleaned_spec = _clean_one_param_spec(leaf_name=leaf_name, spec=spec, date=date)

    if cleaned_spec is None:
        return None
    elif spec["type"] == "scalar":
        return ScalarParam(**cleaned_spec)
    elif spec["type"] == "dict":
        return DictParam(**cleaned_spec)
    elif spec["type"].startswith("piecewise_"):
        cleaned_spec["value"] = get_piecewise_parameters(
            leaf_name=leaf_name,
            func_type=spec["type"],
            parameter_dict=cleaned_spec["value"],
        )
        return PiecewisePolynomialParam(**cleaned_spec)
    elif spec["type"] == "consecutive_int_1d_lookup_table":
        cleaned_spec["value"] = get_consecutive_int_1d_lookup_table_param_value(
            cleaned_spec["value"]
        )
        return ConsecutiveInt1dLookupTableParam(**cleaned_spec)
    elif spec["type"] == "consecutive_int_2d_lookup_table":
        cleaned_spec["value"] = get_consecutive_int_2d_lookup_table_param_value(
            cleaned_spec["value"]
        )
        return ConsecutiveInt1dLookupTableParam(**cleaned_spec)
    elif spec["type"] == "month_based_phase_inout_of_age_thresholds":
        cleaned_spec["value"] = (
            get_month_based_phase_inout_of_age_thresholds_param_value(
                cleaned_spec["value"]
            )
        )
        return ConsecutiveInt1dLookupTableParam(**cleaned_spec)
    elif spec["type"] == "year_based_phase_inout_of_age_thresholds":
        cleaned_spec["value"] = (
            get_year_based_phase_inout_of_age_thresholds_param_value(
                cleaned_spec["value"]
            )
        )
        return ConsecutiveInt1dLookupTableParam(**cleaned_spec)
    elif spec["type"] == "require_converter":
        return RawParam(**cleaned_spec)
    else:
        raise ValueError(f"Unknown parameter type: {spec['type']} for {leaf_name}")


def _clean_one_param_spec(
    leaf_name: str, spec: OrigParamSpec, date: datetime.date
) -> dict[str, Any] | None:
    """Prepare the specification of one parameter for creating a ParamObject."""
    policy_dates = numpy.sort([key for key in spec if isinstance(key, datetime.date)])
    idx = numpy.searchsorted(policy_dates, date, side="right")  # type: ignore[call-overload]
    if idx == 0:
        return None

    out: dict[str, Any] = {}
    out["leaf_name"] = leaf_name
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
    current_spec = copy.deepcopy(spec[policy_dates[idx - 1]])
    out["note"] = current_spec.pop("note", None)
    out["reference"] = current_spec.pop("reference", None)
    if len(current_spec) == 0:
        return None
    elif len(current_spec) == 1 and "updates_previous" in current_spec:
        raise ValueError(
            f"'updates_previous' cannot be specified as the only element, found{spec}"
        )
        # Parameter ceased to exist
    elif spec["type"] == "scalar":
        assert "updates_previous" not in current_spec, (
            "'updates_previous' cannot be specified for scalar parameters"
        )
        out["value"] = current_spec["value"]
    else:
        out["value"] = _get_param_value([spec[d] for d in policy_dates[:idx]])
    return out


def _get_param_value(
    relevant_specs: list[dict[str | int, Any]],
) -> dict[str | int, Any]:
    """Get the value of a parameter.

    Implementation is a recursion in order to handle the 'updates_previous' machinery.

    """
    current_spec = relevant_specs[-1].copy()
    updates_previous = current_spec.pop("updates_previous", False)
    current_spec.pop("note", None)
    current_spec.pop("reference", None)
    if updates_previous:
        assert len(relevant_specs) > 1, (
            "'updates_previous' cannot be missing in the initial spec, found "
            f"{relevant_specs}"
        )
        return upsert_tree(
            base=_get_param_value(relevant_specs=relevant_specs[:-1]),
            to_upsert=current_spec,
        )
    else:
        return current_spec
