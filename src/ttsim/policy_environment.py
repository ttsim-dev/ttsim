from __future__ import annotations

import copy
import datetime
import itertools
from typing import TYPE_CHECKING, Any, Literal

import dags.tree as dt
import numpy

from ttsim.config import numpy_or_jax as np
from ttsim.shared import (
    merge_trees,
    to_datetime,
    upsert_tree,
)
from ttsim.tt_dag_elements.column_objects_param_function import (
    DEFAULT_END_DATE,
)
from ttsim.tt_dag_elements.param_objects import (
    ConsecutiveInt1dLookupTableParam,
    ConsecutiveInt1dLookupTableParamValue,
    ConsecutiveInt2dLookupTableParamValue,
    DictParam,
    ParamObject,
    PiecewisePolynomialParam,
    RawParam,
    ScalarParam,
)
from ttsim.tt_dag_elements.piecewise_polynomial import get_piecewise_parameters

if TYPE_CHECKING:
    from ttsim.tt_dag_elements.typing import (
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


def get_consecutive_int_1d_lookup_table_param_value(
    raw: dict[int, float | int | bool],
) -> ConsecutiveInt1dLookupTableParamValue:
    """Get the parameters for a 1-dimensional lookup table."""
    lookup_keys = numpy.asarray(sorted(raw))
    assert (lookup_keys - min(lookup_keys) == np.arange(len(lookup_keys))).all(), (
        "Dictionary keys must be consecutive integers."
    )

    return ConsecutiveInt1dLookupTableParamValue(
        base_to_subtract=min(lookup_keys),
        values_to_look_up=np.asarray([raw[k] for k in lookup_keys]),
    )


def get_consecutive_int_2d_lookup_table_param_value(
    raw: dict[int, dict[int, float | int | bool]],
) -> ConsecutiveInt2dLookupTableParamValue:
    """Get the parameters for a 2-dimensional lookup table."""
    lookup_keys_rows = numpy.asarray(sorted(raw.keys()))
    lookup_keys_cols = numpy.asarray(sorted(raw[lookup_keys_rows[0]].keys()))
    for col_value in raw.values():
        lookup_keys_this_col = numpy.asarray(sorted(col_value.keys()))
        assert (lookup_keys_cols == lookup_keys_this_col).all(), (
            "Column keys must be the same in each column, got:"
            f"{lookup_keys_cols} and {lookup_keys_this_col}"
        )
    for lookup_keys in lookup_keys_rows, lookup_keys_cols:
        assert (lookup_keys - min(lookup_keys) == np.arange(len(lookup_keys))).all(), (
            f"Dictionary keys must be consecutive integers, got: {lookup_keys}"
        )
    return ConsecutiveInt2dLookupTableParamValue(
        base_to_subtract_rows=min(lookup_keys_rows),
        base_to_subtract_cols=min(lookup_keys_cols),
        values_to_look_up=np.array(
            [
                raw[row][col]
                for row, col in itertools.product(lookup_keys_rows, lookup_keys_cols)
            ]
        ).reshape(len(lookup_keys_rows), len(lookup_keys_cols)),
    )


def _year_fraction(r: dict[Literal["years", "months"], int]) -> float:
    return r["years"] + r["months"] / 12


def get_month_based_phase_inout_of_age_thresholds_param_value(
    raw: dict[str | int, Any],
) -> dict[int, float]:
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
        y=first_year_phase_inout, m=first_month_phase_inout
    )
    last_year_phase_inout: int = max(raw.keys())  # type: ignore[assignment]
    last_month_phase_inout: int = max(raw[last_year_phase_inout].keys())
    last_m_since_ad_phase_inout = _m_since_ad(
        y=last_year_phase_inout, m=last_month_phase_inout
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
            last_m_since_ad_phase_inout + 1, last_m_since_ad_to_consider + 1
        )
    }
    return get_consecutive_int_1d_lookup_table_param_value(
        {**before_phase_inout, **during_phase_inout, **after_phase_inout}
    )


def get_year_based_phase_inout_of_age_thresholds_param_value(
    raw: dict[str | int, Any],
) -> dict[int, float]:
    """Get the parameters for year-based phase-in/phase-out of age thresholds.

    Requires all years to be given.
    """

    first_year_to_consider = raw.pop("first_year_to_consider")
    last_year_to_consider = raw.pop("last_year_to_consider")
    assert all(isinstance(k, int) for k in raw)
    first_year_phase_inout: int = min(raw.keys())  # type: ignore[assignment]
    last_year_phase_inout: int = max(raw.keys())  # type: ignore[assignment]
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
        {**before_phase_inout, **during_phase_inout, **after_phase_inout}
    )
