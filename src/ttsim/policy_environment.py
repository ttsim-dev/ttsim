from __future__ import annotations

import copy
import datetime
import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import dags.tree as dt
import numpy
import optree

from ttsim.column_objects_param_function import (
    DEFAULT_END_DATE,
    ColumnObject,
    ParamFunction,
    policy_function,
)
from ttsim.config import numpy_or_jax as np
from ttsim.loader import (
    orig_params_tree,
    orig_tree_with_column_objects_param_functions,
)
from ttsim.param_objects import (
    ConsecutiveInt1dLookupTableParam,
    ConsecutiveInt1dLookupTableParamValue,
    ConsecutiveInt2dLookupTableParamValue,
    DictParam,
    ParamObject,
    PiecewisePolynomialParam,
    RawParam,
    ScalarParam,
)
from ttsim.piecewise_polynomial import get_piecewise_parameters
from ttsim.shared import (
    assert_valid_ttsim_pytree,
    merge_trees,
    to_datetime,
    upsert_tree,
)

if TYPE_CHECKING:
    from pathlib import Path

    from ttsim.typing import (
        DashedISOString,
        FlatColumnObjectsParamFunctions,
        FlatOrigParamSpecs,
        GenericCallable,
        NestedAnyTTSIMObject,
        NestedColumnObjectsParamFunctions,
        NestedParamObjects,
        OrigParamSpec,
    )


class PolicyEnvironment:
    """
    A container for policy functions and parameters.

    Almost always, instances are created with `set_up_policy_environment()`.

    Parameters
    ----------
    raw_objects_tree
        The pytree of policy inputs, policy functions, agg functions, param functions.
    params_tree
        The pytree of policy parameters.
    """

    def __init__(
        self,
        raw_objects_tree: NestedColumnObjectsParamFunctions,
        params_tree: NestedParamObjects,
    ):
        # Check tree with policy inputs / functions, params functions.
        assert_valid_ttsim_pytree(
            tree=raw_objects_tree,
            leaf_checker=lambda leaf: isinstance(leaf, ColumnObject | ParamFunction),
            tree_name="raw_objects_tree",
        )
        self._raw_objects_tree = optree.tree_map(
            lambda leaf: _convert_to_policy_function_if_not_ttsim_object(leaf),
            raw_objects_tree,
        )
        _fail_if_group_ids_are_outside_top_level_namespace(raw_objects_tree)

        # Check tree with params
        assert_valid_ttsim_pytree(
            tree=params_tree,
            leaf_checker=lambda leaf: isinstance(leaf, ParamObject),
            tree_name="raw_objects_tree",
        )
        self._params_tree = params_tree

    @property
    def raw_objects_tree(self) -> NestedColumnObjectsParamFunctions:
        """The raw column objects and params functions including policy_inputs.

        Does not include automatically added aggregations / time conversions.
        """
        return self._raw_objects_tree

    @property
    def params_tree(self) -> NestedParamObjects:
        """The parameters of the policy environment."""
        return self._params_tree

    @property
    def combined_tree(self) -> NestedAnyTTSIMObject:
        """The combined tree of raw objects and params."""
        return merge_trees(self._raw_objects_tree, self._params_tree)

    @property
    def grouping_levels(self) -> tuple[str, ...]:
        """The grouping levels of the policy environment."""
        return tuple(
            name.rsplit("_", 1)[0]
            for name in self._raw_objects_tree.keys()  # noqa: SIM118
            if name.endswith("_id") and name != "p_id"
        )

    def upsert_objects(
        self, tree_to_upsert: NestedColumnObjectsParamFunctions
    ) -> PolicyEnvironment:
        """Update and insert *tree_to_upsert* into the existing objects tree.

        Adds to or overwrites elements of the policy environment. Note that this
        method does not modify the current policy environment but returns a new one.

        Parameters
        ----------
        tree_to_upsert
            The functions to add or overwrite.

        Returns
        -------
        The policy environment with the upserted functions.
        """

        tree_to_upsert_with_correct_types = optree.tree_map(
            lambda leaf: _convert_to_policy_function_if_not_ttsim_object(leaf),
            tree_to_upsert,
        )
        _fail_if_name_of_last_branch_element_not_leaf_name_of_function(
            tree_to_upsert_with_correct_types
        )

        # Add functions tree to upsert to new functions tree
        new_tree = upsert_tree(
            base={**self._raw_objects_tree},
            to_upsert=tree_to_upsert_with_correct_types,
        )

        _fail_if_group_ids_are_outside_top_level_namespace(new_tree)

        result = object.__new__(PolicyEnvironment)
        result._raw_objects_tree = new_tree  # noqa: SLF001
        result._params_tree = self._params_tree  # noqa: SLF001

        return result

    def replace_params_tree(self, params_tree: NestedParamObjects) -> PolicyEnvironment:
        """
        Replace all parameters of the policy environment. Note that this
        method does not modify the current policy environment but returns a new one.

        Parameters
        ----------
        params:
            The new parameters.

        Returns
        -------
        The policy environment with the new parameters.
        """
        result = object.__new__(PolicyEnvironment)
        result._raw_objects_tree = self._raw_objects_tree  # noqa: SLF001
        result._params_tree = params_tree  # noqa: SLF001

        return result


def set_up_policy_environment(
    root: Path, date: datetime.date | DashedISOString
) -> PolicyEnvironment:
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

    _orig_tree_with_column_objects_param_functions = (
        orig_tree_with_column_objects_param_functions(root)
    )
    _orig_params_tree = orig_params_tree(root)
    # Will move this line out eventually. Just include in tests, do not run every time.
    fail_because_active_periods_overlap(
        orig_tree_with_column_objects_param_functions=_orig_tree_with_column_objects_param_functions,
        orig_params_tree=_orig_params_tree,
    )
    params_tree = active_params_tree(orig_params_tree=_orig_params_tree, date=date)
    assert "evaluationsjahr" not in params_tree, "evaluationsjahr must not be specified"
    params_tree["evaluationsjahr"] = ScalarParam(
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
    return PolicyEnvironment(
        raw_objects_tree=active_tree_with_column_objects_param_functions(
            orig_tree_with_column_objects_param_functions=_orig_tree_with_column_objects_param_functions,
            date=date,
        ),
        params_tree=params_tree,
    )


def fail_because_active_periods_overlap(
    orig_tree_with_column_objects_param_functions: FlatColumnObjectsParamFunctions,
    orig_params_tree: FlatOrigParamSpecs,
) -> None:
    """Fail because active periods of objects / parameters overlap.

    Checks that objects or parameters with the same tree path / qualified name are not
    active at the same time.

    Raises
    ------
    ConflictingActivePeriodsError
        If multiple objects and/or parameters with the same leaf name are active at the
        same time.
    """
    # Create mapping from leaf names to objects.
    overlap_checker: dict[
        tuple[str, ...], list[ColumnObject | ParamFunction | _ParamWithActivePeriod]
    ] = {}
    for orig_path, obj in orig_tree_with_column_objects_param_functions.items():
        path = (*orig_path[:-2], obj.leaf_name)
        if path in overlap_checker:
            overlap_checker[path].append(obj)
        else:
            overlap_checker[path] = [obj]

    for orig_path, obj in orig_params_tree.items():
        path = (*orig_path[:-2], orig_path[-1])
        if path in overlap_checker:
            overlap_checker[path].extend(
                _param_with_active_periods(param_spec=obj, leaf_name=orig_path[-1])
            )
        else:
            overlap_checker[path] = _param_with_active_periods(
                param_spec=obj, leaf_name=orig_path[-1]
            )

    # Check for overlapping start and end dates for time-dependent functions.
    for path, objects in overlap_checker.items():
        active_period = [(f.start_date, f.end_date) for f in objects]
        for (start1, end1), (start2, end2) in itertools.combinations(active_period, 2):
            if start1 <= end2 and start2 <= end1:
                raise ConflictingActivePeriodsError(
                    affected_column_objects=objects,
                    path=path,
                    overlap_start=max(start1, start2),
                    overlap_end=min(end1, end2),
                )


@dataclass(frozen=True)
class _ParamWithActivePeriod(ParamObject):
    """A ParamObject object which mimics a ColumnObject regarding active periods.

    Only used here for checking overlap.
    """

    original_function_name: str


def _param_with_active_periods(
    param_spec: OrigParamSpec,
    leaf_name: str,
) -> list[_ParamWithActivePeriod]:
    """Return parameter with active periods."""

    def _remove_note_and_reference(entry: dict[str | int, Any]) -> dict[str | int, Any]:
        """Remove note and reference from a parameter specification."""
        entry.pop("note", None)
        entry.pop("reference", None)
        return entry

    relevant = sorted(
        [key for key in param_spec if isinstance(key, datetime.date)],
        reverse=True,
    )
    if not relevant:
        raise ValueError(f"No relevant dates found for {param_spec}")

    params_header = {
        "name": param_spec["name"],
        "description": param_spec["description"],
        "unit": param_spec["unit"],
        "reference_period": param_spec["reference_period"],
    }
    out = []
    start_date: datetime.date | None = None
    end_date = DEFAULT_END_DATE
    for date in relevant:
        if _remove_note_and_reference(param_spec[date]):
            start_date = date
        else:
            if start_date:
                out.append(
                    _ParamWithActivePeriod(
                        leaf_name=leaf_name,
                        start_date=start_date,
                        end_date=end_date,
                        original_function_name=leaf_name,
                        **params_header,
                    )
                )
            start_date = None
            end_date = date - datetime.timedelta(days=1)
    if start_date:
        out.append(
            _ParamWithActivePeriod(
                leaf_name=leaf_name,
                original_function_name=leaf_name,
                start_date=start_date,
                end_date=end_date,
                **params_header,
            )
        )

    return out


class ConflictingActivePeriodsError(Exception):
    def __init__(
        self,
        affected_column_objects: list[ColumnObject],
        path: tuple[str, ...],
        overlap_start: datetime.date,
        overlap_end: datetime.date,
    ) -> None:
        self.affected_column_objects = affected_column_objects
        self.path = path
        self.overlap_start = overlap_start
        self.overlap_end = overlap_end

    def __str__(self) -> str:
        overlapping_objects = [
            obj.__getattribute__("original_function_name")
            for obj in self.affected_column_objects
            if obj
        ]
        return f"""
        Functions with path

          {self.path}

        have overlapping start and end dates. The following functions are affected:

          {
            '''
          '''.join(overlapping_objects)
        }

        Overlap from {self.overlap_start} to {self.overlap_end}."""


def active_tree_with_column_objects_param_functions(
    orig_tree_with_column_objects_param_functions: FlatColumnObjectsParamFunctions,
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
        for orig_path, obj in orig_tree_with_column_objects_param_functions.items()
        if obj.is_active(date)
    }

    return dt.unflatten_from_tree_paths(flat_objects_tree)


def _convert_to_policy_function_if_not_ttsim_object(
    input_object: GenericCallable | ColumnObject | ParamFunction,
) -> ColumnObject:
    """Convert an object to a PolicyFunction if it is not already a ColumnObject.

    Parameters
    ----------
    input_object
        The object to convert.

    Returns
    -------
    converted_object
        The converted object.

    """
    if isinstance(input_object, ColumnObject | ParamFunction):
        converted_object = input_object
    else:
        converted_object = policy_function(leaf_name=input_object.__name__)(
            input_object
        )

    return converted_object


def _fail_if_group_ids_are_outside_top_level_namespace(
    raw_objects_tree: NestedColumnObjectsParamFunctions,
) -> None:
    """Fail if group ids are outside the top level namespace."""
    group_ids_outside_top_level_namespace = {
        tree_path
        for tree_path in dt.flatten_to_tree_paths(raw_objects_tree)
        if len(tree_path) > 1 and tree_path[-1].endswith("_id")
    }
    if group_ids_outside_top_level_namespace:
        raise ValueError(
            "Group identifiers must live in the top-level namespace. Got:\n\n"
            f"{group_ids_outside_top_level_namespace}\n\n"
            "To fix this error, move the group identifiers to the top-level namespace."
        )


def active_params_tree(
    orig_params_tree: FlatOrigParamSpecs,
    date: datetime.date,
) -> NestedParamObjects:
    """Parse the original yaml tree."""
    flat_params_tree = {}
    for orig_path, orig_params_spec in orig_params_tree.items():
        path_to_keep = orig_path[:-2]
        leaf_name = orig_path[-1]
        param = get_one_param(
            leaf_name=leaf_name,
            spec=orig_params_spec,
            date=date,
        )
        if param is not None:
            flat_params_tree[(*path_to_keep, leaf_name)] = param
        if orig_params_spec.get("add_jahresanfang", False):
            date_jan1 = date.replace(month=1, day=1)
            leaf_name_jan1 = f"{leaf_name}_jahresanfang"
            param = get_one_param(
                leaf_name=leaf_name_jan1,
                spec=orig_params_spec,
                date=date_jan1,
            )
            if param is not None:
                flat_params_tree[(*path_to_keep, leaf_name_jan1)] = param
    return dt.unflatten_from_tree_paths(flat_params_tree)


def get_one_param(  # noqa: PLR0911
    leaf_name: str,
    spec: OrigParamSpec,
    date: datetime.date,
) -> ParamObject:
    """Parse the original specification found in the yaml tree."""
    cleaned_spec = prep_one_params_spec(leaf_name=leaf_name, spec=spec, date=date)

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
    elif spec["type"] == "birth_month_based_phase_inout":
        cleaned_spec["value"] = get_birth_month_based_phase_inout_param_value(
            cleaned_spec["value"]
        )
        return ConsecutiveInt1dLookupTableParam(**cleaned_spec)
    elif spec["type"] == "birth_year_based_phase_inout":
        cleaned_spec["value"] = get_birth_year_based_phase_inout_param_value(
            cleaned_spec["value"]
        )
        return ConsecutiveInt1dLookupTableParam(**cleaned_spec)
    elif spec["type"] == "require_converter":
        return RawParam(**cleaned_spec)
    else:
        raise ValueError(f"Unknown parameter type: {spec['type']} for {leaf_name}")


def prep_one_params_spec(
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
        out["value"] = _get_params_contents([spec[d] for d in policy_dates[:idx]])
    return out


def _get_params_contents(
    relevant_specs: list[dict[str | int, Any]],
) -> dict[str | int, Any]:
    """Get the contents of the parameters.

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
            base=_get_params_contents(relevant_specs=relevant_specs[:-1]),
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


def get_birth_month_based_phase_inout_param_value(
    raw: dict[str | int, Any],
) -> dict[int, float]:
    """Get the parameters for birth month-based phase-in.

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

    first_m_since_ad_to_consider = _m_since_ad(
        y=raw.pop("first_birthyear_to_consider"), m=1
    )
    last_m_since_ad_to_consider = _m_since_ad(
        y=raw.pop("last_birthyear_to_consider"), m=12
    )
    assert all(isinstance(k, int) for k in raw)
    first_birthyear_phase_inout: int = min(raw.keys())  # type: ignore[assignment]
    first_birthmonth_phase_inout: int = min(raw[first_birthyear_phase_inout].keys())
    first_m_since_ad_phase_inout = _m_since_ad(
        y=first_birthyear_phase_inout, m=first_birthmonth_phase_inout
    )
    last_birthyear_phase_inout: int = max(raw.keys())  # type: ignore[assignment]
    last_birthmonth_phase_inout: int = max(raw[last_birthyear_phase_inout].keys())
    last_m_since_ad_phase_inout = _m_since_ad(
        y=last_birthyear_phase_inout, m=last_birthmonth_phase_inout
    )
    assert first_m_since_ad_to_consider <= first_m_since_ad_phase_inout
    assert last_m_since_ad_to_consider >= last_m_since_ad_phase_inout
    before_phase_inout: dict[int, float] = {
        b_m: _year_fraction(
            raw[first_birthyear_phase_inout][first_birthmonth_phase_inout]
        )
        for b_m in range(first_m_since_ad_to_consider, first_m_since_ad_phase_inout)
    }
    during_phase_inout: dict[int, float] = _fill_phase_inout(
        raw=raw,  # type: ignore[arg-type]
        first_m_since_ad_phase_inout=first_m_since_ad_phase_inout,
        last_m_since_ad_phase_inout=last_m_since_ad_phase_inout,
    )
    after_phase_inout: dict[int, float] = {
        b_m: _year_fraction(
            raw[last_birthyear_phase_inout][last_birthmonth_phase_inout]
        )
        for b_m in range(
            last_m_since_ad_phase_inout + 1, last_m_since_ad_to_consider + 1
        )
    }
    return get_consecutive_int_1d_lookup_table_param_value(
        {**before_phase_inout, **during_phase_inout, **after_phase_inout}
    )


def get_birth_year_based_phase_inout_param_value(
    raw: dict[str | int, Any],
) -> dict[int, float]:
    """Get the parameters for birth year-based phase-in.

    Requires all birth years to be given.
    """

    first_birthyear_to_consider = raw.pop("first_birthyear_to_consider")
    last_birthyear_to_consider = raw.pop("last_birthyear_to_consider")
    assert all(isinstance(k, int) for k in raw)
    first_birthyear_phase_inout: int = min(raw.keys())  # type: ignore[assignment]
    last_birthyear_phase_inout: int = max(raw.keys())  # type: ignore[assignment]
    assert first_birthyear_to_consider <= first_birthyear_phase_inout
    assert last_birthyear_to_consider >= last_birthyear_phase_inout
    before_phase_inout: dict[int, float] = {
        b_y: _year_fraction(raw[first_birthyear_phase_inout])
        for b_y in range(first_birthyear_to_consider, first_birthyear_phase_inout)
    }
    during_phase_inout: dict[int, float] = {
        b_y: _year_fraction(raw[b_y])  # type: ignore[misc]
        for b_y in raw
    }
    after_phase_inout: dict[int, float] = {
        b_y: _year_fraction(raw[last_birthyear_phase_inout])
        for b_y in range(last_birthyear_phase_inout + 1, last_birthyear_to_consider + 1)
    }
    return get_consecutive_int_1d_lookup_table_param_value(
        {**before_phase_inout, **during_phase_inout, **after_phase_inout}
    )


def _fail_if_name_of_last_branch_element_not_leaf_name_of_function(
    functions_tree: NestedColumnObjectsParamFunctions,
) -> None:
    """Raise error if a PolicyFunction does not have the same leaf name as the last
    branch element of the tree path.
    """

    for tree_path, function in dt.flatten_to_tree_paths(functions_tree).items():
        if tree_path[-1] != function.leaf_name:
            raise KeyError(
                f"""
                The name of the last branch element of the functions tree must be the
                same as the leaf name of the PolicyFunction. The tree path {tree_path}
                is not compatible with the PolicyFunction {function.leaf_name}.
                """
            )
