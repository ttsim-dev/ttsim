from __future__ import annotations

import copy
import datetime
import itertools
from typing import TYPE_CHECKING, Any

import dags.tree as dt
import numpy
import optree
import yaml

from ttsim.loader import (
    orig_params_tree,
    orig_ttsim_objects_tree,
)
from ttsim.piecewise_polynomial import (
    check_and_get_thresholds,
    get_piecewise_parameters,
    piecewise_polynomial,
)
from ttsim.shared import (
    assert_valid_ttsim_pytree,
    to_datetime,
    upsert_path_and_value,
    upsert_tree,
)
from ttsim.ttsim_objects import (
    DEFAULT_END_DATE,
    TTSIMObject,
    policy_function,
)
from ttsim.ttsim_params import (
    DictTTSIMParam,
    ListTTSIMParam,
    ScalarTTSIMParam,
    TTSIMParam,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from ttsim.typing import (
        DashedISOString,
        FlatOrigParamSpecDict,
        FlatTTSIMObjectDict,
        FlatTTSIMParamDict,
        NestedTTSIMObjectDict,
        OrigParamSpec,
    )


class PolicyEnvironment:
    """
    A container for policy functions and parameters.

    Almost always, instances are created with `set_up_policy_environment()`.

    Parameters
    ----------
    raw_objects_tree
        The pytree of TTSIM objects (policy inputs, policy functions, agg functions).
    params
        A dictionary with policy parameters.
    """

    def __init__(
        self,
        raw_objects_tree: NestedTTSIMObjectDict,
        params: dict[str, Any] | None = None,
        params_tree: FlatTTSIMParamDict | None = None,
    ):
        # Check functions tree and convert functions to PolicyFunction if necessary
        assert_valid_ttsim_pytree(
            tree=raw_objects_tree,
            leaf_checker=lambda leaf: isinstance(leaf, TTSIMObject),
            tree_name="raw_objects_tree",
        )
        self._raw_objects_tree = optree.tree_map(
            lambda leaf: _convert_to_policy_function_if_not_ttsim_object(leaf),
            raw_objects_tree,
        )
        _fail_if_group_ids_are_outside_top_level_namespace(raw_objects_tree)

        # Read in parameters and aggregation specs
        self._params = params if params is not None else {}
        self._params_tree = params_tree if params_tree is not None else {}

    @property
    def raw_objects_tree(self) -> NestedTTSIMObjectDict:
        """The raw TTSIM objects including policy_inputs.

        Does not include aggregations or time conversions.
        """
        return self._raw_objects_tree

    @property
    def params(self) -> dict[str, Any]:
        """The parameters of the policy environment."""
        return self._params

    @property
    def grouping_levels(self) -> tuple[str, ...]:
        """The grouping levels of the policy environment."""
        return tuple(
            name.rsplit("_", 1)[0]
            for name in self._raw_objects_tree.keys()  # noqa: SIM118
            if name.endswith("_id") and name != "p_id"
        )

    def upsert_objects(
        self, tree_to_upsert: NestedTTSIMObjectDict
    ) -> PolicyEnvironment:
        """Upsert GETTSIM's function tree with (parts of) a new TTSIM objects tree.

        Adds to or overwrites TTSIM objects of the policy environment. Note that this
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
        result._params = self._params  # noqa: SLF001

        return result

    def replace_all_parameters(self, params: dict[str, Any]) -> PolicyEnvironment:
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
        result._params = params  # noqa: SLF001

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

    _orig_ttsim_objects_tree = orig_ttsim_objects_tree(root)
    _orig_yaml_tree = orig_params_tree(root)
    # Will move this line out eventually. Just include in tests, do not run every time.
    fail_because_of_clashes(
        orig_ttsim_objects_tree=_orig_ttsim_objects_tree,
        orig_params_tree=_orig_yaml_tree,
    )

    params = {}
    if "_gettsim" in root.name:
        from _gettsim.config import (
            INTERNAL_PARAMS_GROUPS as internal_params_groups,  # noqa: N811
        )

        for group in internal_params_groups:
            raw_group_data = yaml.load(
                (root / "parameters" / f"{group}.yaml").read_text(encoding="utf-8"),
                Loader=yaml.CLoader,
            )
            params_one_group = _parse_raw_parameter_group(
                raw_group_data=raw_group_data,
                date=date,
                group=group,
                parameters=None,
            )

            # Align parameters for piecewise polynomial functions
            params[group] = _parse_piecewise_parameters(params_one_group)
        params = _parse_kinderzuschl_max(date, params)
        params = _parse_einführungsfaktor_vorsorgeaufwendungen_alter_ab_2005(
            date, params
        )
        params = _parse_vorsorgepauschale_rentenv_anteil(date, params)
        params_tree = {}
    else:
        params = {}
        params_tree = active_ttsim_params_tree(
            orig_params_tree=_orig_yaml_tree, date=date
        )
    return PolicyEnvironment(
        raw_objects_tree=active_ttsim_objects_tree(
            orig_ttsim_objects_tree=_orig_ttsim_objects_tree, date=date
        ),
        params=params,
        params_tree=params_tree,
    )


def fail_because_of_clashes(
    orig_ttsim_objects_tree: FlatTTSIMObjectDict,
    orig_params_tree: FlatOrigParamSpecDict,
) -> None:
    """Fail because of clashes of names.

    Two scenarios are checked:
    - Within TTSIMObjects, active periods of a leaf name could overlap
    - There may be name clashes involving parameters (parameters from multiple files
      or parameters named in the same way as a leaf name).

    Raises
    ------
    ConflictingActivePeriodsError
        If multiple objects with the same leaf name are active at the same time.
    ConflictingNamesError
        If there are name clashes involving parameters.
    """

    # Create mapping from leaf names to TTSIM objects.
    overlap_checker: dict[tuple[str, ...], list[TTSIMObject]] = {}
    for orig_path, obj in orig_ttsim_objects_tree.items():
        path = (*orig_path[:-2], obj.leaf_name)
        if path in overlap_checker:
            overlap_checker[path].append(obj)
        else:
            overlap_checker[path] = [obj]

    # Check for overlapping start and end dates for time-dependent functions.
    for path, objects in overlap_checker.items():
        active_period = [(f.start_date, f.end_date) for f in objects]
        for (start1, end1), (start2, end2) in itertools.combinations(active_period, 2):
            if start1 <= end2 and start2 <= end1:
                raise ConflictingActivePeriodsError(
                    affected_ttsim_objects=objects,
                    path=path,
                    overlap_start=max(start1, start2),
                    overlap_end=min(end1, end2),
                )

    # Start with a single element each.
    name_checker: dict[tuple[str, ...], list[TTSIMObject | OrigParamSpec]] = {}
    for path, objects in overlap_checker.items():
        name_checker[path] = [objects[0]]
    # Now add the yaml objects.
    for orig_path, obj in orig_params_tree.items():
        path = (*orig_path[:-2], orig_path[-1])
        if path in name_checker:
            name_checker[path].append(obj)
        else:
            name_checker[path] = [obj]

    for path, objects in name_checker.items():
        if len(objects) > 1:
            raise ConflictingNamesError(
                affected_objects=objects,
                path=path,
            )


class ConflictingActivePeriodsError(Exception):
    def __init__(
        self,
        affected_ttsim_objects: list[TTSIMObject],
        path: tuple[str, ...],
        overlap_start: datetime.date,
        overlap_end: datetime.date,
    ) -> None:
        self.affected_ttsim_objects = affected_ttsim_objects
        self.path = path
        self.overlap_start = overlap_start
        self.overlap_end = overlap_end

    def __str__(self) -> str:
        overlapping_objects = [
            obj.__getattribute__("original_function_name")
            for obj in self.affected_ttsim_objects
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


class ConflictingNamesError(Exception):
    def __init__(
        self,
        affected_objects: list[TTSIMObject | OrigParamSpec],
        path: tuple[str, ...],
    ) -> None:
        self.affected_objects = affected_objects
        self.path = path

    def __str__(self) -> str:
        objects = []
        for obj in self.affected_objects:
            if isinstance(obj, TTSIMObject):
                objects.append(obj.__getattribute__("original_function_name"))
            else:
                objects.append(str(obj))
        return f"""
        Objects with path

          {self.path}

        clash. The following objects are affected:

          {
            '''
          '''.join(objects)
        }

        """


def active_ttsim_objects_tree(
    orig_ttsim_objects_tree: FlatTTSIMObjectDict, date: datetime.date
) -> NestedTTSIMObjectDict:
    """
    Traverse `root` and return all TTSIMObjects for a given date.

    Parameters
    ----------
    root:
        The directory to traverse.
    date:
        The date for which policy objects should be loaded.

    Returns
    -------
    A tree of active TTSIMObjects.
    """

    flat_objects_tree = {
        (*orig_path[:-2], obj.leaf_name): obj
        for orig_path, obj in orig_ttsim_objects_tree.items()
        if obj.is_active(date)
    }

    return dt.unflatten_from_tree_paths(flat_objects_tree)


def _convert_to_policy_function_if_not_ttsim_object(
    input_object: Callable | TTSIMObject,
) -> TTSIMObject:
    """Convert an object to a PolicyFunction if it is not already a TTSIMObject.

    Parameters
    ----------
    input_object
        The object to convert.

    Returns
    -------
    converted_object
        The converted object.

    """
    if isinstance(input_object, TTSIMObject):
        converted_object = input_object
    else:
        converted_object = policy_function(leaf_name=input_object.__name__)(
            input_object
        )

    return converted_object


def _fail_if_group_ids_are_outside_top_level_namespace(
    raw_objects_tree: NestedTTSIMObjectDict,
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


def active_ttsim_params_tree(
    orig_params_tree: FlatOrigParamSpecDict,
    date: datetime.date,
) -> FlatTTSIMParamDict:
    """Parse the original yaml tree."""
    flat_params_tree = {}
    for orig_path, orig_params_spec in orig_params_tree.items():
        path_to_keep = orig_path[:-2]
        leaf_name = orig_path[-1]
        param = get_one_ttsim_param(
            leaf_name=leaf_name,
            spec=orig_params_spec,
            date=date,
        )
        if param is not None:
            flat_params_tree[(*path_to_keep, leaf_name)] = param
        if orig_params_spec.get("add_jahresanfang", False):
            date_jan1 = date.replace(month=1, day=1)
            leaf_name_jan1 = f"{leaf_name}_jahresanfang"
            param = get_one_ttsim_param(
                leaf_name=leaf_name_jan1,
                spec=orig_params_spec,
                date=date_jan1,
            )
            if param is not None:
                flat_params_tree[(*path_to_keep, leaf_name_jan1)] = param
    return dt.unflatten_from_tree_paths(flat_params_tree)


def get_one_ttsim_param(
    leaf_name: str,
    spec: OrigParamSpec,
    date: datetime.date,
) -> TTSIMParam:
    """Parse the original specification found in the yaml tree."""
    cleaned_spec = prep_one_params_spec(leaf_name=leaf_name, spec=spec, date=date)

    if cleaned_spec is None:
        return None
    elif spec["type"] == "scalar":
        return ScalarTTSIMParam(**cleaned_spec)
    elif spec["type"] == "dict":
        return DictTTSIMParam(**cleaned_spec)
    elif spec["type"] == "list":
        return ListTTSIMParam(**cleaned_spec)
    elif spec["type"] == "piecewise_linear":
        return PiecewiseLinearTTSIMParam(**cleaned_spec)
    elif spec["type"] == "piecewise_quadratic":
        return PiecewiseQuadraticTTSIMParam(**cleaned_spec)
    else:
        raise ValueError(f"Unknown parameter type: {spec['type']} for {leaf_name}")


def prep_one_params_spec(
    leaf_name: str, spec: OrigParamSpec, date: datetime.date
) -> dict[str, Any] | None:
    """Prepare the specification of one parameter for creating a TTSIMParam."""
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
    if relevant_specs[-1].get("updates_previous", False):
        assert len(relevant_specs) > 1, (
            "'updates_previous' cannot be missing in the initial spec, found "
            f"{relevant_specs}"
        )
        params = copy.deepcopy(_get_params_contents(relevant_specs[:-1]))
        params.update(relevant_specs[-1])
        return params
    else:
        return relevant_specs[-1]


def _parse_piecewise_parameters(tax_data: dict[str, Any]) -> dict[str, Any]:
    """Check if parameters are stored in implicit structures and align to general
    structure.

    Parameters
    ----------
    tax_data
        Loaded raw tax data.

    Returns
    -------
    Parsed parameters ready to use in gettsim.

    """
    for param in tax_data:  # noqa: PLC0206
        if isinstance(tax_data[param], dict):
            if "type" in tax_data[param]:
                if tax_data[param]["type"].startswith("piecewise"):
                    if "progressionsfaktor" in tax_data[param]:
                        if tax_data[param]["progressionsfaktor"]:
                            tax_data[param] = add_progressionsfaktor(
                                tax_data[param], param
                            )
                    tax_data[param] = get_piecewise_parameters(
                        tax_data[param],
                        param,
                        func_type=tax_data[param]["type"].split("_")[1],
                    )
            for key in ["type", "progressionsfaktor"]:
                tax_data[param].pop(key, None)

    return tax_data


def _parse_kinderzuschl_max(
    date: datetime.date, params: dict[str, Any]
) -> dict[str, Any]:
    """Prior to 2021, the maximum amount of the Kinderzuschlag was specified directly in
    the laws and directives.

    In 2021, 2022, and from 2024 on, this measure has been derived from
    subsistence levels. This function implements that calculation.

    For 2023 the amount is once again explicitly specified as a parameter.

    Parameters
    ----------
    date
        The date for which the policy parameters are set up.
    params
        A dictionary with parameters from the policy environment.

    Returns
    -------
    updated dictionary

    """

    if 2023 > date.year >= 2021:
        assert {"kinderzuschl", "kindergeld"} <= params.keys()
        params["kinderzuschl"]["maximum"] = (
            params["kinderzuschl"]["existenzminimum"]["regelsatz"]["kinder"]
            + params["kinderzuschl"]["existenzminimum"]["kosten_der_unterkunft"][
                "kinder"
            ]
            + params["kinderzuschl"]["existenzminimum"]["heizkosten"]["kinder"]
        ) / 12 - params["kindergeld"]["kindergeld"][1]

    return params


def _parse_einführungsfaktor_vorsorgeaufwendungen_alter_ab_2005(
    date: datetime.date, params: dict[str, Any]
) -> dict[str, Any]:
    """Calculate introductory factor for pension expense deductions which depends on the
    current year as follows:

    In the years 2005-2025 the share of deductible contributions increases by
    2 percentage points each year from 60% in 2005 to 100% in 2025.

    Reference: § 10 Abs. 1 Nr. 2 Buchst. a und b EStG

    Parameters
    ----------
    date
        The date for which the policy parameters are set up.
    params
        A dictionary with parameters from the policy environment.

    Returns
    -------
    Updated dictionary.

    """
    jahr = date.year
    if jahr >= 2005:
        out = piecewise_polynomial(
            jahr,
            thresholds=params["eink_st_abzuege"]["einführungsfaktor"]["thresholds"],
            rates=params["eink_st_abzuege"]["einführungsfaktor"]["rates"],
            intercepts_at_lower_thresholds=params["eink_st_abzuege"][
                "einführungsfaktor"
            ]["intercepts_at_lower_thresholds"],
        )
        params["eink_st_abzuege"][
            "einführungsfaktor_vorsorgeaufwendungen_alter_ab_2005"
        ] = out
    return params


def _parse_vorsorgepauschale_rentenv_anteil(
    date: datetime.date, params: dict[str, Any]
) -> dict[str, Any]:
    """Calculate the share of pension contributions to be deducted for Lohnsteuer
    increases by year.

    Parameters
    ----------
    date
        The date for which the policy parameters are set up.
    params
        A dictionary with parameters from the policy environment.

    Returns
    -------
    out

    """

    jahr = date.year
    if jahr >= 2005:
        out = piecewise_polynomial(
            jahr,
            thresholds=params["eink_st_abzuege"]["vorsorgepauschale_rentenv_anteil"][
                "thresholds"
            ],
            rates=params["eink_st_abzuege"]["vorsorgepauschale_rentenv_anteil"][
                "rates"
            ],
            intercepts_at_lower_thresholds=params["eink_st_abzuege"][
                "vorsorgepauschale_rentenv_anteil"
            ]["intercepts_at_lower_thresholds"],
        )
        params["eink_st_abzuege"]["vorsorgepauschale_rentenv_anteil"] = out

    return params


def _parse_raw_parameter_group(
    raw_group_data: dict[str, Any],
    date: datetime.date,
    group: str,
    parameters: list[str] | None = None,
) -> dict[str, Any]:
    """Load data from raw yaml group file.

    Parameters
    ----------
    date
        The date for which the policy system is set up.
    group
        Policy system compartment.
    parameters
        List of parameters to be loaded. Only relevant for in function calls.
    yaml_path
        Path to directory of yaml_file. (Used for testing of this function).

    Returns
    -------
    Dictionary of parameters loaded from raw yaml file and striped of unnecessary keys.

    """

    def subtract_years_from_date(date: datetime.date, years: int) -> datetime.date:
        """Subtract one or more years from a date object."""
        try:
            date = date.replace(year=date.year - years)

        # Take care of leap years
        except ValueError:
            date = date.replace(year=date.year - years, day=date.day - 1)
        return date

    def set_date_to_beginning_of_year(date: datetime.date) -> datetime.date:
        """Set date to the beginning of the year."""

        date = date.replace(month=1, day=1)

        return date

    # Load parameters (exclude 'rounding' parameters which are handled at the
    # end of this function)
    not_trans_keys = ["note", "reference", "deviation_from", "access_different_date"]
    out_params: dict[str, Any] = {}
    if not parameters:
        parameters = list(raw_group_data.keys())

    # Load values of all parameters at the specified date
    for param in parameters:
        policy_dates = sorted(
            key for key in raw_group_data[param] if isinstance(key, datetime.date)
        )
        past_policies = [d for d in policy_dates if d <= date]

        if not past_policies:
            # If no policy exists, then we check if the policy maybe agrees right now
            # with another one.
            # Otherwise, do not create an entry for this parameter.
            pass
        else:
            max_past_policy_date = numpy.array(past_policies).max()
            policy_in_place = raw_group_data[param][max_past_policy_date]
            if "scalar" in policy_in_place:
                if policy_in_place["scalar"] == "inf":
                    out_params[param] = numpy.inf
                else:
                    out_params[param] = policy_in_place["scalar"]
            else:
                out_params[param] = {}
                # Keys which if given are transferred
                add_trans_keys = ["type", "progressionsfaktor"]
                for key in add_trans_keys:
                    if key in raw_group_data[param]:
                        out_params[param][key] = raw_group_data[param][key]
                value_keys = (
                    key for key in policy_in_place if key not in not_trans_keys
                )
                if "deviation_from" in policy_in_place:
                    if policy_in_place["deviation_from"] == "previous":
                        new_date = max_past_policy_date - datetime.timedelta(days=1)
                        out_params[param] = _parse_raw_parameter_group(
                            raw_group_data=raw_group_data,
                            date=new_date,
                            group=group,
                            parameters=[param],
                        )[param]
                    elif "." in policy_in_place["deviation_from"]:
                        assert (  # noqa: PT018
                            group == "arbeitsl_geld_2"
                            and param == "eink_anr_frei_kinder"
                        )
                        path_list = policy_in_place["deviation_from"].split(".")
                        out_params[param] = _parse_raw_parameter_group(
                            raw_group_data=raw_group_data,
                            date=date,
                            group=path_list[0],
                            parameters=[path_list[1]],
                        )[path_list[1]]
                    for key in value_keys:
                        key_list: list[str] = []
                        out_params[param][key] = transfer_dictionary(
                            policy_in_place[key],
                            copy.deepcopy(out_params[param][key]),
                            key_list,
                        )
                else:
                    for key in value_keys:
                        out_params[param][key] = policy_in_place[key]

            # Also load earlier parameter values if this is specified in yaml
            if "access_different_date" in raw_group_data[param]:
                if raw_group_data[param]["access_different_date"] == "vorjahr":
                    date_last_year = subtract_years_from_date(date, years=1)
                    params_last_year = _parse_raw_parameter_group(
                        raw_group_data=raw_group_data,
                        date=date_last_year,
                        group=group,
                        parameters=[param],
                    )
                    if param in params_last_year:
                        out_params[f"{param}_vorjahr"] = params_last_year[param]
                elif raw_group_data[param]["access_different_date"] == "jahresanfang":
                    date_beginning_of_year = set_date_to_beginning_of_year(date)
                    if date_beginning_of_year == date:
                        out_params[f"{param}_jahresanfang"] = out_params[param]
                    else:
                        params_beginning_of_year = _parse_raw_parameter_group(
                            raw_group_data=raw_group_data,
                            date=date_beginning_of_year,
                            group=group,
                            parameters=[param],
                        )
                        if param in params_beginning_of_year:
                            out_params[f"{param}_jahresanfang"] = (
                                params_beginning_of_year[param]
                            )
                else:
                    raise ValueError(
                        "Currently, access_different_date is only implemented for "
                        "'vorjahr' (last year) and "
                        "'jahresanfang' (beginning of the year). "
                        f"For parameter {param} a different string is specified."
                    )

    out_params["datum"] = numpy.datetime64(date)

    return out_params


def transfer_dictionary(
    remaining_dict: dict[str, Any] | Any, new_dict: dict[str, Any], key_list: list[str]
) -> dict[str, Any]:
    # To call recursive, always check if object is a dict
    if isinstance(remaining_dict, dict):
        for key in remaining_dict:
            key_list_updated: list[str] = [*key_list, key]
            new_dict = transfer_dictionary(
                remaining_dict[key], new_dict, key_list_updated
            )
    elif len(key_list) == 0:
        return remaining_dict
    else:
        # Now remaining dict is just a scalar
        new_dict = upsert_path_and_value(
            base=new_dict, path_to_upsert=key_list, value_to_upsert=remaining_dict
        )
    return new_dict


def _fail_if_name_of_last_branch_element_not_leaf_name_of_function(
    functions_tree: NestedTTSIMObjectDict,
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


def add_progressionsfaktor(
    params_dict: dict[str | int, Any], parameter: str
) -> dict[str | int, Any]:
    """Quadratic factor of tax tariff function.

    The German tax tariff is defined on several income intervals with distinct
    marginal tax rates at the thresholds. To ensure an almost linear increase of
    the average tax rate, the German tax tariff is defined as a quadratic function,
    where the quadratic rate is the so called linear Progressionsfaktor. For its
    calculation one needs the lower (low_thres) and upper (upper_thres) thresholds of
    the interval as well as the marginal tax rate of the interval (rate_iv) and of the
    following interval (rate_fiv). The formula is then given by:

    (rate_fiv - rate_iv) / (2 * (upper_thres - low_thres))

    """
    out_dict = copy.deepcopy(params_dict)
    interval_keys = sorted(key for key in out_dict if isinstance(key, int))
    # Check and extract lower thresholds.
    lower_thresholds, upper_thresholds = check_and_get_thresholds(
        params_dict, parameter, interval_keys
    )[:2]
    for key in interval_keys:
        if "rate_quadratic" not in out_dict[key]:
            out_dict[key]["rate_quadratic"] = (
                out_dict[key + 1]["rate_linear"] - out_dict[key]["rate_linear"]
            ) / (2 * (upper_thresholds[key] - lower_thresholds[key]))
    return out_dict
