from __future__ import annotations

import copy
import datetime
import re
from typing import TYPE_CHECKING

import dags.tree as dt
import numpy
import pandas as pd
import pytest
from mettsim.config import METTSIM_ROOT

from ttsim import main
from ttsim.interface_dag_elements.fail_if import (
    ConflictingActivePeriodsError,
    _param_with_active_periods,
    _ParamWithActivePeriod,
    active_periods_overlap,
    assert_valid_ttsim_pytree,
    data_paths_are_missing_in_paths_to_column_names,
    foreign_keys_are_invalid_in_data,
    group_ids_are_outside_top_level_namespace,
    group_variables_are_not_constant_within_groups,
    input_data_tree_is_invalid,
    input_df_has_bool_or_numeric_column_names,
    input_df_mapper_has_incorrect_format,
    non_convertible_objects_in_results_tree,
    targets_are_not_in_policy_environment_or_data,
)
from ttsim.tt_dag_elements import (
    ConsecutiveInt1dLookupTableParam,
    ConsecutiveInt1dLookupTableParamValue,
    DictParam,
    PiecewisePolynomialParam,
    PiecewisePolynomialParamValue,
    group_creation_function,
    param_function,
    policy_function,
)

if TYPE_CHECKING:
    from types import ModuleType

    from ttsim.interface_dag_elements.typing import (
        FlatColumnObjectsParamFunctions,
        FlatOrigParamSpecs,
        IntColumn,
        NestedPolicyEnvironment,
        OrigParamSpec,
    )

_GENERIC_PARAM_HEADER = {
    "name": {"de": "foo", "en": "foo"},
    "description": {"de": "foo", "en": "foo"},
    "unit": None,
    "reference_period": None,
}
_GENERIC_PARAM_SPEC = {
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    **_GENERIC_PARAM_HEADER,
}

_SOME_CONSECUTIVE_INT_1D_LOOKUP_TABLE_PARAM = ConsecutiveInt1dLookupTableParam(
    leaf_name="some_consecutive_int_1d_lookup_table_param",
    value=ConsecutiveInt1dLookupTableParamValue(
        base_to_subtract=1,
        values_to_look_up=numpy.array([1, 2, 3]),
    ),
    **_GENERIC_PARAM_SPEC,
)

_SOME_DICT_PARAM = DictParam(
    leaf_name="some_dict_param",
    value={"a": 1, "b": 2},
    **_GENERIC_PARAM_SPEC,
)


_SOME_PIECEWISE_POLYNOMIAL_PARAM = PiecewisePolynomialParam(
    leaf_name="some_piecewise_polynomial_param",
    value=PiecewisePolynomialParamValue(
        thresholds=numpy.array([1, 2, 3]),
        intercepts=numpy.array([1, 2, 3]),
        rates=numpy.array([1, 2, 3]),
    ),
    **_GENERIC_PARAM_SPEC,
)


@pytest.fixture
def minimal_data_tree():
    return {
        "hh_id": numpy.array([1, 2, 3]),
        "p_id": numpy.array([1, 2, 3]),
    }


def identity(x: int) -> int:
    return x


def return_one() -> int:
    return 1


def return_two() -> int:
    return 2


def return_three() -> int:
    return 3


@group_creation_function()
def fam_id() -> int:
    pass


@pytest.fixture(scope="module")
def minimal_input_data():
    n_individuals = 5
    return {
        "p_id": pd.Series(numpy.arange(n_individuals), name="p_id"),
        "fam_id": pd.Series(numpy.arange(n_individuals), name="fam_id"),
    }


@pytest.fixture(scope="module")
def mettsim_environment() -> NestedPolicyEnvironment:
    return main(
        inputs={
            "orig_policy_objects__root": METTSIM_ROOT,
            "date": datetime.date(2025, 1, 1),
        },
        targets=["policy_environment"],
    )["policy_environment"]


def some_x(x):
    return x


@policy_function()
def some_policy_func_returning_array_of_length_2(xnp: ModuleType) -> IntColumn:
    return xnp.array([1, 2])


@param_function()
def some_param_func_returning_list_of_length_2() -> list[int]:
    return [1, 2]


@pytest.mark.parametrize(
    ("tree", "leaf_checker", "err_substr"),
    [
        (
            {"a": 1, "b": 2},
            lambda leaf: leaf is None,
            "Leaf at tree[a] is invalid: got 1 of type <class 'int'>.",
        ),
        (
            {"a": None, "b": {"c": None, "d": 1}},
            lambda leaf: leaf is None,
            "Leaf at tree[b][d] is invalid: got 1 of type <class 'int'>.",
        ),
        (
            [1, 2, 3],
            lambda leaf: leaf is None,
            "tree must be a dict, got <class 'list'>.",
        ),
        (
            {1: 2},
            lambda leaf: leaf is None,
            "Key 1 in tree must be a string but got <class 'int'>.",
        ),
    ],
)
def test_assert_valid_ttsim_pytree(tree, leaf_checker, err_substr):
    with pytest.raises(TypeError, match=re.escape(err_substr)):
        assert_valid_ttsim_pytree(
            tree=tree,
            leaf_checker=leaf_checker,
            tree_name="tree",
        )


@pytest.mark.parametrize(
    "orig_tree_with_column_objects_and_param_functions, orig_tree_with_params",
    [
        # Same global module, no overlapping periods, no name clashes.
        (
            {
                ("c", "a"): policy_function(
                    start_date="2023-01-01",
                    end_date="2023-01-31",
                    leaf_name="f",
                )(identity),
                ("c", "b"): policy_function(
                    start_date="2023-02-01",
                    end_date="2023-02-28",
                    leaf_name="f",
                )(identity),
            },
            {
                ("c", "g"): {  # type: ignore[misc]
                    **_GENERIC_PARAM_HEADER,
                    datetime.date(2023, 1, 1): {"value": 1},
                },
            },
        ),
        # Same submodule, overlapping periods, different leaf names so no name clashes.
        (
            {
                ("x", "c", "a"): policy_function(
                    start_date="2023-01-01",
                    end_date="2023-01-31",
                    leaf_name="f",
                )(identity),
                ("x", "c", "b"): policy_function(
                    start_date="2023-01-01",
                    end_date="2023-02-28",
                    leaf_name="g",
                )(identity),
            },
            {
                ("x", "c", "h"): {  # type: ignore[misc]
                    **_GENERIC_PARAM_HEADER,
                    datetime.date(2023, 1, 1): {"value": 2},
                },
            },
        ),
        # Different submodules, no overlapping periods, no name clashes.
        (
            {
                ("x", "c", "f"): policy_function(
                    start_date="2023-01-01",
                    end_date="2023-01-31",
                )(identity),
                ("x", "d", "f"): policy_function(
                    start_date="2023-02-01",
                    end_date="2023-02-28",
                )(identity),
            },
            {
                ("x", "c", "g"): {  # type: ignore[misc]
                    **_GENERIC_PARAM_HEADER,
                    datetime.date(2023, 1, 1): {"value": 3},
                },
            },
        ),
        # Different paths, overlapping periods, same names but no clashes.
        (
            {
                ("x", "a", "b"): policy_function(
                    start_date="2023-01-01",
                    end_date="2023-01-31",
                    leaf_name="f",
                )(identity),
                ("y", "a", "b"): policy_function(
                    start_date="2023-01-01",
                    end_date="2023-02-28",
                    leaf_name="f",
                )(identity),
            },
            {
                ("z", "a", "f"): {  # type: ignore[misc]
                    **_GENERIC_PARAM_HEADER,
                    datetime.date(2023, 1, 1): {"value": 4},
                },
            },
        ),
        # Different yaml files, no name clashes because of different names.
        (
            {},
            {
                ("x", "a", "f"): {  # type: ignore[misc]
                    **_GENERIC_PARAM_HEADER,
                    datetime.date(2023, 1, 1): {"value": 5},
                },
                ("x", "b", "g"): {  # type: ignore[misc]
                    **_GENERIC_PARAM_HEADER,
                    datetime.date(2023, 1, 1): {"value": 6},
                },
            },
        ),
        # Same leaf names across functions / parameters, but no overlapping periods.
        (
            {
                ("c", "a"): policy_function(
                    start_date="2012-01-01",
                    end_date="2015-12-31",
                    leaf_name="f",
                )(identity),
                ("c", "b"): policy_function(
                    start_date="2023-02-01",
                    end_date="2023-02-28",
                    leaf_name="f",
                )(identity),
            },
            {
                ("c", "f"): {
                    "name": {"de": "foo", "en": "foo"},
                    "description": {"de": "foo", "en": "foo"},
                    "unit": None,
                    "reference_period": None,
                    "type": "scalar",
                    datetime.date(1984, 1, 1): {"value": 1},
                    datetime.date(1985, 1, 1): {"value": 3},
                    datetime.date(1995, 1, 1): {"value": 5},
                    datetime.date(2012, 1, 1): {"note": "more complex, see function"},
                    datetime.date(2016, 1, 1): {"value": 10},
                    datetime.date(2023, 2, 1): {
                        "note": "more complex, see function",
                        "reference": "https://example.com/foo",
                    },
                    datetime.date(2023, 3, 1): {
                        "value": 13,
                        "note": "Complex didn't last long.",
                    },
                },
            },
        ),
        # Different periods specified in different files.
        (
            {},
            {
                ("c", "f"): {
                    "name": {"de": "foo", "en": "foo"},
                    "description": {"de": "foo", "en": "foo"},
                    "unit": None,
                    "reference_period": None,
                    "type": "scalar",
                    datetime.date(1984, 1, 1): {"value": 1},
                    datetime.date(1985, 1, 1): {"value": 3},
                    datetime.date(1995, 1, 1): {"value": 5},
                    datetime.date(2012, 1, 1): {"note": "more complex, see function"},
                },
                ("d", "f"): {
                    "name": {"de": "foo", "en": "foo"},
                    "description": {"de": "foo", "en": "foo"},
                    "unit": None,
                    "reference_period": None,
                    "type": "scalar",
                    datetime.date(2016, 1, 1): {"value": 10},
                    datetime.date(2023, 2, 1): {
                        "note": "more complex, see function",
                        "reference": "https://example.com/foo",
                    },
                    datetime.date(2023, 3, 1): {
                        "value": 13,
                        "note": "Complex didn't last long.",
                    },
                },
            },
        ),
    ],
)
def test_fail_if_active_periods_overlap_passes(
    orig_tree_with_column_objects_and_param_functions: FlatColumnObjectsParamFunctions,
    orig_tree_with_params: FlatOrigParamSpecs,
):
    active_periods_overlap(
        orig_tree_with_column_objects_and_param_functions,
        orig_tree_with_params,
    )


@pytest.mark.parametrize(
    "orig_tree_with_column_objects_and_param_functions, orig_tree_with_params",
    [
        # Exact overlap.
        (
            {
                ("a",): policy_function(
                    start_date="2023-01-01",
                    end_date="2023-01-31",
                    leaf_name="f",
                )(identity),
                ("b",): policy_function(
                    start_date="2023-01-01",
                    end_date="2023-01-31",
                    leaf_name="f",
                )(identity),
            },
            {},
        ),
        # Active period for "a" is subset of "b".
        (
            {
                ("a"): policy_function(
                    start_date="2023-01-01",
                    end_date="2023-01-31",
                    leaf_name="f",
                )(identity),
                ("b"): policy_function(
                    start_date="2021-01-02",
                    end_date="2023-02-01",
                    leaf_name="f",
                )(identity),
            },
            {},
        ),
        # Some overlap.
        (
            {
                ("a",): policy_function(
                    start_date="2023-01-02",
                    end_date="2023-02-01",
                    leaf_name="f",
                )(identity),
                ("b",): policy_function(
                    start_date="2022-01-01",
                    end_date="2023-01-31",
                    leaf_name="f",
                )(identity),
            },
            {},
        ),
        # Same as before, but defined in different modules.
        (
            {
                ("c", "a"): policy_function(
                    start_date="2023-01-02",
                    end_date="2023-02-01",
                    leaf_name="f",
                )(identity),
                ("d", "b"): policy_function(
                    start_date="2022-01-01",
                    end_date="2023-01-31",
                    leaf_name="f",
                )(identity),
            },
            {},
        ),
        # Same as before, but defined in different modules without leaf name.
        (
            {
                ("c", "f"): policy_function(
                    start_date="2023-01-02",
                    end_date="2023-02-01",
                )(identity),
                ("d", "f"): policy_function(
                    start_date="2022-01-01",
                    end_date="2023-01-31",
                )(identity),
            },
            {},
        ),
        # Same global module, no overlap in functions, name clashes leaf name / yaml.
        (
            {
                ("c", "a"): policy_function(
                    start_date="2023-01-01",
                    end_date="2023-01-31",
                    leaf_name="f",
                )(identity),
                ("c", "b"): policy_function(
                    start_date="2023-02-01",
                    end_date="2023-02-28",
                    leaf_name="f",
                )(identity),
            },
            {
                ("c", "f"): {  # type: ignore[misc]
                    **_GENERIC_PARAM_HEADER,
                    datetime.date(2023, 1, 1): {"value": 1},
                },
            },
        ),
        # Same paths, no overlap in functions, name clashes leaf name / yaml.
        (
            {
                ("x", "a", "b"): policy_function(
                    start_date="2023-01-01",
                    end_date="2023-01-31",
                    leaf_name="f",
                )(identity),
                ("x", "a", "c"): policy_function(
                    start_date="2023-02-01",
                    end_date="2023-02-28",
                    leaf_name="f",
                )(identity),
            },
            {
                ("x", "a", "f"): {  # type: ignore[misc]
                    **_GENERIC_PARAM_HEADER,
                    datetime.date(2023, 1, 1): {"value": 2},
                },
            },
        ),
        # Same paths, name clashes within params from different yaml files.
        (
            {},
            {
                ("x", "a", "f"): {  # type: ignore[misc]
                    **_GENERIC_PARAM_HEADER,
                    datetime.date(2023, 1, 1): {"value": 3},
                },
                ("x", "b", "f"): {  # type: ignore[misc]
                    **_GENERIC_PARAM_HEADER,
                    datetime.date(2023, 1, 1): {"value": 4},
                },
            },
        ),
    ],
)
def test_fail_if_active_periods_overlap_raises(
    orig_tree_with_column_objects_and_param_functions: FlatColumnObjectsParamFunctions,
    orig_tree_with_params: FlatOrigParamSpecs,
):
    with pytest.raises(ConflictingActivePeriodsError):
        active_periods_overlap(
            orig_tree_with_column_objects_and_param_functions,
            orig_tree_with_params,
        )


@pytest.mark.parametrize(
    (
        "environment",
        "targets__tree",
    ),
    [
        (
            {
                "some_dict_param": _SOME_DICT_PARAM,
            },
            {"some_dict_param": "res1"},
        ),
    ],
)
def test_fail_if_data_paths_are_missing_in_paths_to_column_names(
    environment,
    targets__tree,
    minimal_data_tree,
):
    results__tree = main(
        inputs={
            "input_data__tree": minimal_data_tree,
            "policy_environment": environment,
            "targets__tree": targets__tree,
            "rounding": False,
        },
        targets=["results__tree"],
    )["results__tree"]
    with pytest.raises(
        ValueError,
        match="failed because the following paths\nare not mapped to a column name",
    ):
        data_paths_are_missing_in_paths_to_column_names(
            results__tree=results__tree,
            targets__tree=targets__tree,
        )


def test_fail_if_foreign_keys_are_invalid_in_data_allow_minus_one_as_foreign_key(
    mettsim_environment: NestedPolicyEnvironment,
):
    flat_objects_tree = dt.flatten_to_qnames(mettsim_environment)
    data = {
        "p_id": pd.Series([1, 2, 3]),
        "p_id_spouse": pd.Series([-1, 1, 2]),
    }

    foreign_keys_are_invalid_in_data(
        labels__root_nodes={n for n in data if n != "p_id"},
        processed_data=data,
        specialized_environment__with_derived_functions_and_processed_input_nodes=flat_objects_tree,
    )


def test_fail_if_foreign_keys_are_invalid_in_data_when_foreign_key_points_to_non_existing_p_id(
    mettsim_environment: NestedPolicyEnvironment,
):
    flat_objects_tree = dt.flatten_to_qnames(mettsim_environment)
    data = {
        "p_id": pd.Series([1, 2, 3]),
        "p_id_spouse": pd.Series([0, 1, 2]),
    }

    with pytest.raises(ValueError, match=r"not a valid p_id in the\sinput data"):
        foreign_keys_are_invalid_in_data(
            labels__root_nodes={n for n in data if n != "p_id"},
            processed_data=data,
            specialized_environment__with_derived_functions_and_processed_input_nodes=flat_objects_tree,
        )


def test_fail_if_foreign_keys_are_invalid_in_data_when_foreign_key_points_to_same_row_if_allowed(
    mettsim_environment: NestedPolicyEnvironment,
):
    flat_objects_tree = dt.flatten_to_qnames(mettsim_environment)
    data = {
        "p_id": pd.Series([1, 2, 3]),
        "p_id_child_": pd.Series([1, 3, 3]),
    }

    foreign_keys_are_invalid_in_data(
        labels__root_nodes={n for n in data if n != "p_id"},
        processed_data=data,
        specialized_environment__with_derived_functions_and_processed_input_nodes=flat_objects_tree,
    )


def test_fail_if_foreign_keys_are_invalid_in_data_when_foreign_key_points_to_same_row_if_not_allowed(
    mettsim_environment: NestedPolicyEnvironment,
):
    flat_objects_tree = dt.flatten_to_qnames(mettsim_environment)
    data = {
        "p_id": pd.Series([1, 2, 3]),
        "child_tax_credit__p_id_recipient": pd.Series([1, 3, 3]),
    }

    foreign_keys_are_invalid_in_data(
        labels__root_nodes={n for n in data if n != "p_id"},
        processed_data=data,
        specialized_environment__with_derived_functions_and_processed_input_nodes=flat_objects_tree,
    )


def test_fail_if_group_ids_are_outside_top_level_namespace():
    with pytest.raises(
        ValueError,
        match="Group identifiers must live in the top-level namespace. Got:",
    ):
        group_ids_are_outside_top_level_namespace({"n1": {"fam_id": fam_id}})


def test_fail_if_group_variables_are_not_constant_within_groups():
    data = {
        "p_id": numpy.array([0, 1, 2]),
        "foo_kin": numpy.array([1, 2, 2]),
        "kin_id": numpy.array([1, 1, 2]),
    }
    with pytest.raises(
        ValueError,
        match="The following data inputs do not have a unique value within",
    ):
        group_variables_are_not_constant_within_groups(
            labels__grouping_levels=("kin",),
            labels__root_nodes={n for n in data if n != "p_id"},
            processed_data=data,
        )


def test_fail_if_input_data_tree_is_invalid(xnp):
    data = {"fam_id": pd.Series(data=numpy.arange(8), name="fam_id")}

    with pytest.raises(
        ValueError,
        match="The input data must contain the `p_id` column.",
    ):
        input_data_tree_is_invalid(input_data__tree=data, xnp=xnp)


def test_fail_if_input_data_tree_is_invalid_via_main():
    data = {"fam_id": pd.Series([1, 2, 3], name="fam_id")}
    with pytest.raises(
        ValueError,
        match="The input data must contain the `p_id` column.",
    ):
        main(
            inputs={
                "input_data__tree": data,
                "policy_environment": {},
                "targets__tree": {},
                "rounding": False,
            },
            targets=["fail_if__input_data_tree_is_invalid"],
        )["fail_if__input_data_tree_is_invalid"]


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame({True: [1, 2]}),
        pd.DataFrame({1: [1, 2]}),
    ],
)
def test_fail_if_input_df_has_bool_or_numeric_column_names(df):
    with pytest.raises(
        ValueError,
        match="DataFrame column names cannot be booleans or numbers.",
    ):
        input_df_has_bool_or_numeric_column_names(df)


@pytest.mark.parametrize(
    (
        "input_data__df_and_mapper__mapper",
        "expected_error_message",
    ),
    [
        (
            [],
            "The inputs tree to column mapping must be a \\(nested\\) dictionary.",
        ),
        (
            {
                "n1": {
                    "n2": pd.Series([1, 2, 3]),
                },
            },
            "n1__n2: Series",
        ),
        (
            {
                "n1": {
                    "n2": None,
                },
            },
            "n1__n2: NoneType",
        ),
        (
            {
                "n1": {
                    True: 2,
                },
            },
            "All path elements of `inputs_tree_to_df_columns` must be strings.",
        ),
    ],
)
def test_fail_if_input_df_mapper_has_incorrect_format(
    input_data__df_and_mapper__mapper,
    expected_error_message,
):
    with pytest.raises(TypeError, match=expected_error_message):
        input_df_mapper_has_incorrect_format(input_data__df_and_mapper__mapper)


@pytest.mark.parametrize(
    (
        "environment",
        "targets__tree",
        "match",
    ),
    [
        (
            {
                "some_piecewise_polynomial_param": _SOME_PIECEWISE_POLYNOMIAL_PARAM,
            },
            {"some_piecewise_polynomial_param": "res1"},
            "The data contains objects that cannot be cast to a pandas.DataFrame",
        ),
        (
            {
                "some_consecutive_int_1d_lookup_table_param": (
                    _SOME_CONSECUTIVE_INT_1D_LOOKUP_TABLE_PARAM
                ),
            },
            {"some_consecutive_int_1d_lookup_table_param": "res1"},
            "The data contains objects that cannot be cast to a pandas.DataFrame",
        ),
        (
            {
                "some_param_func_returning_list_of_length_2": some_param_func_returning_list_of_length_2,
            },
            {"some_param_func_returning_list_of_length_2": "res1"},
            "The data contains objects that cannot be cast to a pandas.DataFrame",
        ),
    ],
)
def test_fail_if_non_convertible_objects_in_results_tree_because_of_object_type(
    environment,
    targets__tree,
    minimal_data_tree,
    match,
    backend,
    xnp,
):
    environment["backend"] = backend
    environment["xnp"] = xnp
    actual = main(
        inputs={
            "input_data__tree": minimal_data_tree,
            "policy_environment": environment,
            "targets__tree": targets__tree,
            "rounding": False,
            "backend": backend,
        },
        targets=["processed_data", "results__tree"],
    )
    with pytest.raises(TypeError, match=match):
        non_convertible_objects_in_results_tree(
            processed_data=actual["processed_data"],
            results__tree=actual["results__tree"],
            xnp=xnp,
        )


@pytest.mark.parametrize(
    (
        "environment",
        "targets__tree",
        "match",
    ),
    [
        (
            {
                "some_policy_func_returning_array_of_length_2": some_policy_func_returning_array_of_length_2,
            },
            {"some_policy_func_returning_array_of_length_2": "res1"},
            "The data contains paths that don't have the same length",
        ),
    ],
)
def test_fail_if_non_convertible_objects_in_results_tree_because_of_object_length(
    environment,
    targets__tree,
    minimal_data_tree,
    match,
    backend,
    xnp,
):
    environment["backend"] = backend
    environment["xnp"] = xnp
    actual = main(
        inputs={
            "input_data__tree": minimal_data_tree,
            "policy_environment": environment,
            "targets__tree": targets__tree,
            "rounding": False,
            "backend": backend,
        },
        targets=["processed_data", "results__tree"],
    )
    with pytest.raises(ValueError, match=match):
        non_convertible_objects_in_results_tree(
            processed_data=actual["processed_data"],
            results__tree=actual["results__tree"],
            xnp=xnp,
        )


def test_fail_if_p_id_does_not_exist(xnp):
    data = {"fam_id": pd.Series(data=numpy.arange(8), name="fam_id")}

    with pytest.raises(
        ValueError,
        match="The input data must contain the `p_id` column.",
    ):
        input_data_tree_is_invalid(input_data__tree=data, xnp=xnp)


def test_fail_if_p_id_does_not_exist_via_main(backend):
    data = {"fam_id": pd.Series([1, 2, 3], name="fam_id")}
    with pytest.raises(
        ValueError,
        match="The input data must contain the `p_id` column.",
    ):
        main(
            inputs={
                "input_data__tree": data,
                "policy_environment": {},
                "targets__tree": {},
                "rounding": False,
                "backend": backend,
            },
            targets=["fail_if__input_data_tree_is_invalid"],
        )["fail_if__input_data_tree_is_invalid"]


def test_fail_if_p_id_is_not_unique(xnp):
    data = {"p_id": pd.Series(data=numpy.arange(4).repeat(2), name="p_id")}

    with pytest.raises(
        ValueError,
        match="The following `p_id`s are not unique in the input data",
    ):
        input_data_tree_is_invalid(input_data__tree=data, xnp=xnp)


def test_fail_if_p_id_is_not_unique_via_main(minimal_input_data, backend):
    data = copy.deepcopy(minimal_input_data)
    data["p_id"][:] = 1

    with pytest.raises(
        ValueError,
        match="The following `p_id`s are not unique in the input data",
    ):
        main(
            inputs={
                "input_data__tree": data,
                "policy_environment": {},
                "targets__tree": {},
                "rounding": False,
                "backend": backend,
            },
            targets=["fail_if__input_data_tree_is_invalid"],
        )["fail_if__input_data_tree_is_invalid"]


def test_fail_if_root_nodes_are_missing_via_main(minimal_input_data, backend):
    def b(a):
        return a

    def c(b):
        return b

    policy_environment = {
        "b": policy_function(leaf_name="b")(b),
        "c": policy_function(leaf_name="c")(c),
    }

    with pytest.raises(
        ValueError,
        match="The following data columns are missing",
    ):
        main(
            inputs={
                "input_data__tree": minimal_input_data,
                "policy_environment": policy_environment,
                "targets__tree": {"c": None},
                "rounding": False,
                "backend": backend,
            },
            targets=["results__tree", "fail_if__root_nodes_are_missing"],
        )


@pytest.mark.parametrize(
    "policy_environment, targets, labels__processed_data_columns, expected_error_match",
    [
        ({"foo": some_x}, {"bar": None}, set(), "('bar',)"),
        ({"foo__baz": some_x}, {"foo__bar": None}, set(), "('foo', 'bar')"),
        ({"foo": some_x}, {"bar": None}, {"spam"}, "('bar',)"),
        ({"foo__baz": some_x}, {"foo__bar": None}, {"spam"}, "('foo', 'bar')"),
    ],
)
def test_fail_if_targets_are_not_in_policy_environment_or_data(
    policy_environment,
    targets,
    labels__processed_data_columns,
    expected_error_match,
):
    with pytest.raises(
        ValueError,
        match="The following targets have no corresponding function",
    ) as e:
        targets_are_not_in_policy_environment_or_data(
            policy_environment=policy_environment,
            targets__qname=targets,
            labels__processed_data_columns=labels__processed_data_columns,
        )
    assert expected_error_match in str(e.value)


def test_fail_if_targets_are_not_in_policy_environment_or_data_via_main(
    minimal_input_data,
):
    with pytest.raises(
        ValueError,
        match="The following targets have no corresponding function",
    ):
        main(
            inputs={
                "input_data__tree": minimal_input_data,
                "policy_environment": {},
                "targets__tree": {"unknown_target": None},
                "rounding": False,
            },
            targets=["fail_if__targets_are_not_in_policy_environment_or_data"],
        )


@pytest.mark.parametrize(
    "param_spec, leaf_name, expected",
    (
        (
            {
                "name": {"de": "spam", "en": "spam"},
                "description": {"de": "spam", "en": "spam"},
                "unit": None,
                "reference_period": None,
                "type": "scalar",
                datetime.date(1984, 1, 1): {"note": "completely empty"},
            },
            "spam",
            [],
        ),
        (
            {
                "name": {"de": "foo", "en": "foo"},
                "description": {"de": "foo", "en": "foo"},
                "unit": None,
                "reference_period": None,
                "type": "scalar",
                datetime.date(1984, 1, 1): {"value": 1},
            },
            "foo",
            [
                _ParamWithActivePeriod(
                    leaf_name="foo",
                    original_function_name="foo",
                    start_date=datetime.date(1984, 1, 1),
                    end_date=datetime.date(2099, 12, 31),
                    **_GENERIC_PARAM_HEADER,
                ),
            ],
        ),
        (
            {
                "name": {"de": "foo", "en": "foo"},
                "description": {"de": "foo", "en": "foo"},
                "unit": None,
                "reference_period": None,
                "type": "scalar",
                datetime.date(1984, 1, 1): {"value": 1},
                datetime.date(1985, 1, 1): {"note": "stop"},
            },
            "foo",
            [
                _ParamWithActivePeriod(
                    leaf_name="foo",
                    original_function_name="foo",
                    start_date=datetime.date(1984, 1, 1),
                    end_date=datetime.date(1984, 12, 31),
                    **_GENERIC_PARAM_HEADER,
                ),
            ],
        ),
        (
            {
                "name": {"de": "bar", "en": "bar"},
                "description": {"de": "bar", "en": "bar"},
                "unit": None,
                "reference_period": None,
                "type": "scalar",
                datetime.date(1984, 1, 1): {"value": 1},
                datetime.date(1985, 1, 1): {"value": 3},
                datetime.date(1995, 1, 1): {"value": 5},
                datetime.date(2012, 1, 1): {"note": "more complex, see function"},
                datetime.date(2016, 1, 1): {"value": 10},
                datetime.date(2023, 2, 1): {
                    "note": "more complex, see function",
                    "reference": "https://example.com/bar",
                },
                datetime.date(2023, 3, 1): {
                    "value": 13,
                    "note": "Complex didn't last long.",
                },
            },
            "bar",
            [
                _ParamWithActivePeriod(
                    leaf_name="bar",
                    original_function_name="bar",
                    start_date=datetime.date(2023, 3, 1),
                    end_date=datetime.date(2099, 12, 31),
                    name={"de": "bar", "en": "bar"},
                    description={"de": "bar", "en": "bar"},
                    unit=None,
                    reference_period=None,
                ),
                _ParamWithActivePeriod(
                    leaf_name="bar",
                    original_function_name="bar",
                    start_date=datetime.date(2016, 1, 1),
                    end_date=datetime.date(2023, 1, 31),
                    name={"de": "bar", "en": "bar"},
                    description={"de": "bar", "en": "bar"},
                    unit=None,
                    reference_period=None,
                ),
                _ParamWithActivePeriod(
                    leaf_name="bar",
                    original_function_name="bar",
                    start_date=datetime.date(1984, 1, 1),
                    end_date=datetime.date(2011, 12, 31),
                    name={"de": "bar", "en": "bar"},
                    description={"de": "bar", "en": "bar"},
                    unit=None,
                    reference_period=None,
                ),
            ],
        ),
    ),
)
def test_ttsim_param_with_active_periods(
    param_spec: OrigParamSpec,
    leaf_name: str,
    expected: list[_ParamWithActivePeriod],
):
    actual = _param_with_active_periods(
        param_spec=param_spec,
        leaf_name=leaf_name,
    )
    assert actual == expected
