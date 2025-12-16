from __future__ import annotations

import copy
import datetime
import re
from typing import TYPE_CHECKING, Literal

import dags.tree as dt
import numpy
import pandas as pd
import pytest
from mettsim import middle_earth

try:
    import jax
except ImportError:
    jax = None

from ttsim import InputData, MainTarget, OrigPolicyObjects, TTTargets, main
from ttsim.interface_dag_elements.fail_if import (
    ConflictingActivePeriodsError,
    _param_with_active_periods,
    _ParamWithActivePeriod,
    active_periods_overlap,
    assert_valid_ttsim_pytree,
    endogenous_p_id_among_targets,
    foreign_keys_are_invalid_in_data,
    group_ids_are_outside_top_level_namespace,
    group_variables_are_not_constant_within_groups,
    input_data_is_invalid,
    input_df_has_bool_or_numeric_column_names,
    input_df_mapper_columns_missing_in_df,
    input_df_mapper_has_incorrect_format,
    input_df_mapper_p_id_is_missing,
    non_convertible_objects_in_results_tree,
    param_function_depends_on_column_objects,
    paths_are_missing_in_targets_tree_mapper,
    policy_environment_is_invalid,
    targets_are_not_in_specialized_environment_or_data,
)
from ttsim.tt import (
    ConsecutiveIntLookupTableParam,
    ConsecutiveIntLookupTableParamValue,
    DictParam,
    PiecewisePolynomialParam,
    PiecewisePolynomialParamValue,
    ScalarParam,
    group_creation_function,
    param_function,
    policy_function,
    policy_input,
)

if TYPE_CHECKING:
    from types import ModuleType

    from jaxtyping import Array, Float

    from ttsim.typing import (
        FlatColumnObjectsParamFunctions,
        FlatOrigParamSpecs,
        IntColumn,
        NestedData,
        OrigParamSpec,
        PolicyEnvironment,
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

some_consecutive_int_lookup_table_param = ConsecutiveIntLookupTableParam(
    value=ConsecutiveIntLookupTableParamValue(
        bases_to_subtract=numpy.array([1]),
        xnp=numpy,
        values_to_look_up=numpy.array([1, 2, 3]),
    ),
    **_GENERIC_PARAM_SPEC,
)

some_dict_param = DictParam(
    value={"a": 1, "b": 2},
    **_GENERIC_PARAM_SPEC,
)


some_piecewise_polynomial_param = PiecewisePolynomialParam(
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
        "p_id_spouse": numpy.array([2, 1, -1]),
    }


def identity(x: int) -> int:
    return x


def return_one() -> int:
    return 1


def return_two() -> int:
    return 2


def return_three() -> int:
    return 3


@pytest.fixture(scope="module")
def minimal_input_data():
    n_individuals = 5
    return {
        "p_id": pd.Series(numpy.arange(n_individuals), name="p_id"),
        "fam_id": pd.Series(numpy.arange(n_individuals), name="fam_id"),
    }


def mettsim_environment(backend) -> PolicyEnvironment:
    return main(
        main_target="policy_environment",
        orig_policy_objects={"root": middle_earth.ROOT_PATH},
        policy_date=datetime.date(2025, 1, 1),
        backend=backend,
    )


@group_creation_function(
    leaf_name="sp_id", fail_msg_if_included="""This should fail."""
)
def should_fail_sp_id(
    p_id: IntColumn, p_id_spouse: IntColumn, xnp: ModuleType
) -> IntColumn:
    """
    Copy of `sp_id` from METTSIM, but with `fail_msg_if_included` set.
    """
    n = xnp.max(p_id)
    p_id_spouse = xnp.where(p_id_spouse < 0, p_id, p_id_spouse)
    return xnp.maximum(p_id, p_id_spouse) + xnp.minimum(p_id, p_id_spouse) * n


@policy_input(fail_msg_if_included="""This should fail.""")
def p_id_spouse() -> IntColumn:
    """Just to test that we can pass a policy input with `fail_msg_if_included` set."""


@group_creation_function(leaf_name="fam_id")
def dummy_fam_id(sp_id: IntColumn, xnp: ModuleType) -> IntColumn:  # noqa: ARG001
    """
    Just want to use this as a drop-in replacement for `fam_id` from METTSIM with
    minimal inputs.
    """
    return sp_id


def some_x(x):
    return x


@param_function()
def some_param_func_returning_array_of_length_2(xnp: ModuleType) -> Float[Array, 2]:
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
    ("orig_tree_with_column_objects_and_param_functions", "orig_tree_with_params"),
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
    ("orig_tree_with_column_objects_and_param_functions", "orig_tree_with_params"),
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
        "tt_targets__tree",
    ),
    [
        (
            {
                "some_dict_param": some_dict_param,
            },
            {"some_dict_param": "res1"},
        ),
    ],
)
def test_fail_if_data_paths_are_missing_in_paths_to_mapped_column_names(
    environment,
    tt_targets__tree,
    minimal_data_tree,
    backend,
):
    results__tree = main(
        main_target="results__tree",
        input_data={"tree": minimal_data_tree},
        policy_environment=environment,
        evaluation_date=datetime.date(2024, 1, 1),
        tt_targets={"tree": tt_targets__tree},
        rounding=False,
        backend=backend,
    )
    with pytest.raises(
        ValueError,
        match="are not mapped to a column name",
    ):
        paths_are_missing_in_targets_tree_mapper(
            results__tree=results__tree,
            tt_targets__tree=tt_targets__tree,
        )


def test_fail_if_foreign_keys_are_invalid_in_data_allow_minus_one_as_foreign_key(
    backend,
):
    flat_objects_tree = dt.flatten_to_qnames(mettsim_environment(backend))
    data = {
        ("p_id",): pd.Series([1, 2, 3]),
        ("p_id_spouse",): pd.Series([-1, 1, 2]),
    }

    foreign_keys_are_invalid_in_data(
        labels__root_nodes={dt.qname_from_tree_path(n) for n in data if n != ("p_id",)},
        input_data__flat=data,
        specialized_environment__without_tree_logic_and_with_derived_functions=flat_objects_tree,
    )


def test_fail_if_foreign_keys_are_invalid_in_data_when_foreign_key_points_to_non_existing_p_id(
    backend,
):
    flat_objects_tree = dt.flatten_to_qnames(mettsim_environment(backend))
    data = {
        ("p_id",): pd.Series([1, 2, 3]),
        ("p_id_spouse",): pd.Series([0, 1, 2]),
    }

    with pytest.raises(ValueError, match=r"not a valid p_id in the\sinput data"):
        foreign_keys_are_invalid_in_data(
            labels__root_nodes={
                dt.qname_from_tree_path(n) for n in data if n != ("p_id",)
            },
            input_data__flat=data,
            specialized_environment__without_tree_logic_and_with_derived_functions=flat_objects_tree,
        )


def test_fail_if_foreign_keys_are_invalid_in_data_when_foreign_key_points_to_same_row_if_allowed(
    backend,
):
    flat_objects_tree = dt.flatten_to_qnames(mettsim_environment(backend))
    data = {
        ("p_id",): pd.Series([1, 2, 3]),
        ("p_id_child_",): pd.Series([1, 3, 3]),
    }

    foreign_keys_are_invalid_in_data(
        labels__root_nodes={dt.qname_from_tree_path(n) for n in data if n != ("p_id",)},
        input_data__flat=data,
        specialized_environment__without_tree_logic_and_with_derived_functions=flat_objects_tree,
    )


def test_fail_if_foreign_keys_are_invalid_in_data_when_foreign_key_points_to_same_row_if_not_allowed(
    backend,
):
    flat_objects_tree = dt.flatten_to_qnames(mettsim_environment(backend))
    data = {
        ("p_id",): pd.Series([1, 2, 3]),
        ("child_tax_credit__p_id_recipient",): pd.Series([1, 3, 3]),
    }

    foreign_keys_are_invalid_in_data(
        labels__root_nodes={dt.qname_from_tree_path(n) for n in data if n != ("p_id",)},
        input_data__flat=data,
        specialized_environment__without_tree_logic_and_with_derived_functions=flat_objects_tree,
    )


def test_fail_if_foreign_keys_point_to_self_when_they_must_not(backend):
    flat_objects_tree = dt.flatten_to_qnames(mettsim_environment(backend))
    data = {
        ("p_id",): pd.Series([1, 2, 3]),
        ("p_id_spouse",): pd.Series([1, 2, -1]),
    }

    with pytest.raises(
        ValueError,
        match="the following are equal to the p_id in the same row",
    ):
        foreign_keys_are_invalid_in_data(
            labels__root_nodes={
                dt.qname_from_tree_path(n) for n in data if n != ("p_id",)
            },
            input_data__flat=data,
            specialized_environment__without_tree_logic_and_with_derived_functions=flat_objects_tree,
        )


def test_fail_if_group_ids_are_outside_top_level_namespace():
    with pytest.raises(
        ValueError,
        match=r"Group identifiers must live in the top-level namespace. Got:",
    ):
        group_ids_are_outside_top_level_namespace({"n1": {"fam_id": dummy_fam_id}})


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
        match=r"DataFrame column names cannot be booleans or numbers.",
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
            "All path elements of `",
        ),
    ],
)
def test_fail_if_input_df_mapper_has_incorrect_format(
    input_data__df_and_mapper__mapper,
    expected_error_message,
    xnp: ModuleType,
):
    with pytest.raises(TypeError, match=expected_error_message):
        input_df_mapper_has_incorrect_format(input_data__df_and_mapper__mapper, xnp=xnp)


@pytest.mark.parametrize(
    (
        "environment",
        "tt_targets__tree",
        "match",
    ),
    [
        (
            {
                "some_piecewise_polynomial_param": some_piecewise_polynomial_param,
            },
            {"some_piecewise_polynomial_param": "res1"},
            "The data contains objects that cannot be cast to a pandas.DataFrame",
        ),
        (
            {
                "some_consecutive_int_lookup_table_param": (
                    some_consecutive_int_lookup_table_param
                ),
            },
            {"some_consecutive_int_lookup_table_param": "res1"},
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
    tt_targets__tree,
    minimal_data_tree,
    match,
    backend,
):
    with pytest.raises(TypeError, match=match):
        main(
            main_target=MainTarget.results.df_with_nested_columns,
            input_data={"tree": minimal_data_tree},
            policy_environment=environment,
            evaluation_date=datetime.date(2024, 1, 1),
            tt_targets={"tree": tt_targets__tree},
            rounding=False,
            backend=backend,
        )


@pytest.mark.parametrize(
    (
        "environment",
        "tt_targets__tree",
        "match",
    ),
    [
        (
            {
                "some_param_func_returning_array_of_length_2": some_param_func_returning_array_of_length_2,
            },
            {"some_param_func_returning_array_of_length_2": "res1"},
            "The data contains paths that don't have the same length",
        ),
    ],
)
def test_fail_if_non_convertible_objects_in_results_tree_because_of_object_length(
    environment,
    tt_targets__tree,
    minimal_data_tree,
    match,
    backend,
):
    with pytest.raises(ValueError, match=match):
        main(
            main_target=MainTarget.results.df_with_nested_columns,
            input_data={"tree": minimal_data_tree},
            policy_environment=environment,
            evaluation_date=datetime.date(2024, 1, 1),
            tt_targets={"tree": tt_targets__tree},
            rounding=False,
            backend=backend,
        )


@pytest.mark.parametrize(
    "results_numbers",
    [
        1.5,
        numpy.array([1.5]),
        numpy.array([1.5, 2.5, 3.5]),
    ],
)
def test_fail_if_non_convertible_objects_in_results_tree_passes_with_correct_length(
    results_numbers,
    minimal_data_tree,
    xnp,
    backend,
):
    if isinstance(results_numbers, numpy.ndarray):
        results__tree = {"some_col": xnp.array(results_numbers)}
    else:
        results__tree = {"some_number": results_numbers}

    processed_data = main(
        main_target=MainTarget.processed_data,
        input_data={"tree": minimal_data_tree},
        tt_targets={"tree": {}},
        backend=backend,
    )
    non_convertible_objects_in_results_tree(
        processed_data=processed_data,
        results__tree=results__tree,
        backend=backend,
        xnp=xnp,
    )


@pytest.mark.skipif_numpy
def test_fail_if_non_convertible_objects_in_results_tree_passes_with_unsized_jax_array(
    minimal_data_tree,
    xnp,
    backend,
):
    results__tree = {"some_col": xnp.array(1.5)}
    assert results__tree["some_col"].shape == ()
    processed_data = main(
        main_target=MainTarget.processed_data,
        input_data={"tree": minimal_data_tree},
        tt_targets={"tree": {}},
        backend=backend,
    )
    non_convertible_objects_in_results_tree(
        processed_data=processed_data,
        results__tree=results__tree,
        backend=backend,
        xnp=xnp,
    )


def test_fail_if_p_id_does_not_exist(xnp):
    data = {("fam_id",): xnp.array([1, 2, 3])}

    with pytest.raises(
        ValueError,
        match=r"The input data must contain the `p_id` column.",
    ):
        input_data_is_invalid(data, xnp)


def test_fail_if_p_id_is_missing_via_main(backend):
    data = {"fam_id": pd.Series([1, 2, 3], name="fam_id")}
    with pytest.raises(
        ValueError,
        match=r"The input data must contain the `p_id` column.",
    ):
        main(
            main_target="fail_if__input_data_is_invalid",
            input_data={"tree": data},
            policy_environment={},
            tt_targets={"tree": {}},
            evaluation_date=datetime.date(2025, 1, 1),
            rounding=False,
            backend=backend,
        )


def test_fail_if_p_id_is_not_unique(xnp):
    data = {("p_id",): xnp.array([1, 1, 3, 4])}

    with pytest.raises(
        ValueError,
        match="The following `p_id`s are not unique in the input data",
    ):
        input_data_is_invalid(data, xnp)


def test_fail_if_p_id_is_not_unique_via_main(minimal_input_data, backend):
    data = copy.deepcopy(minimal_input_data)
    data["p_id"][:] = 1

    with pytest.raises(
        ValueError,
        match="The following `p_id`s are not unique in the input data",
    ):
        main(
            main_target="fail_if__input_data_is_invalid",
            input_data={"tree": data},
            policy_environment={},
            tt_targets={"tree": {}},
            policy_date=datetime.date(2025, 1, 1),
            rounding=False,
            backend=backend,
        )


@pytest.mark.parametrize(
    "data",
    [
        {("p_id",): numpy.array([1, "2", 3])},
        {("p_id",): numpy.array([1, 2, 3.0])},
        {("p_id",): pd.Series([1, 2, 3.0])},
        {("p_id",): pd.Series([1, "2", 3.0])},
    ],
)
def test_fail_if_p_id_is_not_int(data, xnp):
    with pytest.raises(
        ValueError,
        match=r"The `p_id` column must be of integer dtype.",
    ):
        input_data_is_invalid(data, xnp)


@pytest.mark.parametrize(
    "data",
    [
        {("p_id",): numpy.array([1, 2, 3])},
        {("p_id",): pd.Series([1, 2, 3])},
    ],
)
def test_p_id_can_be_specified_as_series_and_numpy_array(data, xnp):
    input_data_is_invalid(data, xnp)


@pytest.mark.skipif_numpy
def test_p_id_can_be_specified_as_jax_array(xnp):
    data = {("p_id",): xnp.array([1, 2, 3])}
    input_data_is_invalid(data, xnp)


@pytest.mark.parametrize("p_id_value", [-100, 0, 1, 42, 999])
def test_input_data_single_person_with_any_p_id_works_correctly(xnp, p_id_value):
    """Test that single-row data works for any p_id value.

    The p_id=0 case is particularly important because this test would fail under the
    implementation of duplicate detection in place at the time of creation (PR #34).
    """
    data = {("p_id",): xnp.array([p_id_value])}
    input_data_is_invalid(data, xnp)


def test_fail_if_input_data_has_different_lengths(backend):
    data = {"p_id": numpy.arange(4), "a": numpy.arange(8)}
    with pytest.raises(
        ValueError,
        match="The lengths of the following columns do not match the length of the",
    ):
        main(
            main_target="fail_if__input_data_is_invalid",
            input_data={"tree": data},
            policy_environment={},
            tt_targets={"tree": {}},
            evaluation_date=datetime.date(2025, 1, 1),
            rounding=False,
            backend=backend,
        )


def test_fail_if_tt_root_nodes_are_missing_via_main(minimal_input_data, backend):
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
            main_targets=["results__tree", "fail_if__tt_root_nodes_are_missing"],
            input_data={"tree": minimal_input_data},
            policy_environment=policy_environment,
            evaluation_date=datetime.date(2024, 1, 1),
            tt_targets={"tree": {"c": None}},
            rounding=False,
            backend=backend,
        )


def test_fail_if_tt_root_nodes_are_missing_asks_for_individual_level_columns(
    minimal_input_data, backend
):
    @policy_function()
    def b(a_fam):
        return a_fam

    @policy_input()
    def a() -> int:
        pass

    policy_environment = {
        "fam_id": dummy_fam_id,
        "a": a,
        "b": b,
    }

    with pytest.raises(
        ValueError,
        match="Note that the missing nodes contain columns that are grouped by ",
    ):
        main(
            main_targets=["results__tree", "fail_if__tt_root_nodes_are_missing"],
            input_data={"tree": minimal_input_data},
            policy_environment=policy_environment,
            evaluation_date=datetime.date(2024, 1, 1),
            tt_targets={"tree": {"b": None}},
            include_warn_nodes=False,
            include_fail_nodes=False,
            rounding=False,
            backend=backend,
        )


@pytest.mark.parametrize(
    (
        "policy_environment",
        "tt_targets",
        "labels__input_columns",
        "expected_error_match",
    ),
    [
        ({"foo": some_x}, {"bar": None}, set(), "('bar',)"),
        ({"foo__baz": some_x}, {"foo__bar": None}, set(), "('foo', 'bar')"),
        ({"foo": some_x}, {"bar": None}, {"spam"}, "('bar',)"),
        ({"foo__baz": some_x}, {"foo__bar": None}, {"spam"}, "('foo', 'bar')"),
    ],
)
def test_fail_if_targets_are_not_in_specialized_environment_or_data(
    policy_environment,
    tt_targets,
    labels__input_columns,
    expected_error_match,
):
    with pytest.raises(
        ValueError,
        match="The following targets have no corresponding function",
    ) as e:
        targets_are_not_in_specialized_environment_or_data(
            specialized_environment__without_tree_logic_and_with_derived_functions=dt.flatten_to_qnames(
                policy_environment
            ),
            tt_targets__qname=tt_targets,
            labels__input_columns=labels__input_columns,
        )
    assert expected_error_match in str(e.value)


def test_fail_if_targets_are_not_in_specialized_environment_or_data_via_main(
    minimal_input_data,
    backend,
):
    with pytest.raises(
        ValueError,
        match="The following targets have no corresponding function",
    ):
        main(
            main_target="fail_if__targets_are_not_in_specialized_environment_or_data",
            input_data={"tree": minimal_input_data},
            policy_environment={},
            tt_targets={"tree": {"unknown_target": None}},
            evaluation_date=datetime.date(2025, 1, 1),
            rounding=False,
            backend=backend,
        )


@pytest.mark.parametrize(
    ("param_spec", "leaf_name", "expected"),
    [
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
                    original_function_name="bar",
                    start_date=datetime.date(2023, 3, 1),
                    end_date=datetime.date(2099, 12, 31),
                    name={"de": "bar", "en": "bar"},
                    description={"de": "bar", "en": "bar"},
                    unit=None,
                    reference_period=None,
                ),
                _ParamWithActivePeriod(
                    original_function_name="bar",
                    start_date=datetime.date(2016, 1, 1),
                    end_date=datetime.date(2023, 1, 31),
                    name={"de": "bar", "en": "bar"},
                    description={"de": "bar", "en": "bar"},
                    unit=None,
                    reference_period=None,
                ),
                _ParamWithActivePeriod(
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
    ],
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


def test_fail_if_input_df_mapper_columns_missing_in_df():
    df = pd.DataFrame({"a": [1]})
    mapper = {"b": "a", "c": "d", "e": 1, "f": 1.5, "g": True, "h": "i"}
    with pytest.raises(
        ValueError,
        match=r"The following columns are missing: \['d', 'i'\]",
    ):
        input_df_mapper_columns_missing_in_df(
            input_data__df_and_mapper__df=df,
            input_data__df_and_mapper__mapper=mapper,
        )


def test_fail_if_input_df_mapper_columns_missing_in_df_via_main(
    backend: Literal["jax", "numpy"],
):
    df = pd.DataFrame({"a": [1]})
    mapper = {"b": "a", "c": "d", "e": 1, "f": 1.5, "g": True, "h": "i"}
    with pytest.raises(
        ValueError,
        match=r"The following columns are missing: \['d', 'i'\]",
    ):
        main(
            input_data=InputData.df_and_mapper(df=df, mapper=mapper),
            main_target=MainTarget.results.df_with_mapper,
            orig_policy_objects={"root": middle_earth.ROOT_PATH},
            tt_targets=TTTargets(qname={"d": None}),
            policy_date_str="2025-01-01",
            backend=backend,
        )


@pytest.mark.parametrize(
    ("mapper", "expected_exception", "expected_match"),
    [
        # Test case 1: Missing p_id mapping
        (
            {"age": "age_column", "income": "income_column"},
            ValueError,
            r"The input mapper must include a mapping for 'p_id'",
        ),
        # Test case 2: Non-string p_id mapping
        (
            {"p_id": 123, "age": "age_column"},
            TypeError,
            r"The p_id mapping must be a string column name",
        ),
        # Test case 3: Valid mapper should not raise
        (
            {"p_id": "person_id", "age": "age_column"},
            None,
            None,
        ),
    ],
)
def test_fail_if_input_df_mapper_p_id_is_missing(
    mapper, expected_exception, expected_match
):
    """Test that the function fails when p_id mapping is missing or invalid."""
    if expected_exception is None:
        # Should not raise any exception
        input_df_mapper_p_id_is_missing(
            input_data__df_and_mapper__mapper=mapper,
        )
    else:
        with pytest.raises(expected_exception, match=expected_match):
            input_df_mapper_p_id_is_missing(
                input_data__df_and_mapper__mapper=mapper,
            )


def test_fail_if_input_df_mapper_p_id_is_missing_via_main(
    backend: Literal["jax", "numpy"],
):
    """Test p_id mapper validation via main function."""
    df = pd.DataFrame({"age_col": [25, 30], "income_col": [1000, 2000]})
    mapper_without_p_id = {"age": "age_col", "income": "income_col"}

    with pytest.raises(
        ValueError,
        match=r"The input mapper must include a mapping for 'p_id'",
    ):
        main(
            main_target="fail_if__input_df_mapper_p_id_is_missing",
            input_data=InputData.df_and_mapper(df=df, mapper=mapper_without_p_id),
            policy_environment={},
            tt_targets={"tree": {}},
            evaluation_date=datetime.date(2025, 1, 1),
            rounding=False,
            backend=backend,
        )


@pytest.mark.parametrize(
    (
        "tt_targets__tree",
        "match",
    ),
    [
        (
            {
                1: None,
                "number_of_individuals_kin": None,
            },
            "Key 1 in tt_targets__tree must be a string but",
        ),
        (
            {
                "number_of_individuals_kin": 1,
            },
            r"Leaf at tt_targets__tree\[number_of_individuals_kin\] is invalid",
        ),
        (
            ["number_of_individuals_kin"],
            "tt_targets__tree must be a dict, got",
        ),
        (
            "number_of_individuals_kin",
            "tt_targets__tree must be a dict, got",
        ),
    ],
)
def test_invalid_tt_targets_tree(
    tt_targets__tree,
    match,
    backend: Literal["jax", "numpy"],
    xnp: ModuleType,
    minimal_data_tree,
):
    with pytest.raises(TypeError, match=match):
        main(
            main_target=MainTarget.results.df_with_nested_columns,
            backend=backend,
            input_data=InputData.tree(
                tree={
                    **minimal_data_tree,
                    "kin_id": xnp.array([0, 1, 2]),
                }
            ),
            orig_policy_objects={"root": middle_earth.ROOT_PATH},
            policy_date_str="2025-01-01",
            tt_targets={"tree": tt_targets__tree},
        )


@pytest.mark.parametrize(
    (
        "input_data_tree",
        "match",
    ),
    [
        (
            {
                "number_of_individuals_kin": [1],
            },
            r"Leaf at input_data__tree\[number_of_individuals_kin\] is invalid",
        ),
        (
            {"number_of_individuals_kin": "1"},
            r"Leaf at input_data__tree\[number_of_individuals_kin\] is invalid",
        ),
    ],
)
def test_invalid_input_data_tree_via_main(
    input_data_tree, match, backend: Literal["jax", "numpy"], xnp: ModuleType
):
    input_data_tree_with_p_id = {
        **input_data_tree,
        "p_id": xnp.array([2]),
    }
    with pytest.raises(TypeError, match=match):
        main(
            main_target=MainTarget.results.df_with_nested_columns,
            policy_date_str="2025-01-01",
            orig_policy_objects={"root": middle_earth.ROOT_PATH},
            input_data=InputData.tree(tree=input_data_tree_with_p_id),
            tt_targets=TTTargets(tree={"p_id": None}),
            backend=backend,
        )


@pytest.mark.parametrize(
    (
        "policy_environment",
        "match",
    ),
    [
        (
            {
                "invalid_leaf": 42,
            },
            r"Leaf at policy_environment\[invalid_leaf\] is invalid",
        ),
        (
            {
                "nested": {
                    "invalid_leaf": "not_allowed_string",
                },
            },
            r"Leaf at policy_environment\[nested\]\[invalid_leaf\] is invalid",
        ),
        (
            {
                "nested": {
                    "another_invalid": [1, 2, 3],
                },
            },
            r"Leaf at policy_environment\[nested\]\[another_invalid\] is invalid",
        ),
        (
            {
                "nested": {
                    "yet_another": {"dict": "not_allowed"},
                },
            },
            r"Leaf at policy_environment\[nested\]\[yet_another\]\[dict\] is invalid",
        ),
        (
            {
                1: "valid_string",
            },
            "Key 1 in policy_environment must be a string but",
        ),
        (
            ["not_a_dict"],
            "policy_environment must be a dict, got",
        ),
    ],
)
def test_fail_if_policy_environment_is_invalid(policy_environment, match):
    with pytest.raises(TypeError, match=match):
        policy_environment_is_invalid(policy_environment)


@pytest.mark.parametrize(
    "policy_environment",
    [
        # Valid environment with policy functions
        {
            "valid_func": policy_function(leaf_name="valid_func")(identity),
            "another_func": policy_function(leaf_name="another_func")(return_one),
        },
        # Valid environment with param functions
        {
            "some_param_func_returning_array_of_length_2": some_param_func_returning_array_of_length_2,
        },
        # Valid environment with param objects
        {
            "some_dict_param": some_dict_param,
            "some_piecewise_polynomial_param": some_piecewise_polynomial_param,
        },
        # Valid environment with module types
        {
            "numpy_module": numpy,
        },
        # Valid environment with nested structure
        {
            "nested": {
                "nested_func": policy_function(leaf_name="nested_func")(identity),
                "some_param_func_returning_array_of_length_2": some_param_func_returning_array_of_length_2,
            },
            "some_dict_param": some_dict_param,
        },
        # Valid environment with mixed types
        {
            "func": policy_function(leaf_name="func")(identity),
            "some_param_func_returning_array_of_length_2": some_param_func_returning_array_of_length_2,
            "some_dict_param": some_dict_param,
            "module": numpy,
        },
    ],
)
def test_policy_environment_is_invalid_passes(policy_environment):
    """Test that valid environments pass the validation."""
    policy_environment_is_invalid(policy_environment)


def test_raises_error_if_p_id_is_passed_as_scalar(backend: Literal["jax", "numpy"]):
    with pytest.raises(
        ValueError,
        match=r"`p_id` must be an array or series.",
    ):
        main(
            main_target=MainTarget.results.df_with_nested_columns,
            policy_date_str="2025-01-01",
            orig_policy_objects={"root": middle_earth.ROOT_PATH},
            input_data=InputData.tree(tree={"p_id": 1}),
            tt_targets=TTTargets(tree={"p_id": None}),
            backend=backend,
        )


def test_invalid_input_data_as_object_via_main(backend: Literal["jax", "numpy"]):
    # Matches both `TypeError` and `ValueError` because on WSL2 the DAG execution order
    # consistently differs from all other tested platforms:
    # 1. `fail_if.input_data_tree_is_invalid` -> TypeError: "input_data__tree must be a dict"
    #    -> runs before `fail_if.any_paths_are_invalid` on all tested platforms except WSL2
    # 2. `fail_if.any_paths_are_invalid` -> ValueError: "argument type ... not in flattenable types"
    #    -> runs before `fail_if.input_data_tree_is_invalid` on WSL2
    with pytest.raises(
        (TypeError, ValueError),
        match=r"(input_data__tree must be a dict, got|argument type .* is not in the flattenalbe types)",
    ):
        main(
            main_target=MainTarget.results.df_with_nested_columns,
            policy_date_str="2025-01-01",
            orig_policy_objects={"root": middle_earth.ROOT_PATH},
            input_data=InputData.tree(tree=object()),
            tt_targets=TTTargets(tree={"p_id": None}),
            backend=backend,
        )


@pytest.mark.parametrize(
    "policy_environment",
    [
        {"foo": policy_function(leaf_name="bar")(return_one)},
    ],
)
def test_fail_if_name_of_last_branch_element_is_not_the_functions_leaf_name(
    policy_environment: PolicyEnvironment,
    xnp: ModuleType,
):
    with pytest.raises(
        ValueError,
        match="The last element of the object's path must be the same as the leaf name",
    ):
        main(
            main_target=MainTarget.results.df_with_nested_columns,
            policy_environment=policy_environment,
            tt_targets=TTTargets(tree={"p_id": None}),
            input_data=InputData.tree(tree={"p_id": xnp.array([0, 1, 2])}),
        )


@pytest.mark.parametrize(
    "main_target",
    [
        MainTarget.tt_function,
        MainTarget.raw_results.columns,
    ],
)
def test_raise_tt_root_nodes_are_missing_without_input_data(
    main_target: MainTarget,
    backend: Literal["jax", "numpy"],
):
    with pytest.raises(
        ValueError,
        match="The following arguments to `main` are missing",
    ):
        main(
            main_target=main_target,
            policy_date_str="2025-01-01",
            backend=backend,
            orig_policy_objects={"root": middle_earth.ROOT_PATH},
        )


def test_raise_some_error_without_input_data(
    backend: Literal["jax", "numpy"],
):
    with pytest.raises(
        ValueError,
        match="The following arguments to `main` are missing",
    ):
        main(
            policy_date_str="2025-01-01",
            main_target=MainTarget.results.df_with_mapper,
            backend=backend,
            orig_policy_objects={"root": middle_earth.ROOT_PATH},
        )


def test_fail_if_tt_dag_includes_function_with_fail_msg_if_included_set(
    minimal_data_tree: NestedData,
    backend: Literal["jax", "numpy"],
):
    env = mettsim_environment(backend)
    env["sp_id"] = should_fail_sp_id
    env["fam_id"] = dummy_fam_id

    with pytest.raises(
        ValueError,
        match="The TT DAG includes the following functions with `fail_msg_if_included`",
    ):
        main(
            main_target=MainTarget.results.df_with_mapper,
            policy_environment=env,
            tt_targets=TTTargets(tree={"fam_id": None}),
            input_data=InputData.tree(tree=minimal_data_tree),
            include_warn_nodes=False,
            backend=backend,
        )


def test_fail_if_tt_dag_includes_policy_input_with_fail_msg_if_included_set(
    minimal_data_tree: NestedData,
    backend: Literal["jax", "numpy"],
):
    env = mettsim_environment(backend)
    env["fam_id"] = dummy_fam_id
    env["p_id_spouse"] = p_id_spouse

    with pytest.raises(
        ValueError,
        match="The TT DAG includes the following functions with `fail_msg_if_included`",
    ):
        main(
            main_target=MainTarget.results.df_with_mapper,
            policy_environment=env,
            tt_targets=TTTargets(tree={"fam_id": None}),
            input_data=InputData.tree(tree=minimal_data_tree),
            include_warn_nodes=False,
            backend=backend,
        )


def test_fail_if_tt_dag_includes_policy_input_with_fail_msg_if_included_set_does_not_fail_if_overriden(
    minimal_data_tree: NestedData,
    backend: Literal["jax", "numpy"],
):
    env = mettsim_environment(backend)
    env["fam_id"] = dummy_fam_id
    env["sp_id"] = should_fail_sp_id

    minimal_data_tree["sp_id"] = numpy.array([0, 0, 1])

    main(
        main_target=MainTarget.results.df_with_mapper,
        policy_environment=env,
        tt_targets=TTTargets(tree={"fam_id": None}),
        input_data=InputData.tree(tree=minimal_data_tree),
        include_warn_nodes=False,
        backend=backend,
    )


@pytest.mark.skipif(jax is None, reason="Jax is not installed")
def test_backend_has_changed_from_jax_to_numpy_passes():
    policy_environment = main(
        main_target=MainTarget.policy_environment,
        policy_date_str="2000-01-01",
        orig_policy_objects=OrigPolicyObjects(root=middle_earth.ROOT_PATH),
        backend="jax",
    )
    input_data = InputData.tree(
        tree={
            "p_id": jax.numpy.array([0, 1, 2]),  # type: ignore[union-attr]
            "property_tax": {
                "acre_size_in_hectares": jax.numpy.array([5, 20, 200]),  # type: ignore[union-attr]
            },
        }
    )
    main(
        main_target=MainTarget.results.df_with_nested_columns,
        input_data=input_data,
        policy_environment=policy_environment,
        tt_targets=TTTargets(tree={"property_tax": {"amount_y": None}}),
        backend="numpy",
    )


@pytest.mark.skipif(jax is None, reason="Jax is not installed")
def test_backend_has_changed_from_numpy_for_processed_data_to_jax_passes():
    input_data = InputData.tree(
        tree={
            "p_id": numpy.array([0, 1, 2]),
            "property_tax": {
                "acre_size_in_hectares": numpy.array([5, 20, 200]),
            },
        }
    )
    processed_data = main(
        main_target=MainTarget.processed_data,
        backend="numpy",
        input_data=input_data,
        tt_targets=TTTargets(tree={"property_tax": {"amount_y": None}}),
    )
    main(
        main_target=MainTarget.results.df_with_nested_columns,
        policy_date_str="2000-01-01",
        orig_policy_objects=OrigPolicyObjects(root=middle_earth.ROOT_PATH),
        input_data=input_data,
        processed_data=processed_data,
        tt_targets=TTTargets(tree={"property_tax": {"amount_y": None}}),
        backend="jax",
    )


@pytest.mark.skipif(jax is None, reason="Jax is not installed")
def test_backend_has_changed_from_numpy_for_policy_environment_to_jax_raises(
    xnp: ModuleType,
):
    policy_environment = main(
        main_target=MainTarget.policy_environment,
        policy_date_str="2000-01-01",
        orig_policy_objects=OrigPolicyObjects(root=middle_earth.ROOT_PATH),
        backend="numpy",
    )
    input_data = InputData.tree(
        tree={
            "p_id": xnp.array([0, 1, 2]),
            "property_tax": {
                "acre_size_in_hectares": xnp.array([5, 20, 200]),
            },
        }
    )
    with pytest.raises(ValueError, match="Backend has changed"):
        main(
            main_target=MainTarget.results.df_with_nested_columns,
            input_data=input_data,
            policy_environment=policy_environment,
            tt_targets=TTTargets(tree={"property_tax": {"amount_y": None}}),
            backend="jax",
        )


@param_function()
def valid_param_function(x: int) -> int:
    """A valid param function that only depends on parameters."""
    return x * 2


@param_function()
def invalid_param_function(some_policy_function: int) -> int:
    """An invalid param function that depends on a column object."""
    return some_policy_function * 2


@policy_function()
def some_policy_function(x: int) -> int:
    """A policy function for testing."""
    return x + 1


@policy_input()
def some_policy_input() -> int:
    """A policy input for testing."""


@pytest.mark.parametrize(
    "specialized_environment",
    [
        # Valid environment with only param functions and no dependencies
        {
            "valid_param": valid_param_function,
        },
        # Valid environment with param functions and column objects but no dependencies
        {
            "valid_param": valid_param_function,
            "some_policy_function": some_policy_function,
        },
        # Valid environment with mixed types but no violations
        {
            "valid_param": valid_param_function,
            "some_policy_function": some_policy_function,
            "policy_input": some_policy_input,
            "some_scalar": 42,
            "some_dict_param": some_dict_param,
        },
    ],
)
def test_param_function_depends_on_column_objects_passes(specialized_environment):
    """Test that valid environments pass the validation."""
    param_function_depends_on_column_objects(specialized_environment)


@pytest.mark.parametrize(
    ("specialized_environment", "expected_error_match"),
    [
        (
            {
                "invalid_param": invalid_param_function,
                "some_policy_function": some_policy_function,
            },
            "`invalid_param` depends on `some_policy_function`",
        ),
        (
            {
                "invalid_param": invalid_param_function,
                "some_policy_function": some_policy_input,
            },
            "`invalid_param` depends on `some_policy_function`",
        ),
        (
            {
                "valid_param": valid_param_function,
                "invalid_param": invalid_param_function,
                "some_policy_function": some_policy_function,
            },
            "`invalid_param` depends on `some_policy_function`",
        ),
    ],
)
def test_param_function_depends_on_column_objects_raises(
    specialized_environment, expected_error_match
):
    """Test that invalid environments raise the expected error."""
    with pytest.raises(ValueError, match=expected_error_match):
        param_function_depends_on_column_objects(specialized_environment)


def test_param_function_depends_on_column_objects_via_main(
    backend: Literal["jax", "numpy"],
    xnp: ModuleType,
):
    """Test that the param_function_depends_on_column_objects check works via main."""

    with pytest.raises(
        ValueError,
        match="`invalid_param_function` depends on `some_policy_function`",
    ):
        main(
            policy_date_str="2025-01-01",
            main_target=MainTarget.results.df_with_mapper,
            tt_targets={
                "tree": {
                    "invalid_param": None,
                },
            },
            input_data={
                "tree": {
                    "p_id": xnp.array([1, 2, 3]),
                    "x": xnp.array([1, 2, 3]),
                },
            },
            backend=backend,
            policy_environment={
                "invalid_param_function": invalid_param_function,
                "some_policy_function": some_policy_function,
            },
        )


def test_endogenous_p_id_among_targets_direct():
    """Test that p_id_* columns are not allowed as targets."""
    targets_with_p_id = ["p_id_child", "valid_target", "p_id_parent"]
    pattern = re.compile(
        r"p_id_\* columns were requested as targets, but these contain internal "
        r"""ID mappings.+p_id_child',\)",\n    "\('p_id_parent.""",
        re.DOTALL,
    )
    with pytest.raises(ValueError, match=pattern):
        endogenous_p_id_among_targets(labels__column_targets=targets_with_p_id)


def test_endogenous_p_id_among_targets_passes_with_valid_targets():
    """Test that validation passes when no p_id_* columns are in targets."""
    valid_targets = ["income", "tax", "benefit"]

    # Should not raise any exception
    endogenous_p_id_among_targets(labels__column_targets=valid_targets)


def test_endogenous_p_id_among_targets_via_main(xnp):
    """Test that p_id_* columns are not allowed as targets via main."""
    date = datetime.date(2023, 1, 1)
    with pytest.raises(
        ValueError,
        match=(
            r"p_id_\* columns were requested as targets, but these contain internal ID"
        ),
    ):
        main(
            main_target=MainTarget.results.df_with_nested_columns,
            policy_environment={
                "p_id_person": policy_function(
                    start_date=date,
                    end_date=date,
                    leaf_name="p_id_person",
                )(identity),
                "policy_year": ScalarParam(
                    value=date.year,
                    start_date=date,
                    end_date=date,
                ),
                "policy_month": ScalarParam(
                    value=date.month,
                    start_date=date,
                    end_date=date,
                ),
                "policy_day": ScalarParam(
                    value=date.day,
                    start_date=date,
                    end_date=date,
                ),
            },
            tt_targets=TTTargets(tree={"p_id": None, "p_id_person": None}),
            input_data=InputData.tree(tree={"p_id": xnp.array([0, 1, 2])}),
        )


def test_pass_scalars_for_natively_vectorized_functions(
    xnp: ModuleType, backend: Literal["numpy", "jax"]
):
    input_data_tree = {
        "age": 30,
        "kin_id": 0,
        "orc_hunting_bounty": {
            "large_orcs_hunted": 10,
            "small_orcs_hunted": 20,
        },
        "p_id": xnp.array([0, 1, 2]),
        "p_id_parent_1": -1,
        "p_id_parent_2": -1,
        "p_id_spouse": -1,
        "parent_is_noble": False,
        "payroll_tax": {
            "child_tax_credit": {
                "p_id_recipient": -1,
            },
            "income": {
                "gross_wage_y": xnp.array([10000, 20000, 30000]),
            },
        },
        "property_tax": {
            "acre_size_in_hectares": 10,
        },
        "wealth": 10000,
    }
    with pytest.raises(
        ValueError,
        match="The following root nodes must be passed as arrays or series, but were "
        "passed as scalars in the input data:\n\n"
        "    - \\('age',\\)\n"
        "    - \\('kin_id',\\)\n"
        "    - \\('p_id_parent_1',\\)\n"
        "    - \\('p_id_parent_2',\\)\n"
        "    - \\('p_id_spouse',\\)\n"
        "    - \\('parent_is_noble',\\)\n"
        "    - \\('payroll_tax', 'child_tax_credit', 'p_id_recipient'\\)\n"
        "    - \\('wealth',\\)\n\n"
        "To fix this, pass them as arrays \\(numpy, jax\\.numpy\\) or pd\\.Series matching "
        "the length of `p_id`\\.",
    ):
        main(
            main_target=MainTarget.results.df_with_nested_columns,
            policy_date_str="2025-01-01",
            input_data=InputData.tree(input_data_tree),
            tt_targets=TTTargets(
                tree={
                    "wealth_tax": {"amount_y": None},
                    "property_tax": {"amount_y": None},
                    "payroll_tax": {"amount_y": None},
                    "orc_hunting_bounty": {"amount": None},
                    "housing_benefits": {"amount_y_fam": None},
                }
            ),
            orig_policy_objects={"root": middle_earth.ROOT_PATH},
            backend=backend,
        )
