from __future__ import annotations

from typing import TYPE_CHECKING

import dags.tree as dt
import pandas as pd
import pytest

from mettsim import middle_earth
from ttsim import InputData, MainTarget, TTTargets, main
from ttsim.tt.column_objects_param_function import policy_function

if TYPE_CHECKING:
    from types import ModuleType
    from typing import Literal


DF_WITH_NESTED_COLUMNS = pd.DataFrame(
    {
        ("age",): [10, 30, 30],  # Reordered to match unsorted p_id
        ("kin_id",): [0, 0, 0],
        ("p_id",): [2, 0, 1],  # Deliberately unsorted: 2, 0, 1
        ("p_id_parent_1",): [0, -1, -1],  # Adjusted for new order
        ("p_id_parent_2",): [1, -1, -1],  # Adjusted for new order
        ("p_id_spouse",): [-1, 1, 0],  # Adjusted for new order
        ("parent_is_noble",): [False, False, False],
        ("wealth",): [0.0, 0.0, 0.0],
        ("payroll_tax", "child_tax_credit", "p_id_recipient"): [
            0,
            -1,
            -1,
        ],  # Adjusted for new order
        ("payroll_tax", "income", "gross_wage_y"): [
            0,
            10000,
            0,
        ],  # Adjusted for new order
    },
)


DF_FOR_MAPPER = pd.DataFrame(
    {
        "age": [10, 30, 30],  # Reordered to match unsorted p_id
        "kin_id": [0, 0, 0],
        "p_id": [2, 0, 1],  # Deliberately unsorted: 2, 0, 1
        "parent_1": [0, -1, -1],  # Adjusted for new order
        "parent_2": [1, -1, -1],  # Adjusted for new order
        "spouse": [-1, 1, 0],  # Adjusted for new order
        "parent_is_noble": [False, False, False],
        "child_tax_credit_recipient": [0, -1, -1],  # Adjusted for new order
        "gross_wage_y": [0, 10000, 0],  # Adjusted for new order
        "wealth": [0.0, 0.0, 0.0],
    },
)


INPUT_DF_MAPPER = {
    "age": "age",
    "kin_id": "kin_id",
    "p_id": "p_id",
    "p_id_parent_1": "parent_1",
    "p_id_parent_2": "parent_2",
    "p_id_spouse": "spouse",
    "parent_is_noble": "parent_is_noble",
    "wealth": "wealth",
    "payroll_tax": {
        "child_tax_credit": {
            "p_id_recipient": "child_tax_credit_recipient",
        },
        "income": {
            "gross_wage_y": "gross_wage_y",
        },
    },
}


TARGETS_TREE = {
    "payroll_tax": {
        "amount_y": "payroll_tax_amount_y",
        "child_tax_credit": {
            "amount_m": "payroll_tax_child_tax_credit_amount_m",
        },
    },
}


EXPECTED_TT_RESULTS = pd.DataFrame(
    {
        "payroll_tax_amount_y": [0.0, 2920.0, 0.0],  # Results for p_id [2, 0, 1]
        "payroll_tax_child_tax_credit_amount_m": [
            0.0,
            8.333333,
            0.0,
        ],  # Results for p_id [2, 0, 1]
    },
    index=pd.Index([2, 0, 1], name="p_id"),  # Matches input order
)


@pytest.mark.parametrize(
    "input_data_arg",
    [
        # Correct way to do it
        InputData.df_and_mapper(df=DF_FOR_MAPPER, mapper=INPUT_DF_MAPPER),
        InputData.df_with_nested_columns(DF_WITH_NESTED_COLUMNS),
        # May or may not continue to work.
        {"df_and_mapper": {"df": DF_FOR_MAPPER, "mapper": INPUT_DF_MAPPER}},
        {"df_with_nested_columns": DF_WITH_NESTED_COLUMNS},
    ],
)
def test_end_to_end(input_data_arg, backend: Literal["numpy", "jax"]):
    result = main(
        main_target=(MainTarget.results.df_with_mapper),
        input_data=input_data_arg,
        tt_targets=TTTargets(tree=TARGETS_TREE),
        policy_date_str="2025-01-01",
        rounding=False,
        orig_policy_objects={"root": middle_earth.ROOT_PATH},
        backend=backend,
    )
    pd.testing.assert_frame_equal(
        EXPECTED_TT_RESULTS,
        result,
        check_dtype=False,
        check_index_type=False,
    )


def test_can_create_input_template(backend: Literal["numpy", "jax"]):
    result_template = main(
        main_target=MainTarget.templates.input_data_dtypes.tree,
        policy_date_str="2025-01-01",
        orig_policy_objects={"root": middle_earth.ROOT_PATH},
        backend=backend,
        tt_targets=TTTargets(tree=TARGETS_TREE),
    )
    flat_result_template = dt.flatten_to_tree_paths(result_template)
    flat_expected = dt.flatten_to_tree_paths(INPUT_DF_MAPPER)
    assert flat_result_template.keys() == flat_expected.keys()


def test_modify_evaluation_date_after_creating_policy_environment(
    backend: Literal["numpy", "jax"],
    xnp: ModuleType,
):
    policy_environment = main(
        main_target=MainTarget.policy_environment,
        policy_date_str="2000-01-01",
        orig_policy_objects={"root": middle_earth.ROOT_PATH},
        backend=backend,
    )
    input_data = InputData.tree(
        tree={
            "p_id": xnp.array([2, 0, 1]),  # Deliberately unsorted
            "property_tax": {
                "acre_size_in_hectares": xnp.array(
                    [200, 5, 20]
                ),  # Reordered to match p_id [2, 0, 1]
            },
        }
    )
    result = main(
        main_target=MainTarget.results.df_with_mapper,
        policy_environment=policy_environment,
        # acre_size_in_hectares capped starting in 2020
        evaluation_date_str="2020-01-01",
        input_data=input_data,
        tt_targets=TTTargets(
            tree={"property_tax": {"amount_y": "property_tax_amount_y"}}
        ),
        backend=backend,
    )
    expected = pd.DataFrame(
        {
            "property_tax_amount_y": [
                1000.0,
                0.0,
                1000.0,
            ],  # Results for p_id [2, 0, 1]
        },
        index=pd.Index([2, 0, 1], name="p_id"),  # Matches input order
    )
    pd.testing.assert_frame_equal(
        expected, result, check_dtype=False, check_index_type=False
    )


def test_different_evaluation_dates_across_data_rows(
    backend: Literal["numpy", "jax"], xnp: ModuleType
):
    @policy_function()
    def f(evaluation_year: int) -> int:
        return evaluation_year

    result = main(
        main_target=MainTarget.results.df_with_nested_columns,
        policy_environment={
            "f": f,
        },
        input_data=InputData.tree(
            tree={
                "p_id": xnp.array([3, 1, 2]),  # Deliberately unsorted
                "evaluation_year": xnp.array(
                    [2022, 2020, 2021]
                ),  # Reordered to match p_id [3, 1, 2]
            }
        ),
        tt_targets=TTTargets(tree={"f": None}),
        backend=backend,
    )

    expected = pd.DataFrame(
        {
            ("f",): [2022, 2020, 2021],  # Results for p_id [3, 1, 2]
        },
        index=pd.Index([3, 1, 2], name="p_id"),  # Matches input order
    )
    pd.testing.assert_frame_equal(
        expected, result, check_dtype=False, check_index_type=False
    )


def test_input_data_as_targets(xnp: ModuleType, backend: Literal["numpy", "jax"]):
    result = main(
        main_target=MainTarget.results.df_with_nested_columns,
        policy_date_str="2025-01-01",
        input_data=InputData.tree(
            {
                "kin_id": xnp.array([0, 0, 0]),
                "payroll_tax": {
                    "amount_y": xnp.array(
                        [0, 1000, 0]
                    ),  # Reordered to match p_id [2, 0, 1]
                },
                "p_id": xnp.array([2, 0, 1]),  # Deliberately unsorted
            }
        ),
        tt_targets=TTTargets(tree={"kin_id": None, "payroll_tax": {"amount_y": None}}),
        orig_policy_objects={"root": middle_earth.ROOT_PATH},
        backend=backend,
        include_warn_nodes=False,
    )
    expected = pd.DataFrame(
        {
            ("kin_id",): [0, 0, 0],
            ("payroll_tax", "amount_y"): [0, 1000, 0],  # Results for p_id [2, 0, 1]
        },
        index=pd.Index([2, 0, 1], name="p_id"),  # Matches input order
    )
    pd.testing.assert_frame_equal(
        expected, result, check_dtype=False, check_index_type=False
    )
