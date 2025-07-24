from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import dags.tree as dt
import pandas as pd
import pytest

from ttsim import InputData, MainTarget, TTTargets, main
from ttsim.tt_dag_elements.column_objects_param_function import policy_function

if TYPE_CHECKING:
    from types import ModuleType
    from typing import Literal


DF_WITH_NESTED_COLUMNS = pd.DataFrame(
    {
        ("age",): [30, 30, 10],
        ("kin_id",): [0, 0, 0],
        ("p_id",): [0, 1, 2],
        ("p_id_parent_1",): [-1, -1, 0],
        ("p_id_parent_2",): [-1, -1, 1],
        ("p_id_spouse",): [1, 0, -1],
        ("parent_is_noble",): [False, False, False],
        ("wealth",): [0.0, 0.0, 0.0],
        ("payroll_tax", "child_tax_credit", "p_id_recipient"): [-1, -1, 0],
        ("payroll_tax", "income", "gross_wage_y"): [10000, 0, 0],
    },
)


DF_FOR_MAPPER = pd.DataFrame(
    {
        "age": [30, 30, 10],
        "kin_id": [0, 0, 0],
        "p_id": [0, 1, 2],
        "parent_1": [-1, -1, 0],
        "parent_2": [-1, -1, 1],
        "spouse": [1, 0, -1],
        "parent_is_noble": [False, False, False],
        "child_tax_credit_recipient": [-1, -1, 0],
        "gross_wage_y": [10000, 0, 0],
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
        "payroll_tax_amount_y": [2920.0, 0.0, 0.0],
        "payroll_tax_child_tax_credit_amount_m": [8.333333, 0.0, 0.0],
    },
    index=pd.Index([0, 1, 2], name="p_id"),
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
        orig_policy_objects={"root": Path(__file__).parent / "mettsim"},
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
        main_target=MainTarget.templates.input_data_dtypes,
        policy_date_str="2025-01-01",
        orig_policy_objects={"root": Path(__file__).parent / "mettsim"},
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
        orig_policy_objects={"root": Path(__file__).parent / "mettsim"},
        backend=backend,
    )
    input_data = InputData.tree(
        tree={
            "p_id": xnp.array([0, 1, 2]),
            "property_tax": {
                "acre_size_in_hectares": xnp.array([5, 20, 200]),
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
            "property_tax_amount_y": [0.0, 1000.0, 1000.0],
        },
        index=pd.Index([0, 1, 2], name="p_id"),
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
                "p_id": xnp.array([1, 2, 3]),
                "evaluation_year": xnp.array([2020, 2021, 2022]),
            }
        ),
        tt_targets=TTTargets(tree={"f": None}),
        backend=backend,
    )

    expected = pd.DataFrame(
        {
            ("f",): [2020, 2021, 2022],
        },
        index=pd.Index([1, 2, 3], name="p_id"),
    )
    pd.testing.assert_frame_equal(
        expected, result, check_dtype=False, check_index_type=False
    )
