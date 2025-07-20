from pathlib import Path
from typing import Literal

import dags.tree as dt
import pandas as pd
import pytest

from ttsim import InputData, MainTarget, TTTargets, main

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
        date_str="2025-01-01",
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
        date_str="2025-01-01",
        orig_policy_objects={"root": Path(__file__).parent / "mettsim"},
        backend=backend,
        tt_targets=TTTargets(tree=TARGETS_TREE),
    )
    flat_result_template = dt.flatten_to_tree_paths(result_template)
    flat_expected = dt.flatten_to_tree_paths(INPUT_DF_MAPPER)
    assert flat_result_template.keys() == flat_expected.keys()
