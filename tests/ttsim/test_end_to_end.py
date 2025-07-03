from pathlib import Path
from typing import Literal

import pandas as pd
import pytest

from ttsim import InputData, Output, main

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


EXPECTED_RESULTS = pd.DataFrame(
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
        input_data=input_data_arg,
        targets={"tree": TARGETS_TREE},
        date_str="2025-01-01",
        rounding=False,
        orig_policy_objects={"root": Path(__file__).parent / "mettsim"},
        backend=backend,
        output=Output.name("results__df_with_mapper"),
    )
    pd.testing.assert_frame_equal(
        EXPECTED_RESULTS,
        result,
        check_dtype=False,
        check_index_type=False,
    )
