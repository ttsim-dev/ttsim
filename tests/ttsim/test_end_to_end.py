from typing import Literal

import pandas as pd
from mettsim.config import METTSIM_ROOT

from ttsim import main

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


def test_end_to_end(backend: Literal["numpy", "jax"]):
    result = main(
        inputs={
            "input_data__df_and_mapper__df": DF_FOR_MAPPER,
            "input_data__df_and_mapper__mapper": INPUT_DF_MAPPER,
            "targets__tree": TARGETS_TREE,
            "date": "2025-01-01",
            "rounding": False,
            "orig_policy_objects__root": METTSIM_ROOT,
            "backend": backend,
        },
        output_names=["results__df_with_mapper"],
    )
    pd.testing.assert_frame_equal(
        EXPECTED_RESULTS,
        result["results__df_with_mapper"],
        check_dtype=False,
        check_index_type=False,
    )
