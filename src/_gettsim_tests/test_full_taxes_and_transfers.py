from __future__ import annotations

from pathlib import Path

import dags.tree as dt
import pytest

from _gettsim.config import GETTSIM_ROOT
from ttsim import main
from ttsim.column_objects_param_function import (
    PolicyInput,
    check_series_has_expected_type,
)
from ttsim.testing_utils import (
    PolicyTest,
    cached_policy_environment,
    load_policy_test_data,
)

TEST_DIR = Path(__file__).parent
POLICY_TEST_IDS_AND_CASES = load_policy_test_data(
    test_dir=TEST_DIR, policy_name="full_taxes_and_transfers"
)


@pytest.mark.parametrize(
    "test", POLICY_TEST_IDS_AND_CASES.values(), ids=POLICY_TEST_IDS_AND_CASES.keys()
)
def test_data_types(test: PolicyTest):
    policy_environment = cached_policy_environment(date=test.date, root=GETTSIM_ROOT)

    qual_name_results = main(
        inputs={
            "data_tree": test.input_tree,
            "policy_environment": policy_environment,
            "targets_tree": test.target_structure,
            "rounding": True,
        },
        targets=["qual_name_results"],
    )["qual_name_results"]

    flat_functions = dt.flatten_to_qual_names(policy_environment)
    flat_types_input_variables = {
        n: pi.data_type
        for n, pi in flat_functions.items()
        if isinstance(pi, PolicyInput)
    }

    for column_name, result_array in qual_name_results.items():
        if column_name in flat_types_input_variables:
            internal_type = flat_types_input_variables[column_name]
        elif column_name in flat_functions:
            internal_type = flat_functions[column_name].__annotations__["return"]
        else:
            # TODO (@hmgaudecker): Implement easy way to find out expected type of
            #     aggregated functions
            # https://github.com/iza-institute-of-labor-economics/gettsim/issues/604
            if column_name.endswith(("_sn", "_hh", "_fg", "_bg", "_eg", "_ehe")):
                internal_type = None
            else:
                raise ValueError(f"Column name {column_name} unknown.")
        if internal_type:
            assert check_series_has_expected_type(result_array, internal_type)
