from __future__ import annotations

import dags.tree as dt
import pytest
from numpy.testing import assert_array_almost_equal

from _gettsim_tests.utils import (
    PolicyTest,
    cached_set_up_policy_environment,
    load_policy_test_data,
)
from ttsim import compute_taxes_and_transfers

test_data = load_policy_test_data("groupings")


@pytest.mark.parametrize("test", test_data, ids=lambda x: x.name)
def test_groupings(test: PolicyTest):
    environment = cached_set_up_policy_environment(date=test.date)

    result = compute_taxes_and_transfers(
        data_tree=test.input_tree,
        environment=environment,
        targets_tree=test.target_structure,
    )

    flat_result = dt.flatten_to_qual_names(result)
    flat_expected_output_tree = dt.flatten_to_qual_names(test.expected_output_tree)

    for col, actual in flat_result.items():
        assert_array_almost_equal(actual, flat_expected_output_tree[col], decimal=2)
