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
from ttsim.config import numpy_or_jax as np

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

    for result, expected in zip(
        flat_result.values(), flat_expected_output_tree.values()
    ):
        assert_array_almost_equal(result, expected, decimal=2)


def test_fail_to_compute_sn_id_if_married_but_gemeinsam_veranlagt_differs():
    data = {
        "p_id": np.array([0, 1]),
        "familie": {
            "p_id_ehepartner": np.array([1, 0]),
        },
        "einkommensteuer": {
            "gemeinsam_veranlagt": np.array([False, True]),
        },
    }

    environment = cached_set_up_policy_environment("2023-01-01")

    with pytest.raises(
        ValueError,
        match="have different values for gemeinsam_veranlagt",
    ):
        compute_taxes_and_transfers(
            data_tree=data,
            environment=environment,
            targets_tree={"sn_id": None},
        )
