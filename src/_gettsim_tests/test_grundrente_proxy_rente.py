from datetime import timedelta

import dags.tree as dt
import pytest
from numpy.testing import assert_array_almost_equal

from _gettsim.config import SUPPORTED_GROUPINGS
from _gettsim_tests.utils import (
    PolicyTest,
    cached_set_up_policy_environment,
    load_policy_test_data,
)
from ttsim import compute_taxes_and_transfers

proxy_rente_test_data = load_policy_test_data(
    "sozialversicherung/rente/grundrente_proxy_rente"
)


@pytest.mark.parametrize("test", proxy_rente_test_data)
def test_grundrente_proxy_rente_vorjahr_comparison(test: PolicyTest):
    environment = cached_set_up_policy_environment(date=test.date)

    result = compute_taxes_and_transfers(
        data_tree=test.input_tree,
        environment=environment,
        targets_tree={
            "sozialversicherung": {
                "rente": {"grundrente": {"proxy_rente_vorjahr_m": None}}
            }
        },
        supported_groupings=SUPPORTED_GROUPINGS,
    )

    # Calculate pension of last year
    environment = cached_set_up_policy_environment(test.date - timedelta(days=365))
    test.input_tree["alter"] -= 1
    result_previous_year = compute_taxes_and_transfers(
        data_tree=test.input_tree,
        environment=environment,
        targets_tree={
            "sozialversicherung": {"rente": {"altersrente": {"bruttorente_m": None}}}
        },
        supported_groupings=SUPPORTED_GROUPINGS,
    )

    flat_result = dt.flatten_to_qual_names(result)
    flat_result_previous_year = dt.flatten_to_qual_names(result_previous_year)
    flat_inputs = dt.flatten_to_qual_names(test.input_tree)
    assert_array_almost_equal(
        flat_result["sozialversicherung__rente__grundrente__proxy_rente_vorjahr_m"],
        flat_result_previous_year[
            "sozialversicherung__rente__altersrente__bruttorente_m"
        ]
        + flat_inputs["sozialversicherung__rente__private_rente_betrag_m"],
        decimal=2,
    )
