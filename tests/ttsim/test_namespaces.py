"""Test namespace-specific function processing."""

import importlib

import pandas as pd
import pytest

from ttsim.aggregation import AggregateByGroupSpec, AggregateByPIDSpec
from ttsim.interface import compute_taxes_and_transfers
from ttsim.policy_environment import PolicyEnvironment


@pytest.fixture
def functions_tree():
    module1 = importlib.import_module("namespaces.module1")
    module2 = importlib.import_module("namespaces.module2")
    return {
        **module1.FUNCTIONS,
        **module2.FUNCTIONS,
    }


@pytest.fixture
def parameters():
    return {
        "module1": {
            "a": 1,
            "b": 1,
            "c": 1,
        },
        "module2": {
            "a": 1,
            "b": 1,
            "c": 1,
        },
    }


@pytest.fixture
def aggregation_tree():
    return {
        "module1": {
            "group_mean_hh": AggregateByGroupSpec(
                source="f",
                aggr="sum",
            ),
        },
        "module2": {
            "p_id_aggregation_target": AggregateByPIDSpec(
                p_id_to_aggregate_by="groupings__some_foreign_keys",
                source="g_hh",
                aggr="sum",
            ),
        },
    }


def test_compute_taxes_and_transfers_with_tree(
    functions_tree, parameters, aggregation_tree
):
    """Test compute_taxes_and_transfers with function tree input."""
    policy_env = PolicyEnvironment(
        functions_tree=functions_tree,
        params=parameters,
        aggregation_specs_tree=aggregation_tree,
    )
    targets = {
        "module1": {
            "g_hh": None,
            "group_mean_hh": None,
        },
        "module2": {
            "g_hh": None,
            "p_id_aggregation_target": None,
        },
    }
    data = {
        "p_id": pd.Series([0, 1, 2]),
        "hh_id": pd.Series([0, 0, 1]),
        "familie": {
            "ehe_id": pd.Series([0, 1, 2]),
        },
        "arbeitslosengeld_2": {
            "bg_id": pd.Series([0, 1, 2]),
            "eg_id": pd.Series([0, 1, 2]),
            "fg_id": pd.Series([0, 1, 2]),
        },
        "wohngeld": {
            "wthh_id": pd.Series([0, 1, 2]),
        },
        "einkommensteuer": {
            "sn_id": pd.Series([0, 1, 2]),
        },
        "groupings": {
            "some_foreign_keys": pd.Series([2, 0, 1]),
        },
        "module1": {
            "f": pd.Series([1, 2, 3]),
        },
    }
    compute_taxes_and_transfers(data, policy_env, targets)
