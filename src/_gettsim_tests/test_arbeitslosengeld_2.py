import dags.tree as dt
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from _gettsim.arbeitslosengeld_2.group_by_ids import bg_id, eg_id, fg_id
from _gettsim_tests._helpers import cached_set_up_policy_environment
from _gettsim_tests._policy_test_utils import PolicyTest, load_policy_test_data
from ttsim import compute_taxes_and_transfers

test_data = load_policy_test_data("arbeitslosengeld_2")


@pytest.mark.parametrize("test", test_data)
def test_arbeitslosengeld_2(test: PolicyTest):
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


@pytest.fixture
def ordered_data():
    data = {
        "p_id_einstandspartner": np.asarray([2, -1, 0]),
        "p_id": np.asarray([0, 1, 2]),
        "hh_id": np.asarray([0, 0, 0]),
        "alter": np.asarray([39, 14, 42]),
        "familie__p_id_elternteil_1": np.asarray([-1, 2, -1]),
        "familie__p_id_elternteil_2": np.asarray([-1, -1, -1]),
        "eigenbedarf_gedeckt": np.asarray([False, False, False]),
    }
    return data


def test_fg_id(ordered_data):
    fg_input = ordered_data.copy()
    fg_input.pop("eigenbedarf_gedeckt")
    fg_ids = fg_id(**fg_input)
    np.testing.assert_equal(fg_ids, np.asarray([0, 0, 0]))


def test_bg_id(ordered_data):
    fg_input = ordered_data.copy()
    fg_input.pop("eigenbedarf_gedeckt")
    bg_ids = bg_id(
        fg_id=fg_id(**fg_input),
        eigenbedarf_gedeckt=ordered_data["eigenbedarf_gedeckt"],
        alter=ordered_data["alter"],
    )
    np.testing.assert_equal(bg_ids, np.asarray([0, 0, 0]))


def test_eg_id(ordered_data):
    eg_ids = eg_id(
        p_id_einstandspartner=ordered_data["p_id_einstandspartner"],
        p_id=ordered_data["p_id"],
    )
    np.testing.assert_equal(eg_ids, np.asarray([0, 1, 0]))
