import pytest

from _gettsim.config import FOREIGN_KEYS, SUPPORTED_GROUPINGS
from _gettsim_tests.utils import (
    PolicyTest,
    cached_set_up_policy_environment,
    load_policy_test_data,
)
from ttsim import compute_taxes_and_transfers

test_data = load_policy_test_data("full_taxes_and_transfers")


@pytest.mark.parametrize("test", test_data, ids=lambda x: x.test_name)
def test_full_taxes_transfers(test: PolicyTest):
    environment = cached_set_up_policy_environment(date=test.date)

    compute_taxes_and_transfers(
        data_tree=test.input_tree,
        environment=environment,
        targets_tree=test.target_structure,
        foreign_keys=FOREIGN_KEYS,
        supported_groupings=SUPPORTED_GROUPINGS,
    )


@pytest.mark.skip(
    reason="Got rid of DEFAULT_TARGETS, there might not be a replacement."
)
@pytest.mark.parametrize("test", test_data, ids=lambda x: x.test_name)
def test_allow_none_as_target_tree(test: PolicyTest):
    environment = cached_set_up_policy_environment(date=test.date)

    compute_taxes_and_transfers(
        data_tree=test.input_tree,
        environment=environment,
        targets_tree=None,
        foreign_keys=FOREIGN_KEYS,
        supported_groupings=SUPPORTED_GROUPINGS,
    )
