import pytest

from _gettsim_tests import TEST_DIR
from _gettsim_tests._policy_test_utils import (
    PolicyTest,
    execute_policy_test,
    load_policy_test_data
)

PARAMS = load_policy_test_data(TEST_DIR / "test_data" / "arbeitslosengeld_2")

@pytest.mark.parametrize("test", PARAMS)
def test(test: PolicyTest):
    execute_policy_test(test)