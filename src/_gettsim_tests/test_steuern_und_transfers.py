import dags.tree as dt
import pytest

from _gettsim_tests._policy_test_utils import PolicyTest, get_policy_test_ids_and_cases, execute_policy_test

policy_test_ids_and_cases = get_policy_test_ids_and_cases()


@pytest.mark.parametrize(
    "test",
    policy_test_ids_and_cases.values(),
    ids=policy_test_ids_and_cases.keys(),
)
def test_policy(test: PolicyTest):
    execute_policy_test(test)
