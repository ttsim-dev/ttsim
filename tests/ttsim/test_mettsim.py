import pytest
from _policy_test_utils import (
    PolicyTest,
    execute_test,
    get_policy_test_ids_and_cases,
)

policy_test_ids_and_cases = get_policy_test_ids_and_cases()


@pytest.mark.parametrize(
    "test",
    policy_test_ids_and_cases.values(),
    ids=policy_test_ids_and_cases.keys(),
)
def test_mettsim(test: PolicyTest):
    execute_test(test)
