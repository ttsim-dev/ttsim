import pytest
from utils import (
    PolicyTest,
    execute_test,
    get_policy_test_ids_and_cases,
)

from ttsim.config import IS_JAX_INSTALLED

policy_test_ids_and_cases = get_policy_test_ids_and_cases()


@pytest.mark.parametrize(
    "test",
    policy_test_ids_and_cases.values(),
    ids=policy_test_ids_and_cases.keys(),
)
def test_mettsim(test: PolicyTest):
    if IS_JAX_INSTALLED:
        execute_test(test, jit=True)
    else:
        execute_test(test, jit=False)
