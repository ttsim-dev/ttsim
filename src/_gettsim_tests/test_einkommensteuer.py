import pytest

from _gettsim_tests import TEST_DIR
from _gettsim_tests._policy_test_utils import (
    PolicyTest,
    execute_policy_test,
    load_policy_test_data
)

PARAMS_ABZÜGE__VORSORGEAUFWENDUNGEN = load_policy_test_data(TEST_DIR / "test_data" / "einkommensteuer" / "abzüge" / "vorsorgeaufwendungen")

@pytest.mark.parametrize("test", PARAMS_ABZÜGE__VORSORGEAUFWENDUNGEN)
def test_abzüge__vorsorgeaufwendungen(test: PolicyTest):
    execute_policy_test(test)

PARAMS_BETRAG = load_policy_test_data(TEST_DIR / "test_data" / "einkommensteuer" / "betrag")

@pytest.mark.parametrize("test", PARAMS_BETRAG)
def test_betrag(test: PolicyTest):
    execute_policy_test(test)

PARAMS_GÜNSTIGERPRÜFUNGEN = load_policy_test_data(TEST_DIR / "test_data" / "einkommensteuer" / "günstigerprüfungen")

@pytest.mark.parametrize("test", PARAMS_GÜNSTIGERPRÜFUNGEN)
def test_betrag(test: PolicyTest):
    execute_policy_test(test)

PARAMS_ZU_VERSTEUERNDES_EINKOMMEN = load_policy_test_data(TEST_DIR / "test_data" / "einkommensteuer" / "zu_versteuerndes_einkommen")

@pytest.mark.parametrize("test", PARAMS_ZU_VERSTEUERNDES_EINKOMMEN)
def test_betrag(test: PolicyTest):
    execute_policy_test(test)
