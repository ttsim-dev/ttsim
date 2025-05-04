from _gettsim.config import GETTSIM_ROOT
from ttsim import set_up_policy_environment


def test_parsing_lohnsteuer_rv_anteil():
    environment = set_up_policy_environment(root=GETTSIM_ROOT, date="2018-01-01")

    assert (
        abs(
            environment.params["eink_st_abzuege"]["vorsorgepauschale_rentenv_anteil"]
            - 0.72
        )
        < 1e-6
    )

    environment = set_up_policy_environment(root=GETTSIM_ROOT, date="2023-01-01")

    assert (
        abs(
            environment.params["eink_st_abzuege"]["vorsorgepauschale_rentenv_anteil"]
            - 1
        )
        < 1e-6
    )
