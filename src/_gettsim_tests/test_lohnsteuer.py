from ttsim import set_up_policy_environment


def test_lohnsteuer_rv_anteil():
    environment = set_up_policy_environment(2018)

    assert (
        environment.params["eink_st_abzuege"]["vorsorgepauschale_rentenv_anteil"]
        == 0.72
    )

    environment = set_up_policy_environment(2023)

    assert (
        environment.params["eink_st_abzuege"]["vorsorgepauschale_rentenv_anteil"] == 1
    )
