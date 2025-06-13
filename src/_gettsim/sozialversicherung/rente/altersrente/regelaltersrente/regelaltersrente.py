"""Regular pathway."""

from __future__ import annotations

from ttsim.tt_dag_elements import ConsecutiveInt1dLookupTableParamValue, policy_function


@policy_function(start_date="2007-04-20", end_date="2030-12-31")
def altersgrenze(
    geburtsjahr: int,
    altersgrenze_gestaffelt: ConsecutiveInt1dLookupTableParamValue,
) -> float:
    """Normal retirement age (NRA) during the phase-in period.

    Just a parameter otherwise.

    The Regelaltersrente cannot be claimed earlier than at the NRA, i.e. the NRA does
    not serve as reference for calculating deductions. However, it serves as reference
    for calculating gains in the Zugangsfakor in case of later retirement.

    Does not check for eligibility for this pathway into retirement.
    """
    return altersgrenze_gestaffelt.values_to_look_up[
        geburtsjahr - altersgrenze_gestaffelt.base_to_subtract
    ]


@policy_function()
def grundsätzlich_anspruchsberechtigt(
    sozialversicherung__rente__mindestwartezeit_erfüllt: bool,
) -> bool:
    """Determining the eligibility for the Regelaltersrente."""
    return sozialversicherung__rente__mindestwartezeit_erfüllt
