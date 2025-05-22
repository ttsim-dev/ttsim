"""Public pension benefits."""

from __future__ import annotations

from ttsim import policy_function


@policy_function()
def alter_bei_renteneintritt(
    jahr_renteneintritt: int,
    monat_renteneintritt: int,
    geburtsjahr: int,
    geburtsmonat: int,
) -> float:
    """Age at retirement in monthly precision.

    Calculates the age of person's retirement in monthly precision.
    As retirement is only possible at first day of month and as
    persons eligible for pension at first of month after reaching the
    age threshold (§ 99 SGB VI) persons who retire in same month will
    be considered a month too young. Hence, subtract 1/12.


    Parameters
    ----------
    geburtsjahr
        See basic input variable :ref:`geburtsjahr <geburtsjahr>`.
    geburtsmonat
        See basic input variable :ref:`geburtsmonat <geburtsmonat>`.
    jahr_renteneintritt
        See basic input variable :ref:`jahr_renteneintritt <jahr_renteneintritt>`.
    monat_renteneintritt
        See basic input variable :ref:`monat_renteneintritt <monat_renteneintritt>`.

    Returns
    -------
    Age at retirement.

    """
    return (
        jahr_renteneintritt
        - geburtsjahr
        + (monat_renteneintritt - geburtsmonat - 1) / 12
    )
