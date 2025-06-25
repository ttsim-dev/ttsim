"""Input columns."""

from __future__ import annotations

from ttsim.tt_dag_elements import policy_input


@policy_input()
def bisherige_bezugsmonate() -> int:
    """Number of months the individual received Elterngeld for the current youngest child."""


@policy_input()
def claimed() -> bool:
    """Individual claims Elterngeld."""


@policy_input()
def nettoeinkommen_vorjahr_m() -> float:
    """Net wage in the 12 months before birth of youngest child."""


@policy_input()
def zu_versteuerndes_einkommen_vorjahr_y_sn() -> float:
    """Taxable income in the calendar year prior to the youngest child's birth year."""
