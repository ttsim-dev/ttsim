"""Input columns."""

from ttsim import policy_input


@policy_input()
def bisherige_bezugsmonate() -> int:
    """Number of months the individual received Elterngeld for the current youngest child."""
    return 0


@policy_input()
def claimed() -> bool:
    """Individual claims Elterngeld."""
    return False


@policy_input()
def nettoeinkommen_vorjahr_m() -> float:
    """Net wage in the 12 months before birth of youngest child."""
    return 0


@policy_input()
def zu_versteuerndes_einkommen_vorjahr_y_sn() -> float:
    """Taxable income in the calendar year prior to the youngest child's birth year."""
    return 0
