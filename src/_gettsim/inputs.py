"""Input columns."""

from ttsim import policy_input


@policy_input()
def alter() -> int:
    """Age in years."""
    return 0


@policy_input()
def arbeitsstunden_w() -> float:
    """Working hours."""
    return 0


@policy_input()
def behinderungsgrad() -> int:
    return 0


@policy_input()
def geburtsjahr() -> int:
    """Birth year."""
    return 0


@policy_input()
def geburtsmonat() -> int:
    """Month of birth (within year)."""
    return 0


@policy_input()
def geburtstag() -> int:
    """Day of birth (within month)."""
    return 0


@policy_input()
def hh_id() -> int:
    return 0


@policy_input()
def p_id() -> int:
    return 0


@policy_input()
def schwerbehindert_grad_g() -> bool:
    return 0


@policy_input()
def vermögen() -> float:
    """Assets for means testing on individual level. {ref}`See this page for more details. <means_testing>`"""
    return 0


@policy_input()
def weiblich() -> bool:
    """Female."""
    return False


@policy_input()
def wohnort_ost() -> bool:
    """Whether the person lives in the Eastern part of Germany."""
    return False
