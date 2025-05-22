"""Input columns."""

from __future__ import annotations

from ttsim import policy_input


@policy_input()
def alter() -> int:
    """Age in years."""


@policy_input()
def arbeitsstunden_w() -> float:
    """Working hours."""


@policy_input()
def behinderungsgrad() -> int:
    pass


@policy_input()
def geburtsjahr() -> int:
    """Birth year."""


@policy_input()
def geburtsmonat() -> int:
    """Month of birth (within year)."""


@policy_input()
def geburtstag() -> int:
    """Day of birth (within month)."""


@policy_input()
def schwerbehindert_grad_g() -> bool:
    pass


@policy_input()
def vermögen() -> float:
    """Assets for means testing on individual level. {ref}`See this page for more details. <means_testing>`"""


@policy_input()
def weiblich() -> bool:
    """Female."""


@policy_input()
def wohnort_ost() -> bool:
    """Whether the person lives in the Eastern part of Germany."""
