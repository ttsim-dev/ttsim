"""Input columns."""

from __future__ import annotations

from ttsim.tt_dag_elements import policy_input


@policy_input()
def monate_beitragspflichtig_versichert_in_letzten_30_monaten() -> int:
    """Number of months of compulsory insurance in the 30 months before claiming unemployment."""


@policy_input()
def arbeitssuchend() -> bool:
    """Looking for employment."""


@policy_input()
def monate_durchgängigen_bezugs_von_arbeitslosengeld() -> float:
    """Number of months the individual already receives Arbeitslosengeld without interruption."""


@policy_input()
def monate_sozialversicherungspflichtiger_beschäftigung_in_letzten_5_jahren() -> float:
    """Months of subjection to compulsory insurance in the 5 years before claiming unemployment."""
