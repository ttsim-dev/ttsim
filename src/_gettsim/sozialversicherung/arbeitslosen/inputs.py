"""Input columns."""

from __future__ import annotations

from ttsim.tt_dag_elements import policy_input


@policy_input()
def mean_nettoeinkommen_in_12_monaten_vor_arbeitslosigkeit_m() -> float:
    """Mean net wage in the 12 months before unemployment.

    To compute this value using GETTSIM:
    1. Use `arbeitslosengeld__mean_nettoeinkommen_für_bemessungsgrundllage_nach_arbeitslosigkeit_y`
    as the TT target
    2. Apply it to data from the 12 months before the unemployment
    3. Use the result as input for this column.
    """


@policy_input()
def monate_beitragspflichtig_versichert_in_letzten_30_monaten() -> int:
    """Number of months of compulsory insurance in the 30 months before claiming unemployment."""


@policy_input()
def arbeitssuchend() -> bool:
    """Looking for employment."""


@policy_input()
def monate_durchgängigen_bezugs_von_arbeitslosengeld() -> int:
    """Number of months the individual already receives Arbeitslosengeld without interruption."""


@policy_input()
def monate_sozialversicherungspflichtiger_beschäftigung_in_letzten_5_jahren() -> int:
    """Months of subjection to compulsory insurance in the 5 years before claiming unemployment."""
