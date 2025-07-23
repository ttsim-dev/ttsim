"""Input columns."""

from __future__ import annotations

from ttsim.tt_dag_elements import policy_input


@policy_input()
def alle_weiteren_m() -> float:
    """Additional income: includes private and public transfers that are not yet
    implemented in GETTSIM (e.g., BAföG, Kriegsopferfürsorge).

    Excludes income from public and private pensions.
    """
