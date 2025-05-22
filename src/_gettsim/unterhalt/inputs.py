"""Input columns."""

from __future__ import annotations

from ttsim import policy_input


@policy_input()
def anspruch_m() -> float:
    """Monthly gross child alimony payments to be received by the child as determined by the court."""


@policy_input()
def tatsächlich_erhaltener_betrag_m() -> float:
    """Child alimony payments the child actually receives."""
