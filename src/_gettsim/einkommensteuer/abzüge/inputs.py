"""Input columns."""

from ttsim import policy_input


@policy_input()
def beitrag_private_rentenversicherung_m() -> float:
    return 0


@policy_input()
def betreuungskosten_m() -> float:
    """Monthly childcare expenses for a particular child under the age of 14."""
    return 0


@policy_input()
def p_id_betreuungskosten_träger() -> int:
    """Identifier of the person who paid childcare expenses."""
    return 0
