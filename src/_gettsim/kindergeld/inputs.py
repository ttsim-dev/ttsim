"""Input columns."""

from ttsim import policy_input


@policy_input()
def in_ausbildung() -> bool:
    """In education according to Kindergeld definition."""


@policy_input()
def p_id_empfänger() -> int:
    """Identifier of person who receives Kindergeld for the particular child."""
