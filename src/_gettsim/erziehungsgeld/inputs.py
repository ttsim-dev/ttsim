"""Input columns."""

from ttsim import policy_input


@policy_input()
def budgetsatz() -> bool:
    """Applied for "Budgetsatz" of parental leave benefit."""


@policy_input()
def p_id_empfänger() -> int:
    pass
