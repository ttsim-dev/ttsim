"""Input columns."""

from ttsim import policy_input


@policy_input()
def alleinerziehend() -> bool:
    """Single parent."""
    return False


@policy_input()
def kind() -> bool:
    """Dependent child living with parents."""
    return False


@policy_input()
def p_id_ehepartner() -> int:
    """Identifier of married partner."""
    return 0


@policy_input()
def p_id_elternteil_1() -> int:
    """Identifier of the first parent."""
    return 0


@policy_input()
def p_id_elternteil_2() -> int:
    """Identifier of the second parent."""
    return 0
