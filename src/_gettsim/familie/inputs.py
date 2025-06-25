"""Input columns."""

from __future__ import annotations

from ttsim.tt_dag_elements import FKType, policy_input


@policy_input()
def alleinerziehend() -> bool:
    """Single parent."""


@policy_input()
def kind() -> bool:
    """Dependent child living with parents."""


@policy_input(foreign_key_type=FKType.MUST_NOT_POINT_TO_SELF)
def p_id_ehepartner() -> int:
    """Identifier of married partner."""


@policy_input(foreign_key_type=FKType.MUST_NOT_POINT_TO_SELF)
def p_id_elternteil_1() -> int:
    """Identifier of the first parent."""


@policy_input(foreign_key_type=FKType.MUST_NOT_POINT_TO_SELF)
def p_id_elternteil_2() -> int:
    """Identifier of the second parent."""
