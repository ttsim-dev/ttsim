"""Input columns."""

from __future__ import annotations

from ttsim import FKType, policy_input


@policy_input()
def budgetsatz() -> bool:
    """Applied for "Budgetsatz" of parental leave benefit."""


@policy_input(foreign_key_type=FKType.MUST_NOT_POINT_TO_SELF)
def p_id_empfänger() -> int:
    pass
