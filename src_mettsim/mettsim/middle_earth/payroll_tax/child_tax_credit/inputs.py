from __future__ import annotations

from ttsim.tt import FKType, policy_input


@policy_input(foreign_key_type=FKType.MAY_POINT_TO_SELF)
def p_id_recipient() -> int:
    """Identifier of the recipient of the child tax credit."""
