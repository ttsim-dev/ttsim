from __future__ import annotations

from ttsim.tt import FKType, Unit, policy_input


@policy_input(foreign_key_type=FKType.MAY_POINT_TO_SELF, unit=Unit.DIMENSIONLESS)
def p_id_recipient() -> int:
    """Identifier of the recipient of the child tax credit."""
