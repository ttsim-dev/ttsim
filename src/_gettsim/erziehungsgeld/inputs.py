"""Input columns."""

from __future__ import annotations

from ttsim.tt_dag_elements import FKType, policy_input


@policy_input(end_date="2008-12-31")
def budgetsatz() -> bool:
    """Applied for "Budgetsatz" of parental leave benefit."""


@policy_input(end_date="2008-12-31", foreign_key_type=FKType.MUST_NOT_POINT_TO_SELF)
def p_id_empfänger() -> int:
    pass
