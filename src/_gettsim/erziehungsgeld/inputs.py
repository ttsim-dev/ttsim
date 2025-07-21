"""Input columns."""

from __future__ import annotations

from ttsim.tt_dag_elements import FKType, policy_input


@policy_input(end_date="2008-12-31")
def bruttolohn_vorjahr_abzüglich_werbungskosten_y() -> float:
    """Gross earnings of the previous year minus Werbungskosten.

    To compute this value using GETTSIM set `('einkommensteuer', 'einkünfte',
    'aus_nichtselbstständiger_arbeit', 'einnahmen_nach_abzug_werbungskosten_y')` as the
    TT target and use input data from the year prior to the youngest child's birth year.
    """


@policy_input(end_date="2008-12-31")
def budgetsatz() -> bool:
    """Applied for "Budgetsatz" of parental leave benefit."""


@policy_input(end_date="2008-12-31", foreign_key_type=FKType.MUST_NOT_POINT_TO_SELF)
def p_id_empfänger() -> int:
    pass
