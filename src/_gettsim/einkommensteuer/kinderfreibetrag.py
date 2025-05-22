"""Tax allowances for individuals or couples with children."""

from __future__ import annotations

from ttsim import AggType, agg_by_p_id_function, policy_function


@agg_by_p_id_function(agg_type=AggType.SUM)
def anzahl_kinderfreibeträge_1(
    kindergeld__grundsätzlich_anspruchsberechtigt: bool,
    p_id_kinderfreibetragsempfänger_1: int,
    p_id: int,
) -> int:
    pass


@agg_by_p_id_function(agg_type=AggType.SUM)
def anzahl_kinderfreibeträge_2(
    kindergeld__grundsätzlich_anspruchsberechtigt: bool,
    p_id_kinderfreibetragsempfänger_2: int,
    p_id: int,
) -> int:
    pass


@policy_function()
def kinderfreibetrag_y(
    anzahl_kinderfreibeträge: int,
    eink_st_abzuege_params: dict,
) -> float:
    """Individual child allowance.

    Parameters
    ----------
    anzahl_kinderfreibeträge
        See :func:`anzahl_kinderfreibeträge`.
    eink_st_abzuege_params
        See params documentation :ref:`eink_st_abzuege_params <eink_st_abzuege_params>`.

    Returns
    -------

    """
    parameter_kinderfreibetrag = list(
        eink_st_abzuege_params["parameter_kinderfreibetrag"].values()
    )
    return sum(parameter_kinderfreibetrag) * anzahl_kinderfreibeträge


@policy_function()
def anzahl_kinderfreibeträge(
    anzahl_kinderfreibeträge_1: int,
    anzahl_kinderfreibeträge_2: int,
) -> int:
    """Return the number of Kinderfreibeträge a person is entitled to.

    The person could be a parent or legal custodian.

    Note: Users should overwrite this function if there are single parents in the data
    who should receive two instead of one Kinderfreibeträge. GETTSIM does not
    automatically do this, even if the p_id of the other parent is set to missing (-1).

    Parameters
    ----------
    anzahl_kinderfreibeträge_1
        See :func:`p_id_kinderfreibetr_empfänger_1 <p_id_kinderfreibetr_empfänger_1>`.
    anzahl_kinderfreibeträge_2
        See :func:`p_id_kinderfreibetr_empfänger_2 <p_id_kinderfreibetr_empfänger_2>`.

    """
    return anzahl_kinderfreibeträge_1 + anzahl_kinderfreibeträge_2


@policy_function()
def p_id_kinderfreibetragsempfänger_1(
    familie__p_id_elternteil_1: int,
) -> int:
    """Assigns child allowance to parent 1.

    Parameters
    ----------
    familie__p_id_elternteil_1
        See :func:`familie__p_id_elternteil_1`.

    Returns
    -------

    """
    return familie__p_id_elternteil_1


@policy_function()
def p_id_kinderfreibetragsempfänger_2(
    familie__p_id_elternteil_2: int,
) -> int:
    """Assigns child allowance to parent 2.

    Parameters
    ----------
    familie__p_id_elternteil_2
        See :func:`familie__p_id_elternteil_2`.

    Returns
    -------

    """
    return familie__p_id_elternteil_2
