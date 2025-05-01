from ttsim import AggType, agg_by_group_function, policy_function


@agg_by_group_function(agg_type=AggType.SUM)
def anzahl_erwachsene_hh(familie__erwachsen: bool, hh_id: int) -> int:
    pass


@agg_by_group_function(agg_type=AggType.SUM)
def anzahl_rentenbezieher_hh(
    sozialversicherung__rente__bezieht_rente: bool, hh_id: int
) -> int:
    pass


@agg_by_group_function(agg_type=AggType.COUNT)
def anzahl_personen_hh(hh_id: int) -> int:
    pass


@policy_function()
def erwachsene_alle_rentenbezieher_hh(
    anzahl_erwachsene_hh: int, anzahl_rentenbezieher_hh: int
) -> bool:
    """Calculate if all adults in the household are pensioners.

    Parameters
    ----------
    anzahl_erwachsene_hh
        See :func:`anzahl_erwachsene_hh`.
    anzahl_rentenbezieher_hh
        See :func:`anzahl_rentenbezieher_hh`.

    Returns
    -------

    """
    return anzahl_erwachsene_hh == anzahl_rentenbezieher_hh
