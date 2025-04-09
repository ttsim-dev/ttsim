from ttsim import AggregateByGroupSpec, AggregationType, policy_function

aggregation_specs = {
    "anzahl_erwachsene_hh": AggregateByGroupSpec(
        source="familie__erwachsen",
        aggr=AggregationType.SUM,
    ),
    "anzahl_rentenbezieher_hh": AggregateByGroupSpec(
        source="sozialversicherung__rente__bezieht_rente",
        aggr=AggregationType.SUM,
    ),
    "anzahl_personen_hh": AggregateByGroupSpec(aggr=AggregationType.COUNT),
}


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
