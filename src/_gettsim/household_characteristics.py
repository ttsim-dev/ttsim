from ttsim import AggregateByGroupSpec, AggregationType, policy_function

aggregation_specs = (
    AggregateByGroupSpec(
        target="anzahl_erwachsene_hh",
        source="familie__erwachsen",
        agg=AggregationType.SUM,
    ),
    AggregateByGroupSpec(
        target="anzahl_rentenbezieher_hh",
        source="sozialversicherung__rente__bezieht_rente",
        agg=AggregationType.SUM,
    ),
    AggregateByGroupSpec(
        target="anzahl_personen_hh", source=None, agg=AggregationType.COUNT
    ),
)


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
