from ttsim.aggregation import AggregateByGroupSpec
from ttsim.function_types import policy_function

aggregation_specs = {
    "anzahl_erwachsene_hh": AggregateByGroupSpec(
        source="familie__erwachsen",
        aggr="sum",
    ),
    "anzahl_rentenbezieher_hh": AggregateByGroupSpec(
        source="sozialversicherung__rente__bezieht_rente",
        aggr="sum",
    ),
    "anzahl_personen_hh": AggregateByGroupSpec(
        aggr="count",
    ),
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
