from dataclasses import dataclass

from ttsim.param_objects import (
    ConsecutiveInt1dLookupTableParamValue,
    ConsecutiveInt2dLookupTableParamValue,
)


@dataclass(frozen=True)
class ElementExistenzminimum:
    single: float
    paar: float
    kind: float


@dataclass(frozen=True)
class ElementExistenzminimumNurKind:
    kind: float


@dataclass(frozen=True)
class ExistenzminimumNachAufwendungenOhneBildungUndTeilhabe:
    regelsatz: ElementExistenzminimum
    kosten_der_unterkunft: ElementExistenzminimum
    heizkosten: ElementExistenzminimum


@dataclass(frozen=True)
class ExistenzminimumNachAufwendungenMitBildungUndTeilhabe:
    regelsatz: ElementExistenzminimum
    kosten_der_unterkunft: ElementExistenzminimum
    heizkosten: ElementExistenzminimum
    bildung_und_teilhabe: ElementExistenzminimumNurKind


@dataclass(frozen=True)
class WohngeldBasisformelParamValues:
    skalierungsfaktor: float
    a: ConsecutiveInt1dLookupTableParamValue
    b: ConsecutiveInt1dLookupTableParamValue
    c: ConsecutiveInt1dLookupTableParamValue
    zusatzbetrag_nach_haushaltsgröße: ConsecutiveInt1dLookupTableParamValue


@dataclass(frozen=True)
class WohngeldMaxMieteNachBaujahrParamValues:
    max_miete_nach_baujahr: dict[int, ConsecutiveInt2dLookupTableParamValue]
